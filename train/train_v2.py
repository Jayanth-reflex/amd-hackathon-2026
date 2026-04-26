"""Production LoRA training v2 — uses the path proven by smoke gate.

Skips Unsloth (which had compat issues with qwen3_5_moe_text) in favor of
the vanilla peft + transformers-from-main path that the H+2 smoke gate
verified works at seq=4096, bsz=1, grad_accum=8.

Loads `train/config.yaml`, reads /data/hf/blend_combined.jsonl, trains 2
epochs with W&B logging, mid-eval checkpoints at 25/50/75/100%, and a
loss-spike kill-switch.

Usage on droplet (inside vllm-openai-rocm:v0.17.1 docker w/ transformers
from main installed):
    docker run --rm \\
      --device=/dev/kfd --device=/dev/dri --group-add video \\
      --shm-size 16g --ipc=host \\
      -v /data:/data -v /root:/host_root \\
      -e HF_TOKEN=$(cat /data/secrets/hf_token) \\
      -e WANDB_API_KEY=$(cat /data/secrets/wandb_key) \\
      -e PYTORCH_ALLOC_CONF=expandable_segments:True \\
      -e UNSLOTH_COMPILE_DISABLE=1 \\
      --entrypoint=/bin/bash \\
      vllm/vllm-openai-rocm:v0.17.1 \\
      -c "pip install --quiet --upgrade git+https://github.com/huggingface/transformers.git peft datasketch && python /host_root/amd-hackathon-2026/train/train_v2.py"
"""
from __future__ import annotations

import os
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_v2")


class JSONLDataset(Dataset):
    """Pre-tokenized blend.jsonl dataset (each row has 'text' and 'n_tokens')."""

    def __init__(self, path: str, tokenizer, max_len: int):
        self.records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("n_tokens", 0) > 16:
                    self.records.append(rec)
        log.info("loaded %d rows from %s", len(self.records), path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        text = self.records[idx]["text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        ids = enc["input_ids"]
        return {"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": ids[:]}


class DataCollator:
    """Pad to max length in batch, mask pad tokens in labels."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            n = max_len - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * n)
            out["attention_mask"].append(b["attention_mask"] + [0] * n)
            out["labels"].append(b["labels"] + [-100] * n)
        return {k: torch.tensor(v) for k, v in out.items()}


class LossSpikeKillSwitch(TrainerCallback):
    def __init__(self, rolling: int, baseline: int, ratio: float):
        self.rolling = rolling
        self.baseline = baseline
        self.ratio = ratio
        self._losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        self._losses.append(float(logs["loss"]))
        if len(self._losses) < self.baseline + self.rolling:
            return
        recent = sum(self._losses[-self.rolling:]) / self.rolling
        prior = sum(self._losses[-self.baseline - self.rolling : -self.rolling]) / self.baseline
        if recent > self.ratio * prior:
            log.error("LOSS-SPIKE: recent=%.4f prior=%.4f ratio=%.2f — interrupting",
                      recent, prior, recent / prior)
            raise KeyboardInterrupt("loss spike kill-switch")


class MidEvalMarker(TrainerCallback):
    """Drops a marker file at 25/50/75/100% so external eval can hook in."""

    def __init__(self, fractions: list[float], output_dir: str):
        self.fractions = sorted(set(fractions))
        self.output_dir = Path(output_dir)
        self._fired: set[float] = set()

    def on_step_end(self, args, state, control, **kwargs):
        total = state.max_steps
        if total <= 0:
            return
        frac = state.global_step / total
        for f in self.fractions:
            if f in self._fired or frac < f:
                continue
            self._fired.add(f)
            marker = self.output_dir / f"mideval_marker_{int(f * 100)}.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({
                "step": state.global_step,
                "frac": f,
                "ts": time.time(),
                "log_history_tail": state.log_history[-5:] if state.log_history else [],
            }))
            log.info("[mid-eval marker] %.0f%% at step %d", f * 100, state.global_step)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    ap.add_argument("--data", default="/data/hf/blend_combined.jsonl")
    ap.add_argument("--base-model-path", default="/data/hf/Qwen3.6-35B-A3B")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    log.info("cfg: %s", json.dumps({k: v for k, v in cfg.items() if k != "blend"}, indent=2)[:2000])

    log.info("[1/5] loading tokenizer + model from %s", args.base_model_path)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=cfg["loader"].get("attn_implementation", "eager"),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    log.info("  loaded in %.1fs, mem=%.1f GB", time.time() - t0,
             torch.cuda.memory_allocated() / 1e9)

    log.info("[2/5] attaching LoRA")
    peft_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    log.info("[3/5] loading dataset %s", args.data)
    ds = JSONLDataset(args.data, tok, cfg["loader"]["max_seq_length"])

    log.info("[4/5] building TrainingArguments")
    out_dir = cfg["output"]["local_dir"]
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_steps=cfg["training"]["warmup_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        optim=cfg["training"]["optim"],
        weight_decay=cfg["training"]["weight_decay"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        dataloader_num_workers=cfg["training"]["dataloader_num_workers"],
        group_by_length=cfg["training"]["group_by_length"],
        report_to=cfg["training"].get("report_to", []),
        run_name=cfg["training"]["run_name"],
        seed=cfg["training"]["seed"],
        gradient_checkpointing=False,  # we already enabled on the model
    )

    callbacks: list[TrainerCallback] = [
        LossSpikeKillSwitch(
            rolling=cfg["loss_spike"]["rolling_mean_window"],
            baseline=cfg["loss_spike"]["comparison_window"],
            ratio=cfg["loss_spike"]["ratio_threshold"],
        ),
        MidEvalMarker(
            fractions=cfg["mid_eval"]["checkpoint_fractions"],
            output_dir=out_dir,
        ),
    ]

    log.info("[5/5] starting training")
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        processing_class=tok,
        data_collator=DataCollator(pad_id=tok.pad_token_id or tok.eos_token_id),
        callbacks=callbacks,
    )

    try:
        out = trainer.train()
    except KeyboardInterrupt:
        log.error("training interrupted (loss-spike or operator)")
        return 3

    log.info("training done: %s", out.metrics)
    log.info("saving adapter to %s", out_dir)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    log.info("DONE — total time %.1f hours", (time.time() - t0) / 3600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
