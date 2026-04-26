"""Unsloth bf16 LoRA training for amd-hackathon-2026.

Loads `train/config.yaml`, fine-tunes `unsloth/Qwen3.6-35B-A3B` (or fallback
ladder) on the §E blend using FastVisionModel, with mid-eval callback,
loss-spike kill-switch, and W&B+TensorBoard logging.

Usage:
    # H+2 smoke gate
    python train/train_lora.py --config train/config.yaml --smoke

    # Full run (H+4 onward)
    python train/train_lora.py --config train/config.yaml \\
        --data /data/hf/blend.jsonl
"""
from __future__ import annotations

# CRITICAL: these env vars MUST be set before any import that pulls torch/unsloth.
import os
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")        # fixes mat1/mat2 dtype mismatch on MoE+LoRA
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import TrainerCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_lora")


def load_model_with_fallback(cfg: dict):
    """Try base_model, fall back through fallback_models on load failure."""
    from unsloth import FastVisionModel
    candidates = [cfg["base_model"], *cfg.get("fallback_models", [])]
    last_err: Exception | None = None
    for name in candidates:
        log.info("loading model: %s", name)
        try:
            model, processor = FastVisionModel.from_pretrained(
                model_name=name,
                max_seq_length=cfg["loader"]["max_seq_length"],
                load_in_4bit=cfg["loader"]["load_in_4bit"],
                load_in_16bit=cfg["loader"]["load_in_16bit"],
                full_finetuning=cfg["loader"]["full_finetuning"],
            )
            log.info("loaded: %s", name)
            return model, processor, name
        except Exception as e:
            log.warning("load failed for %s: %s", name, e)
            last_err = e
    raise RuntimeError(f"all candidates failed; last_err={last_err}")


def attach_lora(model, cfg: dict):
    from unsloth import FastVisionModel
    return FastVisionModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
        random_state=cfg["lora"]["random_state"],
        max_seq_length=cfg["loader"]["max_seq_length"],
    )


class LossSpikeKillSwitch(TrainerCallback):
    """Halts training if rolling-mean loss spikes vs a longer baseline.

    Triggers `KeyboardInterrupt` so the Trainer dumps the optimizer state and
    the operator can decide: lower LR + resume, or revert to last checkpoint.
    """

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
            log.error("LOSS-SPIKE KILL-SWITCH: recent=%.4f prior=%.4f ratio=%.2f", recent, prior, recent / prior)
            raise KeyboardInterrupt("loss spike kill-switch")


class MidEvalCallback(TrainerCallback):
    """Runs a domain-held-out eval at each fractional checkpoint."""

    def __init__(self, fractions: list[float], held_out_path: str, max_new_tokens: int, processor):
        self.fractions = sorted(set(fractions))
        self.held_out_path = held_out_path
        self.max_new_tokens = max_new_tokens
        self.processor = processor
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
            log.info("mid-eval @ frac=%.2f (step %d/%d)", f, state.global_step, total)
            # NOTE: full eval implementation runs offline against domain_qa.jsonl;
            # this callback only marks the checkpoint for the eval harness to pick up.
            Path(args.output_dir, f"mideval_marker_{int(f*100)}.json").write_text(
                json.dumps({"step": state.global_step, "frac": f, "ts": time.time()})
            )


def build_dataset(jsonl_path: str, processor, max_seq_length: int) -> Dataset:
    log.info("loading dataset from %s", jsonl_path)
    ds = load_dataset("json", data_files=jsonl_path, split="train")

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_seq_length,
                        padding=False, return_tensors=None)
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names, num_proc=1)
    log.info("dataset size: %d rows", len(ds))
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    ap.add_argument("--data", default="/data/hf/blend.jsonl",
                    help="Dataset JSONL produced by train/prepare_data.py")
    ap.add_argument("--smoke", action="store_true", help="H+2 smoke gate: 100 steps on 1k rows")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Smoke override
    if args.smoke:
        log.info("=== H+2 SMOKE GATE ===")
        cfg["training"]["max_steps"] = cfg["smoke"]["steps"]
        cfg["training"]["num_train_epochs"] = 1
        cfg["training"]["save_steps"] = 1000  # don't save in smoke
        cfg["training"]["logging_steps"] = 1

    model, processor, model_name = load_model_with_fallback(cfg)
    model = attach_lora(model, cfg)

    # Verify (c) processor.tokenizer extracts cleanly — H+2 gate (c)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    assert tokenizer is not None, "processor.tokenizer extraction FAILED — H+2 gate (c)"
    log.info("tokenizer ok: %s", type(tokenizer).__name__)

    # Memory check (b)
    mem_gb = torch.cuda.memory_allocated() / 1e9
    log.info("post-load GPU memory: %.1f GB", mem_gb)
    if mem_gb > cfg["smoke"]["max_memory_gb"]:
        log.error("H+2 gate (b) FAILED: memory %.1f GB > %d GB", mem_gb, cfg["smoke"]["max_memory_gb"])
        return 2

    # Build dataset
    if args.smoke:
        ds = load_dataset("json", data_files=args.data, split="train").select(range(min(cfg["smoke"]["rows"], 1000)))
        ds = build_dataset(args.data, processor, cfg["loader"]["max_seq_length"]).select(
            range(min(cfg["smoke"]["rows"], 1000))
        )
    else:
        ds = build_dataset(args.data, processor, cfg["loader"]["max_seq_length"])

    # SFTConfig + SFTTrainer — Unsloth-optimized
    from trl import SFTConfig, SFTTrainer
    sft_args = SFTConfig(
        output_dir=cfg["output"]["local_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_steps=cfg["training"]["warmup_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        max_steps=cfg["training"].get("max_steps", -1),
        optim=cfg["training"]["optim"],
        weight_decay=cfg["training"]["weight_decay"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        dataloader_num_workers=cfg["training"]["dataloader_num_workers"],
        dataset_num_proc=cfg["training"]["dataset_num_proc"],
        group_by_length=cfg["training"]["group_by_length"],
        report_to=cfg["training"]["report_to"],
        run_name=cfg["training"]["run_name"],
        seed=cfg["training"]["seed"],
    )

    callbacks: list[TrainerCallback] = []
    if cfg.get("loss_spike", {}).get("rolling_mean_window"):
        callbacks.append(LossSpikeKillSwitch(
            rolling=cfg["loss_spike"]["rolling_mean_window"],
            baseline=cfg["loss_spike"]["comparison_window"],
            ratio=cfg["loss_spike"]["ratio_threshold"],
        ))
    if cfg.get("mid_eval", {}).get("enabled"):
        callbacks.append(MidEvalCallback(
            fractions=cfg["mid_eval"]["checkpoint_fractions"],
            held_out_path=cfg["mid_eval"]["held_out_path"],
            max_new_tokens=cfg["mid_eval"]["max_new_tokens"],
            processor=processor,
        ))

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    log.info("training start: model=%s smoke=%s", model_name, args.smoke)
    initial_loss: float | None = None
    try:
        result = trainer.train()
    except KeyboardInterrupt:
        log.error("training interrupted (loss-spike kill-switch or operator)")
        return 3

    log.info("training done: %s", result.metrics)

    # H+2 gate (a): loss decreased monotonically (rough check)
    if args.smoke:
        history = trainer.state.log_history
        losses = [r["loss"] for r in history if "loss" in r]
        if len(losses) >= 10:
            first10 = sum(losses[:10]) / 10
            last10 = sum(losses[-10:]) / 10
            log.info("H+2 gate (a): first10_mean=%.4f last10_mean=%.4f", first10, last10)
            if last10 >= first10:
                log.error("H+2 gate (a) FAILED: loss did not decrease")
                return 2

    # Save adapter
    trainer.save_model(cfg["output"]["local_dir"])
    log.info("adapter saved to %s", cfg["output"]["local_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
