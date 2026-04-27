"""Merge LoRA adapter into base + push BF16 to HF Hub.

v3: bypasses `PeftModel.from_pretrained` because that path goes through
`convert_peft_adapter_state_dict_for_transformers` which is broken on the
peft↔transformers-main combo we need (peft passes `distributed_operation`
kwarg that transformers main's WeightConverter doesn't accept).

Workaround: recreate the LoRA wrapper using `get_peft_model(base, cfg)`
(the same call training used — works fine), then load adapter weights
directly with `load_state_dict(strict=False)`. This bypasses the conversion.
"""
from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("merge_and_push_v3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    ap.add_argument("--base-model-path", default="/data/hf/Qwen3.6-35B-A3B")
    ap.add_argument("--adapter-dir", default=None, help="default: cfg.output.local_dir")
    ap.add_argument("--out-dir", default="/data/out/merged")
    ap.add_argument("--repo", default=None, help="Override cfg.output.hf_repo")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adapter_dir = Path(args.adapter_dir or cfg["output"]["local_dir"])
    repo = args.repo or cfg["output"]["hf_repo"]

    log.info("[1/5] loading base model from %s", args.base_model_path)
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
        attn_implementation="eager",
    )
    log.info("  base loaded in %.1fs, mem=%.1f GB",
             time.time() - t0, torch.cuda.memory_allocated() / 1e9)

    log.info("[2/5] re-attaching LoRA wrapper from training config")
    t1 = time.time()
    peft_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, peft_cfg)
    log.info("  wrapper attached in %.1fs", time.time() - t1)

    log.info("[3/5] loading adapter weights from %s", adapter_dir)
    sf = adapter_dir / "adapter_model.safetensors"
    sd = load_safetensors(str(sf))
    log.info("  loaded %d adapter tensors from safetensors", len(sd))

    # Adapter saved as 'base_model.model.model.layers.X.q_proj.lora_A.weight'.
    # After get_peft_model the live model expects '...lora_A.default.weight'.
    # Inject '.default' before '.weight' to match.
    converted = {}
    for k, v in sd.items():
        if k.endswith(".weight") and (".lora_A." in k or ".lora_B." in k) and ".default." not in k:
            new_k = k[:-7] + ".default.weight"
        else:
            new_k = k
        converted[new_k] = v

    missing, unexpected = model.load_state_dict(converted, strict=False)
    # Filter expected base-weight misses: model has thousands of base params not in the adapter file
    lora_missing = [m for m in missing if ".lora_" in m]
    if lora_missing:
        log.error("LoRA params still missing after load: %d (first 5: %s)",
                  len(lora_missing), lora_missing[:5])
        return 2
    if unexpected:
        log.warning("unexpected keys: %d (first 5: %s)", len(unexpected), unexpected[:5])
    log.info("  state_dict loaded — base misses are expected (only LoRA weights are in the adapter)")

    log.info("[4/5] merging LoRA into base weights")
    t2 = time.time()
    model = model.merge_and_unload()
    log.info("  merged in %.1fs", time.time() - t2)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("[5/5] saving merged model to %s (5GB shards)", out)
    t3 = time.time()
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(str(out))
    log.info("  saved in %.1fs", time.time() - t3)

    if args.no_push:
        log.info("--no-push set, skipping HF upload")
        return 0

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    log.info("creating + pushing repo: %s (private=%s)", repo, args.private)
    create_repo(repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo,
        repo_type="model",
        commit_message=f"merge LoRA adapter into {cfg['base_model']} (pre-Heretic)",
    )
    log.info("DONE. total time %.1f min", (time.time() - t0) / 60)
    log.info("browse: https://huggingface.co/%s", repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
