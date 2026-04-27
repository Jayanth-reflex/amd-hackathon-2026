"""Merge LoRA adapter into base + push BF16 to HF Hub (peft path, not unsloth).

Runs after training_v2.py completes. Output: `Reflex-jr/Qwen3.6-35B-A3B-Domain`
(Apache-2.0, pre-Heretic). The downstream Heretic step then produces the
`-Aggressive` variant.

Usage on droplet (inside vllm-openai-rocm:v0.17.1 docker w/ transformers main):
    docker run --rm \\
      --device=/dev/kfd --device=/dev/dri --group-add video \\
      --shm-size 16g --ipc=host \\
      -v /data:/data -v /root:/host_root \\
      -e HF_TOKEN=$(cat /data/secrets/hf_token) \\
      -e PYTORCH_ALLOC_CONF=expandable_segments:True \\
      --entrypoint=/bin/bash \\
      vllm/vllm-openai-rocm:v0.17.1 \\
      -c "pip install --quiet --upgrade git+https://github.com/huggingface/transformers.git peft hf_transfer && \\
          HF_HUB_ENABLE_HF_TRANSFER=1 python /host_root/amd-hackathon-2026/train/merge_and_push_v2.py"
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("merge_and_push_v2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    ap.add_argument("--base-model-path", default="/data/hf/Qwen3.6-35B-A3B")
    ap.add_argument("--adapter-dir", default=None, help="default: cfg.output.local_dir")
    ap.add_argument("--out-dir", default="/data/out/merged")
    ap.add_argument("--repo", default=None, help="Override cfg.output.hf_repo")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--private", action="store_true", help="push as private repo")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adapter_dir = args.adapter_dir or cfg["output"]["local_dir"]
    repo = args.repo or cfg["output"]["hf_repo"]

    log.info("[1/4] loading base model from %s", args.base_model_path)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
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

    log.info("[2/4] attaching LoRA adapter from %s", adapter_dir)
    t1 = time.time()
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    log.info("  adapter attached in %.1fs", time.time() - t1)

    log.info("[3/4] merging LoRA into base weights")
    t2 = time.time()
    model = model.merge_and_unload()
    log.info("  merged in %.1fs", time.time() - t2)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("[4/4] saving merged model to %s (5GB shards, safetensors)", out)
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
