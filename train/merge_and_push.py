"""Merge LoRA adapter into base + push BF16 to HF Hub.

Runs at H+22 in the §G timeline. Output: `Reflex-jr/Qwen3.6-35B-A3B-Domain`
(Apache-2.0, pre-Heretic). The downstream Heretic step then produces the
`-Aggressive` variant.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("merge_and_push")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--out-dir", default="/workspace/out/merged")
    ap.add_argument("--repo", default=None, help="Override config.output.hf_repo")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adapter_dir = args.adapter_dir or cfg["output"]["local_dir"]
    repo = args.repo or cfg["output"]["hf_repo"]

    from unsloth import FastVisionModel
    log.info("loading base + adapter from %s", adapter_dir)
    model, processor = FastVisionModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=cfg["loader"]["max_seq_length"],
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    log.info("merging LoRA into base weights")
    model = model.merge_and_unload()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("saving merged model to %s", out)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(str(out))

    if args.no_push:
        log.info("--no-push set, skipping HF upload")
        return 0

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    log.info("creating + pushing repo: %s (Apache-2.0, public)", repo)
    create_repo(repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo,
        repo_type="model",
        commit_message=f"merge LoRA adapter into {cfg['base_model']} (pre-Heretic)",
    )
    log.info("done. browse: https://huggingface.co/%s", repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
