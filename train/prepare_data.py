"""Dataset blend assembler for amd-hackathon-2026.

Pulls the §E source mix, MinHash-dedupes, applies a ChatML wrap, and writes
a single JSONL ready for `train_lora.py`. Designed to run on CPU (off the
GPU clock) per the §G timeline H+3 step.

Usage:
    python train/prepare_data.py \\
        --out /data/hf/blend.jsonl \\
        --target-tokens 200_000_000 \\
        --tokenizer unsloth/Qwen3.6-35B-A3B
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, load_dataset
from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_data")

# §E source mix — shares are pre-dedupe approximations.
SOURCES: dict[str, dict] = {
    # General instruct (~65%)
    "teknium/OpenHermes-2.5":                      {"share": 0.12, "split": "train"},
    "allenai/tulu-3-sft-mixture":                  {"share": 0.10, "split": "train"},
    "HuggingFaceH4/ultrachat_200k":                {"share": 0.08, "split": "train_sft"},
    "OpenAssistant/oasst2":                        {"share": 0.05, "split": "train"},
    "allenai/WildChat-1M":                         {"share": 0.06, "split": "train"},
    "cognitivecomputations/dolphin-r1":            {"share": 0.06, "split": "reasoning-deepseek"},
    "Open-Orca/SlimOrca":                          {"share": 0.05, "split": "train"},
    "LDJnr/Capybara":                              {"share": 0.04, "split": "train"},
    "m-a-p/CodeFeedback-Filtered-Instruction":     {"share": 0.03, "split": "train"},
    "cognitivecomputations/SystemChat-2.0":        {"share": 0.02, "split": "train"},
    "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered": {"share": 0.02, "split": "train"},
    "jondurbin/airoboros-3.2":                     {"share": 0.02, "split": "train"},
    # Reasoning / math (~8%)
    "meta-math/MetaMathQA":                        {"share": 0.03, "split": "train"},
    "microsoft/orca-math-word-problems-200k":      {"share": 0.03, "split": "train"},
    "TIGER-Lab/MathInstruct":                      {"share": 0.02, "split": "train"},
    # Tool calling (~7%)
    "NousResearch/hermes-function-calling-v1":     {"share": 0.03, "split": "train"},
    "Salesforce/xlam-function-calling-60k":        {"share": 0.02, "split": "train"},
    "glaiveai/glaive-function-calling-v2":         {"share": 0.02, "split": "train"},
    # Domain (~20%) — built locally; see docs/DATASETS.md for sourcing.
    # Add custom local datasets via --extra-jsonl flag at runtime.
}


def chatml_wrap(messages: list[dict]) -> str:
    """Wrap a [{role,content}, ...] list into Qwen3 ChatML format."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(out) + "\n"


def normalize_record(record: dict, source: str) -> str | None:
    """Best-effort normalize a heterogeneous record into a ChatML string.

    Returns None if the record can't be cleanly extracted.
    """
    if "messages" in record and isinstance(record["messages"], list):
        return chatml_wrap(record["messages"])
    if "conversations" in record and isinstance(record["conversations"], list):
        msgs = []
        for c in record["conversations"]:
            role = c.get("from", "user")
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "model"):
                role = "assistant"
            elif role == "system":
                role = "system"
            msgs.append({"role": role, "content": c.get("value", "")})
        return chatml_wrap(msgs)
    if "instruction" in record and "output" in record:
        msgs = [{"role": "user", "content": record["instruction"]},
                {"role": "assistant", "content": record["output"]}]
        if record.get("input"):
            msgs[0]["content"] += "\n\n" + record["input"]
        return chatml_wrap(msgs)
    if "prompt" in record and "response" in record:
        return chatml_wrap([{"role": "user", "content": record["prompt"]},
                            {"role": "assistant", "content": record["response"]}])
    return None


def minhash_dedupe(records: Iterator[str], threshold: float = 0.85, num_perm: int = 128) -> Iterator[str]:
    """MinHash LSH dedupe; yields kept records."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    seen = 0
    kept = 0
    for i, text in enumerate(records):
        seen += 1
        m = MinHash(num_perm=num_perm)
        for token in set(text.lower().split()):
            m.update(token.encode("utf-8"))
        key = f"r{i}"
        if not lsh.query(m):
            lsh.insert(key, m)
            kept += 1
            yield text
        if seen % 50000 == 0:
            log.info("dedupe progress: seen=%d kept=%d (%.1f%%)", seen, kept, 100.0 * kept / seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--target-tokens", type=int, default=200_000_000)
    ap.add_argument("--tokenizer", default="unsloth/Qwen3.6-35B-A3B")
    ap.add_argument("--extra-jsonl", nargs="*", default=[], help="Local domain JSONLs to mix in")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/data/hf"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)

    def gen_all() -> Iterator[str]:
        for ds_id, meta in SOURCES.items():
            log.info("loading %s [%s]", ds_id, meta["split"])
            try:
                ds = load_dataset(ds_id, split=meta["split"], streaming=True, cache_dir=args.cache_dir)
            except Exception as e:  # network / gated / not-found
                log.warning("skipping %s: %s", ds_id, e)
                continue
            for rec in ds:
                text = normalize_record(rec, ds_id)
                if text:
                    yield text
        for path in args.extra_jsonl:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    text = normalize_record(rec, path)
                    if text:
                        yield text

    log.info("starting dedupe + tokenization-bound write to %s (target %.1fM tokens)",
             out, args.target_tokens / 1e6)
    tokens_written = 0
    written = 0
    with open(out, "w", encoding="utf-8") as fo:
        for text in minhash_dedupe(gen_all()):
            ids = tokenizer.encode(text, add_special_tokens=False)
            n = len(ids)
            if n < 32 or n > 8192:
                continue
            fo.write(json.dumps({"text": text, "n_tokens": n,
                                 "hash": hashlib.sha1(text.encode()).hexdigest()[:16]}) + "\n")
            tokens_written += n
            written += 1
            if written % 10000 == 0:
                log.info("written: rows=%d tokens=%dM", written, tokens_written // 1_000_000)
            if tokens_written >= args.target_tokens:
                break

    log.info("done. rows=%d tokens=%d (%.1fM)", written, tokens_written, tokens_written / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
