"""Stage-1 domain-source fetcher v4 — targets ~30M domain tokens.

Reads `train/sources_v4.yaml` manifest. For each entry, dispatches to the
right fetcher based on `type`:
  - hf          → load_dataset streaming
  - pg_txt      → urllib download of Project Gutenberg .txt
  - wikipedia   → MediaWiki API extract action

Output: appends new ChatML-wrapped JSONL records to /data/hf/blend.jsonl
(or a separate output file if --out is given). Existing records are
preserved (this is additive).

Usage on droplet:
    HF_TOKEN=$(cat /data/secrets/hf_token) python3 train/fetch_v4.py \\
        --manifest train/sources_v4.yaml \\
        --raw-dir /data/hf/blend_raw_v4 \\
        --out /data/hf/blend_v4_addon.jsonl \\
        --tokenizer /data/hf/Qwen3.6-35B-A3B
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_v4")

UA = "amd-hackathon-2026/1.0 (Jayanth-reflex; jayanthreddy268.jr@gmail.com)"


def chatml_wrap(messages: list[dict]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(out) + "\n"


def normalize_record(record: dict) -> str | None:
    """Same normalize_record as prepare_data.py — handles 8 schemas."""
    if "messages" in record and isinstance(record["messages"], list):
        return chatml_wrap(record["messages"])
    if "conversations" in record and isinstance(record["conversations"], list):
        msgs = []
        if "system" in record and record.get("system"):
            sys_content = str(record["system"])
            if "tools" in record and record.get("tools"):
                sys_content += "\n\nAvailable tools:\n" + str(record["tools"])
            msgs.append({"role": "system", "content": sys_content})
        for c in record["conversations"]:
            role = c.get("from", c.get("role", "user"))
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "model"):
                role = "assistant"
            content = c.get("value", c.get("content", ""))
            msgs.append({"role": role, "content": content})
        return chatml_wrap(msgs) if msgs else None
    if "instruction" in record and "output" in record:
        msgs = [
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["output"]},
        ]
        if record.get("input"):
            msgs[0]["content"] += "\n\n" + record["input"]
        return chatml_wrap(msgs)
    if "prompt" in record and "response" in record:
        return chatml_wrap([
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ])
    if "query" in record and "response" in record:
        return chatml_wrap([
            {"role": "user", "content": record["query"]},
            {"role": "assistant", "content": record["response"]},
        ])
    if "question" in record and "answer" in record:
        return chatml_wrap([
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": str(record["answer"])},
        ])
    if "Instruction" in record and "Response" in record:
        return chatml_wrap([
            {"role": "user", "content": record["Instruction"]},
            {"role": "assistant", "content": record["Response"]},
        ])
    if "chat" in record and isinstance(record["chat"], str):
        return chatml_wrap([{"role": "assistant", "content": record["chat"]}])
    if "text" in record and isinstance(record["text"], str):
        return chatml_wrap([{"role": "assistant", "content": record["text"]}])
    if "Text" in record and "Summary" in record:
        return chatml_wrap([
            {"role": "user", "content": "Summarize this:\n" + record["Text"]},
            {"role": "assistant", "content": record["Summary"]},
        ])
    # InJudgements: 'IndianKanoon URL', 'full_text'
    if "full_text" in record and isinstance(record["full_text"], str):
        return chatml_wrap([{"role": "assistant", "content": record["full_text"]}])
    return None


# ============================================================================
# Fetchers
# ============================================================================

def fetch_hf(src: dict, out_path: Path, target_tokens: int) -> int:
    """Stream HF dataset → write JSONL."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return _approx_tokens(out_path)
    from datasets import load_dataset

    src_id = src["id"]
    split = src.get("split", "train")
    log.info("  hf: %s [%s] target=%d", src_id, split, target_tokens)

    ds = None
    last_err: Exception | None = None
    for attempt_label, attempt in (
        ("split", lambda: load_dataset(src_id, split=split, streaming=True)),
        ("config+split=train", lambda: load_dataset(src_id, name=split, split="train", streaming=True)),
        ("split+trust", lambda: load_dataset(src_id, split=split, streaming=True, trust_remote_code=True)),
    ):
        try:
            ds = attempt()
            log.info("    loaded with %s", attempt_label)
            break
        except (TypeError, ValueError) as e:
            last_err = e
        except Exception as e:
            last_err = e
            break
    if ds is None:
        log.warning("  hf FAILED: %s — %s", src_id, last_err)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_chars, n_rows = 0, 0
    char_target = target_tokens * 4
    filter_subject = src.get("filter_subject")
    with open(out_path, "w", encoding="utf-8") as fo:
        for rec in ds:
            if filter_subject and rec.get("subject_name", "").lower() != filter_subject.lower():
                continue
            text = normalize_record(rec)
            if not text:
                continue
            fo.write(json.dumps({"text": text, "src": src_id, "domain": src.get("domain", "?"),
                                  "license": src.get("license", "?")}) + "\n")
            n_chars += len(text)
            n_rows += 1
            if n_chars >= char_target:
                break
            if n_rows % 5000 == 0:
                log.info("    rows=%d chars=%dM", n_rows, n_chars // 1_000_000)
    log.info("  hf done: %s rows=%d ~%dM tokens", src_id, n_rows, n_chars // 4_000_000)
    return n_chars // 4


def fetch_pg_txt(src: dict, out_path: Path, target_tokens: int) -> int:
    """Download Project Gutenberg .txt and wrap into JSONL."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return _approx_tokens(out_path)
    url = src["url"]
    log.info("  pg_txt: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        log.warning("  pg FAILED: %s — %s", url, e)
        return 0

    # Strip PG header/footer
    start_re = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
    end_re = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
    m_start = start_re.search(text)
    m_end = end_re.search(text)
    if m_start:
        text = text[m_start.end():]
    if m_end:
        text = text[:m_end.start()]
    text = text.strip()

    # Chunk into ~3000-token chunks (~12k chars) — 1 chunk per JSONL row
    chunk_chars = 12_000
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_chars, n_rows = 0, 0
    char_target = target_tokens * 4
    with open(out_path, "w", encoding="utf-8") as fo:
        for i in range(0, len(text), chunk_chars):
            chunk = text[i:i + chunk_chars].strip()
            if len(chunk) < 500:
                continue
            wrapped = chatml_wrap([{"role": "assistant", "content": chunk}])
            fo.write(json.dumps({"text": wrapped, "src": src["id"], "domain": src.get("domain", "?"),
                                  "license": "PD"}) + "\n")
            n_chars += len(chunk)
            n_rows += 1
            if n_chars >= char_target:
                break
    log.info("  pg done: %s rows=%d ~%dM tokens", src["id"], n_rows, n_chars // 4_000_000)
    return n_chars // 4


def fetch_wikipedia(src: dict, out_path: Path, target_tokens: int) -> int:
    """Fetch a single Wikipedia article via MediaWiki API."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return _approx_tokens(out_path)
    title = src["title"]
    log.info("  wp: %s", title)
    api = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
           "&explaintext=1&exsectionformat=plain&redirects=1"
           f"&titles={urllib.parse.quote(title)}&format=json")
    try:
        req = urllib.request.Request(api, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        log.warning("  wp FAILED: %s — %s", title, e)
        return 0

    pages = data.get("query", {}).get("pages", {})
    text = ""
    for _, page in pages.items():
        text = page.get("extract", "") or ""
        break
    if not text or len(text) < 500:
        log.warning("  wp empty: %s", title)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_chars = 8_000  # Wikipedia articles tend to be shorter, smaller chunks
    n_chars, n_rows = 0, 0
    char_target = target_tokens * 4
    with open(out_path, "w", encoding="utf-8") as fo:
        for i in range(0, len(text), chunk_chars):
            chunk = text[i:i + chunk_chars].strip()
            if len(chunk) < 300:
                continue
            wrapped = chatml_wrap([
                {"role": "user", "content": f"Tell me about {title.replace('_', ' ')}."},
                {"role": "assistant", "content": chunk},
            ])
            fo.write(json.dumps({"text": wrapped, "src": f"wikipedia:{title}",
                                  "domain": src.get("domain", "?"),
                                  "license": "CC-BY-SA-4.0"}) + "\n")
            n_chars += len(chunk)
            n_rows += 1
            if n_chars >= char_target:
                break
    log.info("  wp done: %s rows=%d ~%dk tokens", title, n_rows, n_chars // 4_000)
    return n_chars // 4


def _approx_tokens(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                n += len(rec.get("text", "")) // 4
            except Exception:
                pass
    return n


# ============================================================================
# Driver
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="train/sources_v4.yaml")
    ap.add_argument("--raw-dir", default="/data/hf/blend_raw_v4")
    ap.add_argument("--out", default="/data/hf/blend_v4_addon.jsonl")
    ap.add_argument("--tokenizer", default="/data/hf/Qwen3.6-35B-A3B")
    ap.add_argument("--max-seq", type=int, default=4096)
    args = ap.parse_args()

    manifest = yaml.safe_load(open(args.manifest))
    sources = manifest.get("sources", [])
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("manifest: %d sources", len(sources))
    fetched = {}
    t_start = time.time()

    for src in sources:
        src_type = src["type"]
        out_path = raw_dir / f"{src['id'].replace('/', '__').replace('.', '_')}.jsonl"
        target = int(src.get("target_tokens", 100_000))
        try:
            if src_type == "hf":
                got = fetch_hf(src, out_path, target)
            elif src_type == "pg_txt":
                got = fetch_pg_txt(src, out_path, target)
            elif src_type == "wikipedia":
                got = fetch_wikipedia(src, out_path, target)
            else:
                log.warning("unknown type: %s", src_type)
                continue
            fetched[src["id"]] = got
        except Exception as e:
            log.error("FAIL %s — %s", src["id"], e)
            fetched[src["id"]] = 0

    log.info("=== fetch summary (%.1f min) ===", (time.time() - t_start) / 60)
    by_domain: dict[str, int] = {}
    for k, v in sorted(fetched.items(), key=lambda x: -x[1]):
        log.info("  %-50s %12d tokens", k, v)
    for src in sources:
        by_domain[src.get("domain", "?")] = by_domain.get(src.get("domain", "?"), 0) + fetched.get(src["id"], 0)
    log.info("--- by domain ---")
    for d, t in sorted(by_domain.items(), key=lambda x: -x[1]):
        log.info("  %-20s %dM tokens", d, t // 1_000_000)

    # Merge + tokenize-bound write
    log.info("=== merging + tokenize-bound write ===")
    from datasketch import MinHash, MinHashLSH
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    lsh = MinHashLSH(threshold=0.85, num_perm=128)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen, kept, tokens_written = 0, 0, 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for raw_file in sorted(raw_dir.rglob("*.jsonl")):
            with open(raw_file, encoding="utf-8") as fi:
                for line in fi:
                    seen += 1
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    text = rec.get("text", "")
                    if not text or len(text) < 64:
                        continue
                    m = MinHash(num_perm=128)
                    for tk in set(text.lower().split()):
                        m.update(tk.encode("utf-8"))
                    if lsh.query(m):
                        continue
                    lsh.insert(f"r{seen}", m)
                    ids = tok.encode(text, add_special_tokens=False)
                    if len(ids) < 32 or len(ids) > args.max_seq:
                        continue
                    fo.write(json.dumps({
                        "text": text,
                        "n_tokens": len(ids),
                        "src": rec.get("src", "unknown"),
                        "domain": rec.get("domain", "?"),
                        "license": rec.get("license", "?"),
                        "hash": hashlib.sha1(text.encode()).hexdigest()[:16],
                    }) + "\n")
                    kept += 1
                    tokens_written += len(ids)
                    if kept % 5000 == 0:
                        log.info("  merged: kept=%d tokens=%dM", kept, tokens_written // 1_000_000)

    log.info("=== v4 addon ready: %s | rows=%d tokens=%dM ===", out_path, kept, tokens_written // 1_000_000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
