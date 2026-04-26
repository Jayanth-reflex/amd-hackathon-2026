"""Multi-source dataset assembler for amd-hackathon-2026 (Revision 5.1).

Reads `train/config.yaml` blend section and pulls from:
  - HF datasets (`huggingface_hub.load_dataset`)
  - Wikipedia category dumps (REST API)
  - arXiv search (atom feed API)
  - Direct URL/PDF (US Army FMs, CIA Heuer, etc.)

For each source, writes to /data/hf/blend_raw/<id>.jsonl, then merges +
MinHash-dedupes + ChatML-wraps into the final blend.jsonl.

Sources marked as `synthetic` are skipped here — handled by a separate
synthetic-generation pass that runs in parallel with training.

Usage:
    python train/prepare_data.py \\
        --config train/config.yaml \\
        --out /data/hf/blend.jsonl \\
        --raw-dir /data/hf/blend_raw \\
        --tokenizer /data/hf/Qwen3.6-35B-A3B
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_data")

# ============================================================================
# Common ChatML utilities
# ============================================================================

def chatml_wrap(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def normalize_record(record: dict) -> str | None:
    """Best-effort normalize a heterogeneous record to a ChatML string."""
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
    if "text" in record and isinstance(record["text"], str):
        # Plain pretrain text — wrap as a single assistant turn for SFT compat
        return chatml_wrap([{"role": "assistant", "content": record["text"]}])
    return None


# ============================================================================
# Fetchers — one per source type
# ============================================================================

def fetch_hf(source_id: str, split: str, target_tokens: int, raw_path: Path) -> int:
    """Stream HF dataset → write raw text to raw_path. Returns approx token count."""
    if raw_path.exists() and raw_path.stat().st_size > 1024:
        log.info("  hf cached: %s (%.1fMB)", raw_path.name, raw_path.stat().st_size / 1e6)
        return _approx_tokens(raw_path)

    from datasets import load_dataset
    log.info("  hf fetching: %s [%s] target=%d tokens", source_id, split, target_tokens)
    try:
        # Newer datasets versions reject trust_remote_code; try without first
        ds = load_dataset(source_id, split=split, streaming=True)
    except TypeError:
        ds = load_dataset(source_id, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        log.warning("  hf FAILED: %s — %s", source_id, e)
        return 0

    n_chars = 0
    n_rows = 0
    char_target = target_tokens * 4  # ~4 chars/token rough
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as fo:
        for rec in ds:
            text = normalize_record(rec)
            if not text:
                continue
            fo.write(json.dumps({"text": text, "src": source_id}) + "\n")
            n_chars += len(text)
            n_rows += 1
            if n_chars >= char_target:
                break
            if n_rows % 5000 == 0:
                log.info("    rows=%d chars=%dM", n_rows, n_chars // 1_000_000)
    log.info("  hf done: rows=%d chars=%dM ~tokens=%dM", n_rows, n_chars // 1_000_000, n_chars // 4_000_000)
    return n_chars // 4


def fetch_wikipedia_category(category: str, target_tokens: int, raw_path: Path,
                             max_articles: int = 200) -> int:
    """Pull a Wikipedia category's article texts via the REST API."""
    if raw_path.exists() and raw_path.stat().st_size > 1024:
        log.info("  wp cached: %s", raw_path.name)
        return _approx_tokens(raw_path)

    log.info("  wikipedia fetching: %s target=%d tokens", category, target_tokens)
    members = _wp_category_members(category, limit=max_articles)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    n_chars = 0
    with open(raw_path, "w", encoding="utf-8") as fo:
        for title in members:
            text = _wp_extract(title)
            if not text or len(text) < 500:
                continue
            fo.write(json.dumps({"text": chatml_wrap([{"role": "assistant", "content": text}]),
                                  "src": f"wikipedia:{category}", "title": title}) + "\n")
            n_chars += len(text)
            if n_chars >= target_tokens * 4:
                break
    log.info("  wikipedia done: %s ~%dM tokens", category, n_chars // 4_000_000)
    return n_chars // 4


def _wp_category_members(category: str, limit: int = 200) -> list[str]:
    """List article titles in a Wikipedia category (recursive 1 level)."""
    url = ("https://en.wikipedia.org/w/api.php?action=query&list=categorymembers"
           f"&cmtitle=Category:{urllib.parse.quote(category)}&cmlimit={min(limit, 500)}"
           "&cmtype=page&format=json")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
    except Exception as e:
        log.warning("    wp category list failed: %s", e)
        return []


def _wp_extract(title: str) -> str:
    """Fetch the plain-text extract of a Wikipedia article."""
    url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
           "&explaintext=1&exsectionformat=plain&redirects=1"
           f"&titles={urllib.parse.quote(title)}&format=json")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            return page.get("extract", "") or ""
    except Exception as e:
        log.debug("    wp extract failed for %s: %s", title, e)
    return ""


def fetch_arxiv(query: str, target_tokens: int, raw_path: Path, max_results: int = 200) -> int:
    """Pull arXiv abstracts (full PDF later, abstracts now for token volume)."""
    if raw_path.exists() and raw_path.stat().st_size > 1024:
        log.info("  arxiv cached: %s", raw_path.name)
        return _approx_tokens(raw_path)

    log.info("  arxiv fetching: %s target=%d tokens", query, target_tokens)
    url = ("http://export.arxiv.org/api/query?"
           f"search_query={urllib.parse.quote(query)}"
           f"&max_results={max_results}&sortBy=relevance&sortOrder=descending")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            xml = r.read().decode("utf-8")
    except Exception as e:
        log.warning("  arxiv FAILED: %s — %s", query, e)
        return 0

    # Tiny atom parser without lxml dep
    import re
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    n_chars = 0
    with open(raw_path, "w", encoding="utf-8") as fo:
        for e in entries:
            title_m = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
            sum_m = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
            if not (title_m and sum_m):
                continue
            title = title_m.group(1).strip()
            summary = sum_m.group(1).strip()
            text = f"# {title}\n\n{summary}"
            fo.write(json.dumps({"text": chatml_wrap([{"role": "assistant", "content": text}]),
                                  "src": f"arxiv:{query}"}) + "\n")
            n_chars += len(text)
    log.info("  arxiv done: %s ~%dM tokens (%d entries)", query, n_chars // 4_000_000, len(entries))
    return n_chars // 4


def fetch_url(url: str, target_tokens: int, raw_path: Path) -> int:
    """Download a URL (HTML/text/PDF) and store text."""
    if raw_path.exists() and raw_path.stat().st_size > 1024:
        return _approx_tokens(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            content = r.read()
    except Exception as e:
        log.warning("  url FAILED: %s — %s", url, e)
        return 0

    # Naive extraction: try HTML strip, fallback to raw
    try:
        text = content.decode("utf-8", errors="replace")
        # Strip HTML tags crudely
        import re
        text = re.sub(r"<script.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception:
        return 0

    if not text or len(text) < 200:
        return 0

    with open(raw_path, "w", encoding="utf-8") as fo:
        fo.write(json.dumps({"text": chatml_wrap([{"role": "assistant", "content": text}]),
                              "src": url}) + "\n")
    return len(text) // 4


def _approx_tokens(path: Path) -> int:
    """Rough char-based token estimate from JSONL."""
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
# Wikipedia category presets per domain
# ============================================================================

WIKIPEDIA_PRESETS: dict[str, list[str]] = {
    # Survival
    "fire_cooking_offgrid": ["Fire_making", "Open-fire_cooking", "Survival_skills"],
    "water_filtration": ["Water_purification", "Water_supply", "Sanitation"],
    "edible_plants_foraging": ["Edible_plants", "Foraging", "Wild_food"],
    "wilderness_medicine": ["Wilderness_medicine", "First_aid", "Medical_emergencies"],
    "shelter_construction": ["Survival_shelters", "Bushcraft", "Camping"],
    "comms_offgrid": ["Amateur_radio", "Mesh_networking", "Radio_communications"],
    "navigation_lowtech": ["Navigation", "Orienteering", "Celestial_navigation"],
    "urban_disaster_prep": ["Disaster_preparedness", "Emergency_management", "Urban_survival"],
    # History
    "ancient": ["Ancient_history", "Ancient_civilizations", "Indus_Valley_Civilisation"],
    "medieval": ["Medieval_history", "Mughal_Empire", "Delhi_Sultanate", "Byzantine_Empire"],
    "early_modern": ["Early_modern_period", "Renaissance", "British_Raj", "Industrial_Revolution"],
    "modern_post_1900": ["20th_century_history", "World_War_II", "Cold_War", "History_of_independent_India"],
    # Defense forces
    "india_armed_forces": ["Indian_Army", "Indian_Navy", "Indian_Air_Force", "Strategic_Forces_Command_(India)"],
    "tier1_us_china_russia": ["Military_of_the_United_States", "People's_Liberation_Army", "Russian_Armed_Forces"],
    "nato_uk_fr_de": ["NATO", "British_Armed_Forces", "French_Armed_Forces", "Bundeswehr"],
    "regional_powers": ["Pakistan_Armed_Forces", "Israel_Defense_Forces", "Republic_of_Korea_Armed_Forces",
                          "Japan_Self-Defense_Forces", "Turkish_Armed_Forces", "Islamic_Republic_of_Iran_Armed_Forces"],
    "doctrine_warfare_history": ["Military_strategy", "Military_doctrine", "Asymmetric_warfare"],
    # Geopolitics + game theory anchors
    "geopolitics": ["Geopolitics", "International_relations", "Foreign_policy"],
    # Geospatial
    "osm_geospatial": ["OpenStreetMap", "Geographic_information_system", "Remote_sensing"],
    # Indian law (fallback)
    "indian_law_fallback": ["Bharatiya_Nyaya_Sanhita", "Indian_Penal_Code", "Constitution_of_India"],
}


# ============================================================================
# Direct URL presets — public-domain or CC-BY material
# ============================================================================

URL_PRESETS: dict[str, list[str]] = {
    "heuer_psychology_intel_analysis": [
        "https://www.cia.gov/static/9a5f1162fd0932c29bfed1c030edf4ae/Pyschology-of-Intelligence-Analysis.pdf",
    ],
    "heuer_pherson_structured_techniques": [],   # commercial; skip
    "kubark_1963_declass": [
        "https://nsarchive2.gwu.edu/NSAEBB/NSAEBB122/CIA%20Kubark%201-60.pdf",
    ],
    "fm_3-05.30_psyop": ["https://irp.fas.org/doddir/army/fm3-05-30.pdf"],
    "fm_21-76_ch5": ["https://www.bits.de/NRANEU/others/amd-us-archive/FM21-76(57).pdf"],
    "sun_tzu_pd": ["https://www.gutenberg.org/files/132/132-0.txt"],
    "clausewitz_pd": ["https://www.gutenberg.org/files/1946/1946-0.txt"],
    "nash_bargaining": [],
}


# ============================================================================
# Walk the blend config
# ============================================================================

def _flatten_blend(blend: dict, total_tokens: int) -> Iterator[tuple[str, dict, float]]:
    """Yield (logical_id, source_def, share_of_total) for every source in blend."""
    for cat_name in ("general", "reasoning", "tools"):
        cat = blend.get(cat_name, {})
        for src in cat.get("sources", []):
            yield (src["id"], src, src["share"])

    domain = blend.get("domain", {})
    for d_name, d_cfg in domain.items():
        if d_name == "share":
            continue
        if not isinstance(d_cfg, dict):
            continue
        d_share = d_cfg.get("share", 0)

        # Three shapes: flat sources, sub_areas, subdisciplines, languages
        if "sources" in d_cfg:
            for s in d_cfg["sources"]:
                yield (s.get("id") or s.get("url_seed", "unknown"), s, d_share / max(1, len(d_cfg["sources"])))
        elif "sub_areas" in d_cfg:
            for sa_name, sa_cfg in d_cfg["sub_areas"].items():
                yield (f"{d_name}.{sa_name}", sa_cfg, sa_cfg["share"])
        elif "subdisciplines" in d_cfg:
            for sd_name, sd_cfg in d_cfg["subdisciplines"].items():
                yield (f"{d_name}.{sd_name}", sd_cfg, sd_cfg["share"])
        elif "languages" in d_cfg:
            for lang_name, lang_cfg in d_cfg["languages"].items():
                yield (f"{d_name}.{lang_name}", lang_cfg, lang_cfg["share"])


def fetch_one(logical_id: str, source_def: dict, target_tokens: int, raw_dir: Path) -> int:
    """Dispatch to the right fetcher based on source_def shape."""
    raw_path = raw_dir / f"{logical_id.replace('/', '__').replace('.', '__')}.jsonl"

    src_type = source_def.get("type", "auto")
    if src_type == "hf" or "id" in source_def and source_def.get("type", "hf") == "hf":
        return fetch_hf(source_def["id"], source_def.get("split", "train"), target_tokens, raw_path)

    if src_type == "scrape" and "url_seed" in source_def:
        return fetch_url(source_def["url_seed"], target_tokens, raw_path)

    if src_type == "arxiv":
        return fetch_arxiv(source_def.get("query", ""), target_tokens, raw_path)

    if src_type == "synthetic":
        log.info("  synthetic deferred: %s (handled in synth pass)", logical_id)
        return 0

    # sub_area / subdiscipline / language with sources list — try Wikipedia + URL presets
    sources = source_def.get("sources", [])
    n = 0
    for s in sources:
        sub_path = raw_path.parent / f"{logical_id.replace('/', '__').replace('.', '__')}__{s}.jsonl"
        if s in WIKIPEDIA_PRESETS:
            for cat in WIKIPEDIA_PRESETS[s]:
                n += fetch_wikipedia_category(cat, target_tokens // max(1, len(sources)) // len(WIKIPEDIA_PRESETS[s]),
                                              sub_path)
        elif s in URL_PRESETS:
            for url in URL_PRESETS[s]:
                n += fetch_url(url, target_tokens // max(1, len(sources)), sub_path)
        elif s.startswith("flores200_") or s.startswith("tatoeba_") or s.startswith("wiktionary_"):
            log.info("  language source stub: %s (Tatoeba/FLORES handled in lang pass)", s)
        else:
            log.info("  unknown source preset: %s — skipping", s)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    ap.add_argument("--out", default="/data/hf/blend.jsonl")
    ap.add_argument("--raw-dir", default="/data/hf/blend_raw")
    ap.add_argument("--tokenizer", default="/data/hf/Qwen3.6-35B-A3B")
    ap.add_argument("--max-seq", type=int, default=8192)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    blend = cfg["blend"]
    total = blend["total_tokens"]
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = list(_flatten_blend(blend, total))
    log.info("blend: %d sources, total target = %d tokens", len(sources), total)

    fetched = {}
    t_start = time.time()
    for logical_id, source_def, share in sources:
        target = int(total * share)
        if target < 1000:
            log.info("[skip-small] %s share=%.4f target=%d", logical_id, share, target)
            continue
        log.info("[fetch] %s share=%.4f target=%d", logical_id, share, target)
        try:
            got = fetch_one(logical_id, source_def, target, raw_dir)
            fetched[logical_id] = got
        except Exception as e:
            log.error("  FAIL %s — %s", logical_id, e)
            fetched[logical_id] = 0

    log.info("=== fetch summary (%.1f min) ===", (time.time() - t_start) / 60)
    total_fetched = 0
    for k, v in sorted(fetched.items(), key=lambda x: -x[1]):
        log.info("  %-50s %12d tokens", k, v)
        total_fetched += v
    log.info("TOTAL fetched: %dM tokens (target was %dM)", total_fetched // 1_000_000, total // 1_000_000)

    # Merge + dedupe + tokenize-bound write
    log.info("=== merging + dedupe + tokenize-bound write ===")
    from datasketch import MinHash, MinHashLSH
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    seen, kept, tokens_written = 0, 0, 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
                    key = f"r{seen}"
                    if lsh.query(m):
                        continue
                    lsh.insert(key, m)
                    ids = tok.encode(text, add_special_tokens=False)
                    if len(ids) < 32 or len(ids) > args.max_seq:
                        continue
                    fo.write(json.dumps({
                        "text": text,
                        "n_tokens": len(ids),
                        "src": rec.get("src", "unknown"),
                        "hash": hashlib.sha1(text.encode()).hexdigest()[:16],
                    }) + "\n")
                    kept += 1
                    tokens_written += len(ids)
                    if tokens_written >= total:
                        break
                    if kept % 10000 == 0:
                        log.info("  merged: kept=%d tokens=%dM", kept, tokens_written // 1_000_000)
            if tokens_written >= total:
                break

    log.info("=== blend ready: %s | rows=%d tokens=%dM ===", out_path, kept, tokens_written // 1_000_000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
