"""Build the 465-prompt refusal benchmark from public sources.

Sources:
- walledai/AdvBench (520 harmful behaviors, MIT)  — primary
- HarmBench (~400 prompts) is also viable but AdvBench is the canonical
  Heretic/Arditi reference and what `heretic-llm` itself trains on.

We deduplicate against AdvBench's exact text, take the first 445 unique
prompts, and append our 20 scaffold prompts (which include India-specific
harms not in AdvBench like 'tax fraud' specific to Indian context). Total: 465.

Output: eval/refusal_465.jsonl with schema {id:int, prompt:str, source:str}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCAFFOLD_PATH = HERE / "refusal_465.jsonl"
OUT_PATH = HERE / "refusal_465.jsonl"


def load_scaffold() -> list[dict]:
    rows: list[dict] = []
    if not SCAFFOLD_PATH.exists():
        return rows
    for line in SCAFFOLD_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "prompt" in obj:
            rows.append({"prompt": obj["prompt"], "source": "scaffold"})
    return rows


def load_advbench() -> list[dict]:
    """walledai/AdvBench: 520 harmful behaviors, MIT licensed."""
    from datasets import load_dataset
    ds = load_dataset("walledai/AdvBench", split="train")
    rows = []
    for r in ds:
        # AdvBench schema: 'prompt', 'target' — we only need the prompt
        p = r.get("prompt") or r.get("behavior")
        if isinstance(p, str) and p.strip():
            rows.append({"prompt": p.strip(), "source": "advbench"})
    return rows


def main() -> int:
    scaffold = load_scaffold()
    print(f"loaded {len(scaffold)} scaffold prompts", file=sys.stderr)

    advbench = load_advbench()
    print(f"loaded {len(advbench)} AdvBench prompts", file=sys.stderr)

    seen: set[str] = set()
    out: list[dict] = []

    # Scaffold first (preserves any India-specific entries)
    for r in scaffold:
        key = r["prompt"].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    # Pad to 465 from AdvBench
    for r in advbench:
        if len(out) >= 465:
            break
        key = r["prompt"].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    if len(out) < 465:
        print(f"WARN: only {len(out)} unique prompts available "
              f"(scaffold={len(scaffold)} + advbench={len(advbench)})", file=sys.stderr)

    print(f"final corpus: {len(out)} prompts "
          f"(scaffold={sum(1 for r in out if r['source']=='scaffold')}, "
          f"advbench={sum(1 for r in out if r['source']=='advbench')})", file=sys.stderr)

    with OUT_PATH.open("w") as f:
        for i, r in enumerate(out, 1):
            f.write(json.dumps({"id": i, "prompt": r["prompt"], "source": r["source"]}) + "\n")
    print(f"wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
