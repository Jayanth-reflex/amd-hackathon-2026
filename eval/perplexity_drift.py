"""Perplexity drift gate. Runs at H+34 in the §G timeline.

For each quant in the ladder, compute perplexity on WikiText-2 test set
and compare against the BF16 reference. Reject any quant with > 3% drift.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("perplexity_drift")

QUANTS = ("BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "IQ3_M", "IQ2_M")
PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", re.IGNORECASE)


def run_ppl(llamacpp_dir: Path, gguf: Path, wiki_test: Path, ngl: int = 99) -> float:
    """Run llama-perplexity and parse the final PPL number."""
    bin_ = llamacpp_dir / "build" / "bin" / "llama-perplexity"
    if not bin_.exists():
        raise FileNotFoundError(bin_)
    log.info("running ppl: %s", gguf.name)
    proc = subprocess.run(
        [str(bin_), "-m", str(gguf), "-f", str(wiki_test),
         "-ctk", "bf16", "-ctv", "bf16", "-ngl", str(ngl)],
        capture_output=True, text=True, check=True,
    )
    output = proc.stdout + proc.stderr
    m = PPL_RE.search(output)
    if not m:
        raise RuntimeError(f"could not parse PPL from output:\n{output[-1000:]}")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", type=Path, default=Path("/workspace/gguf_out"))
    ap.add_argument("--prefix", default="Qwen3.6-35B-A3B-Domain-Aggressive")
    ap.add_argument("--llamacpp-dir", type=Path, default=Path("/workspace/llama.cpp"))
    ap.add_argument("--wiki-test", type=Path, default=Path("/workspace/wikitext-2-raw/wiki.test.raw"))
    ap.add_argument("--max-drift", type=float, default=0.03, help="Maximum allowed drift fraction (0.03 = 3%)")
    ap.add_argument("--out", type=Path, default=Path("logs/perplexity_drift.json"))
    args = ap.parse_args()

    if not args.wiki_test.exists():
        log.error("WikiText test set missing: %s", args.wiki_test)
        return 2

    bf16_path = args.gguf_dir / f"{args.prefix}-BF16.gguf"
    bf16_ppl = run_ppl(args.llamacpp_dir, bf16_path, args.wiki_test)
    log.info("BF16 reference PPL: %.4f", bf16_ppl)

    results: dict[str, dict] = {"BF16": {"ppl": bf16_ppl, "drift": 0.0, "passed": True}}
    failed: list[str] = []

    for q in QUANTS:
        if q == "BF16":
            continue
        gguf = args.gguf_dir / f"{args.prefix}-{q}.gguf"
        if not gguf.exists():
            log.warning("missing %s, skipping", gguf.name)
            continue
        ppl = run_ppl(args.llamacpp_dir, gguf, args.wiki_test)
        drift = (ppl - bf16_ppl) / bf16_ppl
        passed = drift <= args.max_drift
        results[q] = {"ppl": ppl, "drift": drift, "passed": passed}
        log.info("%-7s PPL=%.4f drift=%+.2f%% %s", q, ppl, drift * 100, "PASS" if passed else "FAIL")
        if not passed:
            failed.append(q)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "bf16_ppl": bf16_ppl,
        "max_drift": args.max_drift,
        "results": results,
        "failed": failed,
        "passed_overall": not failed,
    }, indent=2))
    log.info("wrote %s", args.out)

    return 0 if not failed else 5


if __name__ == "__main__":
    sys.exit(main())
