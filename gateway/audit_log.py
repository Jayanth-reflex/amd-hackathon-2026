"""Tamper-evident append-only audit log.

Each line is a JSON record with a sha256 hash chain — `prev_hash` links to
the previous record's hash. Tampering with any past entry breaks the chain
and is detected on next verify pass.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("audit_log")

DEFAULT_PATH = Path(os.environ.get("AUDIT_LOG_PATH", "/data/audit/decisions.log"))
_lock = threading.Lock()
_last_hash: str | None = None


def _current_last_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = 4096
            offset = max(0, size - chunk)
            f.seek(offset)
            tail = f.read().decode("utf-8", errors="replace")
        last_line = tail.strip().splitlines()[-1] if tail.strip() else ""
        if not last_line:
            return None
        rec = json.loads(last_line)
        return rec.get("hash")
    except Exception as e:
        log.warning("could not read tail of %s: %s", path, e)
        return None


def log_decision(*, action: str, verdict: dict, prompt: str = "", response: str = "",
                 path: Path | None = None) -> dict:
    """Append a decision record to the audit log; return the record."""
    global _last_hash
    target = path or DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        if _last_hash is None:
            _last_hash = _current_last_hash(target)
        rec = {
            "ts": time.time(),
            "action": action,
            "verdict": verdict,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest() if response else "",
            "prompt_head": prompt[:120],
            "response_head": response[:200] if response else "",
            "prev_hash": _last_hash,
        }
        payload = json.dumps(rec, sort_keys=True).encode("utf-8")
        rec["hash"] = hashlib.sha256(payload).hexdigest()
        _last_hash = rec["hash"]
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def verify_chain(path: Path | None = None) -> tuple[bool, int, str | None]:
    """Verify the entire chain. Returns (ok, n_records, first_break_line_or_None)."""
    target = path or DEFAULT_PATH
    if not target.exists():
        return True, 0, None
    prev: str | None = None
    n = 0
    with open(target, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stored = rec.pop("hash", None)
            if rec.get("prev_hash") != prev:
                return False, n, f"line {i}: prev_hash mismatch"
            payload = json.dumps(rec, sort_keys=True).encode("utf-8")
            recomputed = hashlib.sha256(payload).hexdigest()
            if recomputed != stored:
                return False, n, f"line {i}: hash mismatch"
            prev = stored
            n += 1
    return True, n, None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        ok, n, err = verify_chain(args.path)
        print(f"verify: {'OK' if ok else 'FAIL'} records={n} {err or ''}")
        raise SystemExit(0 if ok else 6)
    print(f"audit log: {args.path} (exists={args.path.exists()})")
