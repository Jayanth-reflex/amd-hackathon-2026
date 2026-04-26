#!/usr/bin/env bash
# Upload all GGUF artifacts to HF Hub. Runs at H+31 (end of ladder) in §G.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/workspace/gguf_out}"
REPO="${REPO:-Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF}"

# Ensure repo exists (idempotent)
huggingface-cli repo create "${REPO}" --type model -y || true

echo "[push_gguf] uploading ${OUT_DIR} -> ${REPO}"
huggingface-cli upload "${REPO}" "${OUT_DIR}" . \
    --repo-type=model \
    --commit-message "9-quant ladder + mmproj + imatrix (Aggressive variant)"

# Generate model card pointer
cat > "${OUT_DIR}/README.md" <<'EOF'
---
license: apache-2.0
base_model: Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive
tags:
  - mi300x
  - qwen3.6
  - moe
  - abliterated
  - heretic
  - vision-language
  - hackathon-2026
---

# Qwen3.6-35B-A3B-Domain-Aggressive — GGUF ladder

9-quant GGUF ladder for the Heretic-abliterated, domain-specialized
fine-tune of `unsloth/Qwen3.6-35B-A3B`. Vision tower preserved as paired
`mmproj-*-f16.gguf`.

## Quants

| Quant | Disk size | RAM target | Use case |
|---|---|---|---|
| BF16 | ~70 GB | ≥80 GB | reference / vLLM merged-model GGUF |
| Q8_0 | ~37 GB | ≥48 GB | desktop high-quality |
| Q6_K | ~29 GB | ≥36 GB | balanced quality |
| Q5_K_M | ~25 GB | ≥32 GB | sweet spot |
| **Q4_K_M** | **~21 GB** | **≥24 GB** | **default for M4 failover** |
| IQ4_XS | ~19 GB | ≥22 GB | low-RAM machines |
| Q3_K_M | ~17 GB | ≥20 GB | aggressive |
| IQ3_M | ~15 GB | ≥18 GB | extreme |
| IQ2_M | ~12 GB | ≥14 GB | last resort |

## Refusal posture

Verified **0/465 refusals** post-Heretic. Verification artifacts in the
[parent repo](https://github.com/Jayanth-reflex/amd-hackathon-2026) under
`logs/refusal_results_*.json`.

## Sampling

`temperature=0.7`, `top_p=0.95`, `presence_penalty=1.5`.

## Quick start

```bash
./llama-server \
  -m Qwen3.6-35B-A3B-Domain-Aggressive-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.6-35B-A3B-Domain-Aggressive-f16.gguf \
  --jinja -c 32768 -ngl 99 --host 0.0.0.0 --port 8001
```

## License

Apache-2.0 (preserves upstream Qwen3.6 license).
EOF

huggingface-cli upload "${REPO}" "${OUT_DIR}/README.md" README.md \
    --repo-type=model \
    --commit-message "model card"

echo "[push_gguf] done — https://huggingface.co/${REPO}"
