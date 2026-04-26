#!/usr/bin/env bash
# 9-quant GGUF ladder — runs at H+31 in the §G timeline.
# Bumped from 8 quants in earlier revisions; adds IQ2_M for absolute low-end.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/workspace/gguf_out}"
LCPP="${LCPP:-/workspace/llama.cpp}"
BF16="${OUT_DIR}/Qwen3.6-35B-A3B-Domain-Aggressive-BF16.gguf"
IMATRIX="${OUT_DIR}/Qwen3.6-35B-A3B-Domain-Aggressive-imatrix.gguf"

[[ -f "${BF16}" ]] || { echo "missing ${BF16} — run convert.sh first"; exit 1; }
[[ -f "${IMATRIX}" ]] || { echo "missing ${IMATRIX} — run imatrix.sh first"; exit 1; }

QUANTS=(Q8_0 Q6_K Q5_K_M Q4_K_M IQ4_XS Q3_K_M IQ3_M IQ2_M)

for q in "${QUANTS[@]}"; do
    out="${OUT_DIR}/Qwen3.6-35B-A3B-Domain-Aggressive-${q}.gguf"
    if [[ -f "${out}" ]]; then
        echo "[ladder] ${q} already present, skipping"
        continue
    fi
    echo "[ladder] quantizing ${q}"
    "${LCPP}/build/bin/llama-quantize" \
        --imatrix "${IMATRIX}" \
        "${BF16}" \
        "${out}" \
        "${q}"
done

echo "[ladder] done. summary:"
ls -lh "${OUT_DIR}"/*.gguf
