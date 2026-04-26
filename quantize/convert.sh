#!/usr/bin/env bash
# BF16 GGUF conversion + mmproj generation for the Aggressive variant.
# Runs at H+29 in the §G timeline. Requires llama.cpp built with HIP+ROCWMMA.
set -euo pipefail

SRC_REPO="${SRC_REPO:-Jayanth/Qwen3.6-35B-A3B-Domain-Aggressive}"
LOCAL_DIR="${LOCAL_DIR:-/data/hf/Qwen3.6-35B-A3B-Domain-Aggressive}"
OUT_DIR="${OUT_DIR:-/workspace/gguf_out}"
LCPP="${LCPP:-/workspace/llama.cpp}"

mkdir -p "${OUT_DIR}"

# Pull the Aggressive model locally (skip if already present)
if [[ ! -d "${LOCAL_DIR}" || -z "$(ls -A "${LOCAL_DIR}" 2>/dev/null || true)" ]]; then
    echo "[convert] pulling ${SRC_REPO} -> ${LOCAL_DIR}"
    huggingface-cli download "${SRC_REPO}" --local-dir "${LOCAL_DIR}"
fi

# 1) BF16 GGUF (lossless reference)
echo "[convert] BF16 GGUF"
python "${LCPP}/convert_hf_to_gguf.py" \
    "${LOCAL_DIR}" \
    --outtype bf16 \
    --outfile "${OUT_DIR}/Qwen3.6-35B-A3B-Domain-Aggressive-BF16.gguf"

# 2) mmproj (vision tower, f16)
echo "[convert] mmproj (vision)"
python "${LCPP}/convert_hf_to_gguf.py" \
    "${LOCAL_DIR}" \
    --mmproj \
    --outfile "${OUT_DIR}/mmproj-Qwen3.6-35B-A3B-Domain-Aggressive-f16.gguf"

ls -lh "${OUT_DIR}"
echo "[convert] done — proceed to imatrix.sh"
