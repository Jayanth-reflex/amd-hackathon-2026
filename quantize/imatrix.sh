#!/usr/bin/env bash
# Importance matrix calibration. Runs at H+30 in the §G timeline.
# Bumped to 500 chunks (vs 200 in earlier revisions) — better calibration
# since GPU budget allows during the saturation window.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/workspace/gguf_out}"
LCPP="${LCPP:-/workspace/llama.cpp}"
WIKI_DIR="${WIKI_DIR:-/workspace/wikitext-2-raw}"
CHUNKS="${CHUNKS:-500}"

# Ensure WikiText-2 is available
if [[ ! -f "${WIKI_DIR}/wiki.train.raw" ]]; then
    mkdir -p "$(dirname "${WIKI_DIR}")"
    wget -q -O /tmp/wikitext-2-raw-v1.zip \
        https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
    unzip -q -o /tmp/wikitext-2-raw-v1.zip -d "$(dirname "${WIKI_DIR}")"
fi

echo "[imatrix] computing on BF16 reference, chunks=${CHUNKS}"
"${LCPP}/build/bin/llama-imatrix" \
    -m "${OUT_DIR}/Qwen3.6-35B-A3B-Domain-Aggressive-BF16.gguf" \
    -f "${WIKI_DIR}/wiki.train.raw" \
    --chunks "${CHUNKS}" \
    -ngl 99 \
    -o "${OUT_DIR}/Qwen3.6-35B-A3B-Domain-Aggressive-imatrix.gguf"

ls -lh "${OUT_DIR}"
echo "[imatrix] done — proceed to ladder.sh"
