#!/usr/bin/env bash
# Apple Silicon (M4 24GB) failover server.
# Uses our own Q4_K_M GGUF + mmproj. Confirmed working on M4 Metal builds of llama.cpp.
set -euo pipefail

GGUF_DIR="${GGUF_DIR:-${HOME}/Models/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF}"
LCPP="${LCPP:-${HOME}/llama.cpp}"
MODEL="${MODEL:-${GGUF_DIR}/Qwen3.6-35B-A3B-Domain-Aggressive-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-${GGUF_DIR}/mmproj-Qwen3.6-35B-A3B-Domain-Aggressive-f16.gguf}"

if [[ ! -f "${MODEL}" ]]; then
    echo "model not found: ${MODEL}"
    echo "pull with: huggingface-cli download Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF \\"
    echo "    --include '*Q4_K_M.gguf' '*mmproj*' --local-dir ${GGUF_DIR}"
    exit 1
fi

[[ -d "${LCPP}/build" ]] || {
    echo "llama.cpp not built at ${LCPP}/build"
    echo "build it with: cmake -S ${LCPP} -B ${LCPP}/build -DGGML_METAL=ON && cmake --build ${LCPP}/build --config Release -j"
    exit 1
}

exec "${LCPP}/build/bin/llama-server" \
    -m "${MODEL}" \
    --mmproj "${MMPROJ}" \
    --jinja \
    -c 32768 \
    -ngl 99 \
    -fa \
    -ctk bf16 -ctv bf16 \
    --host 0.0.0.0 \
    --port 8001
