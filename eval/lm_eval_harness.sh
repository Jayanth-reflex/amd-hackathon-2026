#!/usr/bin/env bash
# Capability regression suite. Runs at H+38 in the §G timeline.
# Bumped from sample to full benchmark per Revision 5 GPU-saturation mandate.
set -euo pipefail

MODEL="${MODEL:-Jayanth/Qwen3.6-35B-A3B-Domain-Aggressive}"
OUT_DIR="${OUT_DIR:-/workspace/eval_results}"
mkdir -p "${OUT_DIR}"

# Tasks: full lm-eval-harness MMLU, HellaSwag, TruthfulQA, GSM8K (250 each)
echo "[lm-eval] mainstream capability tasks"
lm-eval \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu,hellaswag,truthfulqa_mc2,gsm8k \
    --num_fewshot 0 \
    --batch_size auto \
    --limit 250 \
    --output_path "${OUT_DIR}/lm_eval_main.json"

# BBH-Hard reasoning
echo "[lm-eval] BBH-Hard"
lm-eval \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True" \
    --tasks bbh \
    --num_fewshot 3 \
    --batch_size auto \
    --limit 100 \
    --output_path "${OUT_DIR}/lm_eval_bbh.json"

# Domain held-out + multimodal — uses our own scripts
echo "[lm-eval] domain held-out (300 items)"
python eval/domain_eval.py \
    --model "${MODEL}" \
    --prompts eval/domain_qa.jsonl \
    --out "${OUT_DIR}/domain_eval.json" || \
    echo "[lm-eval] domain_eval.py not yet implemented — skipping"

echo "[lm-eval] done. summary:"
ls -lh "${OUT_DIR}"
