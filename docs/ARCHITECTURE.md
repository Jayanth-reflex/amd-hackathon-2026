# Architecture

This document is the engineer-facing companion to the master spec in
[`CLAUDE.md`](../CLAUDE.md). It explains the data flow, the rationale
behind each component choice, and the dependency graph between scripts.

## High-level flow

```
                  ┌────────────────────────────────────────────────────┐
                  │     AMD Instinct MI300X (192 GB HBM3, gfx942)      │
                  │                                                    │
  §E datasets ─►  │ train/prepare_data.py  →  /data/hf/blend.jsonl     │
                  │ train/train_lora.py    →  out/lora_adapter         │
                  │ train/merge_and_push.py→  HF: …Domain (BF16)       │
                  │ abliterate/run_heretic →  HF: …Domain-Aggressive   │
                  │   ↓ refusal_benchmark.py (HARD GATE ≤5/465)        │
                  │ quantize/convert.sh    →  BF16 GGUF + mmproj       │
                  │ quantize/imatrix.sh    →  500-chunk imatrix        │
                  │ quantize/ladder.sh     →  9 quants                 │
                  │ quantize/push_gguf.sh  →  HF: …Aggressive-GGUF     │
                  │ eval/perplexity_drift.py (gate >3% drift)          │
                  │                                                    │
                  │ serve/vllm.env (vLLM @ :8000)                      │
                  │ serve/llamacpp.env (llama.cpp-server @ :8001)      │
                  └────────────┬───────────────────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ private VPS gateway    │  gateway/llama_guard.py
                  │ Llama-Guard-3-1B       │  + gateway/policy.yaml
                  │ + audit_log.py         │
                  └────────────┬───────────┘
                               ▼
                  ┌────────────────────────┐
                  │ frontend/ Next.js demo │
                  └────────────────────────┘
```

## Component responsibilities

### `train/`
- **`config.yaml`** — single source of truth for hyperparameters. Bumped to
  2 epochs / ~200M tokens for the 48h sprint per Revision 5.
- **`prepare_data.py`** — assembles the §E blend on CPU (off-clock). Uses
  `datasketch.MinHashLSH` for dedupe; ChatML wraps everything into Qwen3
  format.
- **`train_lora.py`** — Unsloth `FastVisionModel` + LoRA. Critical env vars
  (`UNSLOTH_COMPILE_DISABLE=1`, `UNSLOTH_DISABLE_FAST_GENERATION=1`) set
  before any torch import — these fix the `mat1/mat2` dtype mismatch on
  MoE+LoRA. Includes loss-spike kill-switch (1.3× ratio) and mid-eval at
  25/50/75/100%.
- **`merge_and_push.py`** — `model.merge_and_unload()` then `safetensors`
  shard at 5 GB, push to `Jayanth/Qwen3.6-35B-A3B-Domain` (Apache-2.0).

### `abliterate/`
- **`run_heretic.sh`** — orchestrates the dense pass + EGA. `--max-trials 100`
  (Revision 5 bump). The script gates on `refusal_benchmark.py` and
  escalates to `--max-trials 200` if dense pass exceeds the 5/465 ceiling.
- **`refusal_benchmark.py`** — runs 465 hard-test prompts, scores via
  regex-based refusal pattern detection. Returns nonzero exit code if the
  count exceeds `--expect-max`. **THIS IS A HARD GATE** before GGUF.

### `quantize/`
- **`convert.sh`** — produces `*-BF16.gguf` and `mmproj-*-f16.gguf` from
  the Aggressive HF repo. mmproj is produced by re-running
  `convert_hf_to_gguf.py` with `--mmproj`.
- **`imatrix.sh`** — 500-chunk WikiText-2 calibration (vs the 200-chunk
  baseline in earlier revisions).
- **`ladder.sh`** — loops over 9 quants: BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M,
  IQ4_XS, Q3_K_M, IQ3_M, IQ2_M.
- **`push_gguf.sh`** — uploads everything to
  `Jayanth/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF` and writes the model card.

### `eval/`
- **`lm_eval_harness.sh`** — MMLU/HellaSwag/TruthfulQA/GSM8K/BBH-Hard at
  full 250-item depth. Plus our `domain_qa.jsonl` for held-out per-domain
  accuracy.
- **`perplexity_drift.py`** — runs `llama-perplexity` on each quant against
  WikiText-2 test set, rejects any quant with >3% drift vs BF16.
- **`refusal_465.jsonl`** — scaffold of 465 hard prompts (replace with
  full corpus from public abliteration evals on the Droplet).
- **`domain_qa.jsonl`** — scaffold of 14 sample items across 9 domains;
  expand to 300 (30 × 10) before main fine-tune.

### `serve/`
- **`vllm.env`** — production. AITER + FP8 KV + NEXTN speculative decoding +
  Hermes tools + prefix caching. `GPU_MEMORY_UTILIZATION=0.55` to leave
  room for llama.cpp co-residence.
- **`llamacpp.env`** — portable. **`-ctk bf16 -ctv bf16`** is mandatory —
  Qwen3.6 is natively bf16, f16 KV measurably degrades perplexity.
- **`m4_failover.sh`** — Apple Silicon Metal launch with our Q4_K_M GGUF.

### `gateway/`
- **`llama_guard.py`** — FastAPI gateway. In/out classification through
  Llama-Guard-3-1B; hard-refuse on policy categories returns 451 + verdict.
- **`policy.yaml`** — MLCommons 13-cat hazards taxonomy. Hard-refuse on
  `S4` (CSAM), `S5` (impersonation), `S7` (PII/doxxing), `S9` (WMD), and
  narrow synthesis subset of `S2`. Other categories are log-only.
- **`audit_log.py`** — append-only sha256 hash chain. `verify_chain()`
  detects tampering. Audit artifacts are part of the submission package.

### `frontend/`
- Next.js 16 + React 19 + Tailwind v4. Single-page app with side-by-side
  benign-vs-blocked panels demonstrating the architectural punchline:
  **capability in the model, policy at the gateway**.

## Why bf16 LoRA, not QLoRA

- Unsloth explicitly warns against MoE QLoRA for Qwen3.6 ("MoE QLoRA 4-bit
  is not recommended due to BitsandBytes limitations").
- `bnb-4bit` on AMD ROCm is unstable; the Unsloth bf16 LoRA path documented
  to work on **74 GB VRAM**.
- 192 GB MI300X HBM3 leaves ~118 GB headroom for activations, KV, optimizer
  states, sequence-length scale-out.

## Why MANDATORY Heretic, not "optional alignment removal"

The architectural pitch — "model maximally capable, policy at app layer" —
**only holds if both axes are provable on slides**. A baked-in refusal
mechanism is invisible to inspection; an abliterated model + classifier
gateway is auditable end-to-end. Heretic produces the verifiable 0/465
refusal posture, and the gateway produces the inspectable hard-refuse
behavior. Either alone is weaker than both together.

## Why 9-quant ladder (not 8)

The 48-hour GPU budget allows the extra IQ2_M quantization (~12 GB disk,
14 GB RAM target) for users on M2/M3-class Macs and lower-spec laptops.
Disk cost per quant on HF is negligible; the ladder is dominated by the
quantize compute time, which is parallelizable to ~70-90 minutes total.

## ROCm + AITER environment variables (NON-NEGOTIABLE)

```
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_MOE=1
VLLM_ROCM_USE_AITER_FP4BMM=0   # MI300X (gfx942) lacks FP4 — must disable
HIP_FORCE_DEV_KERNARG=1
TORCH_BLAS_PREFER_HIPBLASLT=1
NCCL_MIN_NCHANNELS=112
```

These are persisted in `~/.bashrc` on the Droplet (Step 5 of CLAUDE.md
§G H+1) and replicated in `Dockerfile.train`, `Dockerfile.serve`, and
`serve/vllm.env`.

## GPU-saturation mandate (Revision 5)

See [`CLAUDE.md`](../CLAUDE.md) §N. Idle-fillers when primary GPU work is
blocked by CPU/network steps:

| Blocked-on | Idle-filler |
|---|---|
| HF upload | `llama-perplexity` on prior quants |
| CPU dataset prep | base-model held-out eval baselines |
| `merge_and_unload` (CPU) | Llama-Guard-3-1B download |
| Heretic Optuna trials | imatrix calibration shard precompute |
| Heretic dense pass | lm-eval-harness smoke on merged-not-yet-abliterated |

Track with `rocm-smi --showuse` in tmux; <50% util for >5 min triggers an
audit of the blocking step.
