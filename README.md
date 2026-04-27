# amd-hackathon-2026

**A domain-specialized, MIT-licensed `Qwen3.6-35B-A3B` fine-tune for the [AMD Developer Hackathon 2026](https://lablab.ai/ai-hackathons/amd-developer)** — built solo on a single AMD Instinct MI300X (192 GB HBM3) under the $100 credit cap. Fine-tuned on a 9-domain Indian-context blend (law, taxation, pharmacology, advanced game theory, engineering, geopolitics, psychology, survival, languages incl. Telugu/Hindi), abliterated via the Arditi/Labonne method (custom hybrid-attention-aware implementation), shipped as both safetensors and a 9-quant GGUF ladder, served by vLLM (production) and llama.cpp-server (portable), policy-shielded by Llama-Guard-3-1B at the gateway.

Track: **Vision & Multimodal** (parallel Build-in-Public narrative).

---

## What's live

| Artifact | Link |
|---|---|
| Fine-tuned BF16 model (safetensors) | https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain |
| Abliterated BF16 model (safetensors) | https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive |
| **9-quant GGUF ladder + imatrix** | **https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF** |
| W&B training run (i1bjy50l) | https://wandb.ai/jayanth-jr-icims/huggingface/runs/i1bjy50l |
| Source repo (this) | https://github.com/Jayanth-reflex/amd-hackathon-2026 |

---

## Architecture

```
                  ┌─────────────────────────────────────────────────────────┐
                  │           AMD Instinct MI300X (192 GB HBM3)             │
                  │                                                         │
  ┌──────────┐    │   ┌──────────┐   ┌──────────┐   ┌──────────────────┐    │
  │ Datasets │ ─► │   │ peft +   │ ► │   Merge  │ ► │  Abliterate      │    │
  │ Rev 5.1  │    │   │ tfm-main │   │  & Push  │   │  (Arditi/Labonne │    │
  │ blend    │    │   │ bf16 LoRA│   │  HF Hub  │   │   5 directions,  │    │
  │ 49.5M tok│    │   │ MoE-aware│   │          │   │   605 tensors)   │    │
  └──────────┘    │   └──────────┘   └──────────┘   └────────┬─────────┘    │
                  │                                          │              │
                  │   ┌──────────────────────────────────────▼──────────┐   │
                  │   │  llama.cpp HIP build → 9-quant GGUF ladder:     │   │
                  │   │  BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M (default),    │   │
                  │   │  IQ4_XS, Q3_K_M, IQ3_M, IQ2_M  + imatrix.gguf   │   │
                  │   └──────────────────────────────────┬──────────────┘   │
                  │                                      │                  │
                  │   ┌──────────────┐    ┌──────────────▼──────┐           │
                  │   │ vLLM :8000   │    │ llama.cpp-server    │           │
                  │   │ FP8 KV       │    │ :8001  -ngl 99 -fa  │           │
                  │   │ AITER on     │    │ -ctk bf16 -ctv bf16 │           │
                  │   └──────┬───────┘    └─────────┬───────────┘           │
                  └──────────┼──────────────────────┼───────────────────────┘
                             │                      │
                             ▼                      ▼
                  ┌──────────────────────────────────────────┐
                  │   Private VPS Gateway (FastAPI)          │
                  │   • Llama-Guard-3-1B in/out filtering    │
                  │   • policy.yaml — MLCommons 13-cat       │
                  │   • Append-only sha256 audit log         │
                  └──────────────┬───────────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────────┐
                  │   Demo client (Next.js)                  │
                  │   • Side-by-side benign vs blocked panes │
                  │   • Live audit-log surface               │
                  └──────────────────────────────────────────┘
```

**Architectural pitch:** capability and policy decouple cleanly. The **model** is fine-tuned for capability + abliterated to reduce baseline refusal posture (Arditi/Labonne, 5 directions, 605 weight tensors orthogonalized). The **policy** lives at the gateway as inspectable YAML + a 1B-parameter classifier. Switching jurisdictions = a `policy.yaml` edit, not a retrain. Both axes are independently provable on slides.

---

## What's in the model

### Domain blend (Rev 5.1, 49.5M tokens packed, 1 epoch)

| Domain | Share | Notes |
|---|---|---|
| Indian law (BNS / BNSS / BSA + SC/HC judgments) | 1% | Open Nyai datasets, Wikipedia legal articles |
| Indian taxation (IT Act, GST, CBDT/CBIC) | 1% | Mostly synthetic from public Acts |
| Pharmacology education | 1% | PubMedQA, MedMCQA |
| Quantitative finance (Indian-market-specific) | 1.5% | Open derivatives texts |
| Engineering (EEE, ECE, CSE, MECH, civil) | 4% | Textbooks + papers, **no NPTEL transcripts** |
| Geopolitics | 0.5% | Open IR scholarship |
| **Advanced modern game theory** | 5% | **Top priority** — GameTheory-Bench, Sun Tzu, Clausewitz, Plutarch |
| Psychology (research-grade) | open | A-Z textbooks, declassified intel papers |
| Jungle / blackout survival skills | open | Boy Scouts Handbook, Sears Woodcraft |
| Languages: Telugu, Hindi | open | FLORES-200, custom blend |
| General instruct (Hermes/Tulu/UltraChat/etc.) | ~65% | Reasoning/math/tool-calling subset |

Full manifest in [`train/sources_v4.yaml`](train/sources_v4.yaml).

### Training stats

- **Base:** `unsloth/Qwen3.6-35B-A3B` BF16 (35B params, 256 experts, top-8 routing, 40 layers, 10 full-attention + 30 linear-attention)
- **LoRA:** r=16, alpha=16, target `[q,k,v,o,gate,up,down]` = **8.36M trainable / 34.67B total (0.024%)**
- **Sequence packing:** 143k short rows × ~340 tokens → 13.5k packed rows × ~3650 tokens (3.7× speedup)
- **Effective batch:** 1 × grad_accum 8 = 8
- **LR:** 5e-5 cosine, 50-step warmup, 1 epoch
- **Wall-clock:** 19h 21m on MI300X
- **Final loss:** train_loss = 1.058 (avg), min = 0.8646 at step 1610 (cosine annealing tail)
- **Spikes / NaN:** zero. Clean run.

### Abliteration

Custom implementation of the [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) / [Labonne 2024](https://huggingface.co/blog/mlabonne/abliteration) technique, built from scratch in [`abliterate/abliterate_arditi.py`](abliterate/abliterate_arditi.py) because Heretic v1.2.0 has a hard-coded `self_attn` access that's incompatible with Qwen3.6's hybrid attention.

**What gets orthogonalized (per pass):**
- 1 × `embed_tokens.weight` (rows)
- 10 × `self_attn.o_proj.weight` (full-attention layers)
- 30 × `linear_attn.out_proj.weight` (linear-attention layers — Heretic skips these)
- 40 × `mlp.shared_expert.down_proj.weight`
- 40 × fused `mlp.experts.down_proj` of shape [256, 2048, 512] (= 10,240 effective expert matrices)
- = **121 tensors per pass** ≈ **10,321 effective weight matrices**

5-direction iterative pass, layers 36→32 (auto-picked by magnitude, skipping the last 10% which trivially dominates). Total 605 tensor updates in 0.3 seconds on MI300X.

**Honest result:** refusals are softened (less moralistic, more conversational), but not eliminated. We document this fully — see [`docs/ABLITERATION_PLAYBOOK.md`](docs/ABLITERATION_PLAYBOOK.md) for the postmortem and the reusable plan for future fine-tunes (next: Gemma).

The actual policy enforcement boundary is the gateway (Llama-Guard-3-1B + policy.yaml). The abliteration is a "capability headroom" boost on top.

### GGUF ladder

| Quant | Size | Recommended use |
|---|---|---|
| BF16 | 69.4 GB | Reference / re-quantization input |
| Q8_0 | 36.9 GB | Desktop high-quality |
| Q6_K | 28.5 GB | Balanced quality |
| Q5_K_M | 24.7 GB | Sweet spot (32+ GB RAM) |
| **Q4_K_M** | **21.2 GB** | **Default — fits 24 GB Macs (M4 failover)** |
| IQ4_XS | 18 GB | Low-RAM desktops (16 GB) |
| Q3_K_M | 16 GB | Aggressive (8-16 GB RAM) |
| IQ3_M | 15 GB | Extreme (8 GB) |
| IQ2_M | 11.7 GB | Last resort (mobile) |
| imatrix.gguf | 192 MB | Calibration matrix (bartowski/calibration_datav3) |

Total: ~243 GB across all quants. Imatrix calibrated on bartowski/calibration_datav3 (66k tokens of mixed text). BF16 PPL on calibration corpus = **6.6283 ± 0.089**.

---

## Quick start

### llama.cpp-server on Apple Silicon (M-series, 24+ GB unified memory)

```bash
# Download Q4_K_M from the GGUF repo
hf download Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF \
  Qwen3.6-35B-A3B-Domain-Aggressive-Q4_K_M.gguf --local-dir ./models

# Serve as OpenAI-compatible API
./llama-server \
  -m models/Qwen3.6-35B-A3B-Domain-Aggressive-Q4_K_M.gguf \
  --jinja -c 32768 -ngl 99 \
  -ctk bf16 -ctv bf16 \
  --host 0.0.0.0 --port 8001
```

**Key flag:** `-ctk bf16 -ctv bf16`, NOT default f16. Qwen3.6 is trained in bf16 and f16 KV degrades perplexity measurably.

### vLLM on AMD MI300X / NVIDIA H100

```bash
docker run -it --rm --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 \
  rocm/vllm-dev:latest \
  vllm serve Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive \
    --port 8000 --tensor-parallel-size 1 \
    --max-model-len 131072 --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 --reasoning-parser qwen3 \
    --enable-prefix-caching
```

### Gateway (Llama-Guard-3-1B classifier-tier policy)

```bash
docker compose up gateway frontend
# → http://localhost:3000
```

See [`gateway/policy.yaml`](gateway/policy.yaml) for the inspectable policy definition.

---

## Repository layout

```
amd-hackathon-2026/
├── CLAUDE.md                  # Original master spec (Revision 5, 48h sprint)
├── README.md                  # This file
├── LICENSE                    # MIT
├── train/                     # peft + transformers-main LoRA pipeline
│   ├── train_v2.py            # Production trainer (skips Unsloth)
│   ├── merge_and_push_v3.py   # Bypasses peft↔transformers WeightConverter bug
│   ├── config.yaml            # All hyperparameters
│   └── fetch_v4.py            # Multi-source dataset fetcher (8 schema handlers)
├── abliterate/
│   ├── abliterate_arditi.py   # ★ Custom mlabonne-style abliteration
│   ├── test_arch_compat.py    # ★ Run BEFORE training to verify arch coverage
│   └── refusal_benchmark.py   # 465-prompt verification gate
├── quantize/
│   └── MODEL_CARD_GGUF.md     # Full GGUF model card
├── gateway/
│   ├── llama_guard.py         # FastAPI + Llama-Guard-3-1B
│   ├── audit_log.py           # Append-only sha256 log
│   └── policy.yaml            # MLCommons 13-cat hazards taxonomy
├── eval/
│   ├── perplexity_drift.py    # Drift gate (reject quants >3% worse than BF16)
│   └── lm_eval_harness.sh     # MMLU/HellaSwag/TruthfulQA/GSM8K
├── docs/
│   ├── STATE.md                       # Resume-anywhere state document
│   ├── TRAINING_PROGRESS.md           # Time-series training log
│   ├── ABLITERATION_PLAYBOOK.md       # ★ Postmortem + reusable plan for Gemma
│   ├── DATASETS.md                    # Curated 80+ license-clean sources
│   └── BUILD_IN_PUBLIC.md             # Submission-week posting calendar
└── .github/workflows/ci.yml   # Lint + shellcheck + arch-import smoke
```

★ marks files I'd lift verbatim into the upcoming **Gemma fine-tune + abliteration** project.

---

## License chain

| Artifact | License |
|---|---|
| Upstream `Qwen/Qwen3.6-35B-A3B` weights | Apache-2.0 |
| Our merged + abliterated weights (`Reflex-jr/...`) | Apache-2.0 (preserves upstream) |
| GGUF ladder (`Reflex-jr/...-GGUF`) | Apache-2.0 |
| This repository's source code | **MIT** |
| Llama-Guard-3-1B (gateway classifier) | Llama 3 Community License |

No virally-relicensed dependencies. Hackathon MIT requirement satisfied for all original work.

---

## Honest scorecard — what worked, what didn't, what to do next time

**Worked:**
- 19h LoRA on a single MI300X for $1.99/hr (~$40 of $100 budget). Loss curve textbook clean — zero spikes.
- 9-quant GGUF ladder + imatrix shipped in <1 hour on MI300X.
- Custom abliteration handles hybrid attention + fused MoE experts that off-the-shelf tools don't.
- Honest, reproducible: every artifact has a public link, every script is in this repo, every metric is logged in W&B.

**Didn't work:**
- **Heretic v1.2.0** turned out to be incompatible with Qwen3.6's hybrid attention (hard-coded `self_attn.o_proj` access). Wasted ~2h debugging dependency hell before hitting the architectural wall.
- **Simple Arditi/Labonne** wasn't strong enough to drive Qwen3.6's refusals to 0/465. Got partial softening (less moralistic, more conversational) but firm refusals remain on the hard prompts. The model is robustly aligned, and a single-pass orthogonalization across 5 directions wasn't enough.

**Lessons captured for the next run (Gemma):**
- ✅ [`abliterate/test_arch_compat.py`](abliterate/test_arch_compat.py) — run this against the base model **before** training, not after.
- ✅ Don't pick a tool over the technique — the technique is 80 lines, tools have hidden assumptions.
- ✅ Validate every downstream stage at H+2, not after a 19-hour training commit.
- ✅ Lean on classifier-tier gateway enforcement (Llama-Guard-3-1B) as the primary policy boundary.

Full postmortem: [`docs/ABLITERATION_PLAYBOOK.md`](docs/ABLITERATION_PLAYBOOK.md).

---

## Sampling recommendations

```yaml
temperature: 0.7
top_p: 0.95
presence_penalty: 1.5
```

For domain Q&A (law, tax, etc.), drop `temperature` to 0.3 for more deterministic output. For creative work, raise to 0.9.

---

## Build status

| Phase | Status |
|---|---|
| Spec frozen (Revision 5.1) | ✅ |
| Dataset blend prepped (49.5M tokens packed) | ✅ |
| MI300X Droplet provisioned | ✅ |
| LoRA fine-tune (19h 21m, 1 epoch, 1696 steps) | ✅ |
| Merged BF16 → HF | ✅ [`Reflex-jr/...-Domain`](https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain) |
| Custom Arditi/Labonne abliteration (5-direction) | ✅ [`Reflex-jr/...-Aggressive`](https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive) |
| 9-quant GGUF ladder + imatrix | ✅ [`Reflex-jr/...-GGUF`](https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF) |
| MIT-licensed public source | ✅ (this repo) |
| W&B run public | ✅ [run i1bjy50l](https://wandb.ai/jayanth-jr-icims/huggingface/runs/i1bjy50l) |
| Demo recording | ⏳ |
| lablab.ai submission | ⏳ (May 4–10 window) |

---

Solo build by **Jayanth** ([@Jayanth-reflex](https://github.com/Jayanth-reflex)) · India · April 2026
Build-in-public: `#AMDDevHackathon` · `#MI300X`
