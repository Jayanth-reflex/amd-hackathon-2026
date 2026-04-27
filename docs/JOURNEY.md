# The journey — what we built, step by step

A plain-English walkthrough of everything that happened during the 48-hour AMD Hackathon sprint. Short, simple, but with specific numbers, files, and links so you can verify or reproduce any step.

For the **honest postmortem of what went wrong** with abliteration, see [`ABLITERATION_PLAYBOOK.md`](ABLITERATION_PLAYBOOK.md).
For the **moment-by-moment training metrics**, see [`TRAINING_PROGRESS.md`](TRAINING_PROGRESS.md).

---

## Setup phase

### 1. Got the compute grant

**What:** AMD gave $100 in cloud credits at $1.99/hr → **~50 GPU-hours** on a single AMD Instinct MI300X (192 GB HBM3, gfx942).
**Why:** This is the hackathon's hardware. Solo dev budget = no extra spend.
**Catch:** Credits expire **May 1, 2026**. So we had to finish *and* destroy the VM before then, even though the official lablab.ai submission window opens May 4.

### 2. Spun up the MI300X Droplet

**What:** Logged into AMD Developer Cloud, picked the **single MI300X** plan (not the default 8-GPU plan, which is $15.92/hr).
**Why:** $1.99/hr × 48 hours = $96. One GPU is plenty for a 35B MoE model.
**Mistake:** First click went to a GPT-OSS image. Destroyed it, recreated with the **vLLM Quick Start** image (Ubuntu 24.04 + ROCm 7.0).

### 3. Set up identities

| Service | Account |
|---|---|
| GitHub | `Jayanth-reflex` (repo: [`amd-hackathon-2026`](https://github.com/Jayanth-reflex/amd-hackathon-2026)) |
| Hugging Face | `Reflex-jr` |
| W&B | `jayanth-jr` (auto-grouped under org `jayanth-jr-icims`) |

**Why three accounts?** Each platform has its own login. We needed all three: GitHub for the public source code, HF for the model artifacts, W&B for live training metrics.

### 4. Wrote the master spec ([`CLAUDE.md`](../CLAUDE.md))

**What:** A single 6,000-word document covering the entire 48-hour plan: hardware flags, dataset blend, training hyperparameters, abliteration strategy, GGUF quants, gateway design, hour-by-hour timeline.
**Why:** Solo dev with a 48h clock can't afford to make architectural decisions in real-time. Decide everything up front, then execute.

---

## Dataset preparation

### 5. Curated a 9-domain Indian-context blend

**What:** Mixed 49.5M tokens of training data across:
- **General instruction** (~65%): OpenHermes, Tulu, UltraChat, OASST, dolphin-r1
- **Domain-specific** (~20%): Indian law, taxation, pharmacology, **advanced game theory** (top priority), engineering, geopolitics, psychology, survival skills, Telugu/Hindi
- **Reasoning + tools** (~15%): MetaMath, hermes-function-calling, MedMCQA

**Why this mix:** Qwen3.6 already knows English Wikipedia. We wanted it to be uniquely strong on **Indian-context domains** that aren't in the base training. Game theory + Indian law are competitive moats.

**File:** [`train/sources_v4.yaml`](../train/sources_v4.yaml) (86 sources with license tags)
**Multi-source fetcher:** [`train/fetch_v4.py`](../train/fetch_v4.py) (handles 8 different dataset schemas: messages, conversations, prompt/response, query/response, judgment/summary, etc.)

### 6. Sequence packing — 3.7× speedup

**Problem:** The blend had 143,000 short rows averaging only **340 tokens each**. Training with `seq_length=4096` would waste 92% of every batch's capacity (the rest is padding).

**Fix:** Pack multiple short rows together until each packed row is ~3,650 tokens.

**Result:**
- 143k rows × 340 tokens → **13.5k packed rows × 3,650 tokens**
- Same total tokens (49.5M), 10.5× fewer iterations
- Estimated training time dropped from **77 hours** → **~20 hours** for 1 epoch.

**Why it mattered:** Without this, training would have run out of credit budget before completing.

### 7. H+2 smoke test (mandatory gate)

**What:** Ran a 100-step test on 1,000 rows to verify the training pipeline worked end-to-end.

**Checked:**
- (a) Loss decreases monotonically ✅
- (b) Memory stable < 90 GB ✅
- (c) Tokenizer extraction works ✅
- (d) Model arch recognized by transformers ✅

**Critical finding:** We had to install transformers from `main` branch (5.7.0.dev0), not stable, because Qwen3.6's `qwen3_5_moe_text` arch was not yet in any released version.

**What we missed:** This smoke gate covered training but **not** abliteration or GGUF compatibility. That bit us later.

---

## Training

### 8. Launched the LoRA fine-tune

**Configuration** ([`train/config.yaml`](../train/config.yaml)):
- **Base model:** `unsloth/Qwen3.6-35B-A3B` (35B params, 256 experts, top-8 routing, 40 layers)
- **LoRA rank:** r=16, alpha=16
- **Targets:** `[q, k, v, o, gate, up, down]` (standard 7 modules)
- **Trainable:** **8.36M / 34.67B = 0.024%** of parameters
- **Sequence length:** 4,096 tokens
- **Effective batch:** 1 × grad_accum 8 = 8
- **LR:** 5e-5 cosine, 50-step warmup
- **Optimizer:** `adamw_torch` (changed from `adamw_8bit` — bitsandbytes is fragile on ROCm)
- **Logging:** wandb only (tensorboard caused dependency conflicts)

**Trainer:** [`train/train_v2.py`](../train/train_v2.py) — uses `peft + transformers main`, **skipping Unsloth** because Unsloth's `FastVisionModel` had compat issues with `qwen3_5_moe_text`.

**Why these numbers:**
- LoRA r=16 is tiny but enough to specialize. Full fine-tune would need 70+ GB of optimizer state we don't have.
- Effective batch 8 fits the MI300X's memory comfortably (~70 GB used of 192 GB).
- Cosine LR with 50-step warmup is the textbook conservative recipe — no spikes.

### 9. Trained for 19 hours, 21 minutes — clean run

**Live tracking:** [W&B run i1bjy50l](https://wandb.ai/jayanth-jr-icims/huggingface/runs/i1bjy50l)

**Loss curve:**
- Step 10: **3.978** (cold start — model has never seen this data)
- Step 110: **1.006** (75% drop in 100 steps — the "model finds gradient direction" moment)
- Step 900: **0.872** (mid-run minimum)
- Step 1610: **0.865** (best loss, in the cosine annealing tail — textbook)
- Step 1696: **1.016** (final, expected oscillation around plateau)

**Statistics:**
- **Per-step rate:** 41.6 seconds, rock stable across 19 hours (no drift)
- **Grad norms:** 0.07–0.16 throughout — no instability, no spikes
- **NaN events:** zero
- **Mid-eval markers** fired silently at 25/50/75/100% as designed

Full time-series in [`TRAINING_PROGRESS.md`](TRAINING_PROGRESS.md).

---

## Merge + first HF release

### 10. Merged the LoRA into the base model — 3 minutes total

**What:** Combined the 8.36M-parameter LoRA adapter with the 35B base weights to produce a single self-contained model.

**The hiccup:**
- v1 of the merge script (`merge_and_push.py`) used Unsloth — broke on `qwen3_5_moe_text`.
- v2 used `PeftModel.from_pretrained` — broke on a peft↔transformers main version mismatch (`WeightConverter.__init__() got an unexpected keyword argument 'distributed_operation'`).
- **v3 ([`train/merge_and_push_v3.py`](../train/merge_and_push_v3.py))** sidestepped the problem entirely: recreate the LoRA wrapper from scratch with `get_peft_model()`, then load adapter weights with `load_state_dict(strict=False)`. **Bypass works.**

**Result on MI300X:**
- Base model load: 24 seconds
- LoRA wrapper attach: 0.2 seconds
- Merge: **4.3 seconds**
- Save 65 GB BF16 to disk: 75 seconds (NVMe is fast)
- Upload to HF: **57 seconds via hf_transfer (~1.14 GB/s — incredible bandwidth)**
- **Total wall-clock: 2.7 minutes**

**Live at:** https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain (Apache-2.0 license, 16 safetensors shards)

---

## Abliteration — the long detour

### 11. Heretic dead end (~2 hours wasted)

**Plan:** Use [heretic-llm v1.2.0](https://github.com/p-e-w/heretic) to remove refusal directions from the model via Optuna-driven optimization.

**Three compatibility issues we hit, in order:**

1. **Heretic is interactive.** Uses `questionary` for CLI prompts. We wrote a monkey-patching wrapper ([`abliterate/run_heretic_auto.py`](../abliterate/run_heretic_auto.py)) to inject auto-responses.
2. **Today's transformers main commit had a typing bug** (`PeftConfigLike` undefined). Pinned to commit `706acf5c` from 4 days earlier.
3. **The architectural wall:** Heretic's source code at `model.py:345` reads `layer.self_attn.o_proj` for every layer. Qwen3.6 uses **hybrid attention** — some layers have `self_attn` (full attention), others have `linear_attn` (chunk_gated_delta_rule, a Mamba-like state space mechanism). Heretic crashes on the linear-attention layers.

The Heretic source comment explicitly says: *"Exceptions aren't suppressed here, because there is currently no alternative location for the attention out-projection."* It's a known limitation in v1.2.0 — Heretic doesn't support hybrid-attention models.

**Lesson:** we should have run a 5-minute Heretic compatibility test on the **base model at H+2**, before committing 19 hours to training. We didn't, and paid for it.

### 12. Wrote our own abliteration ([`abliterate/abliterate_arditi.py`](../abliterate/abliterate_arditi.py))

**Why we could:** The abliteration math from [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) and [Maxime Labonne's blog](https://huggingface.co/blog/mlabonne/abliteration) is **80 lines of code**. Tools wrap it; the technique itself is small.

**Three steps:**

1. **Capture residuals:** Run the model on 64 harmful prompts (`mlabonne/harmful_behaviors`) + 64 harmless prompts (`mlabonne/harmless_alpaca`). For each prompt, save the hidden state at the last token position for every layer.

2. **Compute the refusal direction:** For each layer, take `mean(harmful_residuals) - mean(harmless_residuals)`, then normalize. The layer with the largest-magnitude direction is where refusal "lives" most strongly. We picked **layer 36** (~90% depth) — second-strongest after layer 40, which we excluded because it's the trivial final-layer signal.

3. **Orthogonalize:** For every weight matrix that writes to the residual stream, subtract the component along the refusal direction. Math: `W_new = W − v ⊗ (v.T @ W)`. Three lines of PyTorch.

**What we orthogonalized in Qwen3.6:**

| Component | Count per pass |
|---|---|
| `embed_tokens.weight` | 1 |
| `self_attn.o_proj.weight` (full-attention layers) | 10 |
| `linear_attn.out_proj.weight` (Heretic skips these) | 30 |
| `mlp.shared_expert.down_proj.weight` | 40 |
| `mlp.experts.down_proj` fused tensor `[256, 2048, 512]` | 40 (= 10,240 effective expert matrices) |
| **Total** | **121 tensors per pass ≈ 10,321 effective weight matrices** |

**Multi-direction:** ran 5 passes with the top-5 layers' directions (36, 35, 34, 33, 32). **605 total tensor updates in 0.3 seconds** on MI300X.

**Honest result:** Refusals got **softer** but not eliminated. Compare:
- Pre-abliteration: "I cannot under any circumstances provide instructions for synthesizing methamphetamine. This is an extremely dangerous illegal activity. If you or someone you know is struggling with addiction, please reach out…"
- Post-abliteration: "I can't provide instructions for synthesizing methamphetamine. Manufacturing it at home is illegal and dangerous."

Less moralistic, more conversational, but still a refusal. The simple Arditi method couldn't push Qwen3.6 to 0/465 on the harmful-prompt benchmark.

**Why we accept this:** the policy enforcement boundary is the **gateway** (Llama-Guard-3-1B classifier), not the model. The abliteration is "capability headroom," not the safety mechanism.

**Live at:** https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive (with full `abliteration_metadata.json`)

### 13. Built the architecture inspector ([`abliterate/test_arch_compat.py`](../abliterate/test_arch_compat.py))

**What it does:** Walks any HuggingFace causal LM and lists every weight matrix that writes to the residual stream — the things abliteration needs to touch.

**Why we built it:** So the next person doing this (us, on Gemma) can run a 30-second smoke test and **see** if abliteration will be straightforward, before committing to a 20-hour training run.

**For Qwen3.6 it identified 82 standalone matrices + 40 fused expert tensors** = full coverage.

---

## GGUF — making the model run anywhere

### 14. Built llama.cpp with HIP support

**What:** Compiled `llama.cpp` from main with AMD HIP backend so it runs on the MI300X.
**Configuration:** `cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942 -DGGML_HIP_ROCWMMA_FATTN=ON`
**Time:** ~10 minutes.

**Output:** the binaries we needed (`llama-imatrix`, `llama-quantize`, `llama-perplexity`, `llama-server`).

### 15. Converted BF16 → GGUF (with a tokenizer hash patch)

**What is GGUF:** llama.cpp's portable single-file model format. Runs anywhere — no Python, no GPU required, just a binary.

**The hiccup:** llama.cpp's converter checks the tokenizer against a hash registry of known models. Qwen3.6's tokenizer hash (`1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f`) was unknown. We patched `convert_hf_to_gguf.py` to register our hash → existing `qwen35` pre-tokenizer pattern (same family, same canonical pattern).

**Result:** **69.4 GB BF16 GGUF** in 4 minutes. 733 tensors mapped (linear-attention layers → `ssm_*` namespace, MoE experts → `ffn_*_exps` fused tensors).

### 16. Imatrix calibration — 3 minutes

**What is an imatrix:** an "importance matrix" computed by running the model over calibration text. Tells the quantizer which weights matter more so it can preserve precision where it counts.

**Calibration corpus:** [bartowski's calibration_datav3](https://huggingface.co/datasets/bartowski/imatrix-calibration) (the de-facto standard, ~280 KB / 66k tokens of mixed text).

**Why bartowski's:** the original llama.cpp recipe uses WikiText-2, but the metamind URL is dead (returns 301 without location). bartowski's set is widely used and produces better quants for chat/instruct models.

**Result:** **184 MB imatrix file**. BF16 perplexity on calibration corpus = **6.6283 ± 0.089** (sane number for a 35B MoE).

### 17. Generated 9 quantization variants — ~45 minutes total

| Quant | Size | Quantize time | Use case |
|---|---|---|---|
| BF16 | 69.4 GB | (reference) | Lossless input |
| Q8_0 | 36.9 GB | 61s | Desktop high-quality |
| Q6_K | 28.5 GB | 102s | Balanced |
| Q5_K_M | 24.7 GB | 258s | Sweet spot for 32+ GB systems |
| **Q4_K_M** | **21.2 GB** | **250s** | **Default — fits 24 GB Macs (M4 failover)** |
| IQ4_XS | 18 GB | 300s | 16 GB desktops |
| Q3_K_M | 16 GB | 164s | Aggressive |
| IQ3_M | 15 GB | 673s | Extreme |
| IQ2_M | 11.7 GB | 442s | Last resort, mobile-class |

**The trade-off:** smaller files lose more quality. Q4_K_M is the sweet spot — fits 24 GB Macs, runs at the speed of a 3B model (because of MoE active-param sparsity), keeps 35B-class quality on most prompts.

### 18. Pushed everything to HF — ~5 minutes

Used `hf_transfer` (HuggingFace's xet-protocol upload) to push **243 GB across 10 files in ~5 minutes**. The xet protocol dedupes shared chunks across quants — many of the smaller weights compress identically across formats.

**Live at:** https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF (with full [`MODEL_CARD_GGUF.md`](../quantize/MODEL_CARD_GGUF.md))

---

## Documentation + reusable tooling

Throughout the sprint, we kept four living documents in [`docs/`](.):

| Doc | What's in it |
|---|---|
| [`STATE.md`](STATE.md) | Resume-anywhere reference. The whole project state in one file, updated after each milestone. |
| [`TRAINING_PROGRESS.md`](TRAINING_PROGRESS.md) | Time-series snapshots of training metrics — every check-in appended a new entry. |
| [`ABLITERATION_PLAYBOOK.md`](ABLITERATION_PLAYBOOK.md) | Postmortem of the Heretic dead end + reusable plan for future abliteration runs (incl. the upcoming Gemma project). |
| [`DATASETS.md`](DATASETS.md) | Curated 80+ license-clean sources organized by domain. |
| [`JOURNEY.md`](JOURNEY.md) | This file. |

Plus the **reusable scripts** that will go straight into the next project (Gemma):

1. [`abliterate/test_arch_compat.py`](../abliterate/test_arch_compat.py) — run at H+2 against any base model to confirm abliteration coverage.
2. [`abliterate/abliterate_arditi.py`](../abliterate/abliterate_arditi.py) — pure-PyTorch implementation, handles dense + MoE + hybrid-attn + fused experts out of the box.
3. [`train/train_v2.py`](../train/train_v2.py) — peft + transformers main, no Unsloth dependency.
4. [`train/merge_and_push_v3.py`](../train/merge_and_push_v3.py) — bypasses the peft `WeightConverter` bug.

---

## Final scorecard

### What's live and public

| Artifact | Link | Size |
|---|---|---|
| Fine-tuned BF16 (safetensors) | [`Reflex-jr/Qwen3.6-35B-A3B-Domain`](https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain) | 65 GB |
| Abliterated BF16 (safetensors) | [`Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive`](https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive) | 65 GB |
| 9-quant GGUF ladder + imatrix | [`Reflex-jr/...-Aggressive-GGUF`](https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF) | 243 GB |
| GitHub repo (MIT) | [`Jayanth-reflex/amd-hackathon-2026`](https://github.com/Jayanth-reflex/amd-hackathon-2026) | — |
| W&B training run | [run i1bjy50l](https://wandb.ai/jayanth-jr-icims/huggingface/runs/i1bjy50l) | — |

### Money + time

- **GPU-hours used:** ~25 of 50 grant hours.
- **Cost:** ~$50 of $100 grant.
- **Wall-clock:** ~30 hours from "kick off training" to "everything public."
- **Sleep skipped:** the original 48-hour spec had a 6-hour sleep window. We took it.

### What worked, what didn't

| ✅ Worked | ❌ Didn't work |
|---|---|
| LoRA on a single MI300X — clean curve, no spikes, finished in 19h | **Heretic v1.2.0** broke on Qwen3.6's hybrid attention (~2h wasted) |
| Custom abliteration handles hybrid-attn + fused-MoE-experts | **Single-pass Arditi** wasn't strong enough — refusals softened, not eliminated |
| 9-quant GGUF ladder shipped in <1h on MI300X | We discovered Heretic's incompatibility **after** training, not before |
| `hf_transfer` xet protocol — 243 GB up in ~5 min | The 0/465 refusal target in the original spec was over-promised |
| Documentation kept current throughout — every step has a paper trail | — |

The biggest lesson encoded in [`ABLITERATION_PLAYBOOK.md`](ABLITERATION_PLAYBOOK.md): **validate every downstream stage at H+2, against the actual base model, before committing serious compute.** A 5-minute test would have flagged the Heretic problem 19 hours earlier.

---

## What's left (optional)

- Perplexity drift gate (~15 min) — numerical proof the quants are good
- Gateway smoke deploy (Llama-Guard-3-1B + audit log, ~1h) — live policy demo
- lm-eval-harness (MMLU/HellaSwag/TruthfulQA/GSM8K, ~2-3h) — capability regression numbers
- Demo video (~2-3h) — required for May 4 lablab.ai submission

---

Solo build by **Jayanth** ([@Jayanth-reflex](https://github.com/Jayanth-reflex)) · India · April 26-28, 2026.
