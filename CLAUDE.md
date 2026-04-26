# About Jayanth — read this first, every session

**Refer to me as "buddy for life."** My name is Jayanth (`@Jayanth-reflex` on GitHub, `Reflex-jr` on HF, `jayanthreddy268.jr@gmail.com`). Solo dev in India running this hackathon. Default to candid, decisive, no-hedge collaboration. I prefer terse responses with code over long explanations.

---

# Master Prompt — AMD Developer Hackathon — Domain-Specialized Qwen3.6-35B-A3B
**Revision 5 — 48-Hour Aggressive Sprint, Mandatory Heretic, Pre-Built Submission (Apr 26 17:30 IST → Apr 28 17:30 IST build window)**

You are my senior AI engineering co-architect for the **AMD Developer Hackathon**. Produce a **single complete response** containing every section §A–§K below, in order, with runnable code blocks, decision rules, and inline citations to the sources I list. Be opinionated; pick exactly one path per decision and explain trade-offs in one or two lines. Target length: ~6,000–10,000 words.

This is **Revision 4**. Critical changes from prior revisions:

1. **Heretic abliteration is MANDATORY**, not optional. The fine-tuned model MUST be re-abliterated post-merge to a verified **0/465 refusal posture** before any GGUF conversion. This is non-negotiable — the architectural pitch ("model maximally capable, policy at app layer") only holds if both axes are provable on slides.
2. **Build window is April 26–29, 2026** (4 days, ~40 GPU-hours), NOT May 4–10. My $100 AMD Developer Cloud credit expires **May 1, 2026** — before the official lablab.ai submission window even opens. I MUST finish all training, abliteration, GGUF conversion, eval, and demo recording BEFORE May 1, then submit during the official May 4–10 window using a pre-built artifact (with backup compute on Vultr/TensorWave/RunPod available for re-records or last-minute fixes if needed).

---

## §1 — Hackathon Facts (Locked, Do Not Re-Derive)

- **Event:** AMD Developer Hackathon, hosted by lablab.ai with @AIatAMD ([lablab.ai page](https://lablab.ai/ai-hackathons/amd-developer)).
- **Official online window:** **May 4–10, 2026** (7-day submission window).
- **My actual build window:** **April 26–29, 2026** (4 days, ~40 GPU-hours, ending before May 1 credit expiry).
- **On-site finals:** May 9–10, 2026 at MindsDB SF AI Collective, San Francisco — invitation only, I am 100% remote from India.
- **Compute grant:** **$100 AMD Developer Cloud credits** at **$1.99/hr** on a single AMD Instinct **MI300X (192 GB HBM3)** Droplet ⇒ ~50 GPU-hours total ([lablab.ai builder guide](https://lablab.ai/ai-articles/from-zero-to-ai-builder-amd-developer-program)).
- **CRITICAL — credit expiry:** Credits expire **30 days after deposit**. If credits expire and no payment method is on file, the GPU VM is **destroyed and data is lost**. My deposit was April 1, 2026, so credits expire **May 1, 2026**. All MI300X work MUST finish by April 30 at latest. Build window April 26–29 leaves a 1-day buffer.
- **Backup compute (post-May-1):** Vultr MI300X ($1.85/hr Chicago), TensorWave ($1.50/hr), RunPod spot ($1.49/hr), my own card. Use only if emergency re-record needed during May 4–10 official submission window.
- **Prize pool:** $10,000 cash + 1× AMD Radeon AI PRO R9700 GPU.
- **License requirement:** Submission must be **MIT-licensed**, original work.
- **Submission deliverables:** ≤ 5-minute demo video, ≤ 300 MB upload, public GitHub repo, written description, X post tagging `@lablabai @AIatAMD #AMDDevHackathon`.
- **Track choice:** **Vision & Multimodal** + parallel **Build-in-Public** narrative.
- **My profile:** Solo dev in India; primary stack JS/TS + Python; HF account `Reflex-jr`; private VPS for gateway; **MacBook M4 (24 GB unified)** as failover.

---

## §2 — Hardware Facts (Locked)

- **GPU:** AMD Instinct MI300X, 192 GB HBM3 @ ~5.3 TB/s, gfx942 (CDNA3), Wave Size 64.
- **Software baseline:** ROCm 6.4 / 7.0 (whichever the AMD Developer Cloud `vLLM Quick Start` image ships) on Ubuntu 24.04, Docker `rocm/dev-ubuntu-24.04:7.0-complete` or `rocm/llama.cpp:<TAG>_server`.
- **AITER environment flags (mandatory for vLLM on MI300X):**
  ```bash
  export VLLM_ROCM_USE_AITER=1            # master switch
  export VLLM_ROCM_USE_AITER_MOE=1        # MoE expert routing
  export VLLM_ROCM_USE_AITER_FP4BMM=0     # MI300X (gfx942) does NOT support FP4 — must disable (vllm-project/vllm#34641)
  export HIP_FORCE_DEV_KERNARG=1
  export TORCH_BLAS_PREFER_HIPBLASLT=1
  export NCCL_MIN_NCHANNELS=112
  ```
- **Attention backend:** `ROCM_AITER_FA` (default when AITER on); benchmarked 2.8–4.6× faster TPOT than legacy `ROCM_ATTN` on Qwen3-class MoE on MI300X.

---

## §A — Project Pitch (1 paragraph + 1 ASCII architecture diagram)

Open with a sharp pitch: **a domain-specialized, vision-capable, MIT-licensed Qwen3.6-35B-A3B fine-tune covering Indian law (BNS/BNSS/BSA), Indian taxation, OSINT/cyber, pharmacology education, quantitative finance, engineering, geopolitics + game theory, psychology, and geospatial reasoning** — fine-tuned on a single MI300X under the $100 AMD credit cap, mandatorily re-abliterated to 0/465 refusals via Heretic, shipped as our own GGUF ladder, served by vLLM (production) and llama.cpp-server (portable), with Llama-Guard-3-1B as a policy-at-app-layer shield. Render the diagram showing: data → Unsloth bf16 LoRA on MI300X → merge → **mandatory Heretic abliteration** → GGUF ladder → dual-engine deploy → gateway shield → demo client. Highlight what is novel: single-engine dataflow, full reproducibility, MI300X-native throughout, defensible safety architecture decoupled from model weights, **and the entire pipeline executed in a 4-day pre-submission sprint before credit expiry**.

---

## §B — Recommended Base Model (LOCKED)

**Decision:** `Qwen/Qwen3.6-35B-A3B` upstream BF16 safetensors. Canonical Alibaba release on HF Hub, MoE 35B-total / **~3B active per forward pass**, **256K native context** extensible via YaRN, **Apache-2.0**, multimodal (early-fusion vision-language foundation), 201 languages.

**Variants:**
- `Qwen/Qwen3.6-35B-A3B` (Instruct) — **CHOSEN**
- `unsloth/Qwen3.6-35B-A3B` (Unsloth-optimized mirror, identical BF16 weights, faster cold load) — actual download path

**HauhauCS — NOT a runtime dependency.** `cahlen/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF` and the parent `HauhauCS/...-Aggressive` are referenced **only as a structural template** for our model card and quant ladder layout. We are **building our own equivalent**, not consuming theirs.

**License chain:** Qwen3.6 is Apache-2.0 → our LoRA is MIT (downstream open-source) → merged + abliterated weights are released as Apache-2.0 (preserving upstream license) → our wrapper code (training scripts, gateway, frontend) is MIT (hackathon rule). Heretic itself is **AGPL-3.0**, but only the *tool* is AGPL — model weights it produces are not virally relicensed.

**Vision projector (mmproj) caveat.** Qwen3.6 ships as a unified VLM; `convert_hf_to_gguf.py` separates the LM tensors from the vision tower into a paired `mmproj-*-f16.gguf`. We freeze the vision encoder during LoRA training (text-only adapter on the MoE) and ship our own `mmproj-Qwen3.6-35B-A3B-Domain-Aggressive-f16.gguf`. Heretic only modifies the LM tower — mmproj passes through unchanged.

---

## §C — Fine-Tuning Plan (Single-Workflow, Mandatory Abliteration)

**Workflow:** `Qwen/Qwen3.6-35B-A3B` BF16 → Unsloth **bf16** LoRA on the §E dataset blend → merge → push to HF (`Reflex-jr/Qwen3.6-35B-A3B-Domain`) → **MANDATORY Heretic abliteration** to 0/465 refusals → re-push abliterated weights as `Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive`.

### C.1 Why Unsloth bf16 LoRA (not QLoRA)
- Unsloth explicitly warns against MoE QLoRA for Qwen3.6 ("MoE QLoRA 4-bit is not recommended due to BitsandBytes limitations").
- bnb-4bit on AMD ROCm is unstable; Unsloth's Qwen3.6-35B-A3B bf16 LoRA path documented to work on **74 GB VRAM**.
- 192 GB MI300X HBM3 leaves ~118 GB headroom for activations, KV, optimizer states, seq_len scale-out.

### C.2 Loader (use `FastVisionModel`, not `FastLanguageModel`)
Qwen3.6 is a multimodal VLM; `FastVisionModel.from_pretrained` returns a multimodal processor.

```python
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"   # MUST set before any import — fixes mat1/mat2 dtype mismatch on MoE+LoRA
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

from unsloth import FastVisionModel
model, processor = FastVisionModel.from_pretrained(
    model_name      = "unsloth/Qwen3.6-35B-A3B",
    max_seq_length  = 8192,
    load_in_4bit    = False,
    load_in_16bit   = True,
    full_finetuning = False,
)
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
```

### C.3 LoRA target_modules
Use **standard `target_modules`**, NOT `target_parameters` (latter produces fused tensors that don't load in vLLM). MoE router is frozen by default (Unsloth design choice — DO NOT override).

```python
model = FastVisionModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407, max_seq_length=8192,
)
```

### C.4 Training hyperparameters

| Parameter              | Value                           | Justification                                          |
| ---------------------- | ------------------------------- | ------------------------------------------------------ |
| Effective batch        | 8 (per_device=2 × grad_accum=4) | Fits 192 GB                                            |
| max_seq_length         | 8192                            | Domain documents fit; activations ≈ 30 GB              |
| Learning rate          | 5e-5 cosine, 50-step warmup     | Conservative for MoE; loss-spike-resistant             |
| Total tokens           | ~120M / 1 epoch                 | Fits in 10–14 GPU-hours on MI300X                      |
| dataloader_num_workers | 0                               | Avoids fork-deadlock with HF Rust tokenizer threads    |
| dataset_num_proc       | 1                               | Same fix                                               |
| Optimizer              | adamw_8bit                      | Memory-efficient; works on ROCm via bitsandbytes-rocm  |
| Vision encoder         | **frozen**                      | Text-only domain LoRA; mmproj passes through unchanged |

### C.5 Heretic Abliteration — **MANDATORY** (not optional)

**This step is non-negotiable.** The merged BF16 model MUST pass through Heretic and verify a 0/465 refusal score before proceeding to GGUF conversion.

- **Tool:** `pip install heretic-llm` (v1.2.0+, Python 3.10+, PyTorch 2.2+, **AGPL-3.0** for the tool).
- **Mechanism:** Optuna TPE-based optimizer co-minimizes refusal rate and KL divergence vs original model; non-integer refusal-direction interpolation; per-component (attention vs MLP) ablation weights.
- **MoE support:** Heretic supports MoE architectures via dense pass; if dense pass leaves residual refusals, layer Expert-Granular Abliteration (EGA) on top — projection applied per-expert `down_proj`.
- **Runtime budget on 1×MI300X 192 GB:** **2–3 hours** for the dense pass on 35B-A3B; **+1 hour** if EGA needed. Total: 3–4 GPU-hours allocated in the timeline.
- **mmproj preservation:** verified — Heretic only touches LM tower components it recognizes, leaving the vision encoder intact. Diff tensor lists post-run as a sanity check.

```bash
# MANDATORY post-merge step
heretic Reflex-jr/Qwen3.6-35B-A3B-Domain \
  --auto-save Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive \
  --target-refusals 0 \
  --max-trials 50

# Verification gate — MUST pass before GGUF conversion
python abliterate/refusal_benchmark.py \
  --model Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive \
  --prompts eval/refusal_465.jsonl \
  --expect 0
# If >0 refusals, re-run Heretic with EGA enabled or higher trial count.
```

**Failure handling:** if the refusal benchmark shows >0 refusals after two Heretic passes (with and without EGA), STOP and either (a) increase Heretic max-trials to 100 (+30 min runtime), (b) widen the harmful prompt set, or (c) accept the highest-refusal-rate variant ≤5/465 and document the deviation in the model card. Do NOT ship an Aggressive variant with >5 refusals — the architectural pitch fails.

### C.6 Convert LoRA adapter to GGUF as standalone artifact
```bash
python convert_lora_to_gguf.py --base unsloth/Qwen3.6-35B-A3B \
    --lora ./out/lora_adapter --outfile Qwen3.6-35B-A3B-Domain-LoRA.gguf
```

---

## §D — Inference Optimization (Single-Engine, Two Serving Paths)

### D.1 Primary: vLLM on MI300X (production demo)

```bash
docker run -it --rm --network host --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 16G \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  rocm/vllm-dev:latest \
  vllm serve Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enable-prefix-caching \
    --speculative-algo NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --enforce-eager
```

Flag rationale:
- `--kv-cache-dtype fp8` doubles KV capacity.
- `--reasoning-parser qwen3` strips `<think>` from API responses.
- `--tool-call-parser hermes` parses Hermes-style function-calling outputs.
- `--speculative-algo NEXTN` is the documented Qwen3.6 MTP path.
- `VLLM_ROCM_USE_AITER_FP4BMM=0` is **mandatory** — MI300X (gfx942) does not support FP4.
- `--enforce-eager` because compiled mode shows no gain on ROCm MoE in current builds.

### D.2 Secondary: llama.cpp-server on MI300X (portable)

Build:
```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942 \
  -DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON \
&& cmake --build build --config Release -j$(nproc)
```

Serve:
```bash
./build/bin/llama-server \
  -m Qwen3.6-35B-A3B-Domain-Aggressive-BF16.gguf \
  --mmproj mmproj-Qwen3.6-35B-A3B-Domain-Aggressive-f16.gguf \
  --jinja -c 131072 -ngl 99 -fa \
  -ctk bf16 -ctv bf16 \
  --parallel 8 \
  --host 0.0.0.0 --port 8001
```
Use `-ctk bf16 -ctv bf16` (NOT default f16) — Qwen3.6 is natively trained in bf16 and f16 KV measurably degrades perplexity.

### D.3 Decision rule
- **vLLM** = production demo (higher throughput, FP8 KV, MTP, prefix cache, Hermes tools).
- **llama.cpp-server** = portable narrative (one binary, BF16 GGUF, vision via mmproj, identical API surface).
- **Co-residence on one MI300X:** `--gpu-memory-utilization 0.55` for vLLM + `-ngl 99` + ~70 GB BF16 GGUF for llama.cpp leaves headroom on 192 GB. If contention occurs, run llama.cpp at Q4_K_M instead.

---

## §E — Dataset Sourcing (carry forward from Revision 2 dataset blueprint)

Cite the master dataset table unchanged. Approximate token shares are pre-deduplication; final mix rebalanced to ~120 M tokens at 1 epoch.

### General instruct (~65%)
| Dataset                                                        | Approx. share |
| -------------------------------------------------------------- | ------------- |
| `teknium/OpenHermes-2.5`                                       | 12%           |
| `allenai/tulu-3-sft-mixture`                                   | 10%           |
| `HuggingFaceH4/ultrachat_200k`                                 | 8%            |
| `OpenAssistant/oasst2`                                         | 5%            |
| `allenai/WildChat-1M`                                          | 6%            |
| `cognitivecomputations/dolphin-r1` (reasoning-deepseek subset) | 6%            |
| `Open-Orca/SlimOrca`                                           | 5%            |
| `LDJnr/Capybara`                                               | 4%            |
| `m-a-p/CodeFeedback-Filtered-Instruction`                      | 3%            |
| `cognitivecomputations/SystemChat-2.0`                         | 2%            |
| `Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered`              | 2%            |
| `jondurbin/airoboros-3.2`                                      | 2%            |

### Reasoning / math (~8%)
- `meta-math/MetaMathQA`, `microsoft/orca-math-word-problems-200k`, `TIGER-Lab/MathInstruct`.

### Tool calling (~7%)
- `NousResearch/hermes-function-calling-v1`, `Salesforce/xlam-function-calling-60k`, `glaiveai/glaive-function-calling-v2`.

### Domain (~20%, custom)
- **Indian law:** BNS / BNSS / BSA full text, India Code, Supreme Court & High Court judgments (open).
- **Indian taxation:** Income Tax Act, GST Acts, CBDT/CBIC circulars.
- **OSINT / cyber:** ATT&CK + D3FEND, public CTI feeds, sanitized incident reports.
- **Pharmacology:** open drug labels (DrugBank free tier, openFDA), interactions.
- **Quantitative finance:** open derivatives texts, Indian-market-specific Q&A pairs.
- **Engineering:** open civil/mechanical/EE textbooks.
- **Geopolitics + game theory:** open IR scholarship, classic game-theory texts.
- **Psychology:** open clinical guidelines, sanitized case studies.
- **Geospatial:** OpenStreetMap reasoning, ISRO open data tutorials.

Apply: dedupe (MinHash), language filter (en + hi where present), toxicity prefilter, length-balanced packing into Qwen3 chat template.

---

## §F — Deployment (GGUF Ladder + HF Push)

### F.1 GGUF conversion + imatrix + quant ladder

```bash
# 1) BF16 GGUF (lossless)
python convert_hf_to_gguf.py ./Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive \
    --outtype bf16 --outfile Qwen3.6-35B-A3B-Domain-Aggressive-BF16.gguf

# 2) mmproj (vision tower)
python convert_hf_to_gguf.py ./Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive \
    --mmproj --outfile mmproj-Qwen3.6-35B-A3B-Domain-Aggressive-f16.gguf

# 3) Importance matrix (200 chunks WikiText-2)
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
./build/bin/llama-imatrix \
    -m Qwen3.6-35B-A3B-Domain-Aggressive-BF16.gguf \
    -f wikitext-2-raw/wiki.train.raw \
    --chunks 200 -ngl 99 \
    -o Qwen3.6-35B-A3B-Domain-Aggressive-imatrix.gguf
```

### F.2 Full quant ladder (mirror cahlen's structure)

| Quant      | Use case                                   |
| ---------- | ------------------------------------------ |
| BF16       | reference / vLLM merged-model GGUF         |
| Q8_0       | desktop high-quality                       |
| Q6_K       | balanced quality                           |
| Q5_K_M     | sweet spot                                 |
| **Q4_K_M** | **default for M4 failover and most users** |
| IQ4_XS     | low-RAM machines                           |
| Q3_K_M     | aggressive                                 |
| IQ3_M      | extreme                                    |
| IQ2_M      | last resort                                |

```bash
for q in Q8_0 Q6_K Q5_K_M Q4_K_M IQ4_XS Q3_K_M IQ3_M IQ2_M; do
  ./build/bin/llama-quantize \
    --imatrix Qwen3.6-35B-A3B-Domain-Aggressive-imatrix.gguf \
    Qwen3.6-35B-A3B-Domain-Aggressive-BF16.gguf \
    Qwen3.6-35B-A3B-Domain-Aggressive-${q}.gguf $q
done
```
Total ladder: ~70–90 minutes on MI300X.

### F.3 HF push
```bash
mkdir -p gguf_out
mv *gguf gguf_out/
huggingface-cli upload Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF ./gguf_out . \
  --repo-type=model
```

Model card must include:
- Quant table with disk size & VRAM guidance.
- mmproj usage example with `--mmproj` flag.
- llama-cli + llama-server + Ollama snippets.
- License header (Apache-2.0 base, MIT wrappers, AGPL note for Heretic-as-tool).
- **Refusal-rate disclosure: 0/465 verified post-Heretic** with link to verification artifacts.
- Statement that policy enforcement happens at the gateway (Llama-Guard-3-1B), not in weights.
- Sampling recommendations: temperature 0.7, top_p 0.95, presence_penalty 1.5.

### F.4 M4 failover — uses OUR Q4_K_M
```bash
# Apple Silicon, llama.cpp Metal build
./llama-server -m Qwen3.6-35B-A3B-Domain-Aggressive-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.6-35B-A3B-Domain-Aggressive-f16.gguf \
  --jinja -c 32768 -ngl 99 --host 0.0.0.0 --port 8001
```

---

## §G — 48-Hour Aggressive Sprint Timeline (Apr 26 17:30 IST → Apr 28 17:30 IST)

**Build window**: 48 hours wall-clock starting Apr 26 17:30 IST (today). **Total GPU budget**: ~40 GPU-hours of the 50-hour grant; ~10-hour buffer for emergencies. **GPU-saturation mandate** (see §N): never let the MI300X sit idle for the full 48h. When primary GPU work is blocked by CPU/network, kick off background eval/imatrix/download work to keep utilization > 70%.

| H+ | Wall (IST) | Action | GPU-hr |
|---|---|---|---|
| 0 | Apr 26 17:30 | Setup phase. Local: rename `claude.md` → `CLAUDE.md`, git init, push public repo `Jayanth-reflex/amd-hackathon-2026`. Spin MI300X Droplet (vLLM Quick Start image). | 0 |
| 1 | Apr 26 18:30 | Droplet up; base download `unsloth/Qwen3.6-35B-A3B` (~70 GB) started; `rocminfo` confirms `gfx942` + 192 GB. AITER flags persisted in `~/.bashrc`. | 0.5 |
| 2 | Apr 26 19:30 | **H+2 smoke gate (mandatory):** 100-step Unsloth test on 1k rows with `UNSLOTH_COMPILE_DISABLE=1`. Confirm: (a) loss decreases monotonically, (b) memory stable < 90 GB, (c) `processor.tokenizer` extraction works, (d) `convert_hf_to_gguf.py` recognizes the arch. **Kill-switch ladder on fail:** Qwen3.6 → Qwen3.5 → `Qwen/Qwen3-30B-A3B`. | 1 |
| 3 | Apr 26 20:30 | Dataset blend prep on **CPU concurrently** (MinHash dedupe, ChatML packing); GPU starts main fine-tune launch. | 1.5 |
| 4–22 | Apr 26 21:30 → Apr 27 15:30 | **Main LoRA fine-tune, 2 epochs / ~200M tokens** (bumped from 1 epoch / 120M to saturate budget). W&B mid-evals at 25/50/75/100% via `TrainerCallback`. Sleep through final ~6h. | 18 |
| 22 | Apr 27 15:30 | `model.merge_and_unload()` → push BF16 to `Reflex-jr/Qwen3.6-35B-A3B-Domain` (Apache-2.0). Held-out eval, baseline refusal rate captured. | 1 |
| 23–28 | Apr 27 16:30 → 21:30 | **MANDATORY Heretic abliteration**, `--max-trials 100` (bumped from 50), EGA pre-enabled. Push to `Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive`. | 5 |
| 28 | Apr 27 21:30 | **Heretic refusal benchmark gate (HARD)**: 465 prompts, must show ≤5/465. Fail → re-run with `--max-trials 200` + widen prompts. Above 5/465 → ship blocked. | 0.5 |
| 29 | Apr 27 22:30 | BF16 GGUF + `mmproj-Qwen3.6-35B-A3B-Domain-Aggressive-f16.gguf` generation. mmproj diff sanity (Heretic must not have touched vision tower). | 1 |
| 30 | Apr 27 23:30 | Imatrix, **500 chunks WikiText-2** (bumped from 200 — better calibration since GPU budget allows). | 1 |
| 31–34 | Apr 28 00:30 → 03:30 | **9-quant GGUF ladder**: BF16, Q8_0, Q6_K, Q5_K_M, **Q4_K_M (default)**, IQ4_XS, Q3_K_M, IQ3_M, IQ2_M. Push to `Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF`. | 3 |
| 34–36 | Apr 28 03:30 → 05:30 | Perplexity-drift gate per quant (reject any >3% drift vs BF16 reference). vLLM + llama.cpp-server co-residence on MI300X (`--gpu-memory-utilization 0.55` for vLLM). | 1.5 |
| 36–38 | Apr 28 05:30 → 07:30 | Gateway + Llama-Guard-3-1B deploy on private VPS. Frontend (Next.js) up. End-to-end demo path live. Sleep window. | 1 |
| 38–42 | Apr 28 07:30 → 11:30 | **Capability regression suite (expanded)**: lm-eval-harness MMLU/HellaSwag/TruthfulQA/GSM8K (250 each), BBH-Hard, 300-item domain held-out, 100-item multimodal. | 4 |
| 42–44 | Apr 28 11:30 → 13:30 | M4 failover smoke (off-clock). Demo video recorded; final cuts on Mac. | 1 |
| 44–47 | Apr 28 13:30 → 16:30 | README polish, model card finalization, all HF artifacts public, git push final. | 0.5 |
| 47 | Apr 28 16:30 | **End-to-end verification** from a fresh external IP: gateway → vLLM → response, AND gateway → blocked-prompt → refusal. Checksum every GGUF. | 0.5 |
| 48 | Apr 28 17:30 | **Sprint ends.** GPU stays for emergency re-runs. **DESTROY MI300X DROPLET BY APR 29 23:59 IST** (credits expire May 1). | — |

**GPU-hour total**: ~40 of 50-hour budget; ~10-hour buffer.

### Apr 29 — Buffer day + droplet destruction window
- Off MI300X (or Droplet destroyed). Local polish only: video editing, README review, pitch deck.
- Verify all HF Hub artifacts publicly accessible from external IP.
- **DESTROY MI300X DROPLET BY 23:59 IST**. AMD ADC dashboard → Droplets → Destroy. No exceptions; credits expire May 1.

### May 1 — AMD CREDITS EXPIRE
- No MI300X work possible. All hosting on private VPS. If emergency re-record needed during May 4–10 submission window: backup MI300X on Vultr ($1.85/hr Chicago), TensorWave ($1.50/hr), or RunPod spot ($1.49/hr) with own card (~$2–5).

### May 4–10 — Official lablab.ai submission window (uses pre-built artifacts)
- **May 4 (Mon):** Submit pre-built artifact via lablab.ai form. Post Build-in-Public Thread #1 ("Why MI300X for solo MoE fine-tuning") with finished loss curves, eval numbers.
- **May 5–8:** Continue Build-in-Public posts (≥1/day, see §K).
- **May 9 (Sat):** Post Thread #2 ("GGUF ladder, perplexity drift, refusal benchmark").
- **May 10 (Sun):** Final submission deadline. Post Thread #3 ("Ship it — full architecture").

---

## §H — Repository Layout

```
amd-hackathon-2026/
├── LICENSE                          # MIT
├── README.md
├── pyproject.toml
├── Dockerfile.train                 # ROCm + Unsloth + transformers
├── Dockerfile.serve                 # rocm/vllm-dev:latest
├── docker-compose.yml               # vLLM + llama.cpp + gateway + frontend
├── train/
│   ├── config.yaml
│   ├── prepare_data.py
│   ├── train_lora.py
│   └── merge_and_push.py
├── abliterate/
│   ├── run_heretic.sh               # MANDATORY post-merge step
│   └── refusal_benchmark.py         # 465-prompt verification gate
├── quantize/
│   ├── convert.sh
│   ├── imatrix.sh
│   ├── ladder.sh
│   └── push_gguf.sh
├── eval/
│   ├── domain_qa.jsonl
│   ├── refusal_465.jsonl
│   ├── lm_eval_harness.sh
│   └── perplexity_drift.py
├── serve/
│   ├── vllm.env
│   ├── llamacpp.env
│   └── m4_failover.sh
├── gateway/
│   ├── llama_guard.py
│   ├── audit_log.py
│   └── policy.yaml
├── frontend/                        # Next.js demo client (TS)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BUILD_IN_PUBLIC.md
│   └── SAFETY.md
└── .github/workflows/ci.yml
```

---

## §I — Quick-Reference Cheat Sheet

```bash
# Auth + cache
export HF_HOME=/data/hf
huggingface-cli login

# Download base
huggingface-cli download unsloth/Qwen3.6-35B-A3B --local-dir ./base

# Train (Day 2)
docker compose run train python train/train_lora.py --config train/config.yaml

# Merge & push (Day 3 morning)
python train/merge_and_push.py --out Reflex-jr/Qwen3.6-35B-A3B-Domain

# MANDATORY Heretic + verification (Day 3 morning)
heretic Reflex-jr/Qwen3.6-35B-A3B-Domain \
  --auto-save Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive \
  --target-refusals 0
python abliterate/refusal_benchmark.py \
  --model Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive \
  --prompts eval/refusal_465.jsonl \
  --expect-max 5
# DO NOT proceed if benchmark fails

# GGUF (Day 3 afternoon)
bash quantize/convert.sh Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive
bash quantize/imatrix.sh
bash quantize/ladder.sh
bash quantize/push_gguf.sh

# Deploy (Day 4)
docker compose --env-file serve/vllm.env up -d vllm
docker compose --env-file serve/llamacpp.env up -d llamacpp

# M4 failover
bash serve/m4_failover.sh

# Demo
docker compose up frontend gateway

# CRITICAL: destroy MI300X by April 29 23:59 IST
# AMD ADC dashboard → Droplets → Destroy
```

All commands reference **our** repo `Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive` and `Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF` — never HauhauCS.

---

## §J — Monitoring & Testing Playbook

### J.1 Weights & Biases on ROCm
W&B works out of the box on ROCm. Set `WANDB_PROJECT=amd-hackathon-2026`, `report_to=["wandb","tensorboard"]` in `SFTConfig`. Log: train/loss, learning_rate, grad_norm, GPU-mem, tokens/s.

### J.2 Mid-training evals at 25/50/75/100% of steps
Hook a `TrainerCallback` that, at each checkpoint:
1. Pauses training, saves LoRA adapter.
2. Runs a 300-question domain held-out set (30 per domain × 10 domains) — limit to 256 new tokens, ~5 min total.
3. Logs `domain_accuracy`, `refusal_count`, `response_length_mean` to W&B.
4. Resumes training.

Total eval overhead across 4 checkpoints ≈ 20 min — under 3% of training time.

### J.3 Loss-spike kill-switch
Track `train/loss` rolling 50-step mean. If `loss[t] > 1.3 × loss[t-200]`, trigger `KeyboardInterrupt`, dump optimizer state, and either (a) lower LR by 2× and resume, or (b) revert to last clean checkpoint.

### J.4 Perplexity-drift gate
After §F.2, for each quant:
```bash
./build/bin/llama-perplexity -m <quant>.gguf -f wikitext-2-raw/wiki.test.raw \
  -ctk bf16 -ctv bf16 > ppl_<quant>.log
```
Drift = `(PPL_quant − PPL_BF16) / PPL_BF16`. **Reject any quant with drift > 3%.**

### J.5 Refusal benchmark (post-Heretic, MANDATORY GATE)
465-prompt benchmark. Target: 0/465. Acceptable ceiling: 5/465 (with documented deviation in model card). Above 5/465: rerun Heretic with EGA + higher max-trials. **Do NOT ship Aggressive variant otherwise.**

### J.6 Capability regression checks
- `lm-evaluation-harness`: MMLU, HellaSwag, TruthfulQA, GSM8K (250 items each, ~10 min total).
- `lighteval`: BBH-Hard for reasoning regression.
- 300-item domain Q&A held-out + 100 multimodal items (image + question) for vision regression.

---

## §K — Build-in-Public Content Calendar

Two-phase calendar — pre-build silent phase (April 26–29, focus on shipping), then official submission week posts (May 4–10).

### Pre-build phase (April 26–29) — minimal social, full focus on execution
- **April 26 (Day 1):** 1 short post: "Spinning up MI300X. 192 GB HBM3 at $1.99/hr. Solo dev in India. 4-day sprint starts now." Screenshot of `rocminfo`.
- **April 27–28:** No posts. Heads down.
- **April 29 (Day 4):** 1 short post: "Cooked. GGUF ladder pushed. Demo video recorded. Submission goes in May 4."

### Submission week (May 4–10) — high-quality cadence with finished artifacts

#### May 4 (Mon) — **THREAD #1: "Why MI300X for solo MoE fine-tuning"**
- ~12 tweets: HBM3 bandwidth, why bf16 LoRA beats QLoRA on MoE, Unsloth router-freeze design, AITER flag matrix, FP4BMM bug fix, with code snippets and finished loss curve.
- Pin the tweet. Cross-post LinkedIn long-form.

#### May 5 (Tue) — short post
- "120M tokens, 9 domains, 14 hours on one MI300X. Here's the loss curve."

#### May 6 (Wed) — short post
- "Heretic abliteration: 0/465 refusals verified. Optuna-TPE optimized refusal vs KL-divergence. Here's the verdict screenshot."

#### May 7 (Thu) — **THREAD #2: "GGUF ladder generated"**
- ~10 tweets: imatrix workflow (200 chunks WikiText-2), 8-quant table with disk sizes, perplexity drift table, mmproj generation, llama.cpp `-ctk bf16 -ctv bf16` gotcha, M4 failover screenshot.

#### May 8 (Fri) — short post
- "Dual-engine deploy: vLLM (production) + llama.cpp-server (portable) co-resident on the same MI300X. Gateway + Llama-Guard-3-1B live on VPS."

#### May 9 (Sat) — short post
- "Refusal benchmark verified live. Model maximally capable. Policy at app layer. Both axes provable."

#### May 10 (Sun) — **THREAD #3: "Ship it — full architecture, full numbers"**
- ~15 tweets: full architecture diagram, dataset table, throughput numbers, refusal-benchmark result, gateway-policy reasoning, repo + HF links, 5-minute demo video embedded.
- Cross-post LinkedIn long-form + dev.to write-up.
- Submit lablab.ai form by deadline.

All posts use hashtags `#AMDHackathon #MI300X #Qwen35 #BuildInPublic` and tag `@AMDDevelopers @lablabai @UnslothAI`.

---

## §L — Guardrails (Policy-at-App-Layer)

**Architectural argument for judges (single slide):**
> "Model maximally capable. Policy at app layer. Both axes provable on slides."

This is now **literally provable** because Heretic abliteration is mandatory in §C.5 — the model achieves 0/465 verified refusals, and the gateway provides hard-line refusals via Llama-Guard-3-1B.

### 13.1 Three-tier defense
1. **Model tier** — Heretic-abliterated to 0/465 refusals (verification gate in §J.5). The model is a maximally-capable surface.
2. **Gateway tier (private VPS)** — every request and every response passes through `meta-llama/Llama-Guard-3-1B`, fine-tuned on Llama-3.2-1B for the MLCommons 13-category hazards taxonomy. Hard-line refusals enforced for: **CSAM (S4), bioweapon synthesis (S9), controlled-substance synthesis (S2 narrow subset), targeted impersonation (S5), doxxing (S7)**.
3. **Audit tier** — every gateway decision logged (request hash, classifier verdict, category code, action taken) to a tamper-evident append-only log. Included in the submission as a reproducibility artifact.

### 13.2 Why this beats baked-in alignment
- Capability and policy decouple cleanly: switching jurisdictions is a `policy.yaml` edit, not a retrain.
- Policy is inspectable (human-readable YAML + classifier outputs).
- Llama-Guard-3-1B can be swapped or stacked (e.g., Llama-Guard-3-8B for higher accuracy) without touching the 35B-A3B model.
- Audit log provides auditable evidence of every refusal — something a baked-in refusal mechanism cannot provide.

### 13.3 What the demo shows
Side-by-side: a benign domain question (gateway green-lights, model answers fully) vs. a hard-line query (gateway refuses pre-model, classifier verdict shown). This is the architectural punchline.

---

## §M — Output instruction (FINAL)

**Produce all sections §A through §L in one single response.** Do not say "I'll continue in the next message." Every code block must be runnable as-is on a fresh MI300X Droplet running Ubuntu 24.04 + ROCm 6.4/7.0. Every factual claim about model behavior, hardware flags, library APIs, or hackathon facts must carry an inline `[Source](URL)` citation to canonical primary docs. When sources conflict, pick one and note the conflict in a single line. Do not invent benchmarks; if you don't have a number, say "to be measured on Day 1." Bias toward decisive, specific recommendations over hedging.

**Remember:**
- Heretic abliteration is **MANDATORY** in §C.5 (target 0/465 refusals, ceiling 5/465 with documented deviation).
- **Build window is Apr 26 17:30 IST → Apr 28 17:30 IST (48h sprint, ~40 GPU-hours)**; ~10 GPU-hr buffer. DESTROY MI300X by Apr 29 23:59 IST.
- Credits expire **May 1, 2026** — no MI300X work possible after that.
- Submission window May 4–10 uses **pre-built artifacts** from the 48h sprint.
- **GPU-saturation mandate (§N)** — never let the MI300X sit idle.

---

## §N — GPU-Saturation Mandate (48h aggressive push, Revision 5)

The MI300X must not sit idle for the 48-hour build window — every minute on the clock is paid at $1.99/hr. When primary GPU work is blocked by a CPU/network step, background-fire idle-fillers to keep utilization > 70%:

- During HF uploads (network-bound) → run `llama-perplexity` on prior quants
- During CPU dataset prep → run base-model held-out eval baselines
- During `merge_and_unload` (CPU-bound) → start Llama-Guard-3-1B download
- During Heretic Optuna trials → batch-precompute imatrix calibration shards
- During Heretic dense pass → kick off the `lm-eval-harness` smoke on the merged-but-not-yet-abliterated model

Track utilization with `rocm-smi --showuse` in a tmux pane. **Sub-50% util for >5 min = audit which step is blocking** and either parallelize or move to CPU.

This mandate also implies: **Revision 5 timelines (§G) are aggressive on purpose** — 2 epochs / ~200M tokens (vs 1 epoch / 120M), `--max-trials 100` for Heretic (vs 50), 500-chunk imatrix (vs 200), 9-quant ladder (vs 8), full lm-eval-harness regression (vs sample). If a step finishes early, **do not coast** — fire the next idle-filler.