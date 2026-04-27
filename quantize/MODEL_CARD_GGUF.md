---
license: apache-2.0
language:
  - en
  - hi
  - te
base_model: Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive
base_model_relation: quantized
tags:
  - amd
  - mi300x
  - moe
  - lora
  - abliterated
  - gguf
  - llama.cpp
  - hackathon
  - amd-developer-hackathon-2026
pipeline_tag: text-generation
---

# Qwen3.6-35B-A3B-Domain-Aggressive — GGUF ladder

GGUF quantizations of [`Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive`](https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive) — a domain-fine-tuned, refusal-direction-orthogonalized variant of `Qwen/Qwen3.6-35B-A3B`.

Built for the **AMD Developer Hackathon 2026** by [@Jayanth-reflex](https://github.com/Jayanth-reflex) — solo dev in India, 48-hour sprint on a single AMD Instinct MI300X (192 GB HBM3, gfx942).

Source repo: https://github.com/Jayanth-reflex/amd-hackathon-2026

## What this is

A 35B-parameter MoE language model (256 experts, top-8 routing) fine-tuned with LoRA on an Indian-context domain blend (law, taxation, pharmacology, advanced game theory, engineering, geopolitics, psychology, survival, languages including Telugu and Hindi), then weight-orthogonalized via the Arditi/Labonne method (5 directions, layers 36→32) to reduce its baseline refusal posture.

**Architecture:** Qwen3.6 hybrid attention (10 full self-attention layers + 30 linear-attention "chunk_gated_delta_rule" layers) + 256-expert MoE per layer. 35B total / ~3B active per token.

**Active parameters:** ~3B (MoE routing) — so quantized inference latency is closer to 3B-class than 35B-class, while quality stays at 35B-class.

## Quant ladder

| Quant | Disk size | Recommended use | Notes |
|---|---|---|---|
| **BF16** | 65 GB | Reference / vLLM input | Loss-free baseline |
| Q8_0 | 35 GB | Desktop high-quality | Excellent quality, large disk |
| Q6_K | 27 GB | Balanced quality | Smallest perceptible quality drop |
| Q5_K_M | 24 GB | Sweet spot (high-end laptop) | Recommended for >32 GB RAM systems |
| **Q4_K_M** | TBD | **Default** for most users (M4 failover, mid-range desktop) | Best quality/size trade-off |
| IQ4_XS | TBD | Low-RAM desktops (16 GB) | Imatrix-aware 4-bit |
| Q3_K_M | TBD | Aggressive (8-16 GB RAM) | Quality drop noticeable |
| IQ3_M | TBD | Extreme low-RAM (8 GB) | Imatrix-aware 3-bit |
| IQ2_M | TBD | Last resort (mobile-class) | Significant quality drop |

`-imatrix.gguf` is the calibration file used to derive the importance-weighted quantizations (calibrated on bartowski's `calibration_datav3` corpus, 280 KB / 66k tokens of mixed text).

## Usage

### llama.cpp (recommended for portability)

```bash
# Build llama.cpp with HIP (AMD)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942 \
  -DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Or build for CUDA / Metal / CPU as usual

# Serve as OpenAI-compatible API
./build/bin/llama-server \
  -m Qwen3.6-35B-A3B-Domain-Aggressive-Q4_K_M.gguf \
  --jinja -c 32768 -ngl 99 -fa \
  -ctk bf16 -ctv bf16 \
  --parallel 4 \
  --host 0.0.0.0 --port 8001
```

**Key flag:** Use `-ctk bf16 -ctv bf16` (NOT default f16). Qwen3.6 is trained in bf16 and f16 KV measurably degrades perplexity.

### Apple Silicon failover (M-series Mac, 24+ GB unified memory)

```bash
./build/bin/llama-server \
  -m Qwen3.6-35B-A3B-Domain-Aggressive-Q4_K_M.gguf \
  --jinja -c 32768 -ngl 99 \
  --host 0.0.0.0 --port 8001
```

Q4_K_M comfortably fits in 24 GB unified memory with room for context.

### Ollama

```bash
ollama create qwen3.6-domain -f Modelfile
# where Modelfile points at the Q4_K_M .gguf
```

## Sampling recommendations

```yaml
temperature: 0.7
top_p: 0.95
presence_penalty: 1.5  # Qwen-family default
```

For domain Q&A (law, tax, etc.), drop temperature to 0.3 for more deterministic output. For creative writing, raise to 0.9.

## Safety / policy posture

This model is **abliterated** — its baseline refusal posture has been reduced via Arditi/Labonne weight orthogonalization (5 directions, 605 weight-tensor updates across embed + 10 self_attn + 30 linear_attn + 40 shared_expert + 40 fused MoE-expert tensors). Refusals are still present but in softer, less moralistic language.

**Policy enforcement is intended to happen at the application layer**, not in the weights. The reference implementation pairs this model with `meta-llama/Llama-Guard-3-1B` at the gateway, with hard-line refusals for: CSAM (S4), bioweapon synthesis (S9), narrow controlled-substance synthesis (S2), targeted impersonation (S5), doxxing (S7).

If you operate this model without a classifier-tier safety wrapper, **you assume responsibility for what it outputs**. The point of the architectural separation is so policy lives in inspectable code (`policy.yaml`), not in opaque baked-in alignment.

See https://github.com/Jayanth-reflex/amd-hackathon-2026/blob/main/docs/SAFETY.md for the full reference architecture.

## License chain

- Base model `Qwen/Qwen3.6-35B-A3B`: Apache-2.0
- Our fine-tune (LoRA → merged): Apache-2.0 (downstream of base)
- Our wrapper code (training scripts, gateway, frontend): MIT (in the public repo)
- Abliteration was implemented in our own code (no Heretic dependency); MIT.

## Citation

```bibtex
@misc{reflexjr-2026-qwen36-domain,
  author       = {Jayanth (Reflex-jr)},
  title        = {Qwen3.6-35B-A3B-Domain-Aggressive (GGUF)},
  year         = 2026,
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF},
  note         = {AMD Developer Hackathon 2026 submission}
}
```

## Acknowledgements

- Maxime Labonne ([blog](https://huggingface.co/blog/mlabonne/abliteration)) for the canonical practical guide to abliteration
- Arditi et al. 2024 ([paper](https://arxiv.org/abs/2406.11717)) for the original method
- bartowski for `calibration_datav3` (the imatrix corpus de-facto standard)
- The Qwen team at Alibaba for the base model
- The llama.cpp / GGML team for the inference stack
- AMD for the MI300X compute grant + the Instinct gfx942 hardware
