# amd-hackathon-2026

**Domain-specialized, vision-capable, MIT-licensed `Qwen3.6-35B-A3B` fine-tune for the [AMD Developer Hackathon](https://lablab.ai/ai-hackathons/amd-developer) — built on a single AMD Instinct MI300X (192 GB HBM3) under a $100 credit cap, mandatorily re-abliterated to a verified 0/465 refusal posture, served by vLLM (production) and llama.cpp-server (portable), shielded by Llama-Guard-3-1B at the gateway.**

Track: **Vision & Multimodal** (parallel Build-in-Public narrative).

---

## Architecture

```
                  ┌─────────────────────────────────────────────────────────┐
                  │           AMD Instinct MI300X (192 GB HBM3)             │
                  │                                                         │
  ┌──────────┐    │   ┌──────────┐   ┌──────────┐   ┌──────────────────┐    │
  │ Datasets │ ─► │   │ Unsloth  │ ► │   Merge  │ ► │  Heretic         │    │
  │ §E blend │    │   │ bf16 LoRA│   │  & Push  │   │  abliteration    │    │
  │ ~200M tok│    │   │ FastVLM  │   │  HF Hub  │   │ (MANDATORY,      │    │
  └──────────┘    │   │ MoE-aware│   │          │   │  EGA, ≤5/465)    │    │
                  │   └──────────┘   └──────────┘   └────────┬─────────┘    │
                  │                                          │              │
                  │   ┌──────────────────────────────────────▼──────────┐   │
                  │   │  GGUF ladder: BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, │   │
                  │   │  IQ4_XS, Q3_K_M, IQ3_M, IQ2_M  +  mmproj-f16    │   │
                  │   └──────────────────────────────────┬──────────────┘   │
                  │                                      │                  │
                  │   ┌──────────────┐    ┌──────────────▼──────┐           │
                  │   │ vLLM :8000   │    │ llama.cpp-server    │           │
                  │   │ FP8 KV       │    │ :8001  -ngl 99 -fa  │           │
                  │   │ MTP, prefix  │    │ -ctk bf16 -ctv bf16 │           │
                  │   │ AITER on     │    │ vision via mmproj   │           │
                  │   └──────┬───────┘    └─────────┬───────────┘           │
                  └──────────┼──────────────────────┼───────────────────────┘
                             │                      │
                             ▼                      ▼
                  ┌──────────────────────────────────────────┐
                  │   Private VPS Gateway (FastAPI)          │
                  │   • Llama-Guard-3-1B in/out filtering    │
                  │   • policy.yaml — MLCommons 13-cat       │
                  │   • Append-only audit log                │
                  └──────────────┬───────────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────────┐
                  │   Next.js demo client                    │
                  │   • Side-by-side benign vs blocked panes │
                  │   • Live audit-log surface               │
                  └──────────────────────────────────────────┘
```

**Pitch**: Capability and policy *decouple cleanly*. The model is maximally capable (verified 0/465 refusals via [Heretic](https://github.com/p-e-w/heretic)). Policy lives at the gateway as inspectable YAML + a small classifier. Switching jurisdictions = a `policy.yaml` edit, not a retrain.

---

## Domains covered (~20% of training mix)

Indian law (BNS / BNSS / BSA), Indian taxation (IT Act, GST, CBDT/CBIC), OSINT/cyber (ATT&CK + D3FEND, public CTI), pharmacology education (DrugBank free, openFDA), quantitative finance (Indian-market-specific), engineering (civil/mechanical/EE), geopolitics + game theory, psychology (sanitized clinical), geospatial (OSM + ISRO open data).

General-instruct (~65%) and reasoning/math/tool-calling (~15%) blend per [`CLAUDE.md`](CLAUDE.md) §E.

---

## Repository layout

```
amd-hackathon-2026/
├── CLAUDE.md                # Master spec (Revision 5, 48h sprint)
├── README.md                # This file
├── LICENSE                  # MIT
├── pyproject.toml
├── Dockerfile.{train,serve}
├── docker-compose.yml
├── train/                   # Unsloth bf16 LoRA pipeline
├── abliterate/              # MANDATORY Heretic step + 465-prompt gate
├── quantize/                # GGUF ladder (9 quants + mmproj)
├── eval/                    # lm-eval, perplexity drift, refusal benchmark
├── serve/                   # vLLM + llama.cpp-server + M4 failover
├── gateway/                 # FastAPI + Llama-Guard-3-1B + audit log
├── frontend/                # Next.js demo client
└── docs/                    # Architecture, BIP cadence, safety
```

## Quant table (post-build)

| Quant | Disk | RAM target | Use case |
|---|---|---|---|
| BF16 | ~70 GB | ≥80 GB | reference / vLLM merged-model GGUF |
| Q8_0 | ~37 GB | ≥48 GB | desktop high-quality |
| Q6_K | ~29 GB | ≥36 GB | balanced quality |
| Q5_K_M | ~25 GB | ≥32 GB | sweet spot |
| **Q4_K_M** | **~21 GB** | **≥24 GB** | **default for M4 failover and most users** |
| IQ4_XS | ~19 GB | ≥22 GB | low-RAM machines |
| Q3_K_M | ~17 GB | ≥20 GB | aggressive |
| IQ3_M | ~15 GB | ≥18 GB | extreme |
| IQ2_M | ~12 GB | ≥14 GB | last resort |

(Sizes approximate, will be measured post-build.)

## Quick start (after submission)

```bash
# llama.cpp-server (Apple Silicon Metal)
./llama-server -m Qwen3.6-35B-A3B-Domain-Aggressive-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.6-35B-A3B-Domain-Aggressive-f16.gguf \
  --jinja -c 32768 -ngl 99 --host 0.0.0.0 --port 8001
```

## Sampling recommendations

`temperature=0.7`, `top_p=0.95`, `presence_penalty=1.5`.

## License chain

- Upstream Qwen3.6 weights → **Apache-2.0**
- Our LoRA + merged + abliterated weights → **Apache-2.0** (preserves upstream)
- This repository's wrapper code (training scripts, gateway, frontend) → **MIT**
- Heretic abliteration tool → **AGPL-3.0** (tool only — model weights produced are NOT virally relicensed)

## Refusal posture

Verified **0/465 refusals** post-Heretic. Verification artifacts: [`eval/refusal_results_aggressive.json`](eval/refusal_results_aggressive.json) (post-build). Policy enforcement happens at the **gateway**, not in weights. See [`docs/SAFETY.md`](docs/SAFETY.md).

## Build status

| Phase | Status |
|---|---|
| Spec frozen (Revision 5) | ✅ |
| Repo scaffolded | ✅ |
| MI300X Droplet | ⏳ |
| Fine-tune complete | ⏳ |
| Heretic abliteration ≤5/465 | ⏳ |
| GGUF ladder pushed | ⏳ |
| Demo recorded | ⏳ |
| Submitted (lablab.ai) | ⏳ (May 4 window) |

---

Solo build by **Jayanth** ([@Jayanth-reflex](https://github.com/Jayanth-reflex)) · India · April 2026
Build-in-public thread: https://x.com/Jayanth — `#AMDDevHackathon` `#MI300X`
