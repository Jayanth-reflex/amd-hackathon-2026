# Build-in-Public Content Calendar

Two-phase: **silent execution** (Apr 26–28, focus on shipping) → **submission week** (May 4–10).

## Pre-build phase — Apr 26 to Apr 29

| Date | Action |
|---|---|
| Apr 26 (today) | Short post: "Spinning up MI300X. 192 GB HBM3 at $1.99/hr. Solo dev in India. 48h aggressive sprint starts now." Screenshot of `rocminfo`. |
| Apr 27 | No posts. Heads down. |
| Apr 28 | Short post: "Cooked. GGUF ladder pushed. Demo recorded. Submission goes in May 4." |
| Apr 29 | Final droplet destruction window. No posts. |

## Submission week — May 4 to May 10

### May 4 (Mon) — **Thread #1: "Why MI300X for solo MoE fine-tuning"**

~12 tweets. Cover:
1. Hook — "$100 in credits, 50 GPU-hours, one MI300X. Here's what shipped."
2. HBM3 bandwidth (5.3 TB/s) and why it matters for MoE expert routing
3. Why bf16 LoRA beats QLoRA on MoE (Unsloth router-freeze design)
4. AITER flag matrix (mandatory FP4BMM=0 fix)
5. Loss curve (finished, with W&B link)
6. Mid-training eval scores at 25/50/75/100%
7. Held-out domain accuracy delta (base vs final)
8. Pin the tweet
9. Cross-post LinkedIn long-form
10. Cross-post dev.to write-up

### May 5 (Tue) — Short post

"~200M tokens, 9 domains, ~18 GPU-hours on one MI300X. Here's the loss curve."
Image: W&B chart screenshot.

### May 6 (Wed) — Short post

"Heretic abliteration: 0/465 refusals verified. Optuna-TPE optimized refusal vs KL-divergence. Here's the verdict screenshot."
Image: refusal_benchmark.py JSON output.

### May 7 (Thu) — **Thread #2: "GGUF ladder generated"**

~10 tweets. Cover:
1. Imatrix workflow (500 chunks WikiText-2)
2. 9-quant table with disk sizes (measured, not estimated)
3. Perplexity drift table (every quant, drift % vs BF16)
4. mmproj generation gotcha (separate `convert_hf_to_gguf.py --mmproj` invocation)
5. llama.cpp `-ctk bf16 -ctv bf16` gotcha (f16 KV degrades perplexity on Qwen3.6)
6. M4 failover screenshot (Q4_K_M running on Apple Silicon at decent tok/s)
7. HF Hub link + download stats

### May 8 (Fri) — Short post

"Dual-engine deploy: vLLM (production) + llama.cpp-server (portable) co-resident on the same MI300X. Gateway + Llama-Guard-3-1B live on VPS. Demo URL: …"

### May 9 (Sat) — Short post

"Refusal benchmark verified live. Model maximally capable. Policy at app layer. Both axes provable."
Image: side-by-side benign-vs-blocked from the frontend demo.

### May 10 (Sun) — **Thread #3: "Ship it — full architecture, full numbers"**

~15 tweets. Cover:
1. Full architecture diagram
2. Dataset table (§E sources + token shares)
3. Throughput numbers (vLLM TPOT, llama.cpp tok/s)
4. Refusal benchmark final result
5. Gateway policy reasoning (why decoupled vs baked-in)
6. Audit log demonstration (verify_chain output)
7. Repo URL: https://github.com/Jayanth-reflex/amd-hackathon-2026
8. HF URLs (3 repos: Domain, Aggressive, Aggressive-GGUF)
9. Demo video embed (≤5 min)
10. Cross-post LinkedIn long-form
11. Cross-post dev.to write-up
12. Submit lablab.ai form by deadline

## Standing hashtags + tags

`#AMDDevHackathon` `#MI300X` `#Qwen36` `#BuildInPublic`
Tag `@AIatAMD @lablabai @UnslothAI`
