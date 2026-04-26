# State Snapshot — amd-hackathon-2026

> **Resume-anywhere doc.** Read this first if you're picking up the project from a compacted conversation, a new session, or another collaborator. Updated 2026-04-26 22:15 IST.

---

## 0. TL;DR — where we are right this second

- **Sprint hour:** H+4.7 of 48 hours (started Apr 26 17:30 IST)
- **GPU credit used:** ~$3 of $100 (rate: $1.99/hr on idle/active MI300X)
- **Current phase:** Dataset assembly running on CPU; main 19h training launches when assembly hits ~30M tokens
- **Blocking:** none on user side
- **What's running:** `tmux dataprep` session on droplet → `python3 train/prepare_data.py` (v3 with HF token + improved normalize_record)
- **What's next:** wait for `blend.jsonl` to reach ~30M+ tokens (~20-30 more min), then launch main fine-tune

---

## 1. Sprint context (locked, do not re-derive)

| Field | Value |
|---|---|
| Hackathon | [AMD Developer Hackathon](https://lablab.ai/ai-hackathons/amd-developer) (lablab.ai × @AIatAMD) |
| Track | Vision & Multimodal + Build-in-Public narrative |
| Build window | **Apr 26 17:30 IST → Apr 28 17:30 IST** (48h sprint) |
| Buffer | Apr 29 — destroy droplet by 23:59 IST |
| Credits expire | **May 1 2026** (deposit Apr 1) |
| Submission window | May 4–10 2026 (uses pre-built artifacts) |
| Compute | Single AMD MI300X 192 GB HBM3 @ $1.99/hr |
| Budget | $100 = ~50 GPU-hr; targeting ~40 GPU-hr |
| Prize | $10K cash + 1× AMD Radeon AI PRO R9700 GPU |
| Submission deliverables | ≤5-min demo video, ≤300 MB upload, public GitHub repo, written description, X post |
| License requirement | MIT for code, Apache-2.0 for weights |

---

## 2. Identities & secrets (three different handles)

| Service | Handle | Verified | Token location |
|---|---|---|---|
| GitHub | `Jayanth-reflex` | `gh auth status` (scopes: `gist, read:org, repo, workflow`) | local Mac keyring |
| Hugging Face | `Reflex-jr` | HF whoami-v2 API (`role: write`) | droplet `/data/secrets/hf_token` (chmod 600) |
| Weights & Biases | `jayanth-jr` | W&B GraphQL viewer query | droplet `/data/secrets/wandb_key` (chmod 600) |
| Email | `jayanthreddy268.jr@gmail.com` | git config global | — |

Tokens are sourced into `/root/.bashrc` via `export X=$(cat /data/secrets/X)` pattern so the secret never appears in shell history or `env` inline output. Both tokens **must be rotated after May 10** since they were pasted in chat.

---

## 3. Critical paths & URLs

### Local Mac
- Project root: `/Users/jayanth/Desktop/resume/projects/reflex-amd/`
- Plan file: `/Users/jayanth/.claude/plans/let-s-push-all-the-groovy-crown.md`
- Memory dir: `/Users/jayanth/.claude/projects/-Users-jayanth-Desktop-resume-projects-reflex-amd/memory/`
- SSH key: `~/.ssh/id_ed25519` (ed25519, no passphrase, generated 2026-04-26)

### Remote
- GitHub: https://github.com/Jayanth-reflex/amd-hackathon-2026 (public, MIT)
- HF model repos (target):
  - `Reflex-jr/Qwen3.6-35B-A3B-Domain` — pre-Heretic merged
  - `Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive` — post-Heretic
  - `Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF` — 9-quant ladder
- W&B project: `jayanth-jr/amd-hackathon-2026`

### Droplet (MI300X)
- IP: `165.245.134.90` (region ATL1)
- SSH: `ssh -i ~/.ssh/id_ed25519 root@165.245.134.90`
- Image: vLLM Quick Start (Ubuntu 24.04 + ROCm 7.0 + Docker pre-pulled `vllm/vllm-openai-rocm:v0.17.1`)
- Specs: 1× MI300X (gfx942, 192 GB HBM3 / 205.8 GB exposed), 20 vCPU, 235 GiB RAM
- Storage: 720 GB root (`/dev/vda1`) + 5 TB scratch mounted at `/data` (`/dev/vdc1`, ext4, fstab persisted)
- Repo: `~/amd-hackathon-2026`
- Base model: `/data/hf/Qwen3.6-35B-A3B/` (67 GB BF16, 26 shards, downloaded 2026-04-26)
- Active sessions:
  - `tmux dataprep` → `python3 train/prepare_data.py …` (v3 run since 16:40 UTC)

### Droplet must be destroyed by Apr 29 23:59 IST
AMD Developer Cloud dashboard → GPU Droplets → 0.17.1-gpu-mi300x1-192gb-devcloud-atl1 → Destroy.

---

## 4. What's been built (chronological)

### 4.1 Local repo + GitHub (Apr 26 17:30–18:30 IST)
- Renamed `claude.md` → `CLAUDE.md` (case-corrected)
- `CLAUDE.md` updated to **Revision 5.2** (preamble "buddy for life", §G timeline replaced with H+0→H+48 table, §N GPU-saturation mandate appended, all Qwen3.5 → Qwen3.6 refs, HF account `Reflex-jr`)
- 39 files scaffolded across:
  - `train/` — prepare_data.py, train_lora.py, merge_and_push.py, profile_8k.py, config.yaml
  - `abliterate/` — run_heretic.sh, refusal_benchmark.py
  - `quantize/` — convert.sh, imatrix.sh, ladder.sh, push_gguf.sh
  - `eval/` — lm_eval_harness.sh, perplexity_drift.py, refusal_465.jsonl, domain_qa.jsonl
  - `serve/` — vllm.env, llamacpp.env, m4_failover.sh
  - `gateway/` — llama_guard.py, policy.yaml, audit_log.py
  - `frontend/` — Next.js 16 + React 19 + Tailwind v4 demo client
  - `docs/` — ARCHITECTURE.md, BUILD_IN_PUBLIC.md, SAFETY.md, **this file**
  - `Dockerfile.{train,serve}`, `docker-compose.yml`, `pyproject.toml`, `LICENSE`, `README.md`, `.gitignore`, `.github/workflows/ci.yml`
- Public GitHub repo `Jayanth-reflex/amd-hackathon-2026` created and force-pushed (MIT license, populated)

### 4.2 Memory files saved
At `/Users/jayanth/.claude/projects/-Users-jayanth-Desktop-resume-projects-reflex-amd/memory/`:
- `MEMORY.md` (index)
- `user_jayanth.md` — user profile, "buddy for life" directive, terse-response preference
- `project_amd_hackathon.md` — sprint window, hard deadlines, repo + HF URLs
- `feedback_aggressive_pushes.md` — GPU-saturation preference, decisive recommendations
- `reference_amd_cloud.md` — AMD Cloud dashboard, lablab.ai page, backup compute
- `reference_qwen36_arch.md` — Qwen3.6 architecture quirks (256 experts top-8, linear-attn, OOM at 8K)

### 4.3 Browser-driven Droplet provisioning (Apr 26 18:30–19:30 IST)
Used Claude-in-Chrome MCP to drive Comet browser through AMD Developer Cloud:
- Logged into amd.digitalocean.com (cookies carried)
- Navigated GPU Droplets section
- Selected single MI300X plan ($1.99/hr) — bypassed default 8×MI300X plan ($15.92/hr would've burned credit in 6h)
- Selected vLLM Quick Start image (`appId=221160341, image=vllm-0-17-1`)
- Generated ed25519 SSH key on Mac, pasted public key into AMD as `jayanth-mac-m4`
- One mis-click created wrong-image droplet (GPT-OSS); Jayanth fixed manually
- Final droplet: `0.17.1-gpu-mi300x1-192gb-devcloud-atl1` Active at IP `165.245.134.90`

### 4.4 Droplet bootstrap (Apr 26 19:30–20:00 IST)
- Verified GPU: gfx942, MI300X VF, 205.8 GB HBM
- Mounted `/dev/vdc1` (5 TB scratch) at `/data` + persisted in `/etc/fstab` via UUID
- AITER + Unsloth env vars persisted in `/root/.bashrc`
- Cloned repo to `~/amd-hackathon-2026`
- Downloaded base model (67 GB) via Docker container running `hf download` with `hf_transfer` enabled — completed in **~80 seconds**

### 4.5 H+2 Smoke gate ✅ ALL FOUR PASS (Apr 26 20:00–20:50 IST)
Used `vllm/vllm-openai-rocm:v0.17.1` Docker image with `transformers` from `git+https://github.com/huggingface/transformers.git` (the released 4.57.6 doesn't recognize the model; main branch has it).

- **Gate (a) Loss decreasing:** PASS — 100 LoRA steps on dummy data, loss `[3.13, 2.20, 1.32, 0.79, 0.56, 0.42, 0.39, 0.36, 0.34, 0.34]`. First-third mean 2.22 → last-third 0.35 (6.4× drop).
- **Gate (b) Memory:** PASS — peak 71.2 GB during training (well under 90 GB threshold)
- **Gate (c) Tokenizer:** PASS — Qwen2Tokenizer extracts cleanly, vocab 248,077
- **Gate (d) Architecture:** PASS — `qwen3_5_moe_text` recognized via transformers main; **256 experts, top-8 routing, 40 layers, hidden=2048**
- LoRA size: 8.36M trainable / 34.67B total = 0.024%

### 4.6 8K seq profiling — found a real constraint (Apr 26 21:00–22:00 IST)

| Config | Result |
|---|---|
| seq=8192, bsz=2, default mem | OOM (allocated 189 GB, needed +512 MiB → fail in MoE grouped-GEMM) |
| seq=8192, bsz=1, expandable_segments | OOM (linear-attn `chunk_gated_delta_rule` recurrent state grows past 192 GB) |
| **seq=4096, bsz=1, grad_accum=8** | **PASS** — 169.4 GB peak, 3.53s/micro-step, 28.3s/effective-step |

The model uses **linear attention with chunk-gated delta rule** which requires the `flash-linear-attention` (FLA) library for memory efficiency. Without FLA installed, the pure-PyTorch fallback OOMs at seq≥8192. Installing FLA might enable 8K but unknown ROCm compatibility — not worth the H+1 detour.

**Locked: seq=4096, bsz=1, grad_accum=8 (effective batch 8) → 80M tokens × 28.3s/step ÷ 65,536 tokens/step = 19.2 hours.**

### 4.7 Dataset blend Rev 5.1 (locked Apr 26 21:30 IST)

Macro: General 55% / Reasoning 7% / Tools 6% / **Domain 32%** (heavier specialty bias than typical 20%).

**Domain breakdown** (32% = 25.6M tokens):
| Sub-area | % | Tokens | Notes |
|---|---|---|---|
| Indian law | 1.0% | 0.80M | Dropped from 2% per Jayanth |
| Indian taxation | 1.0% | 0.80M | |
| Pharmacology | 1.0% | 0.80M | |
| Quantitative finance | 1.5% | 1.20M | |
| Engineering (5 sub-disciplines, 0.8% each) | 4.0% | 3.20M | **No NPTEL transcripts** per Jayanth (filler-heavy); textbooks/papers/MIT-OCW-CC-BY only |
| Geopolitics | 0.5% | 0.40M | |
| **Game theory** ⭐ TOP-OF-LINE | 5.0% | 4.00M | 8 sub-areas: mech design, algo GT, behavioral, network, climate, evolutionary, cooperative, AI/ML applications |
| OSINT/cyber | 2.5% | 2.00M | ATT&CK/D3FEND, Bellingcat |
| Psychology | 1.5% | 1.20M | Top-researcher OA papers + A-Z textbooks + CIA Heuer + KUBARK + MOSSAD memoirs |
| Geospatial | 2.0% | 1.60M | OSM, ISRO Bhuvan, GDAL |
| Survival skills | 5.0% | 4.00M | Fire/cooking/water/edible plants/wilderness medicine/shelter/comms/navigation/disaster prep |
| World + Indian history | 3.0% | 2.40M | Ancient/medieval/early modern/modern post-1900 |
| Defense forces worldwide | 2.0% | 1.60M | India tier-1 + US/China/Russia + NATO + regional powers |
| Languages (top-11 practical) | 2.5% | 2.00M | Hindi, Spanish, Arabic, Bengali, **Telugu (added)**, Portuguese, German, Russian, French, Japanese, Mandarin (only 5% — Qwen already heavy on Chinese) |

---

## 5. What's running NOW

### 5.1 `tmux dataprep` on droplet (started 2026-04-26 16:40 UTC)
Command:
```
python3 train/prepare_data.py \
  --config train/config.yaml \
  --out /data/hf/blend.jsonl \
  --raw-dir /data/hf/blend_raw \
  --tokenizer /data/hf/Qwen3.6-35B-A3B
```
With env: `HF_TOKEN`, `HF_HOME=/data/hf`, `HF_HUB_ENABLE_HF_TRANSFER=1`.

Logs at `/data/hf/prepare3.log`. Inspect via `tmux attach -t dataprep` or `tail -f /data/hf/prepare3.log`.

### 5.2 Existing artifacts
- `/data/hf/blend.jsonl` — 13.7M tokens (32,394 rows, 6 sources from v2 partial run)
- `/data/hf/blend_raw/` — 178 MB across 20 raw files
- v3 will overwrite `blend.jsonl` when it reaches its merge step

---

## 6. Locked decisions (ordered, with rationale)

| # | Decision | Why |
|---|---|---|
| 1 | Public GitHub repo from day 0 | lablab.ai requires public + MIT |
| 2 | HF namespace `Reflex-jr` (not `Jayanth`) | His actual HF account; verified via whoami |
| 3 | W&B entity `jayanth-jr` | Verified via GraphQL viewer query |
| 4 | Base model `unsloth/Qwen3.6-35B-A3B` | April 2026 release, supersedes Qwen3.5 spec |
| 5 | Fallback ladder: Qwen3.6 → Qwen3.5 → Qwen3-30B-A3B | Three levels of kill-switch by H+2 |
| 6 | Heretic abliteration MANDATORY | The architectural pitch ("model max-capable, policy at app layer") only holds if both axes provable on slides |
| 7 | seq_len = 4096 (not 8192) | OOM at 8K w/o flash-linear-attention library; 4K fits at 169 GB peak |
| 8 | bsz=1, grad_accum=8 (effective batch 8) | Memory-bound at 4K; same training dynamics |
| 9 | 80M tokens / 1 epoch | Profile shows 19.2h at 28.3s/step — fits 18h GPU window |
| 10 | Macro blend 55/7/6/32 | Heavy domain bias to differentiate; reasoning + tools preserved |
| 11 | Domain blend Rev 5.1 (14 sub-areas) | Per Jayanth's Apr 26 spec |
| 12 | No NPTEL transcripts | Jayanth flagged filler-word problem; use textbooks/papers/MIT-OCW instead |
| 13 | Game theory at 5% (top-of-line) | Jayanth: "must be top of line we need to be very focused on this" |
| 14 | Telugu added to Languages, Mandarin trimmed to 5% of D14 | Telugu speakers ~85M Indian; Qwen already trained heavy on Mandarin |
| 15 | Psychology sources rewritten | Top-researcher OA + A-Z textbooks + CIA Heuer + KUBARK + MOSSAD memoirs |
| 16 | License hygiene: only CC-BY/SA, CC0, PD, MIT, Apache-2.0 | Compatibility with our Apache-2.0 model weights downstream |
| 17 | Synthetic data deferred to during-training pass | Generates from non-abliterated teacher (DeepSeek-V3 or pre-LoRA Qwen) |
| 18 | Heretic `--max-trials 100` (not 50) | Saturate budget per GPU-saturation mandate |
| 19 | Imatrix 500 chunks (not 200) | Better calibration since GPU budget allows |
| 20 | 9-quant GGUF ladder (added IQ2_M) | Cover M2/M3-class Macs and lower-spec laptops |

---

## 7. Lessons learned (avoid these next time)

| Issue | Fix |
|---|---|
| `transformers` 4.57.6 doesn't know `qwen3_5_moe` | Install from `git+https://github.com/huggingface/transformers.git` |
| `vllm` Docker default ENTRYPOINT runs `vllm` command | Use `--entrypoint=/bin/bash` for one-off Python tasks |
| `hf_transfer` not pre-installed in vllm image | `pip install hf_transfer` inline before `hf download` |
| `huggingface_hub` 1.12 dropped `[cli]` extra; `huggingface-cli` not on PATH host-side | Use `hf` command (newer CLI) or run via Docker |
| Datasets new version rejects `trust_remote_code` for non-script datasets | Cascade: try without → with `name=X, split=train` → with `trust_remote_code` |
| MoE grouped-GEMM in transformers main OOMs at seq≥8192/bsz=2 | Drop bsz=1; if still OOM, drop seq to 4096 |
| Linear attention recurrent state in `chunk_gated_delta_rule` accumulates → also OOMs at 8K bsz=1 | Either install `flash-linear-attention` or drop to seq=4096 |
| `PYTORCH_HIP_ALLOC_CONF` is deprecated | Use `PYTORCH_ALLOC_CONF=expandable_segments:True` |
| Pip install of `huggingface_hub` on host conflicts with apt-installed `rich` | Use `--break-system-packages --ignore-installed` |
| `gh` token `repo` scope alone can't push to `.github/workflows/` | Need additional `workflow` scope: `gh auth refresh -h github.com -s workflow` |
| Initial Droplet creation had wrong "Create" button click trigger pattern in dialogs | After registering SSH key, double-check active radio button before clicking Create |
| AMD Developer Cloud defaults to **8×MI300X** plan ($15.92/hr), not single | Always click the lower MI300X plan radio explicitly |
| MI300X exposes 192 GB HBM3 but transformers reports 205.8 GB | Use 192 GB as planning anchor; ~14 GB overhead is normal |
| Some HF datasets (Salesforce/xlam-fc-60k) are gated | Need HF_TOKEN env var set before fetching |
| `dolphin-r1` `reasoning-deepseek` is a config name, not a split | Try `name=X, split=train` as second attempt |
| Indian gov URLs (incometaxindia.gov.in/cbic-gst.gov.in/indiankanoon API) often 404/403 | Don't rely on these scrapes; use synthetic generation or scrape Wikipedia for Indian law/tax content |

---

## 8. Pending — what still needs human input

| Item | Status | When needed |
|---|---|---|
| HF token | ✅ done | — |
| W&B token | ✅ done | — |
| `gh auth refresh -h github.com -s workflow` | ✅ done | — |
| Decision on training when to launch | ⏸ | Auto: when blend.jsonl ≥ 30M tokens |
| Demo video review | ⏸ | Apr 28 ~16:00 IST |
| Destroy droplet | ⏸ | Apr 29 23:59 IST (Jayanth via AMD dashboard) |
| Pitch deck approval | ⏸ | Apr 30 |
| Submission form (lablab.ai) | ⏸ | May 4 |

**Nothing blocking** as of Apr 26 22:15 IST.

---

## 9. Next steps — phase by phase

### Phase 5 — Tonight (Apr 26 22:15 IST → ~01:00 IST = ~3h)
- [ ] Wait for `blend.jsonl` to grow past ~30M tokens (auto, no action needed)
- [ ] Verify blend has source diversity (general + reasoning + tools + at least some domain)
- [ ] Launch main training run in tmux session `train` on droplet
- [ ] Monitor first ~30 min to confirm loss decreases monotonically

### Phase 6 — Training run (Apr 26 ~22:30 → Apr 27 17:30 IST = ~19h)
- LoRA training on blend.jsonl, 1 epoch, ~80M tokens
- W&B logging to `jayanth-jr/amd-hackathon-2026`
- Mid-evals at 25/50/75/100% via `TrainerCallback`
- Loss-spike kill-switch (1.3× ratio threshold over 200-step rolling baseline)
- Auto checkpoint to `/workspace/out/lora_adapter` every 500 steps

### Phase 7 — Merge + Heretic (Apr 27 17:30 → 23:00 IST = ~5.5h)
- `python train/merge_and_push.py` → push BF16 to `Reflex-jr/Qwen3.6-35B-A3B-Domain` (Apache-2.0)
- Run held-out eval, capture baseline refusal rate
- `bash abliterate/run_heretic.sh` — dense pass (`--max-trials 100`) + EGA if needed
- **HARD GATE**: `abliterate/refusal_benchmark.py` must show ≤5/465 refusals
- Push to `Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive`

### Phase 8 — GGUF ladder (Apr 27 23:00 → Apr 28 03:00 IST = ~4h)
- `bash quantize/convert.sh` — BF16 GGUF + mmproj-f16.gguf (verify mmproj integrity post-Heretic)
- `bash quantize/imatrix.sh` — 500-chunk WikiText-2 calibration
- `bash quantize/ladder.sh` — 9 quants: BF16, Q8_0, Q6_K, Q5_K_M, **Q4_K_M (default)**, IQ4_XS, Q3_K_M, IQ3_M, IQ2_M
- `bash quantize/push_gguf.sh` — upload to `Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive-GGUF`
- `python eval/perplexity_drift.py` — reject any quant >3% drift vs BF16

### Phase 9 — Deploy + eval (Apr 28 03:00 → 11:30 IST = ~8.5h)
- `docker compose --env-file serve/vllm.env up -d vllm` on droplet
- `docker compose --env-file serve/llamacpp.env up -d llamacpp` co-resident
- Gateway + Llama-Guard-3-1B + frontend on private VPS (or droplet for demo)
- `bash eval/lm_eval_harness.sh` — MMLU/HellaSwag/TruthfulQA/GSM8K (250 each), BBH-Hard, domain held-out
- Verify end-to-end: gateway → benign → 200 + answer; gateway → blocked → 451 + verdict

### Phase 10 — Final polish (Apr 28 11:30 → 17:30 IST = ~6h)
- M4 failover smoke test (`bash serve/m4_failover.sh` on local Mac with Q4_K_M GGUF)
- Record ≤5-min demo video (gateway side-by-side panels)
- README polish, model card on HF
- Final git push
- **Sprint ends Apr 28 17:30 IST**

### Phase 11 — Buffer (Apr 29)
- Local-only work: video editing, README review, pitch deck draft
- Verify all HF artifacts publicly accessible from external IP
- **DESTROY MI300X droplet by 23:59 IST** — AMD ADC dashboard → Droplets → Destroy

### Phase 12 — Apr 30
- Pitch deck final
- Pre-write 3 X threads (Why MI300X, GGUF ladder, Ship it)
- LinkedIn long-form draft

### Phase 13 — May 1 (CREDITS EXPIRE)
- No more cloud GPU. Hosting on private VPS or local M4.
- Backup compute (if needed): Vultr MI300X $1.85/hr, TensorWave $1.50/hr, RunPod spot $1.49/hr

### Phase 14 — Submission week (May 4–10)
| Day | Action |
|---|---|
| May 4 Mon | Submit on lablab.ai. Post Thread #1 ("Why MI300X for solo MoE fine-tuning") with finished loss curves, eval numbers. Tag `@AIatAMD @lablabai @UnslothAI`. |
| May 5 Tue | Short post: loss curve highlight |
| May 6 Wed | Short post: 0/465 refusal verification screenshot |
| May 7 Thu | Thread #2 ("GGUF ladder, perplexity drift, refusal benchmark") |
| May 8 Fri | Short post: dual-engine deploy demo |
| May 9 Sat | Short post: gateway architecture punchline |
| May 10 Sun | Thread #3 ("Ship it — full architecture, full numbers"). Final submission. |

---

## 10. Recovery instructions — if conversation context is lost

### To resume from any new session:
1. Read this file: `cat docs/STATE.md` (or read at https://github.com/Jayanth-reflex/amd-hackathon-2026/blob/main/docs/STATE.md)
2. Read CLAUDE.md (master spec, Revision 5)
3. Check droplet status: `ssh -i ~/.ssh/id_ed25519 root@165.245.134.90 'tmux ls; nvidia-smi 2>/dev/null || rocm-smi --showuse | head -20'`
4. Check active tmux sessions: `tmux attach -t dataprep` or `tmux attach -t train`
5. Check git log: `git log --oneline -20`
6. Check W&B project: https://wandb.ai/jayanth-jr/amd-hackathon-2026

### If droplet is unreachable:
- Has it been destroyed? Check AMD ADC dashboard.
- Is credits expired? Check May 1 deadline status.
- If needed, spin backup on Vultr/TensorWave/RunPod with own card and migrate state via HF Hub artifacts.

### If training crashed:
1. Check `~/amd-hackathon-2026/logs/` and `/data/hf/wandb/` for last loss values
2. Check `/workspace/out/lora_adapter/checkpoint-*` for last good checkpoint
3. Resume with `python train/train_lora.py --resume_from_checkpoint /workspace/out/lora_adapter/checkpoint-N`

### If Heretic refusal gate fails (>5/465):
1. Re-run with `--max-trials 200`
2. If still fail, enable EGA + widen prompts
3. If still fail, revert to last clean checkpoint + retrain with stricter LoRA target_modules subset
4. **Do not ship Aggressive variant with >5/465** — the architectural pitch fails

---

## 11. Quick command reference

### SSH + droplet
```bash
ssh -i ~/.ssh/id_ed25519 root@165.245.134.90
# Once in:
tmux attach -t dataprep        # watch dataset prep
tmux attach -t train           # watch training (when running)
docker ps                       # see what's running
rocm-smi --showuse              # GPU utilization
df -h /data                     # scratch disk usage
tail -f /data/hf/prepare3.log   # latest dataset prep log
```

### Local + git
```bash
cd /Users/jayanth/Desktop/resume/projects/reflex-amd
git log --oneline -10
git pull origin main
gh repo view --web              # opens repo on github.com
```

### HF + W&B verification
```bash
# HF whoami
ssh root@165.245.134.90 'curl -s -H "Authorization: Bearer $(cat /data/secrets/hf_token)" https://huggingface.co/api/whoami-v2'

# W&B viewer
ssh root@165.245.134.90 'curl -s -X POST -u "api:$(cat /data/secrets/wandb_key)" https://api.wandb.ai/graphql -H "Content-Type: application/json" -d "{\"query\":\"{ viewer { username } }\"}"'
```

### To launch main training (when blend.jsonl is ready)
```bash
ssh root@165.245.134.90 'tmux new-session -d -s train "
  source /root/.bashrc;
  cd ~/amd-hackathon-2026;
  docker compose --profile train run --rm \
    -e WANDB_API_KEY=\$(cat /data/secrets/wandb_key) \
    -e HF_TOKEN=\$(cat /data/secrets/hf_token) \
    train python train/train_lora.py \
      --config train/config.yaml \
      --data /data/hf/blend.jsonl \
      2>&1 | tee /data/hf/train.log
"'
```

### Destroy droplet (Apr 29)
- Open AMD Developer Cloud dashboard
- GPU Droplets → `0.17.1-gpu-mi300x1-192gb-devcloud-atl1` → Actions → Destroy
- Type the droplet name to confirm
- Click red Destroy button

---

## 12. Document changelog

| Date | What changed | By |
|---|---|---|
| 2026-04-26 22:15 IST | Initial creation — captures full state through smoke gate, profile, and dataset prep v3 launch | Claude |

---

*This document is the resume-anywhere reference. Update it after each major phase transition (training start, merge, Heretic, GGUF, deploy, demo, submission).*
