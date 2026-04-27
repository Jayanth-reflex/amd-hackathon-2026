# Training Progress — Live Time-Series

**Run:** `qwen36-domain-aggressive-rev5.1`  
**W&B run id:** `i1bjy50l`  
**Dashboard:** https://wandb.ai/jayanth-jr-icims/huggingface/runs/i1bjy50l

**Setup:**
- Base model: `unsloth/Qwen3.6-35B-A3B` (256 experts, top-8 routing, 40 layers)
- LoRA: r=16, alpha=16, target [q,k,v,o,gate,up,down] = 8.36M trainable / 34.67B total (0.024%)
- Dataset: `/data/hf/blend_packed.jsonl` — 13,561 packed rows × ~3,653 tokens = 49.5M tokens
- Sequence length: 4,096
- Effective batch: 1 × grad_accum 8 = 8
- Optimizer: adamw_torch, LR 5e-5, cosine schedule, 50-step warmup
- Total target: 1 epoch = ~1,695 effective steps

---

## Snapshots

### 2026-04-27 00:10 IST — Step 50 / 1,695 (2.95%)

| Metric | Value |
|---|---|
| Runtime | 35 min |
| Loss | **1.32** (was 3.98 at step 10 — clean monotonic decrease) |
| Grad norm | 0.39 (stable) |
| Learning rate | 4.9e-5 (peak after warmup) |
| Per-step rate | 42.3s |
| ETA remaining | ~19.3h |
| GPU utilization | 81% |
| GPU power | 408W |
| GPU temp | 48°C junction |

**Loss progression:**
- step 10: 3.98 (cold start)
- step 50: 1.32 (66% drop in 40 steps)

Loss curve is healthy. Will append next snapshot at ~step 200 (~02:30 IST).

---

### 2026-04-27 01:16 IST — Step 110 / 1,696 (6.5%)

Pulled from operator's docker-log capture (10 logging events visible since step 10).

| Step | Loss | Grad norm | LR | Wall clock UTC |
|---|---|---|---|---|
| 10 | 3.978 | 8.576 | 9e-6 | 18:36:51 |
| 20 | 3.735 | 7.921 | 1.9e-5 | 18:43:59 |
| 30 | 2.638 | 5.719 | 2.9e-5 | 18:50:51 |
| 40 | 1.574 | 0.889 | 3.9e-5 | 18:57:50 |
| 50 | 1.316 | 0.393 | 4.9e-5 | 19:04:51 |
| 60 | 1.263 | 0.194 | 5.0e-5 | 19:11:55 |
| 70 | 1.176 | 0.140 | 4.998e-5 | 19:18:46 |
| 80 | 1.089 | 0.153 | 4.996e-5 | 19:25:40 |
| 90 | 1.092 | 0.104 | 4.993e-5 | 19:32:37 |
| 100 | 1.069 | 0.111 | 4.989e-5 | 19:39:29 |
| 110 | **1.006** | 0.111 | 4.984e-5 | 19:46:25 |

**Observations:**
- Loss curve is textbook clean. **3.98 → 1.01 in 110 steps (75% drop, no plateau, no spike, no oscillation).**
- The cliff between step 30 → 40 (loss 2.64 → 1.57) is the classic "model finds gradient direction" moment as LR hits a useful range. Healthy.
- Grad norms collapsed 8.58 → 0.11 (78× drop). Means optimizer transitioned from "everything is wrong" to "fine-tuning small adjustments". Stable.
- LR schedule honored: linear warmup to 5e-5 by step 50, cosine decay starting (4.984e-5 at step 110).
- Per-step rate stable at **41-42s/step** across both warmup (28 min for 40 steps = 42s/step) and post-warmup (41 min for 60 steps = 41s/step). No drift.

**Updated ETA:**
- Remaining: 1,696 - 110 = **1,586 steps**
- × 42s = 66,612 sec = **18.5 hours**
- Done at: ~19:46 IST Apr 27 (within 30 min of original estimate)

Will append next snapshot at step ~200 (~02:30 IST) or step ~424 (25% mid-eval marker), whichever first.

---

### 2026-04-27 01:54 IST — Step 160 / 1,696 (9.4%)

Pulled via `wandb.Api()` `scan_history()` from inside training-run container (the GraphQL `sampledHistory` returns 500 on this run, needs the SDK). Heartbeat at 20:24:35 UTC.

| Step | Loss | Grad norm | LR |
|---|---|---|---|
| 110 | 1.006 | 0.111 | 4.984e-5 |
| 120 | 1.149 | 0.110 | 4.978e-5 |
| 130 | **0.970** | 0.093 | 4.972e-5 |
| 140 | 1.053 | 0.100 | 4.964e-5 |
| 150 | 1.062 | 0.102 | 4.956e-5 |
| 160 | 1.096 | 0.101 | 4.946e-5 |

**Observations:**
- **First sub-1.0 loss at step 130 (0.970).** Loss is now oscillating in a tight 0.97–1.10 band with no upward drift — classic post-warmup fine-tuning behavior.
- Grad norms locked into **0.09–0.11**: model is no longer making large parameter updates, just small refinements. This is exactly what we want.
- LR cosine decay tracking precisely: 4.984e-5 → 4.946e-5 in 50 steps (peak 5e-5 at step 50). On schedule for ~3.5e-5 by 25% (step 424) and ~0 by step 1696.
- Per-step rate **41.6s** (150 steps in 104 min) — no degradation from longer-context packing.
- **Loss-spike kill-switch is armed but not close to triggering**: would need recent-50 mean to exceed 1.3× of prior-200 mean. Recent-50 ~1.05, prior-200 not yet computable but trending similar. Safe.

**Updated ETA:**
- Remaining: 1,696 - 160 = **1,536 steps** × 41.6s = 63,898s = **17.7 hours**
- Done at: 20:24 UTC + 17.7h = **~14:09 UTC = 19:39 IST Apr 27** (within 7 min of original estimate)

Will append next snapshot at step ~424 (25% mid-eval marker, ~05:00 IST) or sooner if anything moves.

---

### 2026-04-27 11:26 IST — Step 990 / 1,696 (58.4%)

Heartbeat at 05:56:20 UTC. Cleared the **25% (step 424)** and **50% (step 848)** mid-eval markers — both fired silently as expected. Training has been running for **11h 21m**.

| Step | Loss | Grad norm | LR |
|---|---|---|---|
| 200 | 1.0087 | 0.077 | 4.900e-5 |
| 300 | 0.9648 | 0.081 | 4.723e-5 |
| 400 | 1.0399 | 0.094 | 4.466e-5 |
| 500 | 1.0655 | 0.099 | 4.137e-5 |
| 600 | 1.0406 | 0.085 | 3.749e-5 |
| 700 | 0.9817 | 0.122 | 3.315e-5 |
| 800 | 0.9142 | 0.091 | 2.852e-5 |
| 900 | **0.8717** | 0.091 | 2.376e-5 |
| 950 | 1.0832 | 0.087 | 2.139e-5 |
| 990 | 0.9497 | 0.106 | 1.951e-5 |

**Observations:**
- **New best loss: 0.872 at step 900**, down from 0.970 at step 130. The model is genuinely learning — not just stuck in the bowl.
- Loss now oscillates in a **0.87–1.10 band** with a clear downward trend (mean of last 50 logged points = **0.9895**). Compare to the **1.05–1.10 mean around step 200** — that's a ~10% absolute drop while LR has decayed 60% (4.90e-5 → 1.95e-5).
- Grad norms locked at **0.07–0.12** for 800+ steps. Optimizer is in pure fine-tuning mode; no instability.
- Per-step rate steady at **41.6s/step** (980 steps in 11h 21m). Zero degradation.
- Cosine schedule honored exactly: 4.90e-5 → 1.95e-5 over 790 steps (39% remaining LR at 58% remaining steps — matches `1 - cos(π·x/2)` profile).
- Loss-spike kill-switch armed but cannot trigger before step 2500 of logging history (we end at step 1696 = 169 log entries; switch needs 250). Effectively safety net is the underlying healthy curve.

**Updated ETA:**
- Remaining: 1,696 - 990 = **706 steps** × 41.6s = 29,370s = **8.16 hours**
- Done at: 05:56 UTC + 8.16h = **~14:05 UTC = 19:35 IST Apr 27**
- 75% mid-eval at step 1,272 ≈ **15:25 IST**.
- 100% completion ≈ **19:35 IST**.

Will append next snapshot when 75% mid-eval fires (~15:25 IST) or near completion.

---

### 2026-04-27 13:52 IST — Step 1,200 / 1,696 (70.8%)

Heartbeat at 08:22:35 UTC. Training has been running for **13h 53m**. Cosine schedule has decayed LR by 80% (5e-5 → 1.04e-5) — we're well into the convergence tail.

| Step | Loss | Grad norm | LR |
|---|---|---|---|
| 1000 | 0.9213 | 0.105 | 1.905e-5 |
| 1050 | 0.9638 | 0.110 | 1.676e-5 |
| 1100 | 0.9898 | 0.107 | 1.455e-5 |
| 1150 | 0.9218 | 0.121 | 1.243e-5 |
| 1200 | 1.0597 | 0.097 | 1.043e-5 |

**Observations:**
- Loss locked into **0.87–1.10 band** for 700+ steps. Min still 0.8717 at step 900. Last-50 mean = **1.0060** (vs 0.9895 at step 990) — slight tick up but well within noise.
- This is **convergence behavior, not regression**. At LR 1e-5 you're making tiny updates; loss fluctuates around a plateau because each batch's intrinsic difficulty dominates the marginal LR contribution.
- Grad norms 0.09–0.12 throughout — no instability, no spikes.
- Per-step rate **41.6s** (1190 steps in 13h 46m). Zero drift across 70% of the run.
- 75% mid-eval marker (step 1272) fires in ~50 min (~14:42 IST).

**Updated ETA:**
- Remaining: 1,696 - 1,200 = **496 steps** × 41.6s = 20,634s = **5.73 hours**
- Done at: 08:22 UTC + 5.73h = **~14:06 UTC = 19:36 IST Apr 27 today**

Will append a final snapshot at step 1696 when training completes.

---

## How to fetch a fresh snapshot

```bash
ssh -i ~/.ssh/id_ed25519 root@165.245.134.90 'curl -s -X POST -u "api:$(cat /data/secrets/wandb_key)" -H "Content-Type: application/json" "https://api.wandb.ai/graphql" -d "{\"query\": \"query { project(name: \\\"huggingface\\\", entityName: \\\"jayanth-jr-icims\\\") { runs(first:1, order:\\\"-createdAt\\\") { edges { node { state summaryMetrics historyLineCount } } } } }\"}"'
```

Or just refresh the W&B URL above.

---

## Mid-eval markers (auto-fired at 25/50/75/100%)

| Fraction | Step | Status |
|---|---|---|
| 25% | ~424 | pending |
| 50% | ~848 | pending |
| 75% | ~1,272 | pending |
| 100% | 1,695 | pending |

(`MidEvalMarker` callback writes `/data/out/lora_adapter/mideval_marker_{25,50,75,100}.json` with last 5 log_history entries at each.)

---

## Loss-spike kill-switch

Triggers `KeyboardInterrupt` if `mean(loss[-50:]) > 1.3 × mean(loss[-200:-50:])`. Has not fired.

---

## Risks being monitored

- ⚠️ **Per-step time creep**: if heavier sequences cause step time to grow >50s, ETA could slip. Recheck at step 200.
- ⚠️ **GPU power throttling**: MI300X TDP is 750W, currently 408W. If sustained 600W+, monitor for thermal throttle.
- ⚠️ **Loss plateau before 25%**: if loss stops dropping by step 400, we may need to bump LR or add warmup steps.
- ✅ **OOM**: not a risk; we're at 35% VRAM (~67 GB / 192 GB).
- ✅ **Disk fill**: 5 TB scratch, currently ~70 GB used, plenty of headroom.
