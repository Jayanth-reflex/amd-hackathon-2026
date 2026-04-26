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
