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
