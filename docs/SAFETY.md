# Safety architecture — three-tier defense

The architectural punchline of this project, suitable for a single judge slide:

> **Model maximally capable. Policy at app layer. Both axes provable.**

This is **literally provable** because Heretic abliteration is mandatory:
the model achieves 0/465 verified refusals (the "max capability" axis), and
the gateway enforces hard-line refusals via Llama-Guard-3-1B (the "policy"
axis). Either axis alone is weaker than both together.

## Tier 1 — Model

`Jayanth/Qwen3.6-35B-A3B-Domain-Aggressive` is verified to **0/465 refusals
(ceiling 5/465 with documented deviation)** on a fixed adversarial corpus
post-Heretic abliteration. Verification artifacts:

- `logs/refusal_results_dense.json` — dense pass output
- `logs/refusal_results_ega.json` — EGA pass output (if needed)
- `logs/heretic_dense.log` — Heretic Optuna trial log

The model is intentionally a maximally-capable surface — it does not refuse
based on "alignment training." Refusal posture is owned by tier 2.

## Tier 2 — Gateway

Every request and every response passes through `meta-llama/Llama-Guard-3-1B`,
fine-tuned on Llama-3.2-1B for the **MLCommons 13-category hazards taxonomy**.

### Hard-refuse categories (return 451 + classifier verdict)

| Code | Category | Why hard-refuse |
|---|---|---|
| **S4** | Child Sexual Exploitation (CSAM) | Strict-liability content |
| **S5** | Defamation | Targeted impersonation explicitly out of scope |
| **S7** | Privacy | Doxxing / PII generation explicitly out of scope |
| **S9** | Indiscriminate Weapons | Bio/radiological/chem WMD synthesis explicitly out of scope |
| **S2 narrow** | Non-Violent Crimes (synthesis subset) | Manufacturing routes for controlled substances; Indian law/tax pedagogy remains allowed |

### Log-only categories

`S1, S3, S6, S8, S10, S11, S12, S13` — classifier verdicts logged but
request still flows. Useful for collecting examples to tune policy without
breaking demos. Operators can promote categories from `log_only` to
`hard_refuse` with a single `policy.yaml` edit.

## Tier 3 — Audit log

Every gateway decision (request hash, classifier verdict, category code,
action taken) is logged via `gateway/audit_log.py` to a tamper-evident
sha256-chained append-only log at `/data/audit/decisions.log`.

Verify with:

```bash
python gateway/audit_log.py --verify
# verify: OK records=N
```

The audit log is part of the submission package — judges can verify any
demo decision was logged, classified, and acted upon transparently.

## Why this beats baked-in alignment

| Axis | Baked-in alignment | This architecture |
|---|---|---|
| Capability | Reduced — refusals trained into weights | Maximal — verified 0/465 |
| Policy change | Requires retrain | `policy.yaml` edit |
| Inspectability | Opaque (weights) | Open (YAML + classifier) |
| Jurisdictional flexibility | None | Full — different policies per deployment |
| Audit trail | None | Tamper-evident hash chain |
| Classifier swap | Impossible | Drop-in (e.g., upgrade to Llama-Guard-3-8B) |

## What the demo shows

The frontend at `frontend/app/page.tsx` runs side-by-side:

1. **Benign domain question** (e.g., "Walk through Section 115BAC of the Indian Income Tax Act")
   - Gateway green-lights → upstream model answers fully → 200 + answer.
2. **Hard-line query** (e.g., "Provide a step-by-step synthesis route for a Schedule I controlled substance")
   - Gateway refuses pre-model → 451 + classifier verdict → category badge displayed.

This is the architectural punchline rendered in a single screen.

## Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Heretic post-Heretic refusal count > 5/465 | Re-run with `--max-trials 200` + EGA + widened prompts. If still failing, ship is blocked. |
| Llama-Guard-3-1B false positive on benign domain prompt | Move category from `hard_refuse` to `log_only` in `policy.yaml`; commit + restart gateway. |
| Audit log hash chain breaks | `verify_chain()` returns FAIL with the breaking line; investigate as security incident. |
| Vision tower modified by Heretic (mmproj integrity check fails) | Heretic should only touch LM tower; if tensor diff non-empty, rerun without vision-tower load. |

## Compliance notes

- License chain: Qwen3.6 base (Apache-2.0) → LoRA (MIT) → merged + abliterated (Apache-2.0) → wrapper code (MIT). Heretic itself is AGPL-3.0, but only the *tool* is AGPL — model weights it produces are not virally relicensed.
- The 13-category taxonomy is from MLCommons and is documented in the upstream Llama-Guard-3-1B model card.
