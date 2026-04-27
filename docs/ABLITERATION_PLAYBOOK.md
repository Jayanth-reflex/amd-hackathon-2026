# Abliteration playbook — what to do, what NOT to do

**Audience:** future Jayanth + future Claude doing the Gemma fine-tune (and any other model). Read this BEFORE picking a tool. Reusable across architectures.

---

## TL;DR — pick the technique first, the tool second

The abliteration **technique** is small (Arditi et al. 2024 / Labonne 2024). Three things:

1. Compute a **refusal direction** = `mean(harmful_residuals) - mean(harmless_residuals)`, normalized, at the last token position, for every layer.
2. Pick the layer with the largest-magnitude direction (mlabonne's blog uses ~30% depth).
3. **Orthogonalize** every weight matrix that writes to the residual stream against that direction:
   - `W_new = W - (proj_v(W))` where the projection is taken along the d_model axis that exits to the residual.
   - In code (3 lines): `proj = (W @ v.unsqueeze(-1)) * v; W_new = W - proj`.

Reference: https://huggingface.co/blog/mlabonne/abliteration · Arditi et al. 2024 (https://arxiv.org/abs/2406.11717).

The **tool** (heretic-llm, failspy/abliterator, custom) is just an implementation detail. Pick based on whether the tool's assumptions match your model's architecture. **Verify this BEFORE training.**

---

## What weight matrices to orthogonalize

For ANY decoder-only transformer:

| What writes to the residual stream | Always present |
|---|---|
| `model.embed_tokens.weight` (token embedding) | yes |
| `layer.self_attn.o_proj.weight` (attention output) | most arches |
| `layer.mlp.down_proj.weight` (MLP output) | dense MLP |
| For MoE: `mlp.shared_expert.down_proj` + each `mlp.experts[i].down_proj` | MoE arches |
| For hybrid attention (Qwen3.6, Mamba-Transformer): `layer.linear_attn.<output>` | hybrid arches |

**The architecture-agnostic rule:** *anything whose output dimension equals `d_model` and is added to the residual must be orthogonalized.* Walk the model, list every such weight, orthogonalize all of them.

---

## Mistake we made (Apr 27, 2026) — read this BEFORE picking a tool

### What happened

1. CLAUDE.md spec called for **Heretic abliteration** as MANDATORY (target ≤5/465 refusals).
2. We trained Qwen3.6-35B-A3B for 19h, merged the LoRA, pushed to HF.
3. Tried to run Heretic. **It blew up** at `layer.self_attn.o_proj` because Qwen3.6 uses **hybrid attention**: some layers have `self_attn`, some have `linear_attn` (chunk_gated_delta_rule).
4. Heretic's source code at `model.py:345` has the comment: *"Exceptions aren't suppressed here, because there is currently no alternative location for the attention out-projection."*
5. Heretic v1.2.0 simply doesn't support hybrid-attention models.

We burned ~1.5 hours debugging dependency hell (transformers main vs. heretic's pin, peft↔transformers `WeightConverter` mismatch, today's transformers main `PeftConfigLike` typing bug) before hitting the real architectural wall.

### Why we didn't catch it earlier

Three failures, in increasing order of seriousness:

1. **The CLAUDE.md spec asserted "Heretic supports MoE architectures via dense pass" as a fact**, based on heretic's README mentioning MoE. We never validated against Qwen3.6 specifically.
2. **The H+2 smoke gate (defined in CLAUDE.md §G) checked training compatibility** (loss decreases, memory stable, tokenizer extraction works, GGUF arch recognition) — but **nothing about Heretic**. Could have run `heretic <base_model> --n-trials 1` as part of H+2 in 5 minutes.
3. **The fact "Qwen3.6 uses linear_attn for some layers" was already in memory** (`reference_qwen36_arch.md`: *"linear attn, OOM at 8K w/o FLA"*) — we knew it would matter for FLA install, but didn't connect it to "Heretic doesn't handle this."

### What we should have done

For every multi-stage pipeline, **validate every stage against the actual artifact the previous stage will produce, BEFORE committing serious compute to that previous stage.**

Concrete: before kicking off training, the H+2 smoke gate should have included:
```bash
# (e) verify abliteration tool works on the base model
heretic <base_model> --n-trials 1 --no-save 2>&1 | tee /tmp/heretic_smoke.log
grep -q "Optimization finished\|Trial complete" /tmp/heretic_smoke.log || exit 1

# (f) verify Llama-Guard-3-1B classifier loads
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-Guard-3-1B', dtype='bfloat16')" || exit 1
```

### Tooling fix going forward

Add `abliterate/test_arch_compat.py` that, given any model path:

1. Loads the model with `AutoModelForCausalLM`
2. Walks every decoder layer
3. Lists every weight matrix that writes to the residual stream
4. Verifies all of them are accessible (not wrapped in something opaque)
5. Reports a **count and list** so a human can sanity-check coverage

```bash
python abliterate/test_arch_compat.py /path/to/model
# Output:
#   Model: Qwen3_5MoeForCausalLM, 40 layers, d_model=2048
#   Embedding: model.embed_tokens.weight [248064, 2048]  ✓
#   Layer 0:
#     attention: self_attn (full) → o_proj.weight [2048, 2048]  ✓
#     mlp: shared_expert.down_proj.weight [2048, 512]  ✓
#     mlp: 256× experts[i].down_proj.weight [2048, 512]  ✓
#   Layer 1:
#     attention: linear_attn (chunk_gated_delta_rule) → ???  ⚠ NEEDS INSPECTION
#     ...
```

This script should be **run at H+2** for any new abliteration target, before training starts.

---

## Reusable plan for the Gemma fine-tune

When we do Gemma later, the H+2 smoke gate **must** include:

1. ✅ Train smoke (loss decreases, mem stable) — same as Qwen3.6 sprint
2. ✅ GGUF arch recognition — same
3. ➕ **Architecture mapping via `test_arch_compat.py`** (new — write it before Gemma run)
4. ➕ **Custom abliteration script smoke** — load Gemma base, capture residuals on 4 prompts, orthogonalize embedding only, verify model still produces coherent text. ~3 min.

The actual abliteration script should be built around the technique, not a tool:

- **Don't** depend on heretic-llm or failspy/abliterator unless you've verified they iterate every weight matrix listed by `test_arch_compat.py`.
- **Do** write your own ~80-line `abliterate_arditi.py` that:
  - Captures last-token residuals on AdvBench (harmful) + harmless_alpaca (harmless), 32 of each
  - Computes per-layer refusal direction
  - Picks best layer by magnitude
  - Walks the model, orthogonalizes every residual-writing weight matrix
  - Saves and pushes

**Per-architecture knobs to tune in the abliteration script:**

| Architecture | Attention output | MLP output | Notes |
|---|---|---|---|
| Llama / Qwen2 / Qwen3 (dense) | `self_attn.o_proj.weight` | `mlp.down_proj.weight` | Standard |
| Qwen3-30B-A3B (MoE, full attn only) | `self_attn.o_proj.weight` | `mlp.shared_expert.down_proj` + `mlp.experts[i].down_proj` for each expert | MoE shape, single attn type |
| Qwen3.6-35B-A3B (MoE + hybrid attn) | `self_attn.o_proj.weight` OR `linear_attn.<output>.weight` (per layer's `layer_type`) | `mlp.shared_expert.down_proj` + `mlp.experts[i].down_proj` | Hybrid attn — mix per layer |
| Gemma 2 / Gemma 3 (dense, RMSNorm-based) | `self_attn.o_proj.weight` | `mlp.down_proj.weight` | Standard layout. Should be straightforward. Verify with `test_arch_compat.py` first. |

**Likely Gemma layout** (based on Gemma 2 9B / 27B): standard dense decoder, no hybrid attention. Heretic *might* work directly. **But still run `test_arch_compat.py` first — and if it works, prefer the custom script anyway because we control it.**

---

## Hard rules going forward

1. **No abliteration tool runs without `test_arch_compat.py` greenlighting it first.**
2. **No training run starts without an end-to-end smoke gate that validates EVERY downstream tool against the base model.**
3. **No "trust the spec" — verify every assumption with a 5-minute test before committing GPU-hours.**
4. **When picking a tool over a technique, ask: "what assumptions does this tool make about my model, and have I verified them?"**

---

## What we shipped today (Apr 27)

- Trained Qwen3.6-35B-A3B → public at https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain
- Wrote `abliterate/abliterate_arditi.py` — custom mlabonne-style abliteration, hybrid-attention aware
- Pushed abliterated variant → https://huggingface.co/Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive (subject to refusal benchmark passing ≤5/465)

Time spent on Heretic dead end: ~2 hours. Time spent writing the custom solution: ~3 hours. Net loss: ~1 hour, but with a defensible "we built our own" pitch and a reusable script for Gemma.
