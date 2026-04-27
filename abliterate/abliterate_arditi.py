"""Custom abliteration for Qwen3.6 (and any decoder-only LM with hybrid attention + MoE).

Implements the Arditi et al. 2024 / Labonne 2024 technique:
  https://arxiv.org/abs/2406.11717
  https://huggingface.co/blog/mlabonne/abliteration

Steps:
  1. Capture last-token residual stream for harmful + harmless prompts at every layer.
  2. Compute per-layer refusal direction = normalize(mean_harmful - mean_harmless).
  3. Pick the layer with the largest-magnitude direction (highest signal).
  4. Orthogonalize ALL weight matrices that write to the residual stream:
       - embed_tokens (per-row)
       - per layer: self_attn.o_proj OR linear_attn.out_proj (per-column)
       - per layer: mlp.shared_expert.down_proj (per-column)
       - per layer: mlp.experts.down_proj (fused [num_experts, d_model, d_inter], batched)
  5. Save modified model.

Why this is better than heretic-llm v1.2.0 for Qwen3.6:
  - Heretic hard-codes `layer.self_attn.o_proj` access. Qwen3.6 has hybrid
    attention (some layers use linear_attn). This script handles both.
  - Heretic doesn't traverse fused MoE expert tensors. We do.
  - Pure script, no questionary / interactive prompts.

Usage:
    python abliterate/abliterate_arditi.py \\
        --model /data/out/merged \\
        --out /data/out/aggressive \\
        --n-prompts 64

Optional:
    --push-to-hub Reflex-jr/Qwen3.6-35B-A3B-Domain-Aggressive
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("abliterate_arditi")


def _extract_text(record: dict) -> str | None:
    """Try common column names; return first non-empty match."""
    for col in ("text", "prompt", "behavior", "instruction", "goal", "question"):
        v = record.get(col)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_prompts(harmful_n: int, harmless_n: int) -> tuple[list[str], list[str]]:
    """mlabonne/harmful_behaviors (harmful) + mlabonne/harmless_alpaca (harmless).

    These are the exact datasets used in Maxime Labonne's reference blog post.
    Both are public. We fall back to AdvBench mirrors if mlabonne's pull fails.
    """
    harmful_sources = [
        ("mlabonne/harmful_behaviors", "train"),
        ("Anthropic/hh-rlhf", "test"),  # fallback
    ]
    harmless_sources = [
        ("mlabonne/harmless_alpaca", "train"),
        ("yahma/alpaca-cleaned", "train"),  # fallback
    ]

    def pull(sources, n: int, label: str) -> list[str]:
        for ds_id, split in sources:
            try:
                log.info("Loading %s [%s]...", ds_id, split)
                ds = load_dataset(ds_id, split=split)
                rows = [_extract_text(r) for r in ds]
                rows = [r for r in rows if r][:n]
                if rows:
                    log.info("  got %d %s prompts from %s", len(rows), label, ds_id)
                    return rows
            except Exception as e:
                log.warning("  %s failed: %s", ds_id, e)
        raise RuntimeError(f"could not load any {label} dataset")

    harmful = pull(harmful_sources, harmful_n, "harmful")
    harmless = pull(harmless_sources, harmless_n, "harmless")
    return harmful, harmless


@torch.no_grad()
def get_last_token_residuals(
    model, tokenizer, prompts: list[str], device: str = "cuda"
) -> torch.Tensor:
    """Run model on each prompt, capture last-token hidden state at every layer.

    Returns: [n_prompts, n_layers+1, d_model] (CPU, float32 for stable accumulation)
    """
    all_residuals = []
    n_layers_plus = None
    for i, prompt in enumerate(prompts):
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple of (n_layers + 1) tensors, each [1, seq, d_model]
        # Take last token from each, stack
        last_per_layer = torch.stack(
            [h[0, -1, :].float().cpu() for h in out.hidden_states]
        )  # [n_layers+1, d_model]
        all_residuals.append(last_per_layer)
        if n_layers_plus is None:
            n_layers_plus = last_per_layer.shape[0]
        if (i + 1) % 16 == 0:
            log.info("  captured residuals for %d/%d prompts", i + 1, len(prompts))

    return torch.stack(all_residuals)  # [n_prompts, n_layers+1, d_model]


def compute_refusal_directions(
    harmful_res: torch.Tensor, harmless_res: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Returns (per-layer normalized direction, per-layer magnitude, best layer index).

    harmful_res, harmless_res: [n_prompts, n_layers+1, d_model] (float32, CPU)
    """
    mean_h = harmful_res.mean(dim=0)  # [n_layers+1, d_model]
    mean_l = harmless_res.mean(dim=0)
    diff = mean_h - mean_l
    magnitudes = diff.norm(dim=-1)  # [n_layers+1]
    directions = diff / magnitudes.unsqueeze(-1).clamp(min=1e-8)  # [n_layers+1, d_model]

    # Pick layer with highest magnitude (skip layer 0 = raw embedding, less signal)
    # mlabonne picks from top-20 candidates with human inspection; we go for max magnitude.
    # Skip the first 10% and last 10% of layers — refusal direction is usually mid-stack.
    n_layers_plus = magnitudes.shape[0]
    skip_lo = max(1, n_layers_plus // 10)
    skip_hi = n_layers_plus - max(1, n_layers_plus // 10)
    best = int(magnitudes[skip_lo:skip_hi].argmax().item()) + skip_lo

    return directions, magnitudes, best


def orthogonalize_rows(W: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """W: [num_rows, d_model], v: [d_model]. Subtract v-component from each row."""
    proj_coefs = W @ v  # [num_rows]
    projection = proj_coefs.unsqueeze(-1) * v  # [num_rows, d_model]
    return W - projection


def orthogonalize_columns(W: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """W: [d_model, d_in], v: [d_model]. Make output orthogonal to v.

    For nn.Linear: y = x @ W.T. To ensure y · v = 0 for any x:
      W_new = W - v ⊗ (v.T @ W)
    """
    proj_coefs = v @ W  # [d_in]
    projection = v.unsqueeze(-1) * proj_coefs.unsqueeze(0)  # [d_model, d_in]
    return W - projection


def orthogonalize_fused_experts(W: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """W: [n_experts, d_model, d_inter], v: [d_model]. Per-expert orthogonalize."""
    # proj_coefs[e, i] = sum_d v[d] * W[e, d, i]
    proj_coefs = torch.einsum("d,nde->ne", v, W)  # [n_experts, d_inter]
    projection = v.view(1, -1, 1) * proj_coefs.unsqueeze(1)  # [n_experts, d_model, d_inter]
    return W - projection


def apply_abliteration(model, refusal_dir: torch.Tensor) -> dict:
    """Walk the model and orthogonalize every residual-writing weight.

    Returns a dict with counts per kind for logging.
    """
    counts = {
        "embed": 0, "attn_full": 0, "attn_linear": 0,
        "mlp_shared": 0, "mlp_expert_fused": 0, "mlp_expert_listed": 0, "mlp_dense": 0,
    }
    base = getattr(model, "model", model)

    # 1. Token embedding
    embed = getattr(base, "embed_tokens", None)
    if embed is not None and hasattr(embed, "weight"):
        device = embed.weight.device
        v_dev = refusal_dir.to(device=device, dtype=embed.weight.dtype)
        embed.weight.data = orthogonalize_rows(embed.weight.data, v_dev)
        counts["embed"] = 1

    # 2. Walk decoder layers
    layers = getattr(base, "layers", [])
    for i, layer in enumerate(layers):
        # 2a. Attention output (full or linear)
        for attr_name, kind, proj_attr in (
            ("self_attn", "attn_full", "o_proj"),
            ("linear_attn", "attn_linear", "out_proj"),
        ):
            attn = getattr(layer, attr_name, None)
            if attn is None:
                continue
            proj = getattr(attn, proj_attr, None)
            if proj is None or not hasattr(proj, "weight"):
                continue
            device = proj.weight.device
            v_dev = refusal_dir.to(device=device, dtype=proj.weight.dtype)
            proj.weight.data = orthogonalize_columns(proj.weight.data, v_dev)
            counts[kind] += 1

        # 2b. MLP — shared expert + (fused experts | listed experts | dense)
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        # Dense MLP
        dp = getattr(mlp, "down_proj", None)
        if dp is not None and hasattr(dp, "weight"):
            device = dp.weight.device
            v_dev = refusal_dir.to(device=device, dtype=dp.weight.dtype)
            dp.weight.data = orthogonalize_columns(dp.weight.data, v_dev)
            counts["mlp_dense"] += 1

        # MoE shared expert (Qwen3 family)
        shared = getattr(mlp, "shared_expert", None)
        if shared is not None:
            sdp = getattr(shared, "down_proj", None)
            if sdp is not None and hasattr(sdp, "weight"):
                device = sdp.weight.device
                v_dev = refusal_dir.to(device=device, dtype=sdp.weight.dtype)
                sdp.weight.data = orthogonalize_columns(sdp.weight.data, v_dev)
                counts["mlp_shared"] += 1

        # MoE routed experts — fused or listed
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            fused_dp = getattr(experts, "down_proj", None)
            if isinstance(fused_dp, torch.nn.Parameter) or isinstance(fused_dp, torch.Tensor):
                device = fused_dp.device
                v_dev = refusal_dir.to(device=device, dtype=fused_dp.dtype)
                new_dp = orthogonalize_fused_experts(fused_dp.data, v_dev)
                fused_dp.data.copy_(new_dp)
                counts["mlp_expert_fused"] += 1
            else:
                try:
                    for exp in experts:
                        edp = getattr(exp, "down_proj", None)
                        if edp is not None and hasattr(edp, "weight"):
                            device = edp.weight.device
                            v_dev = refusal_dir.to(device=device, dtype=edp.weight.dtype)
                            edp.weight.data = orthogonalize_columns(edp.weight.data, v_dev)
                            counts["mlp_expert_listed"] += 1
                except (TypeError, AttributeError):
                    pass

    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to merged model")
    ap.add_argument("--out", required=True, help="path to save abliterated model")
    ap.add_argument("--n-prompts", type=int, default=64,
                    help="number of harmful + harmless prompts each (default 64)")
    ap.add_argument("--push-to-hub", default=None, help="HF repo to push to")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    log.info("=" * 70)
    log.info("Abliteration (Arditi/Labonne method)")
    log.info("  model: %s", args.model)
    log.info("  out:   %s", args.out)
    log.info("  n_prompts (each side): %d", args.n_prompts)
    log.info("=" * 70)

    t0 = time.time()
    log.info("[1/6] loading tokenizer + model")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    log.info("  loaded in %.1fs", time.time() - t0)

    log.info("[2/6] loading harmful + harmless prompts")
    harmful, harmless = load_prompts(args.n_prompts, args.n_prompts)

    log.info("[3/6] capturing last-token residuals (harmful)")
    t1 = time.time()
    h_residuals = get_last_token_residuals(model, tok, harmful)
    log.info("  done in %.1fs, shape=%s", time.time() - t1, tuple(h_residuals.shape))

    log.info("[3/6] capturing last-token residuals (harmless)")
    t2 = time.time()
    l_residuals = get_last_token_residuals(model, tok, harmless)
    log.info("  done in %.1fs, shape=%s", time.time() - t2, tuple(l_residuals.shape))

    log.info("[4/6] computing refusal directions")
    directions, magnitudes, best_layer = compute_refusal_directions(h_residuals, l_residuals)
    log.info("  best layer: %d  magnitude: %.4f", best_layer, magnitudes[best_layer].item())
    # Print top 5 candidates for transparency
    top5 = magnitudes.argsort(descending=True)[:5].tolist()
    for rank, layer_idx in enumerate(top5):
        log.info("    [#%d] layer=%d magnitude=%.4f", rank + 1, layer_idx, magnitudes[layer_idx].item())

    refusal_dir = directions[best_layer]
    log.info("  refusal direction: shape=%s norm=%.4f",
             tuple(refusal_dir.shape), refusal_dir.norm().item())

    log.info("[5/6] orthogonalizing all residual-writing weights")
    t3 = time.time()
    counts = apply_abliteration(model, refusal_dir)
    log.info("  done in %.1fs", time.time() - t3)
    for k, v in counts.items():
        log.info("    %-22s %4d", k, v)
    total = sum(counts.values())
    log.info("    %-22s %4d", "TOTAL", total)

    log.info("[6/6] saving abliterated model to %s", args.out)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    t4 = time.time()
    model.save_pretrained(str(out_path), safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(str(out_path))
    log.info("  saved in %.1fs", time.time() - t4)

    # Save abliteration metadata for the model card
    meta = {
        "method": "Arditi et al. 2024 / Labonne 2024 weight orthogonalization",
        "best_layer": best_layer,
        "best_magnitude": float(magnitudes[best_layer].item()),
        "top5_candidates": [
            {"layer": int(l), "magnitude": float(magnitudes[l].item())} for l in top5
        ],
        "n_harmful_prompts": len(harmful),
        "n_harmless_prompts": len(harmless),
        "harmful_source": "walledai/AdvBench",
        "harmless_source": "mlabonne/harmless_alpaca",
        "matrices_orthogonalized": counts,
        "refusal_direction_norm": float(refusal_dir.norm().item()),
    }
    (out_path / "abliteration_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("  metadata: %s", out_path / "abliteration_metadata.json")

    if args.push_to_hub:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        log.info("pushing to https://huggingface.co/%s (private=%s)",
                 args.push_to_hub, args.private)
        create_repo(args.push_to_hub, repo_type="model", exist_ok=True, private=args.private)
        api.upload_folder(
            folder_path=str(out_path),
            repo_id=args.push_to_hub,
            repo_type="model",
            commit_message="abliterated via Arditi/Labonne method (custom hybrid-attn + MoE handling)",
        )
        log.info("DONE. browse: https://huggingface.co/%s", args.push_to_hub)

    log.info("Total time: %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
