"""Architecture compatibility inspector for abliteration.

Walks a model and lists every weight matrix that writes to the residual
stream (the things we need to orthogonalize). Run this BEFORE any
abliteration attempt to know exactly what to target — especially for
non-standard architectures (MoE, hybrid attention, mamba-transformer mix).

Usage:
    python abliterate/test_arch_compat.py /path/to/model

Output:
    Lists every residual-writing weight matrix with:
    - Module path (e.g., model.layers.0.self_attn.o_proj.weight)
    - Shape
    - Layer type (full attn / linear attn / MLP shared / MLP expert / embed)

Use this output to drive abliterate_arditi.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class ResidualWriter:
    path: str
    shape: tuple[int, ...]
    kind: str  # "embed" | "attn_full" | "attn_linear" | "mlp_shared" | "mlp_expert" | "mlp_dense" | "lm_head"


def find_residual_writers(model) -> list[ResidualWriter]:
    """Walk the model and collect every weight matrix that writes to residual."""
    out: list[ResidualWriter] = []

    def add(path: str, weight: torch.Tensor, kind: str):
        out.append(ResidualWriter(path=path, shape=tuple(weight.shape), kind=kind))

    # 1. Token embedding
    embed = getattr(getattr(model, "model", model), "embed_tokens", None)
    if embed is not None and hasattr(embed, "weight"):
        add("model.embed_tokens.weight", embed.weight, "embed")

    # 2. LM head (sometimes tied to embedding, sometimes not — orthogonalize if untied)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        # Check if it's tied to embed_tokens; if so, skip (already handled)
        if embed is None or lm_head.weight.data_ptr() != embed.weight.data_ptr():
            add("lm_head.weight", lm_head.weight, "lm_head")

    # 3. Walk each decoder layer
    layers = getattr(getattr(model, "model", model), "layers", [])
    for i, layer in enumerate(layers):
        # 3a. Attention output projection (full attention)
        if hasattr(layer, "self_attn") and layer.self_attn is not None:
            attn = layer.self_attn
            for cand in ("o_proj", "out_proj", "dense", "wo"):
                proj = getattr(attn, cand, None)
                if proj is not None and hasattr(proj, "weight"):
                    add(f"model.layers.{i}.self_attn.{cand}.weight", proj.weight, "attn_full")
                    break

        # 3b. Linear attention output projection (hybrid models)
        if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
            la = layer.linear_attn
            for cand in ("o_proj", "out_proj", "dense", "wo", "out", "output", "proj_out"):
                proj = getattr(la, cand, None)
                if proj is not None and hasattr(proj, "weight"):
                    add(f"model.layers.{i}.linear_attn.{cand}.weight", proj.weight, "attn_linear")
                    break
            else:
                # Couldn't find a standard output projection — dump the structure for human inspection
                attrs = [n for n in dir(la) if not n.startswith("_")]
                module_attrs = [n for n in attrs if hasattr(getattr(la, n, None), "weight")]
                add(f"model.layers.{i}.linear_attn.<UNKNOWN: candidates={module_attrs}>", torch.zeros(1), "attn_linear")

        # 3c. MLP outputs — handle dense + MoE variants
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            # Dense MLP
            dp = getattr(mlp, "down_proj", None)
            if dp is not None and hasattr(dp, "weight"):
                add(f"model.layers.{i}.mlp.down_proj.weight", dp.weight, "mlp_dense")

            # MoE shared expert (Qwen3 family)
            shared = getattr(mlp, "shared_expert", None)
            if shared is not None:
                shared_dp = getattr(shared, "down_proj", None)
                if shared_dp is not None and hasattr(shared_dp, "weight"):
                    add(f"model.layers.{i}.mlp.shared_expert.down_proj.weight",
                        shared_dp.weight, "mlp_shared")

            # MoE routed experts
            experts = getattr(mlp, "experts", None)
            if experts is not None:
                try:
                    n_experts = len(experts)
                    # Check if it's a list (Qwen3 style) or fused (DeepSeek-V2 style with parameter tensor)
                    if n_experts > 0 and hasattr(experts[0], "down_proj"):
                        for j, exp in enumerate(experts):
                            dp = getattr(exp, "down_proj", None)
                            if dp is not None and hasattr(dp, "weight"):
                                add(f"model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                                    dp.weight, "mlp_expert")
                except (TypeError, AttributeError):
                    # Fused experts? Look for a single weight tensor.
                    for cand in ("w2", "down_proj_weight", "experts_down_proj"):
                        ew = getattr(experts, cand, None) or getattr(mlp, cand, None)
                        if ew is not None and hasattr(ew, "weight"):
                            add(f"model.layers.{i}.mlp.experts.{cand}.weight", ew.weight, "mlp_expert")

            # block_sparse_moe (Phi-3.5-MoE style)
            bsm = getattr(layer, "block_sparse_moe", None)
            if bsm is not None and hasattr(bsm, "experts"):
                for j, exp in enumerate(bsm.experts):
                    w2 = getattr(exp, "w2", None)
                    if w2 is not None and hasattr(w2, "weight"):
                        add(f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight",
                            w2.weight, "mlp_expert")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--meta", action="store_true",
                    help="Use meta device (don't actually load weights)")
    args = ap.parse_args()

    print(f"Inspecting: {args.model_path}")
    print()

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Architecture:    {cfg.architectures}")
    print(f"Model type:      {cfg.model_type}")
    print(f"Hidden size:     {getattr(cfg, 'hidden_size', '?')}")
    print(f"Num layers:      {getattr(cfg, 'num_hidden_layers', '?')}")
    if hasattr(cfg, "num_experts"):
        print(f"Num experts:     {cfg.num_experts}")
    if hasattr(cfg, "num_experts_per_tok"):
        print(f"Top-k routing:   {cfg.num_experts_per_tok}")
    print()

    print("Loading model (may take ~30s)...")
    if args.meta:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    print()
    writers = find_residual_writers(model)

    # Group + report
    by_kind: dict[str, list[ResidualWriter]] = {}
    for w in writers:
        by_kind.setdefault(w.kind, []).append(w)

    print(f"{'='*70}")
    print(f"RESIDUAL-WRITING WEIGHT MATRICES")
    print(f"{'='*70}")
    for kind in ("embed", "lm_head", "attn_full", "attn_linear",
                 "mlp_dense", "mlp_shared", "mlp_expert"):
        items = by_kind.get(kind, [])
        if not items:
            continue
        print(f"\n[{kind}]  ({len(items)} matrices)")
        if len(items) <= 6:
            for w in items:
                print(f"  {w.path:<70} {w.shape}")
        else:
            # Print first 2 + last 2 + count
            for w in items[:2]:
                print(f"  {w.path:<70} {w.shape}")
            print(f"  ... [{len(items) - 4} more] ...")
            for w in items[-2:]:
                print(f"  {w.path:<70} {w.shape}")

    print()
    print(f"{'='*70}")
    print(f"TOTAL: {len(writers)} residual-writing weight matrices")
    print(f"{'='*70}")

    # Architecture sanity: did we hit any UNKNOWN linear_attn?
    unknown_la = [w for w in writers if "UNKNOWN" in w.path]
    if unknown_la:
        print()
        print("WARNING: linear_attn has unknown output projection structure:")
        for w in unknown_la:
            print(f"  {w.path}")
        print("Inspect the layer manually before running abliteration.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
