"""8K seq_len profiling — validates the 18h training budget assumption.

Loads Qwen3.6 with flash_attention_2, runs 5 micro-steps at seq_len=8192
batch_size=2 grad_accum=4 (effective batch = 8 × 8192 = 65,536 tokens/step),
and projects total wall-time for 80M tokens.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MP = "/data/hf/Qwen3.6-35B-A3B"
SEQ = int(os.environ.get("SEQ", "8192"))
BSZ = int(os.environ.get("BSZ", "1"))   # bumped from 2 → 1: MoE grouped-GEMM OOMs at bsz=2/seq=8192
GA = int(os.environ.get("GA", "8"))     # keep effective batch = 8

print(f"=== 8K seq profile: bsz={BSZ} grad_accum={GA} seq_len={SEQ} ===")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
tok.pad_token = tok.eos_token

print("[1/4] Loading model with flash_attention_2")
m = AutoModelForCausalLM.from_pretrained(
    MP,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
)
m.config.use_cache = False
m.gradient_checkpointing_enable()
m.enable_input_require_grads()
print(f"  loaded in {time.time()-t0:.1f}s, mem={round(torch.cuda.memory_allocated()/1e9, 1)} GB")

print("[2/4] Attaching LoRA")
m = get_peft_model(
    m,
    LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
)

print("[3/4] Building 8K-token random input batch")
input_ids = torch.randint(0, len(tok), (BSZ, SEQ), device=m.device)
attn = torch.ones_like(input_ids)
labels = input_ids.clone()

opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-5)

print("[4/4] Timing 5 micro-step iterations (2 warmup + 3 measured)")
torch.cuda.synchronize()
times = []
for i in range(5):
    t = time.time()
    opt.zero_grad()
    out = m(input_ids=input_ids, attention_mask=attn, labels=labels)
    out.loss.backward()
    if (i + 1) % GA == 0:
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t
    times.append(dt)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  iter {i}: {dt:.2f}s  loss={out.loss.item():.4f}  peak_mem={peak:.1f} GB")

measured = times[2:]
avg = sum(measured) / len(measured)
print(f"=== avg per micro-step (warmed): {avg:.2f}s ===")
print(f"=== effective step (×{GA} grad_accum): {avg*GA:.1f}s ===")

total_tokens = 80_000_000
tokens_per_step = BSZ * SEQ * GA
n_steps = total_tokens / tokens_per_step
total_h = n_steps * avg * GA / 3600
print(f"=== 80M tokens projection: {n_steps:.0f} steps × {avg*GA:.1f}s/step = {total_h:.1f} hours ===")
