"""Policy gateway for amd-hackathon-2026.

Three-tier defense (CLAUDE.md §L):
1. Model tier  — Heretic-abliterated to 0/465 refusals (MAX CAPABILITY).
2. Gateway tier — THIS MODULE. Llama-Guard-3-1B classifies in/out content
   against the MLCommons 13-category hazards taxonomy. Hard-line refusals
   for: CSAM (S4), bioweapon synthesis (S9), targeted impersonation (S5),
   doxxing (S7), narrow controlled-substance synthesis (S2 subset).
3. Audit tier  — every decision logged via gateway/audit_log.py.

Architecture:
    user → /v1/chat/completions →
        guard_in()  -> Llama-Guard-3-1B classifies prompt
        if blocked  -> 451 + classifier verdict
        else        -> upstream vLLM (or llama.cpp)
        guard_out() -> Llama-Guard-3-1B classifies response
        if blocked  -> 451 + classifier verdict (and original blanked)
        else        -> stream/return upstream response
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llama_guard")

POLICY_PATH = Path(os.environ.get("POLICY_PATH", "/app/policy.yaml"))
VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
LLAMACPP_URL = os.environ.get("LLAMACPP_BASE_URL", "http://localhost:8001/v1")
GUARD_MODEL = os.environ.get("GUARD_MODEL", "meta-llama/Llama-Guard-3-1B")

# Category code -> friendly name (MLCommons 13-cat)
CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
}


_guard_state: dict[str, Any] = {}


def load_policy() -> dict:
    if not POLICY_PATH.exists():
        log.warning("policy not found at %s — using empty policy", POLICY_PATH)
        return {"hard_refuse": [], "log_only": []}
    return yaml.safe_load(POLICY_PATH.read_text()) or {"hard_refuse": [], "log_only": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading guard model: %s", GUARD_MODEL)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GUARD_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    _guard_state["tokenizer"] = tokenizer
    _guard_state["model"] = model
    _guard_state["policy"] = load_policy()
    log.info("guard ready. hard-refuse categories: %s", _guard_state["policy"].get("hard_refuse", []))
    yield


app = FastAPI(lifespan=lifespan, title="amd-hackathon-2026 gateway")


def classify(messages: list[dict], side: str) -> dict:
    """Run Llama-Guard-3-1B on `messages`. side ∈ {prompt, response}."""
    import torch
    tok = _guard_state["tokenizer"]
    model = _guard_state["model"]

    chat = tok.apply_chat_template(messages, tokenize=False, return_tensors=None)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    verdict = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    safe = verdict.lower().startswith("safe")
    categories: list[str] = []
    if not safe:
        for line in verdict.splitlines():
            for token in line.replace(",", " ").split():
                if token.startswith("S") and token[1:].isdigit():
                    categories.append(token)
    return {"side": side, "safe": safe, "verdict": verdict, "categories": sorted(set(categories))}


def is_hard_refuse(categories: list[str], policy: dict) -> bool:
    hard = set(policy.get("hard_refuse", []))
    return bool(hard.intersection(categories))


@app.get("/health")
async def health():
    return {"status": "ok", "guard_model": GUARD_MODEL,
            "policy_categories": sorted(_guard_state["policy"].get("hard_refuse", []))}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    from audit_log import log_decision   # local import; gateway/audit_log.py
    body = await req.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(400, "missing messages")

    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user is None:
        raise HTTPException(400, "no user message")

    # In-bound classification
    in_verdict = classify([last_user], side="prompt")
    if not in_verdict["safe"] and is_hard_refuse(in_verdict["categories"], _guard_state["policy"]):
        log_decision(action="refuse_in", verdict=in_verdict, prompt=last_user["content"])
        return JSONResponse(
            status_code=451,
            content={
                "error": {
                    "code": "policy.hard_refuse",
                    "message": "Request blocked by gateway policy.",
                    "categories": [CATEGORIES.get(c, c) for c in in_verdict["categories"]],
                    "verdict": in_verdict["verdict"],
                }
            },
        )

    # Forward to upstream
    upstream = body.pop("_upstream", "vllm")
    base = LLAMACPP_URL if upstream == "llamacpp" else VLLM_URL
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{base}/chat/completions", json=body)
        r.raise_for_status()
        upstream_json = r.json()

    response_text = ""
    try:
        response_text = upstream_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass

    # Out-bound classification
    out_verdict = classify(
        [last_user, {"role": "assistant", "content": response_text}],
        side="response",
    )
    if not out_verdict["safe"] and is_hard_refuse(out_verdict["categories"], _guard_state["policy"]):
        log_decision(action="refuse_out", verdict=out_verdict, prompt=last_user["content"], response=response_text)
        return JSONResponse(
            status_code=451,
            content={
                "error": {
                    "code": "policy.hard_refuse_response",
                    "message": "Response blocked by gateway policy.",
                    "categories": [CATEGORIES.get(c, c) for c in out_verdict["categories"]],
                    "verdict": out_verdict["verdict"],
                }
            },
        )

    log_decision(action="allow", verdict=in_verdict, prompt=last_user["content"], response=response_text[:200])
    return upstream_json
