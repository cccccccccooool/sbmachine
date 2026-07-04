"""Phase 3b local Ollama backend."""
from __future__ import annotations

import os

from sbmachine.common import _output_cap, _post_with_retry


def _ctx_hint(log_ctx: dict | None) -> str:
    if not log_ctx:
        return ""
    r = log_ctx.get("round", "")
    s = log_ctx.get("scene", "")
    if r and s:
        return f" [{r} - {s}]"
    if r:
        return f" [{r}]"
    return ""


def generate(
    prompt: str,
    llm_cfg: dict,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
) -> str:
    cap = _output_cap(llm_cfg, max_tokens)
    timeout = int(llm_cfg.get("timeout_sec", 300))
    print(f"  >> [LLM API] 正在请求 ollama 后端{_ctx_hint(log_ctx)}... (timeout: {timeout}s)", flush=True)

    url = os.getenv("AI6657_OLLAMA_URL") or llm_cfg.get("ollama_url", "http://127.0.0.1:11434/api/generate")
    model = os.getenv("AI6657_LLM_MODEL") or llm_cfg.get("model", "qwen3:8b")
    options: dict = {
        "temperature": float(llm_cfg.get("temperature", 0.75)),
        "num_ctx": int(llm_cfg.get("num_ctx", 16384)),
        "repeat_penalty": float(llm_cfg.get("repeat_penalty", 1.15)),
        "repeat_last_n": int(llm_cfg.get("repeat_last_n", 256)),
    }
    if cap:
        options["num_predict"] = cap
    if "qwen3" in model.lower():
        options["think"] = False
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    response = _post_with_retry(url, payload, timeout)
    data = response.json()
    return (data.get("response") or "").strip()
