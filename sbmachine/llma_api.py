"""Phase 3a OpenAI-compatible API backend."""
from __future__ import annotations

from sbmachine.common import _output_cap
from sbmachine.llm_shim import _execute_openai_chat


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
    timeout = int(llm_cfg.get("timeout_sec", 120))
    print(f"  >> [LLM API] 正在请求 api 后端{_ctx_hint(log_ctx)}... (timeout: {timeout}s)", flush=True)
    messages = [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": prompt},
    ]
    return _execute_openai_chat(messages, llm_cfg, max_tokens=cap, log_ctx=log_ctx, secret_scope="llma")
