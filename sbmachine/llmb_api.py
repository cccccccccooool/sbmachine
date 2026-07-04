"""Phase 3b OpenAI-compatible API backend."""
from __future__ import annotations

from pathlib import Path

from sbmachine.common import PROJECT_ROOT, _output_cap, resolve_path
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


def load_style_skill(config: dict) -> str:
    skill_path = (config.get("paths", {}) or {}).get("style_skill")
    path = resolve_path(skill_path, base=PROJECT_ROOT)
    if path is None or not Path(path).exists():
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


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
    return _execute_openai_chat(messages, llm_cfg, max_tokens=cap, log_ctx=log_ctx, secret_scope="llmb")
