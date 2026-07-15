"""Phase 3a 的 OpenAI 兼容 API 后端。"""
from __future__ import annotations

from sbmachine.common import _output_cap
from sbmachine.llm_shim import _execute_openai_chat


def _ctx_hint(log_ctx: dict | None) -> str:
    """把日志上下文里的 round/scene 拼成一段可读的进度提示。"""
    if not log_ctx:
        return ""
    round_label = log_ctx.get("round", "")
    scene_label = log_ctx.get("scene", "")
    if round_label and scene_label:
        return f" [{round_label} - {scene_label}]"
    if round_label:
        return f" [{round_label}]"
    return ""


def generate(
    prompt: str,
    llm_cfg: dict,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
) -> str:
    """向 Phase 3a 的 OpenAI 兼容后端发起一次生成请求。"""
    cap = _output_cap(llm_cfg, max_tokens)
    timeout = int(llm_cfg.get("timeout_sec", 120))
    print(f"  >> [LLM API] 正在请求 api 后端{_ctx_hint(log_ctx)}... (timeout: {timeout}s)", flush=True)
    messages = [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": prompt},
    ]
    return _execute_openai_chat(messages, llm_cfg, max_tokens=cap, log_ctx=log_ctx, secret_scope="llma")
