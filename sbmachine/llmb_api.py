"""Phase 3b 的 OpenAI 兼容 API 后端。"""
from __future__ import annotations

import os
from pathlib import Path

from sbmachine.common import PROJECT_ROOT, _output_cap, resolve_path
from sbmachine.llm_shim import _execute_openai_chat


STYLE_DEFAULTS = {
    "style_temperature": 0.55,
    "style_top_p": 0.85,
    "style_frequency_penalty": 0.2,
    "style_output_max_tokens": 256,
    "style_max_retries": 2,
    "style_recent_window_count": 12,
    "style_phrase_max_reuse": 2,
    "style_budget_hard_tolerance": 0.5,
    "style_empty_window_threshold": 0.30,
    "style_k_enabled": False,
}


def style_runtime_config(config: dict) -> tuple[dict, dict]:
    """读取 Phase3b 专属采样/验收配置；旧配置使用明确默认值。"""
    llm_cfg = dict(config.get("llm", {}) if isinstance(config.get("llm"), dict) else {})
    semantic_config = config.get("semantic", {}) if isinstance(config.get("semantic"), dict) else {}
    effective_config = {
        key: semantic_config.get(key, default) for key, default in STYLE_DEFAULTS.items()
    }
    # 云端特化键不在 STYLE_DEFAULTS 中，需单独透传，否则 phase3b 云端 max_tokens 放开失效。
    if semantic_config.get("cloud_style_output_max_tokens"):
        effective_config["cloud_style_output_max_tokens"] = int(
            semantic_config["cloud_style_output_max_tokens"]
        )
    llm_cfg.update({
        "temperature": float(effective_config["style_temperature"]),
        "top_p": float(effective_config["style_top_p"]),
        "frequency_penalty": float(effective_config["style_frequency_penalty"]),
    })
    if semantic_config.get("style_request_interval_sec"):
        llm_cfg["request_interval_sec"] = float(semantic_config["style_request_interval_sec"])
    return llm_cfg, effective_config


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


def load_style_skill(config: dict) -> str:
    """从配置里读取解说风格 skill 文件内容，缺失时返回空串。"""
    skill_path = (config.get("paths", {}) or {}).get("style_skill")
    resolved_skill_path = resolve_path(skill_path, base=PROJECT_ROOT)
    if resolved_skill_path is None or not Path(resolved_skill_path).exists():
        return ""
    return Path(resolved_skill_path).read_text(encoding="utf-8").strip()


def generate(
    prompt: str,
    llm_cfg: dict,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
    response_format: dict | None = None,
) -> str:
    """向 Phase 3b 的 OpenAI 兼容后端发起一次生成请求。"""
    output_cap = _output_cap(llm_cfg, max_tokens)
    timeout = int(llm_cfg.get("timeout_sec", 120))
    if os.getenv("AI6657_DEBUG_PHASE3"):
        print(f"  >> [LLM API] 正在请求 api 后端{_ctx_hint(log_ctx)}... (timeout: {timeout}s)", flush=True)
    messages = [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": prompt},
    ]
    return _execute_openai_chat(
        messages,
        llm_cfg,
        max_tokens=output_cap,
        log_ctx=log_ctx,
        secret_scope="llmb",
        response_format=response_format,
    )
