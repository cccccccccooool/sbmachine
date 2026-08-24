"""云端 OpenAI 请求：为一整个小局返回结构化输出。"""
from __future__ import annotations

import time
import uuid

from sbmachine import llm_shim
from sbmachine.phase3a_cloud_prompt import cloud_response_format


def generate_cloud_round(prompt: str, llm_cfg: dict, *, system_prompt: str, max_tokens: int, log_ctx: dict | None = None) -> str:
    secrets = llm_shim._load_secrets()
    scoped = secrets.get("llma", {}) if isinstance(secrets.get("llma"), dict) else {}
    base_url = str(scoped.get("base_url") or secrets.get("base_url") or llm_cfg.get("base_url") or "")
    if not base_url:
        raise ValueError("LLM base_url is not configured; set BASE_URL (or AI6657_CLOUD_BASE_URL) in .env")
    api_key = llm_shim._resolve_api_key(base_url, str(scoped.get("api_key") or secrets.get("api_key") or ""))
    model = str(scoped.get("model") or secrets.get("model") or llm_cfg.get("model") or "")
    if not model:
        raise ValueError("LLM model is not configured; set MODEL (or AI6657_CLOUD_MODEL) in .env")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_cfg.get("temperature", 0.2)),
        "max_tokens": int(max_tokens) * 10,  # S6 前保持 ×10 兜底；cloud 路径独立于 Phase3a
        "response_format": cloud_response_format(),
    }
    if llm_cfg.get("top_p") is not None:
        payload["top_p"] = float(llm_cfg["top_p"])
    url = f"{base_url.rstrip('/')}/chat/completions"
    ctx = dict(log_ctx or {})
    ctx["run_id"] = str(ctx.get("run_id") or uuid.uuid4().hex)
    started = time.perf_counter()
    try:
        data = llm_shim._post_openai_with_retry(url, payload, {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, int(llm_cfg.get("timeout_sec", 300)))
    except Exception as exc:
        llm_shim._dump_api_log(url, payload, {}, log_ctx=ctx, scope="llma", http_status=getattr(getattr(exc, "response", None), "status_code", None), duration_ms=round((time.perf_counter() - started) * 1000), error=type(exc).__name__)
        raise
    llm_shim._dump_api_log(url, payload, data, log_ctx=ctx, scope="llma", http_status=int(getattr(data, "http_status", 200)), duration_ms=round((time.perf_counter() - started) * 1000))
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("cloud API response is missing choices")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("cloud API response is missing message content")
    return content.strip()
