"""Local Adapter —— 本地 vLLM 回环端点的私有调用层。

仅 backend == "vllm" 路径使用。负责（计划书 §2.3 本地私有层）：
- 回环地址、非流式请求
- 全局请求节流 request_interval_sec（本地重负载挤占显存/带宽时压制速率）

不包含：vLLM 服务生命周期（service_manager/compose_manager）、GPU guard、显存管理。
"""
from __future__ import annotations

import threading
import time
import uuid

import requests

from sbmachine.llm_protocol import (
    _build_chat_payload,
    _dump_api_log,
    _finalize_chat_result,
    _is_loopback_url,
    _load_secrets,
    _post_openai_with_retry,
    _request_error_category,
    _resolve_api_key,
)

# 全局 LLM 请求节流：相邻请求至少间隔 request_interval_sec（由调用方经
# llm_config["request_interval_sec"] 下发）。拆分为窗口请求时（本地 Phase3a）
# 用它把请求速率压到限流阈值以下，避免短时间连发触发服务端封禁。
_throttle_lock = threading.Lock()
_last_request_ts = 0.0


def _request_throttle(min_interval_sec: float | None) -> None:
    """相邻 LLM 请求至少间隔 min_interval_sec 秒（<=0 时不节流）。"""
    if min_interval_sec is None or float(min_interval_sec) <= 0:
        return
    interval = float(min_interval_sec)
    global _last_request_ts
    with _throttle_lock:
        now = time.monotonic()
        wait = interval - (now - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_request_ts = now


def local_generate(
    messages: list[dict],
    llm_config: dict,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
    secret_scope: str | None = None,
    response_format: dict | None = None,
) -> str:
    """向本地 vLLM 回环端点发起一次无状态非流式请求。

    与云端 adapter 的差异（计划书 §2.3）：
    - 非流式（不注入 stream/stream_options）
    - request_interval_sec 全局节流（仅回环端点生效）
    """
    secrets = _load_secrets()
    raw_scoped = secrets.get(secret_scope, {}) if secret_scope else {}
    scoped = raw_scoped if isinstance(raw_scoped, dict) else {}
    base_url = str(scoped.get("base_url") or secrets.get("base_url") or llm_config.get("base_url") or "")
    if not base_url:
        raise ValueError("LLM base_url is not configured; set BASE_URL (or AI6657_CLOUD_BASE_URL) in .env")
    if not _is_loopback_url(base_url):
        raise ValueError("local_generate requires a loopback endpoint; remote URL must use cloud_adapter")
    # 全局请求节流仅对本机回环端点（本地 vLLM）生效：本地推理无服务端限流，
    # 但重负载时会挤占显存/带宽；云端官方 API 自带配额管理，客户端节流纯属拖慢。
    _request_throttle(float(llm_config.get("request_interval_sec", 0.0) or 0.0))
    url = f"{base_url.rstrip('/')}/chat/completions"
    api_key = _resolve_api_key(base_url, str(scoped.get("api_key") or secrets.get("api_key") or ""))
    model = str(scoped.get("model") or secrets.get("model") or llm_config.get("model") or "")
    if not model:
        raise ValueError("LLM model is not configured; set MODEL (or AI6657_CLOUD_MODEL) in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = _build_chat_payload(messages, llm_config, model, max_tokens=max_tokens, response_format=response_format)

    timeout_base = float(llm_config.get("timeout_sec", 120) or 120)
    connect_timeout = float(llm_config.get("connect_timeout_sec", 0) or 0) or timeout_base
    read_idle_timeout = float(llm_config.get("read_idle_timeout_sec", 0) or 0) or timeout_base

    request_ctx = dict(log_ctx or {})
    source_run_id = str(request_ctx.get("run_id") or uuid.uuid4().hex)
    request_ctx["run_id"] = source_run_id
    request_id = uuid.uuid4().hex
    request_ctx["request_id"] = request_id
    started = time.perf_counter()
    try:
        data = _post_openai_with_retry(
            url, payload, headers,
            timeout=(connect_timeout, read_idle_timeout),
        )
    except requests.RequestException as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        status = exc.response.status_code if exc.response is not None else None
        _dump_api_log(
            url, payload, {}, log_ctx=request_ctx, scope=secret_scope,
            http_status=status, duration_ms=duration_ms, error=type(exc).__name__,
            request_id=request_id, model=model, streaming=False,
            retry_category=_request_error_category(exc),
        )
        raise
    duration_ms = round((time.perf_counter() - started) * 1000)
    status = int(getattr(data, "http_status", 200))
    transport = getattr(data, "transport", None)
    transport_meta = transport if isinstance(transport, dict) else {}
    _dump_api_log(
        url, payload, data, log_ctx=request_ctx, scope=secret_scope,
        http_status=status, duration_ms=duration_ms,
        request_id=request_id, model=model,
        streaming=transport_meta.get("streaming"),
        connect_ms=transport_meta.get("connect_ms"),
        ttfb_ms=transport_meta.get("ttfb_ms"),
        usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
    )
    return _finalize_chat_result(
        data,
        url=url,
        payload=payload,
        log_ctx=request_ctx,
        request_id=request_id,
        secret_scope=secret_scope,
    )
