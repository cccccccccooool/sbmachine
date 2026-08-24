"""Cloud Adapter —— 云端 OpenAI 兼容 API 的私有调用层。

仅 backend == "api" 路径使用。负责（计划书 §2.2 云端私有层）：
- 远端密钥/端点解析、SSE 流式聚合、总时限
- scope 级有界信号量（cloud_request_concurrency）与队列等待上限
- 在途请求计数（诊断）、脱敏日志
- 连通性预检 probe_api_connectivity

不包含：会话历史（cloud_memory）、Prompt 组合（cloud_prompts）、成功缓存（cloud_cache）。
"""
from __future__ import annotations

import threading
import time
import uuid

import requests

from sbmachine.llm_protocol import (
    _ApiChatResult,
    _build_chat_payload,
    _dump_api_log,
    _finalize_chat_result,
    _is_loopback_url,
    _load_secrets,
    _post_openai_with_retry,
    _request_error_category,
    _resolve_api_key,
)

# scope 级有界信号量（阶段 1 云端护栏）：cloud_request_concurrency>0 时生效，
# 队列等待按 cloud_queue_timeout_sec 限制，超时抛 Timeout（可重试基础设施错误）。
_SCOPE_SEMAPHORES: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_SCOPE_SEMAPHORES_LOCK = threading.Lock()

# 全局在途请求计数（诊断用）：记录每个请求开始时已有的并发数。
_in_flight_lock = threading.Lock()
_in_flight_count = 0


def _scope_semaphore(scope: str, concurrency: int) -> threading.BoundedSemaphore:
    key = (str(scope or "default"), int(concurrency))
    with _SCOPE_SEMAPHORES_LOCK:
        sem = _SCOPE_SEMAPHORES.get(key)
        if sem is None:
            sem = threading.BoundedSemaphore(int(concurrency))
            _SCOPE_SEMAPHORES[key] = sem
        return sem


def _in_flight_start() -> int:
    """进入一次请求：返回请求开始时的在途请求数。"""
    global _in_flight_count
    with _in_flight_lock:
        before = _in_flight_count
        _in_flight_count += 1
        return before


def _in_flight_end() -> None:
    global _in_flight_count
    with _in_flight_lock:
        _in_flight_count = max(0, _in_flight_count - 1)


def probe_api_connectivity(
    llm_cfg: dict,
    *,
    attempts: int = 3,
    timeout_sec: float = 30.0,
) -> None:
    """Phase3 前的 LLM API 连通性预检。

    对齐本地 vLLM 服务的 startup_timeout 健康检查语义：发一次 max_tokens=1 的
    ping，连续失败 attempts 次（或超时）直接抛错中止流水线，避免整场阶段在
    无效请求上白跑。调用方传入的 llm_cfg 不会被修改。
    """
    probe_cfg = dict(llm_cfg or {})
    probe_cfg["timeout_sec"] = max(5, int(timeout_sec))
    last_error: BaseException | None = None
    total = max(1, int(attempts))
    for attempt in range(total):
        try:
            cloud_generate(
                [{"role": "user", "content": "ping"}],
                probe_cfg,
                max_tokens=1,
                log_ctx=None,
                secret_scope=None,
            )
            return
        except BaseException as exc:  # noqa: BLE001 - 预检只关心连通性结果
            last_error = exc
            if attempt + 1 < total:
                time.sleep(min(float(2 ** attempt), 10.0))
    raise RuntimeError(
        f"LLM API connectivity probe failed after {total} attempts "
        f"({type(last_error).__name__}: {last_error})"
    )


def cloud_generate(
    messages: list[dict],
    llm_config: dict,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
    secret_scope: str | None = None,
    response_format: dict | None = None,
) -> str:
    """向云端 OpenAI 兼容端点发起一次无状态流式请求（SSE）。

    与本地 adapter 的差异（计划书 §2.2）：
    - 强制 stream=true + stream_options.include_usage，SSE 聚合后归一为 _ApiChatResult
    - total_timeout_sec 硬总时限（超过抛 requests.Timeout，可重试基础设施错误）
    - cloud_request_concurrency 有界信号量 + cloud_queue_timeout_sec 排队上限
    """
    secrets = _load_secrets()
    raw_scoped = secrets.get(secret_scope, {}) if secret_scope else {}
    scoped = raw_scoped if isinstance(raw_scoped, dict) else {}
    base_url = str(scoped.get("base_url") or secrets.get("base_url") or llm_config.get("base_url") or "")
    if not base_url:
        raise ValueError("LLM base_url is not configured; set BASE_URL (or AI6657_CLOUD_BASE_URL) in .env")
    if _is_loopback_url(base_url):
        raise ValueError("cloud_generate requires a remote endpoint; loopback URL must use local_adapter")
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
    # 云端官方 API 走流式（SSE）：思考模型首 token 更早返回，避免整段等待；
    # stream_options.include_usage 让末块携带 usage，训练样本字段不丢。
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    timeout_base = float(llm_config.get("timeout_sec", 120) or 120)
    connect_timeout = float(llm_config.get("connect_timeout_sec", 0) or 0) or timeout_base
    read_idle_timeout = float(llm_config.get("read_idle_timeout_sec", 0) or 0) or timeout_base
    total_timeout_sec = llm_config.get("total_timeout_sec")
    total_timeout = float(total_timeout_sec) if total_timeout_sec else None

    request_ctx = dict(log_ctx or {})
    source_run_id = str(request_ctx.get("run_id") or uuid.uuid4().hex)
    request_ctx["run_id"] = source_run_id
    request_id = uuid.uuid4().hex
    request_ctx["request_id"] = request_id
    started = time.perf_counter()
    # 阶段 1：scope 级有界并发（cloud_request_concurrency>0 时生效）；
    # 队列等待不计入模型生成耗时，单独记录 queue_ms。
    concurrency = int(llm_config.get("cloud_request_concurrency", 0) or 0)
    semaphore = _scope_semaphore(secret_scope, concurrency) if concurrency > 0 else None
    queue_ms = 0
    if semaphore is not None:
        queue_timeout = float(llm_config.get("cloud_queue_timeout_sec", 0) or 0) or None
        if not semaphore.acquire(timeout=queue_timeout):
            raise requests.Timeout(
                f"scope {secret_scope} queue wait exceeded cloud_queue_timeout_sec"
            )
        queue_ms = round((time.perf_counter() - started) * 1000)
    in_flight_before = _in_flight_start()
    try:
        data = _post_openai_with_retry(
            url, payload, headers,
            timeout=(connect_timeout, read_idle_timeout),
            total_timeout_sec=total_timeout,
        )
    except requests.RequestException as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        status = exc.response.status_code if exc.response is not None else None
        _dump_api_log(
            url, payload, {}, log_ctx=request_ctx, scope=secret_scope,
            http_status=status, duration_ms=duration_ms, error=type(exc).__name__,
            request_id=request_id, model=model, streaming=bool(payload.get("stream")),
            in_flight=in_flight_before, retry_category=_request_error_category(exc),
            queue_ms=queue_ms,
        )
        raise
    finally:
        _in_flight_end()
        if semaphore is not None:
            semaphore.release()
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
        in_flight=in_flight_before,
        queue_ms=queue_ms,
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
