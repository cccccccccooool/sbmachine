"""共享的 OpenAI 兼容 Chat Completions 客户端（薄兼容入口）。

整洁迁移（计划书 §3.1）后本模块只做三件事：
1. re-export LLM Protocol Core（sbmachine.llm_protocol）的全部共用符号，
   供既有调用方（phase3a/3b、cloud_memory、cloud_cache、llma_api、llmb_api、llm_backends）零改动继续使用。
2. re-export cloud/local adapter 的全局状态符号（信号量/在途计数/节流锁），
   兼容旧测试与调用方的 `llm_shim._scope_semaphore` 等属性访问。
3. 保留 `_execute_openai_chat` 分流函数：按 base_url 是否回环委托给
   `cloud_adapter.cloud_generate` / `local_adapter.local_generate`。

云端/本地各自的端点策略、Prompt、并发、会话、缓存与密钥归属已移入对应 adapter。
"""
from __future__ import annotations

from sbmachine import cloud_adapter
from sbmachine import local_adapter
from sbmachine.llm_protocol import (  # noqa: F401  - 兼容 re-export
    _ApiChatResult,
    _HttpJson,
    _LOG_DIR,
    _LOG_LOCK,
    _MAX_RETRY_AFTER_SEC,
    _PROJECT_ROOT,
    _RETRY_WAIT,
    _assert_json_content_type,
    _build_chat_payload,
    _consume_sse_response,
    _dump_accepted_api_sample,
    _dump_api_log,
    _endpoint_metadata,
    _finalize_chat_result,
    _is_loopback_url,
    _is_retryable_request_error,
    _last_user_prompt,
    _load_secrets,
    _payload_sha256,
    _post_openai_with_retry,
    _request_error_category,
    _resolve_api_key,
    _retry_after_seconds,
    _usage_tokens,
    _validate_api_base_url,
    _wait_retry_after_or_exponential,
    accept_api_response,
    infra_backoff_delay,
    record_validation_reason,
    retry_after_seconds,
)
from sbmachine.cloud_adapter import (  # noqa: F401  - 兼容 re-export
    _in_flight_count,
    _in_flight_end,
    _in_flight_lock,
    _in_flight_start,
    _scope_semaphore,
    probe_api_connectivity,
)
from sbmachine.local_adapter import (  # noqa: F401  - 兼容 re-export
    _last_request_ts,
    _request_throttle,
    _throttle_lock,
)

# 兼容旧调用方对 _SCOPE_SEMAPHORES 系列的引用。
_SCOPE_SEMAPHORES = cloud_adapter._SCOPE_SEMAPHORES
_SCOPE_SEMAPHORES_LOCK = cloud_adapter._SCOPE_SEMAPHORES_LOCK


def _execute_openai_chat(
    messages: list[dict],
    llm_config: dict,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
    secret_scope: str | None = None,
    response_format: dict | None = None,
) -> str:
    """向 OpenAI 兼容的 Chat 端点发起一次无状态请求（统一分流入口）。

    回环端点（本地 vLLM）走 local_adapter.local_generate（非流式、chat template、
    节流）；远端端点走 cloud_adapter.cloud_generate（SSE、总时限、信号量）。
    response_format 仅在调用方显式传入时才写入 payload（如 {"type": "json_object"}）。
    """
    secrets = _load_secrets()
    raw_scoped = secrets.get(secret_scope, {}) if secret_scope else {}
    scoped = raw_scoped if isinstance(raw_scoped, dict) else {}
    base_url = str(scoped.get("base_url") or secrets.get("base_url") or llm_config.get("base_url") or "")
    if not base_url:
        raise ValueError("LLM base_url is not configured; set BASE_URL (or AI6657_CLOUD_BASE_URL) in .env")
    if _is_loopback_url(base_url):
        return local_adapter.local_generate(
            messages, llm_config, max_tokens=max_tokens, log_ctx=log_ctx,
            secret_scope=secret_scope, response_format=response_format,
        )
    return cloud_adapter.cloud_generate(
        messages, llm_config, max_tokens=max_tokens, log_ctx=log_ctx,
        secret_scope=secret_scope, response_format=response_format,
    )
