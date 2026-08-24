"""LLM Protocol Core —— 云端/本地共用的最小 OpenAI 兼容协议层。

只保留无论 backend 为何都必须存在的能力（计划书 §2.1 共用层清单）：
- 请求/响应 DTO（_ApiChatResult、_HttpJson）与业务验收样本落盘
- OpenAI 响应归一（SSE 聚合、think 剥离、usage/finish_reason 提取）
- 脱敏日志接口（_dump_api_log、record_validation_reason）
- 端点/密钥校验（_load_secrets、_validate_api_base_url、_resolve_api_key、_is_loopback_url）
- 错误分类与退避工具（retry_after_seconds、infra_backoff_delay、_request_error_category）
- 传输层重试（_post_openai_with_retry：流式/非流式统一入口）

端点策略、Prompt、并发、会话、显存和密钥归属由 cloud/local adapter 决定。
"""
from __future__ import annotations

import datetime
import hashlib
import ipaddress
import json
import os
import threading
import time
import uuid
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_LOCK = threading.Lock()
_RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=10)
_MAX_RETRY_AFTER_SEC = 120.0


def _load_secrets() -> dict:
    """按 profile→scope→fallback 优先级加载端点配置。

    返回结构（迁移窗口合同，§4.3）：
    {
      "api_key"/"base_url"/"model": cloud 通用值（裸键 BASE_URL/API_KEY/MODEL 或旧 AI6657_* 回退）,
      "llma"/"llmb"/"vlm": {"api_key","base_url","model"}（scope 覆盖或回退通用）,
      "local": {"base_url","model","api_key"}（本地 vLLM 回环，允许占位 key）,
      "warnings": [str]  // 旧 scoped 键使用提示（不阻断运行）
    }
    优先级：进程环境变量 > .env > 旧 AI6657_* 通用键 > 裸键 BASE_URL/API_KEY/MODEL。
    """
    dotenv: dict[str, str] = {}
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or not key.replace("_", "a").isalnum():
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            dotenv[key] = value

    def env_value(name: str) -> str:
        return os.environ[name] if name in os.environ else dotenv.get(name, "")

    def value(*names: str) -> str:
        for name in names:
            val = env_value(name)
            if val:
                return val
        return ""

    def value_or(name: str, default: str) -> str:
        return env_value(name) or default

    warnings: list[str] = []
    for legacy in ("AI6657_LLMA_", "AI6657_LLMB_", "AI6657_VLM_"):
        if any(key.startswith(legacy) and env_value(key) for key in (*dotenv, *os.environ)):
            warnings.append(
                f"{legacy}* scoped keys are legacy; use AI6657_CLOUD_* (cloud) or "
                f"AI6657_CLOUD_LLM*/VLM_* instead"
            )
            break

    cloud_key = value("AI6657_CLOUD_API_KEY", "AI6657_API_KEY", "API_KEY")
    cloud_base = value("AI6657_CLOUD_BASE_URL", "AI6657_BASE_URL", "BASE_URL")
    cloud_model = value("AI6657_CLOUD_MODEL", "AI6657_LLM_MODEL", "MODEL")

    def scope_entry(scope_prefix: str) -> dict:
        """scope 专用键优先，其次 cloud 通用，再其次旧 scope 键与裸键。"""
        key = value(f"AI6657_CLOUD_{scope_prefix}_API_KEY", f"AI6657_CLOUD_API_KEY", f"AI6657_{scope_prefix}_API_KEY", "AI6657_API_KEY", "API_KEY")
        base = value(f"AI6657_CLOUD_{scope_prefix}_BASE_URL", f"AI6657_CLOUD_BASE_URL", f"AI6657_{scope_prefix}_BASE_URL", "AI6657_BASE_URL", "BASE_URL")
        model = value(f"AI6657_CLOUD_{scope_prefix}_MODEL", f"AI6657_CLOUD_MODEL", f"AI6657_{scope_prefix}_MODEL", "AI6657_LLM_MODEL", "MODEL")
        return {"api_key": key, "base_url": base, "model": model}

    return {
        "api_key": cloud_key,
        "base_url": cloud_base,
        "model": cloud_model,
        "llma": scope_entry("LLMA"),
        "llmb": scope_entry("LLMB"),
        "llmc": scope_entry("LLMC"),  # Phase3c/LLM-C 逻辑域：凭证复用同一 .env 云端配置（无独立密钥时回退 cloud 通用键）
        "vlm": scope_entry("VLM"),
        "local": {
            "api_key": value_or("AI6657_LOCAL_API_KEY", "EMPTY"),
            "base_url": value_or("AI6657_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
            "model": value_or("AI6657_LOCAL_MODEL", env_value("AI6657_LLM_MODEL") or env_value("MODEL")),
        },
        "warnings": warnings,
    }


def _is_loopback_url(base_url: str) -> bool:
    """判断 base_url 是否指向本机回环地址（localhost 或 loopback IP）。"""
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_api_base_url(base_url: str) -> None:
    """校验 base_url：必须是绝对 HTTP(S) URL，远端强制 HTTPS，仅回环端点允许 HTTP。"""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API base_url must be an absolute HTTP(S) URL")
    if parsed.scheme != "https" and not _is_loopback_url(base_url):
        raise ValueError("remote API base_url must use HTTPS; HTTP is allowed only for loopback endpoints")


def _resolve_api_key(base_url: str, api_key: str) -> str:
    """解析 API key：显式 key 优先，回环端点允许占位 key，其余必须提供 key。"""
    _validate_api_base_url(base_url)
    if api_key:
        return api_key
    if _is_loopback_url(base_url):
        return "EMPTY"
    raise ValueError("API key is required; set API_KEY (or AI6657_CLOUD_API_KEY) in .env")


def _payload_sha256(payload: object) -> str:
    """对 payload 做稳定序列化后取 sha256，用于日志脱敏指纹。"""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _endpoint_metadata(url: str) -> dict[str, str]:
    """只保留端点路由信息（origin/path），剥离凭据与查询参数。"""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    origin = f"{parsed.scheme}://{hostname}" if parsed.scheme and hostname else ""
    if port is not None:
        origin += f":{port}"
    if not origin:
        return {"endpoint_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest()}
    return {"endpoint_origin": origin, "endpoint_path": parsed.path or "/"}


def _usage_tokens(usage: object) -> dict[str, int]:
    """从 usage 提取纯数字 token 字段；缺失字段不输出（旧日志向后兼容）。"""
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning = details.get("reasoning_tokens")
        if isinstance(reasoning, int) and not isinstance(reasoning, bool):
            out["reasoning_tokens"] = reasoning
    return out


def _dump_api_log(
    url: str,
    request_payload: dict,
    response_raw: dict,
    log_ctx: dict | None = None,
    scope: str | None = None,
    *,
    http_status: int | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
    request_id: str | None = None,
    model: str | None = None,
    streaming: bool | None = None,
    connect_ms: int | None = None,
    ttfb_ms: int | None = None,
    in_flight: int | None = None,
    queue_ms: int | None = None,
    finish_reason: str | None = None,
    retry_category: str | None = None,
    usage: object = None,
    validation_reason: str | None = None,
) -> None:
    """把一次 API 调用写进当日 debug 日志，只落脱敏指纹与元数据。

    阶段 0 诊断字段均为可选：旧日志/旧调用缺字段时按缺失读取，不改变业务产物。
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y%m%d")
    log_path = _LOG_DIR / f"api_debug_{today}.jsonl"
    entry: dict[str, object] = {
        key: log_ctx[key]
        for key in ("run_id", "round", "scene")
        if log_ctx and log_ctx.get(key) is not None
    }
    entry["run_id"] = str(entry.get("run_id") or uuid.uuid4().hex)
    entry.update({
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "scope": scope,
        "http_status": http_status,
        "request_sha256": _payload_sha256(request_payload),
        "response_sha256": _payload_sha256(response_raw),
        "duration_ms": duration_ms,
    })
    entry.update(_endpoint_metadata(url))
    if request_id:
        entry["request_id"] = request_id
    if model:
        entry["model"] = model
    if streaming is not None:
        entry["streaming"] = bool(streaming)
    if connect_ms is not None:
        entry["connect_ms"] = connect_ms
    if ttfb_ms is not None:
        entry["ttfb_ms"] = ttfb_ms
    if in_flight is not None:
        entry["in_flight"] = in_flight
    if queue_ms is not None:
        entry["queue_ms"] = queue_ms
    if finish_reason:
        entry["finish_reason"] = finish_reason
    if retry_category:
        entry["retry_category"] = retry_category
    if validation_reason:
        entry["validation_reason"] = validation_reason
    token_fields = _usage_tokens(usage)
    if token_fields:
        entry["usage_tokens"] = token_fields
    if error:
        entry["error"] = error
    with _LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def record_validation_reason(response: object, reason: str) -> None:
    """业务验收失败后，把 validation_reason 关联到原请求追加一条诊断。

    仅写当日 api_debug JSONL，不修改原请求记录；
    旧消费者按未知字段忽略，缺字段时按 unknown 解析。
    """
    if not isinstance(response, _ApiChatResult) or not reason:
        return
    payload = getattr(response, "request_payload", None)
    if not isinstance(payload, dict):
        payload = {}
    _dump_api_log(
        str(getattr(response, "endpoint_url", "") or "unknown"),
        payload,
        {},
        log_ctx=response.log_ctx,
        scope=response.scope,
        http_status=getattr(response, "http_status", None),
        request_id=getattr(response, "request_id", None) or None,
        model=str(payload.get("model") or ""),
        finish_reason=getattr(response, "finish_reason", None),
        usage=getattr(response, "usage", None),
        validation_reason=reason,
    )


def _last_user_prompt(messages: object) -> str:
    """取消息列表里最后一条 user 文本内容。"""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"].strip()
    return ""


def _dump_accepted_api_sample(
    *,
    scope: str,
    source_run_id: str,
    request_payload: dict,
    output: str,
    log_ctx: dict | None,
) -> None:
    """把一条被业务采纳的输入输出对写入 Phase 3 训练样本日志。"""
    input_text = _last_user_prompt(request_payload.get("messages"))
    output_text = output.strip()
    if not input_text or not output_text:
        return
    identity = "\0".join((scope, source_run_id, input_text, output_text))
    entry: dict[str, object] = {
        "sample_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "scope": scope,
        "accepted": True,
        "source_run_id": source_run_id,
        "input": input_text,
        "output": output_text,
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    for key in ("round", "scene"):
        if log_ctx and log_ctx.get(key) is not None:
            entry[key] = log_ctx[key]
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y%m%d")
    log_path = _LOG_DIR / f"api_training_{today}.jsonl"
    with _LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class _ApiChatResult(str):
    def __new__(
        cls,
        value: str,
        *,
        scope: str | None,
        source_run_id: str,
        request_payload: dict,
        log_ctx: dict | None,
        raw_response: dict | None = None,
        reasoning_content: str | None = None,
        finish_reason: str | None = None,
        http_status: int | None = None,
        usage: dict | None = None,
        budget_silence: bool = False,
        request_id: str | None = None,
        endpoint_url: str | None = None,
    ) -> "_ApiChatResult":
        result = super().__new__(cls, value)
        result.scope = scope
        result.source_run_id = source_run_id
        result.request_payload = request_payload
        result.log_ctx = dict(log_ctx or {})
        result.accepted = False
        result.raw_response = raw_response
        result.reasoning_content = reasoning_content
        result.finish_reason = finish_reason
        result.http_status = http_status
        result.usage = usage
        result.budget_silence = bool(budget_silence)
        result.request_id = request_id
        result.endpoint_url = endpoint_url
        return result

    def accept(self, output: str | None = None) -> None:
        if self.accepted or self.scope not in {"llma", "llmb"}:
            return
        _dump_accepted_api_sample(
            scope=self.scope,
            source_run_id=self.source_run_id,
            request_payload=self.request_payload,
            output=str(self) if output is None else output,
            log_ctx=self.log_ctx,
        )
        self.accepted = True


def accept_api_response(response: object, *, output: str | None = None) -> None:
    """仅在调用方业务校验通过后，才落盘一条 Phase 3 训练样本。"""
    if isinstance(response, _ApiChatResult):
        response.accept(output)


class _HttpJson(dict):
    http_status: int


def _is_retryable_request_error(exc: BaseException) -> bool:
    """判断请求异常是否可重试：连接/超时、408、429、5xx 才重试；其余 4xx 不重试。"""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in {408, 429} or exc.response.status_code >= 500
    return False


def _request_error_category(exc: BaseException) -> str:
    """把请求异常归类为诊断用的 retry_category（未知按 unknown，不改变异常类型）。"""
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.ConnectionError):
        return "connection_error"
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        if status in {408, 429}:
            return "rate_limit"
        if status >= 500:
            return "server_error"
        return "client_error"
    return "unknown"


def retry_after_seconds(response: object) -> float | None:
    """从 HTTP 响应对象解析 Retry-After（秒数或 HTTP 日期），返回被封顶的等待秒数。"""
    headers = getattr(response, "headers", {}) or {}
    value = str(headers.get("Retry-After", "")).strip()
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=datetime.timezone.utc)
            delay = (retry_at - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not delay >= 0:
        return None
    return min(delay, _MAX_RETRY_AFTER_SEC)


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    """解析响应头 Retry-After（秒数或 HTTP 日期），返回被封顶的等待秒数。"""
    if not isinstance(exc, requests.HTTPError) or exc.response is None:
        return None
    return retry_after_seconds(exc.response)


def infra_backoff_delay(attempt: int, retry_after_sec: float | None = None) -> float:
    """方案 R 基建错误退避：默认 2**attempt 秒（1/2/4）；http_error 尊重 Retry-After。"""
    if retry_after_sec is not None:
        return min(retry_after_sec, _MAX_RETRY_AFTER_SEC)
    return float(2 ** max(0, attempt))


def _wait_retry_after_or_exponential(retry_state) -> float:
    """优先遵循服务端 Retry-After，缺失时退回指数退避。"""
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    retry_after = _retry_after_seconds(exc)
    return retry_after if retry_after is not None else float(_RETRY_WAIT(retry_state))


@retry(
    stop=stop_after_attempt(3),
    wait=_wait_retry_after_or_exponential,
    retry=retry_if_exception(_is_retryable_request_error),
    reraise=True,
)
def _post_openai_with_retry(url: str, payload: dict, headers: dict, timeout, total_timeout_sec: float | None = None):
    _validate_api_base_url(url)
    merged_headers = dict(headers or {})
    # opencode 网关（https://opencode.ai/zen/go/v1）经 Cloudflare 按 UA 签名校验，
    # 默认 requests UA 会被 403(1010) 拦截；统一补上 opencode 客户端 UA。
    merged_headers.setdefault("User-Agent", "opencode/1.0")
    connect_started = time.perf_counter()
    if payload.get("stream"):
        response = requests.post(url, json=payload, headers=merged_headers, timeout=timeout, stream=True)
        response.encoding = "utf-8"
        connect_ms = round((time.perf_counter() - connect_started) * 1000)
        response.raise_for_status()
        _assert_json_content_type(response, url)
        timing: list = []
        data = _consume_sse_response(response, timing, total_timeout_sec=total_timeout_sec)
        ttfb_ms = round((timing[0] - connect_started) * 1000) if timing else connect_ms
    else:
        response = requests.post(url, json=payload, headers=merged_headers, timeout=timeout)
        response.encoding = "utf-8"
        connect_ms = round((time.perf_counter() - connect_started) * 1000)
        response.raise_for_status()
        _assert_json_content_type(response, url)
        data = response.json()
        ttfb_ms = connect_ms
        if not isinstance(data, dict):
            raise ValueError("OpenAI-compatible endpoint returned a non-object JSON response")
    result = _HttpJson(data)
    result.http_status = response.status_code
    result.transport = {
        "streaming": bool(payload.get("stream")),
        "connect_ms": connect_ms,
        "ttfb_ms": ttfb_ms,
    }
    return result


def _assert_json_content_type(response, url: str) -> None:
    """防御：base_url 缺 /v1 等路径错误时，Web 前端常对 POST 返回 200 HTML，
    会被静默解析成空响应（流式）或 JSON 解析失败（非流式），且不计费。"""
    headers = getattr(response, "headers", None)
    content_type = str(headers.get("Content-Type", "")).lower() if isinstance(headers, dict) else ""
    if content_type and not any(kind in content_type for kind in ("application/json", "text/event-stream", "text/json", "+json")):
        raise ValueError(
            f"endpoint returned non-JSON content-type {content_type!r} "
            f"(check base_url path, e.g. missing /v1): {url}"
        )


def _consume_sse_response(response, timing: list | None = None, total_timeout_sec: float | None = None) -> dict:
    """把流式 Chat Completions 响应（SSE）聚合为与普通响应同构的 dict。

    total_timeout_sec 是 SSE 的硬总时限：超过后中断并抛 requests.Timeout
    （可重试基础设施错误），防止思考模型持续吐字让单请求无限挂起。
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict | None = None
    first_chunk_at: float | None = None
    deadline = time.monotonic() + float(total_timeout_sec) if total_timeout_sec else None
    # SSE 全链路统一按 UTF-8 解码：iter_lines(decode_unicode=True) 依赖 response.encoding，
    # 服务端未带 charset 时 requests 退回 ISO-8859-1，会把 UTF-8 中文错解成 mojibake。
    for raw_line in response.iter_lines(decode_unicode=False):
        if deadline is not None and time.monotonic() > deadline:
            raise requests.Timeout(
                f"SSE stream exceeded total_timeout_sec={float(total_timeout_sec):g}"
            )
        if not raw_line:
            continue
        raw_text = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = raw_text.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter()
            if timing is not None:
                timing.append(first_chunk_at)
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        if isinstance(choice.get("finish_reason"), str) and choice["finish_reason"]:
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta")
        if isinstance(delta, dict):
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning_parts.append(delta["reasoning_content"])
    message: dict[str, object] = {"role": "assistant", "content": "".join(content_parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    return {
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _build_chat_payload(
    messages: list[dict],
    llm_config: dict,
    model: str,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> dict:
    """构造 OpenAI Chat Completions 请求体（共用：两端一致的采样/格式字段）。"""
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_config.get("temperature", 0.75)),
    }
    if max_tokens and int(max_tokens) > 0:
        payload["max_tokens"] = int(max_tokens)
    frequency_penalty = float(llm_config.get("frequency_penalty", 0.0) or 0.0)
    if frequency_penalty:
        payload["frequency_penalty"] = frequency_penalty
    if llm_config.get("top_p") is not None:
        payload["top_p"] = float(llm_config["top_p"])
    if llm_config.get("repeat_penalty") is not None:
        payload["repetition_penalty"] = float(llm_config["repeat_penalty"])
    if response_format is not None:
        # 上游协议要求：json_object 模式时 prompt 必须包含 "json" 字样，
        # 否则 opencode 网关/上游 provider 直接 400（本地 vLLM 宽容不校验）。
        if response_format.get("type") == "json_object":
            prompt_text = "\n".join(
                str(m.get("content") or "") for m in messages if isinstance(m, dict)
            )
            if "json" not in prompt_text.lower():
                added = False
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "system" and isinstance(m.get("content"), str):
                        m["content"] = m["content"].rstrip() + "\nOutput JSON."
                        added = True
                        break
                if not added:
                    messages = [{"role": "system", "content": "Output JSON."}] + list(messages)
        payload["response_format"] = response_format
    return payload


def _finalize_chat_result(
    data: dict,
    *,
    url: str,
    payload: dict,
    log_ctx: dict,
    request_id: str,
    secret_scope: str | None,
) -> _ApiChatResult:
    """把归一后的响应转成 _ApiChatResult（共用：提取 content/reasoning/finish_reason/usage）。"""
    status = int(getattr(data, "http_status", 200))
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    reasoning = message.get("reasoning_content")
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    # 兜底剥离思考块：上游未关思考时 content 形如 "<think>…</think>\n{json}"，
    # 剥离后才能通过 phase3a/3b 的严格 JSON 解析；服务端已关思考时此处为无操作。
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1]
    content = content.strip()
    return _ApiChatResult(
        content,
        scope=secret_scope,
        source_run_id=str(log_ctx.get("run_id") or uuid.uuid4().hex),
        request_payload=payload,
        log_ctx=log_ctx,
        raw_response=dict(data),
        reasoning_content=reasoning if isinstance(reasoning, str) and reasoning else None,
        finish_reason=finish_reason if isinstance(finish_reason, str) and finish_reason else None,
        http_status=status,
        usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
        request_id=request_id,
        endpoint_url=url,
    )
