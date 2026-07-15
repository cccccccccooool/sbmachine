"""共享的 OpenAI 兼容 Chat Completions 客户端。"""
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
    """从环境变量或根目录 .env 加载通用及分域（llma/llmb/vlm）的端点配置。"""
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

    secrets: dict[str, object] = {
        "api_key": env_value("AI6657_API_KEY"),
        "base_url": env_value("AI6657_BASE_URL"),
        "model": env_value("AI6657_LLM_MODEL"),
    }
    for scope in ("llma", "llmb", "vlm"):
        prefix = f"AI6657_{scope.upper()}_"
        scoped = {
            "api_key": env_value(prefix + "API_KEY"),
            "base_url": env_value(prefix + "BASE_URL"),
            "model": env_value(prefix + "MODEL"),
        }
        if any(scoped.values()):
            secrets[scope] = scoped
    return secrets


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
    raise ValueError("API key is required; set AI6657_API_KEY or a scoped key in .env")


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
) -> None:
    """把一次 API 调用写进当日 debug 日志，只落脱敏指纹与元数据。"""
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
    if error:
        entry["error"] = error
    with _LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
    ) -> "_ApiChatResult":
        result = super().__new__(cls, value)
        result.scope = scope
        result.source_run_id = source_run_id
        result.request_payload = request_payload
        result.log_ctx = dict(log_ctx or {})
        result.accepted = False
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


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    """解析响应头 Retry-After（秒数或 HTTP 日期），返回被封顶的等待秒数。"""
    if not isinstance(exc, requests.HTTPError) or exc.response is None:
        return None
    headers = getattr(exc.response, "headers", {}) or {}
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
def _post_openai_with_retry(url: str, payload: dict, headers: dict, timeout: int):
    _validate_api_base_url(url)
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("OpenAI-compatible endpoint returned a non-object JSON response")
    result = _HttpJson(data)
    result.http_status = response.status_code
    return result


def _execute_openai_chat(
    messages: list[dict],
    llm_config: dict,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
    secret_scope: str | None = None,
) -> str:
    """向 OpenAI 兼容的 Chat 端点发起一次无状态请求。"""
    secrets = _load_secrets()
    raw_scoped = secrets.get(secret_scope, {}) if secret_scope else {}
    scoped = raw_scoped if isinstance(raw_scoped, dict) else {}
    base_url = str(scoped.get("base_url") or secrets.get("base_url") or llm_config.get("base_url", "https://api.openai.com/v1"))
    url = f"{base_url.rstrip('/')}/chat/completions"
    api_key = _resolve_api_key(base_url, str(scoped.get("api_key") or secrets.get("api_key") or ""))
    model = str(scoped.get("model") or secrets.get("model") or llm_config.get("model", "gpt-4o-mini"))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_config.get("temperature", 0.75)),
    }
    # Qwen3 系思考默认开启，只能用 chat template 参数关闭（Qwen2.5 时代的 /no_think 前缀已失效）。
    # chat_template_kwargs 是 vLLM 私有字段，远端第三方 API 可能拒收未知字段，仅对本机回环端点注入；
    # 远端后端仍输出思考块时由下方的 </think> 剥离兜底。
    if not bool(llm_config.get("enable_thinking", False)) and "qwen" in model.lower() and _is_loopback_url(base_url):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if max_tokens and int(max_tokens) > 0:
        payload["max_tokens"] = int(max_tokens)
    frequency_penalty = float(llm_config.get("frequency_penalty", 0.0) or 0.0)
    if frequency_penalty:
        payload["frequency_penalty"] = frequency_penalty
    if llm_config.get("top_p") is not None:
        payload["top_p"] = float(llm_config["top_p"])
    if llm_config.get("repeat_penalty") is not None:
        payload["repetition_penalty"] = float(llm_config["repeat_penalty"])

    timeout = int(llm_config.get("timeout_sec", 120))
    request_ctx = dict(log_ctx or {})
    source_run_id = str(request_ctx.get("run_id") or uuid.uuid4().hex)
    request_ctx["run_id"] = source_run_id
    started = time.perf_counter()
    try:
        data = _post_openai_with_retry(url, payload, headers, timeout)
    except requests.RequestException as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        status = exc.response.status_code if exc.response is not None else None
        _dump_api_log(
            url, payload, {}, log_ctx=request_ctx, scope=secret_scope,
            http_status=status, duration_ms=duration_ms, error=type(exc).__name__,
        )
        raise
    duration_ms = round((time.perf_counter() - started) * 1000)
    status = int(getattr(data, "http_status", 200))
    _dump_api_log(
        url, payload, data, log_ctx=request_ctx, scope=secret_scope,
        http_status=status, duration_ms=duration_ms,
    )
    content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
    # 兜底剥离思考块：上游未关思考时 content 形如 "<think>…</think>\n{json}"，
    # 剥离后才能通过 phase3a/3b 的严格 JSON 解析；服务端已关思考时此处为无操作。
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1]
    content = content.strip()
    return _ApiChatResult(
        content,
        scope=secret_scope,
        source_run_id=source_run_id,
        request_payload=payload,
        log_ctx=request_ctx,
    )
