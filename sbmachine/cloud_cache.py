"""阶段 4：云端成功响应缓存（cloud success-response cache）。

仅缓存“业务验收成功”的 cloud 请求：网络请求返回 HTTP 200 后先落 pending 条目，
业务侧验收成功（cloud_memory.commit_round 的调用方 accept 路径）后调用
`confirm_cached(response)` 转正；运行结束未确认的 pending 由 `cleanup_pending`
清理，且 pending 条目绝不被当作命中读取。

- 缓存键：scope / model / endpoint / prompt_hash / system_prompt_hash /
  generation_config（temperature/max_tokens/top_p/frequency_penalty/
  repetition_penalty/response_format 等稳定序列化）/ source_rounds_sha256
  （从 (scope, run_id) 关联的已验收轮次哈希取，或由 semantic_cfg 传入，缺省 ""）。
- 命中结果与实时 `_ApiChatResult` 同构并标记 `cache_hit=True`；不落盘完整 prompt
  （只存哈希），命中时用调用方当前 payload 重建 request_payload。
- 脱敏：缓存文件只存生成文本与元数据（content/finish_reason/usage/http_status），
  不存请求正文全文。
- 并发安全：写缓存先写临时文件再 os.replace（原子替换，Windows 短暂占用时重试）。
- 向后兼容：目录缺失、cache_version 不匹配、条目损坏一律视为 miss，
  自动回退到实时网络请求。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from sbmachine.common import PROJECT_ROOT, _output_cap
from sbmachine.llm_shim import _ApiChatResult, _load_secrets

_CACHE_VERSION = 1
_DEFAULT_PENDING_TTL_SEC = 86400.0  # 惰性清理兜底：超 1 天的 pending 自动删除
_SWEEP_INTERVAL_SEC = 60.0  # 同一目录惰性清理的节流间隔
_WRITE_RETRIES = 5  # Windows 下目标文件短暂占用时重试 os.replace 的次数

# ── (scope, run_id) → 已验收轮次哈希注册表（由 cloud_memory 在 commit_round 维护） ──
_source_rounds: dict[tuple[str, str], str] = {}
_source_rounds_lock = threading.Lock()
_last_sweep: dict[str, float] = {}
_last_sweep_lock = threading.Lock()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_source_rounds(scope: str, run_id: str, rounds_sha: str) -> None:
    """登记某 (scope, run_id) 当前的 source_rounds 哈希；空值清除。"""
    with _source_rounds_lock:
        if rounds_sha:
            _source_rounds[(scope, run_id)] = rounds_sha
        else:
            _source_rounds.pop((scope, run_id), None)


def get_source_rounds(scope: str, run_id: str) -> str:
    with _source_rounds_lock:
        return _source_rounds.get((scope, run_id), "")


def clear_registry() -> None:
    """清空注册表与清扫节流状态（测试隔离用）。"""
    with _source_rounds_lock:
        _source_rounds.clear()
    with _last_sweep_lock:
        _last_sweep.clear()


def _canonical_key(key: dict) -> str:
    """缓存键的稳定序列化：排序键 + 紧凑分隔符，跨进程/平台稳定。"""
    return json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _key_filename(key: dict) -> str:
    return f"{_sha256_text(_canonical_key(key))}.json"


def build_cache_key(
    *,
    scope: str,
    model: str,
    endpoint: str,
    prompt: str,
    system_prompt: str | None,
    llm_cfg: dict | None,
    max_tokens: int | None,
    response_format: dict | None,
    source_rounds_sha: str,
) -> dict | None:
    """构造缓存键（dict）。内容无法稳定序列化时返回 None（调用方跳过缓存）。

    prompt_hash 计算后不落盘原文；generation_config 与 llm_shim 实际发送的
    采样参数保持一致（temperature 恒有，其余仅在实际发送时才进键）。"""
    llm = dict(llm_cfg or {})
    cap = _output_cap(llm, max_tokens)
    gen_cfg: dict[str, object] = {
        "temperature": float(llm.get("temperature", 0.75)),
        "max_tokens": cap,
        "frequency_penalty": float(llm.get("frequency_penalty", 0.0) or 0.0),
    }
    if llm.get("top_p") is not None:
        gen_cfg["top_p"] = float(llm["top_p"])
    if llm.get("repeat_penalty") is not None:
        gen_cfg["repetition_penalty"] = float(llm["repeat_penalty"])
    if response_format is not None:
        gen_cfg["response_format"] = response_format
    key = {
        "cache_version": _CACHE_VERSION,
        "scope": scope,
        "model": model,
        "endpoint": endpoint,
        "prompt_hash": _sha256_text(f"{prompt or ''}{system_prompt or ''}"),
        "system_prompt_hash": _sha256_text(system_prompt or ""),
        "generation_config": gen_cfg,
        "source_rounds_sha256": source_rounds_sha,
    }
    try:
        _canonical_key(key)
    except (TypeError, ValueError):
        return None
    return key


def _resolve_cache_dir(semantic_cfg: dict | None, run_id: str) -> Path:
    """缓存目录：semantic_cfg.cloud_cache_dir（支持 {run_id} 占位）> 默认 output/cloud_cache/<run_id>/。"""
    semantic = semantic_cfg if isinstance(semantic_cfg, dict) else {}
    override = semantic.get("cloud_cache_dir")
    if override:
        path = Path(str(override).replace("{run_id}", run_id))
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "output" / "cloud_cache" / run_id


def _resolve_endpoint(secret_scope: str, llm_cfg: dict | None) -> str:
    """与 llm_shim._execute_openai_chat 相同的 base_url 解析顺序，拼出 /chat/completions URL。"""
    secrets = _load_secrets()
    scoped = secrets.get(secret_scope) if isinstance(secrets.get(secret_scope), dict) else {}
    base_url = str(scoped.get("base_url") or secrets.get("base_url") or (llm_cfg or {}).get("base_url") or "")
    return f"{base_url.rstrip('/')}/chat/completions" if base_url else "unknown"


def _atomic_write_json(path: Path, entry: dict) -> bool:
    """先写临时文件再 os.replace（原子替换）；Windows 短暂占用时小退避重试。"""
    payload = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    for _ in range(_WRITE_RETRIES):
        try:
            os.replace(str(tmp), str(path))
            return True
        except OSError:
            time.sleep(0.01)
    try:
        tmp.unlink()
    except OSError:
        pass
    return False


def _build_entry(key: dict, response: object, *, status: str, model: str, scope: str) -> dict:
    """构造缓存条目；usage 仅保留 dict 形态（脱敏：不存 request_payload 全文）。"""
    usage = getattr(response, "usage", None)
    return {
        "cache_version": _CACHE_VERSION,
        "status": status,
        "key": key,
        "ts": time.time(),
        "response": {
            "content": str(response),
            "finish_reason": getattr(response, "finish_reason", None),
            "usage": usage if isinstance(usage, dict) else None,
            "http_status": int(getattr(response, "http_status", 200) or 200),
        },
        "model": model,
        "scope": scope,
    }


def load_cache(key: dict, cache_dir: Path) -> dict | None:
    """读取已确认的缓存条目；缺失/损坏/版本不匹配/未转正一律返回 None（视为 miss）。"""
    if key is None:
        return None
    path = cache_dir / _key_filename(key)
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entry, dict):
        return None
    if entry.get("cache_version") != _CACHE_VERSION or entry.get("status") != "confirmed":
        return None
    if entry.get("key") != key:
        return None
    response = entry.get("response")
    if not isinstance(response, dict) or not isinstance(response.get("content"), str):
        return None
    if int(response.get("http_status") or 0) != 200:
        return None
    return entry


def _sweep_stale_pending(cache_dir: Path, ttl_sec: float) -> None:
    """惰性清理过期 pending（TTL 兜底）；正式清理走 cleanup_pending。"""
    key = str(cache_dir)
    now = time.monotonic()
    with _last_sweep_lock:
        if now - _last_sweep.get(key, 0.0) < _SWEEP_INTERVAL_SEC:
            return
        _last_sweep[key] = now
    threshold = time.time() - float(ttl_sec)
    for path in cache_dir.glob("*.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(entry, dict) and entry.get("status") == "pending" and float(entry.get("ts") or 0) < threshold:
            try:
                path.unlink()
            except OSError:
                pass


def store_pending(
    key: dict,
    cache_dir: Path,
    response: object,
    *,
    model: str,
    scope: str,
    pending_ttl_sec: float = _DEFAULT_PENDING_TTL_SEC,
) -> bool:
    """网络请求成功后写 pending 缓存条目，并把转正所需元数据挂到 response 上。

    失败/超时/异常（非 _ApiChatResult、非 HTTP 200、预算静默、重复条目）绝不写缓存。"""
    if not isinstance(response, _ApiChatResult):
        return False
    if getattr(response, "cache_hit", False) or getattr(response, "_cache_status", None) in ("pending", "confirmed"):
        return False
    if int(getattr(response, "http_status", 0) or 0) != 200 or getattr(response, "budget_silence", False):
        return False
    _sweep_stale_pending(cache_dir, pending_ttl_sec)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not _atomic_write_json(cache_dir / _key_filename(key), _build_entry(key, response, status="pending", model=model, scope=scope)):
        return False
    response._cache_key = key
    response._cache_dir = str(cache_dir)
    response._cache_status = "pending"
    return True


def confirm_cached(response: object) -> bool:
    """业务验收通过后调用：把该 response 的 pending 条目转正为 confirmed。

    pending 与 confirmed 共用同一哈希文件：转正用 response 自身内容重写，
    保证只有被验收的响应进入成功缓存（被拒尝试留下的 pending 被覆盖）。
    缓存命中结果（cache_hit=True）与未挂 pending 元数据的结果均为无操作。"""
    if not isinstance(response, _ApiChatResult) or getattr(response, "cache_hit", False):
        return False
    key = getattr(response, "_cache_key", None)
    cache_dir = getattr(response, "_cache_dir", None)
    if not isinstance(key, dict) or not cache_dir:
        return False
    path = Path(cache_dir) / _key_filename(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = _build_entry(
        key, response, status="confirmed",
        model=str(key.get("model", "")), scope=str(key.get("scope", "")),
    )
    if not _atomic_write_json(path, entry):
        return False
    response._cache_status = "confirmed"
    return True


def cleanup_pending(run_id: str | None = None, semantic_cfg: dict | None = None) -> int:
    """删除未确认的 pending 缓存条目（运行结束清理用），返回删除数。

    run_id 为空时清理默认缓存根下所有运行目录的 pending 条目。"""
    semantic = semantic_cfg if isinstance(semantic_cfg, dict) else {}
    if run_id:
        directories = [_resolve_cache_dir(semantic, run_id)]
    elif semantic.get("cloud_cache_dir"):
        directories = [_resolve_cache_dir(semantic, "default")]
    else:
        root = PROJECT_ROOT / "output" / "cloud_cache"
        directories = [p for p in root.glob("*") if p.is_dir()] if root.is_dir() else []
    removed = 0
    for directory in directories:
        for path in directory.glob("*.json"):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(entry, dict) and entry.get("status") == "pending":
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


def _build_hit_result(
    entry: dict,
    *,
    prompt: str,
    system_prompt: str | None,
    log_ctx: dict | None,
    run_id: str,
) -> _ApiChatResult:
    """把确认条目还原为与实时 _ApiChatResult 同构的结果（cache_hit=True）。

    request_payload 不落盘：用调用方当前 payload 重建最小结构（model + messages）。"""
    key = entry["key"]
    response = entry["response"]
    content = str(response.get("content") or "")
    finish_reason = response.get("finish_reason")
    usage = response.get("usage")
    model = str(entry.get("model") or "")
    scope = str(entry.get("scope") or "")
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": prompt},
        ],
    }
    result = _ApiChatResult(
        content,
        scope=scope or None,
        source_run_id=run_id,
        request_payload=request_payload,
        log_ctx=dict(log_ctx or {}),
        raw_response={
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        },
        finish_reason=finish_reason,
        http_status=int(response.get("http_status") or 200),
        usage=usage,
        request_id=f"cache:{_sha256_text(_canonical_key(key))[:12]}",
        endpoint_url=str(key.get("endpoint") or ""),
    )
    result.cache_hit = True
    result._cache_status = "hit"
    return result


def make_cached_generate(secret_scope: str, semantic_cfg: dict | None, inner_generate: Callable) -> Callable:
    """装饰一个 generate 闭包：命中直接返回缓存结果，未命中调用 inner 并写 pending。

    inner 签名与 cloud_memory.generate 兼容（prompt, llm_cfg, system_prompt,
    max_tokens, log_ctx, response_format）。缓存键按调用参数独立计算，
    source_rounds_sha256 由调用方经 set_source_rounds 按 (scope, run_id) 维护，
    或由 semantic_cfg.cloud_source_rounds_sha256 显式传入（缺省 ""）。"""
    semantic = semantic_cfg if isinstance(semantic_cfg, dict) else {}
    scope = secret_scope if secret_scope in {"llma", "llmb"} else "llma"
    cloud_model = str(semantic.get("cloud_model") or "").strip()
    source_rounds_override = str(semantic.get("cloud_source_rounds_sha256") or "")
    pending_ttl = float(semantic.get("cloud_cache_pending_ttl_sec") or _DEFAULT_PENDING_TTL_SEC)

    def cached_generate(
        prompt: str,
        llm_cfg: dict,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        log_ctx: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        run_id = str((log_ctx or {}).get("run_id") or "default")
        cache_dir = _resolve_cache_dir(semantic, run_id)
        # 与 make_generate 内闭包一致：cloud_model 覆盖 llm_cfg.model
        model = cloud_model or str((llm_cfg or {}).get("model") or "")
        endpoint = _resolve_endpoint(scope, llm_cfg)
        source_rounds = source_rounds_override or get_source_rounds(scope, run_id)
        key = build_cache_key(
            scope=scope, model=model, endpoint=endpoint,
            prompt=prompt, system_prompt=system_prompt,
            llm_cfg=llm_cfg, max_tokens=max_tokens,
            response_format=response_format, source_rounds_sha=source_rounds,
        )
        if key is not None:
            entry = load_cache(key, cache_dir)
            if entry is not None:
                return _build_hit_result(
                    entry, prompt=prompt, system_prompt=system_prompt,
                    log_ctx=log_ctx, run_id=run_id,
                )
        result = inner_generate(
            prompt, llm_cfg, system_prompt=system_prompt,
            max_tokens=max_tokens, log_ctx=log_ctx, response_format=response_format,
        )
        if key is not None:
            store_pending(key, cache_dir, result, model=model, scope=scope, pending_ttl_sec=pending_ttl)
        return result

    return cached_generate
