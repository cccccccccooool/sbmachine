"""阶段 4：云端成功响应缓存（cloud_cache）单元测试。"""
import json
import threading
import time

import pytest

from sbmachine import cloud_cache, cloud_memory
from sbmachine.llm_shim import _ApiChatResult

_DEFAULT_SEMANTIC = {"cloud_cache_enabled": True}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """每个用例隔离：secrets 指向假端点，缓存根指向 tmp_path，清空注册表与会话。"""
    monkeypatch.setattr(
        cloud_cache, "_load_secrets",
        lambda: {"api_key": "k", "base_url": "https://api.example.test/v1", "model": "qwen3"},
    )
    monkeypatch.setattr(cloud_cache, "PROJECT_ROOT", tmp_path)
    cloud_cache.clear_registry()
    cloud_memory.clear_sessions()
    yield
    cloud_cache.clear_registry()
    cloud_memory.clear_sessions()


def _make_result(
    content: str = '{"neutral":"ok"}',
    *,
    scope: str = "llma",
    run_id: str = "run-1",
    http_status: int = 200,
    finish_reason: str = "stop",
    budget_silence: bool = False,
) -> _ApiChatResult:
    return _ApiChatResult(
        content,
        scope=scope,
        source_run_id=run_id,
        request_payload={
            "model": "qwen3",
            "messages": [{"role": "user", "content": "窗口投影"}],
        },
        log_ctx={"run_id": run_id, "round": "round1", "scene": "win1"},
        raw_response={"choices": [{"message": {"content": content, "role": "assistant"}, "finish_reason": finish_reason}]},
        finish_reason=finish_reason,
        http_status=http_status,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        request_id="req-1",
        endpoint_url="https://api.example.test/v1/chat/completions",
        budget_silence=budget_silence,
    )


def _make_inner(results=None, errors=()):
    state = {"calls": 0, "results": list(results or ()), "errors": list(errors)}

    def inner(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, response_format=None):
        state["calls"] += 1
        if state["errors"]:
            raise state["errors"].pop(0)
        if state["results"]:
            return state["results"].pop(0)
        return _make_result('{"neutral":"cached-ok"}', run_id=str((log_ctx or {}).get("run_id") or "run-1"))

    return inner, state


def _default_kwargs():
    return {"prompt": "窗口投影", "llm_cfg": {"model": "qwen3", "temperature": 0.5},
            "system_prompt": "云端系统提示", "log_ctx": {"run_id": "run-1"}}


def _cache_files(run_id: str = "run-1"):
    return sorted(str(p) for p in cloud_cache._resolve_cache_dir(_DEFAULT_SEMANTIC, run_id).glob("*.json"))


# ── 命中 / 未命中 / pending→confirm 生命周期 ────────────────────────────────


def test_miss_pending_confirm_then_hit():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)

    first = gen(**_default_kwargs())
    assert state["calls"] == 1
    assert str(first) == '{"neutral":"cached-ok"}'
    assert getattr(first, "cache_hit", False) is False
    assert first._cache_status == "pending"
    assert len(_cache_files()) == 1

    assert cloud_cache.confirm_cached(first) is True
    assert first._cache_status == "confirmed"

    hit = gen(**_default_kwargs())
    assert state["calls"] == 1  # 命中：不发起网络请求
    assert str(hit) == '{"neutral":"cached-ok"}'
    assert hit.cache_hit is True
    assert hit._cache_status == "hit"


def test_pending_is_not_served_until_confirmed():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)

    gen(**_default_kwargs())
    again = gen(**_default_kwargs())  # pending 未转正：仍视为 miss，走网络
    assert state["calls"] == 2
    assert getattr(again, "cache_hit", False) is False


def test_confirm_uses_accepted_response_content_not_pending_file():
    inner, state = _make_inner([_make_result('{"neutral":"rejected-first"}'), _make_result('{"neutral":"accepted-retry"}')])
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)

    first = gen(**_default_kwargs())
    second = gen(**_default_kwargs())  # 同键重试：pending 覆盖，仍 miss
    assert state["calls"] == 2
    assert first._cache_status == "pending" and second._cache_status == "pending"

    cloud_cache.confirm_cached(second)  # 只有被验收的响应内容转正

    hit = gen(**_default_kwargs())
    assert state["calls"] == 2
    assert str(hit) == '{"neutral":"accepted-retry"}'


def test_confirm_is_noop_for_hit_or_plain_results():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    plain = _make_result()
    assert cloud_cache.confirm_cached(plain) is False  # 未挂 pending 元数据

    first = gen(**_default_kwargs())
    assert cloud_cache.confirm_cached(first) is True
    hit = gen(**_default_kwargs())
    assert cloud_cache.confirm_cached(hit) is False  # 命中结果不再重写


# ── 未确认清理 / 失败不缓存 ────────────────────────────────────────────────


def test_cleanup_pending_removes_unconfirmed_entries():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)

    first = gen(**_default_kwargs())
    gen(prompt="另一个窗口", llm_cfg=_default_kwargs()["llm_cfg"], system_prompt=_default_kwargs()["system_prompt"], log_ctx=_default_kwargs()["log_ctx"])
    assert len(_cache_files()) == 2
    cloud_cache.confirm_cached(first)  # 只转正第一个

    removed = cloud_cache.cleanup_pending("run-1")
    assert removed == 1
    assert len(_cache_files()) == 1  # 只剩已确认条目
    gen(prompt="另一个窗口", llm_cfg=_default_kwargs()["llm_cfg"], system_prompt=_default_kwargs()["system_prompt"], log_ctx=_default_kwargs()["log_ctx"])
    assert state["calls"] == 3  # 被清理的 pending 键重新走网络


def test_cleanup_pending_all_runs():
    inner, _ = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    gen(**_default_kwargs())
    gen(prompt="窗口B", llm_cfg={"model": "qwen3"}, system_prompt="s", log_ctx={"run_id": "run-2"})
    assert cloud_cache.cleanup_pending() == 2


def test_failed_requests_never_cached():
    inner, state = _make_inner(errors=[RuntimeError("boom")])
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    with pytest.raises(RuntimeError, match="boom"):
        gen(**_default_kwargs())
    assert _cache_files() == []  # 异常不写任何缓存（含 pending）

    bad_inner, _ = _make_inner([_make_result(http_status=500)])
    gen_bad = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, bad_inner)
    gen_bad(**_default_kwargs())
    assert _cache_files() == []  # HTTP 非 200 不写缓存

    str_inner = lambda prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, response_format=None: "plain str"
    gen_str = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, str_inner)
    gen_str(**_default_kwargs())
    assert _cache_files() == []  # 非 _ApiChatResult 不写缓存


def test_budget_silence_never_cached():
    inner, _ = _make_inner([_make_result(budget_silence=True)])
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    gen(**_default_kwargs())
    assert _cache_files() == []


# ── 版本不匹配 / 损坏条目 / 键敏感性 ───────────────────────────────────────


def test_cache_version_mismatch_is_miss():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    first = gen(**_default_kwargs())
    cloud_cache.confirm_cached(first)
    path = cloud_cache._resolve_cache_dir(_DEFAULT_SEMANTIC, "run-1") / cloud_cache._key_filename(first._cache_key)
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["cache_version"] = 999
    path.write_text(json.dumps(entry), encoding="utf-8")

    gen(**_default_kwargs())
    assert state["calls"] == 2  # 版本不匹配 → miss → 回退网络


def test_corrupted_or_wrong_key_entry_is_miss():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    first = gen(**_default_kwargs())
    cloud_cache.confirm_cached(first)
    cache_dir = cloud_cache._resolve_cache_dir(_DEFAULT_SEMANTIC, "run-1")
    path = cache_dir / cloud_cache._key_filename(first._cache_key)
    path.write_text("{not valid json", encoding="utf-8")
    gen(**_default_kwargs())
    assert state["calls"] == 2

    path.unlink()
    path.write_text(json.dumps({"cache_version": 1, "status": "confirmed", "key": {"wrong": True}, "response": {"content": "x", "http_status": 200}}), encoding="utf-8")
    gen(**_default_kwargs())
    assert state["calls"] == 3  # 键字段不符 → miss


def test_key_changes_with_prompt_config_and_source_rounds():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    base_llm_cfg = {"model": "qwen3", "temperature": 0.5}
    base_system = "云端系统提示"
    gen(prompt="窗口投影", llm_cfg=base_llm_cfg, system_prompt=base_system, log_ctx={"run_id": "run-1"})
    gen(prompt="完全不同的提示", llm_cfg=base_llm_cfg, system_prompt=base_system, log_ctx={"run_id": "run-1"})
    gen(prompt="窗口投影", llm_cfg={"model": "qwen3", "temperature": 0.9}, system_prompt=base_system, log_ctx={"run_id": "run-1"})
    gen(prompt="窗口投影", llm_cfg=base_llm_cfg, system_prompt="不同的系统提示", log_ctx={"run_id": "run-1"})
    cloud_cache.set_source_rounds("llma", "run-1", "abc123")
    gen(prompt="窗口投影", llm_cfg=base_llm_cfg, system_prompt=base_system, log_ctx={"run_id": "run-1"})
    assert state["calls"] == 5  # prompt/system/temperature/source_rounds 变化全部导致 miss


def test_pending_entry_is_sanitized_no_prompt_text_on_disk():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    gen(prompt="机密窗口投影内容-勿落盘", llm_cfg={"model": "qwen3"}, system_prompt="机密系统提示-勿落盘", log_ctx={"run_id": "run-1"})
    files = _cache_files()
    assert len(files) == 1
    raw = open(files[0], encoding="utf-8").read()
    assert "机密窗口投影内容" not in raw
    assert "机密系统提示" not in raw


# ── 命中结果与 _ApiChatResult 同构 ─────────────────────────────────────────


def test_hit_result_is_isomorphic_to_apichatresult():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    first = gen(**_default_kwargs())
    cloud_cache.confirm_cached(first)
    hit = gen(**_default_kwargs())

    assert isinstance(hit, _ApiChatResult)
    assert isinstance(hit, str)
    assert str(hit) == '{"neutral":"cached-ok"}'
    assert hit.scope == "llma"
    assert hit.source_run_id == "run-1"
    assert hit.finish_reason == "stop"
    assert hit.http_status == 200
    assert hit.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert hit.budget_silence is False
    assert hit.cache_hit is True
    assert hit.request_id.startswith("cache:")
    assert hit.endpoint_url == "https://api.example.test/v1/chat/completions"
    assert hit.log_ctx == {"run_id": "run-1"}  # 命中结果 log_ctx = 当前调用上下文
    assert isinstance(hit.request_payload, dict) and hit.request_payload["model"] == "qwen3"
    assert hit.request_payload["messages"][-1]["content"] == "窗口投影"
    envelope = hit.raw_response
    assert isinstance(envelope, dict)
    assert envelope["choices"][0]["message"]["content"] == '{"neutral":"cached-ok"}'
    assert envelope["choices"][0]["finish_reason"] == "stop"
    assert envelope["usage"] == hit.usage
    assert hit.accepted is False  # 与实时结果一样是 str 子类实例


# ── 并发写入安全 ───────────────────────────────────────────────────────────


def test_concurrent_writes_are_safe():
    inner, state = _make_inner()
    gen = cloud_cache.make_cached_generate("llma", _DEFAULT_SEMANTIC, inner)
    results = []
    lock = threading.Lock()
    errors = []

    def worker():
        try:
            result = gen(**_default_kwargs())
            with lock:
                results.append(result)
        except Exception as exc:  # pragma: no cover - 断言层兜底
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert state["calls"] == 8  # 并发首跑全部 miss
    assert len(results) == 8
    for path in _cache_files():
        entry = json.loads(open(path, encoding="utf-8").read())  # 每条都是合法 JSON
        assert entry["status"] == "pending"

    for result in results:
        assert cloud_cache.confirm_cached(result) is True

    hit = gen(**_default_kwargs())
    assert state["calls"] == 8  # 转正后可命中
    assert hit.cache_hit is True


# ── cloud_memory 集成 ──────────────────────────────────────────────────────


def _install_fake_execute(monkeypatch, state):
    def fake_execute(messages, llm_config, max_tokens=None, log_ctx=None, secret_scope=None, response_format=None):
        state["calls"] += 1
        last_user = ""
        for m in reversed(list(messages or [])):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                last_user = m["content"]
                break
        return _ApiChatResult(
            '{"neutral":"ok-%d"}' % state["calls"],
            scope=secret_scope or "llmb",
            source_run_id=str((log_ctx or {}).get("run_id") or "run-1"),
            request_payload={"model": "qwen3", "messages": list(messages or [])},
            log_ctx=dict(log_ctx or {}),
            raw_response={"choices": [{"message": {"content": "x", "role": "assistant"}}]},
            finish_reason="stop",
            http_status=200,
            usage={"total_tokens": 5},
            request_id="req-%d" % state["calls"],
            endpoint_url="https://api.example.test/v1/chat/completions",
        )

    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", fake_execute)
    return fake_execute


def test_make_generate_llma_stateless_commit_round_confirms(monkeypatch):
    state = {"calls": 0}
    _install_fake_execute(monkeypatch, state)
    gen = cloud_memory.make_generate("llma", _DEFAULT_SEMANTIC)
    kwargs = {
        "prompt": "窗口投影", "llm_cfg": {"model": "qwen3"},
        "system_prompt": "sys", "log_ctx": {"run_id": "run-1"},
    }

    first = gen(**kwargs)
    assert state["calls"] == 1
    cloud_memory.commit_round(first)  # 业务验收成功路径：自动转正
    hit = gen(**kwargs)
    assert state["calls"] == 1  # 重跑命中，零网络
    assert hit.cache_hit is True
    assert str(hit) == str(first)


def test_make_generate_llmb_source_rounds_chain_hits_on_rerun(monkeypatch):
    state = {"calls": 0}
    _install_fake_execute(monkeypatch, state)
    kwargs = {
        "prompt": "窗口A", "llm_cfg": {"model": "qwen3"},
        "system_prompt": "sys", "log_ctx": {"run_id": "run-1", "round": "round1", "scene": "win1"},
    }

    # ── 首跑：窗口A 验收后会话历史增长，下一窗同键也应 miss（历史不同） ──
    gen = cloud_memory.make_generate("llmb", _DEFAULT_SEMANTIC)
    first = gen(**kwargs)
    cloud_memory.commit_round(first)
    again = gen(**kwargs)
    assert state["calls"] == 2  # source_rounds 变化 → 键变化 → miss
    cloud_memory.commit_round(again)
    assert cloud_cache.get_source_rounds("llmb", "run-1")  # 注册表已记录

    # ── 重跑同一场比赛（清空会话后重新建 generate）：历史按同样顺序重建 → 全命中 ──
    cloud_memory.clear_sessions()
    gen_rerun = cloud_memory.make_generate("llmb", _DEFAULT_SEMANTIC)
    rr1 = gen_rerun(**kwargs)
    assert state["calls"] == 2  # 第一窗命中
    assert rr1.cache_hit is True
    cloud_memory.commit_round(rr1)
    rr2 = gen_rerun(**kwargs)
    assert state["calls"] == 2  # 第二窗（历史已重建）同样命中
    assert rr2.cache_hit is True


def test_make_generate_cache_disabled_has_zero_overhead(monkeypatch, tmp_path):
    state = {"calls": 0}
    _install_fake_execute(monkeypatch, state)
    gen = cloud_memory.make_generate("llmb", {"cloud_model": "qwen3"})  # 未启用缓存
    result = gen(prompt="p", llm_cfg={"model": "qwen3"}, system_prompt="s", log_ctx={"run_id": "run-1"})
    assert state["calls"] == 1
    assert not hasattr(result, "cache_hit")
    assert not hasattr(result, "_cache_key")
    assert not (tmp_path / "output" / "cloud_cache").exists()  # 零落盘


def test_confirm_cache_and_cleanup_pending_cache_public_entry(monkeypatch):
    state = {"calls": 0}
    _install_fake_execute(monkeypatch, state)
    gen = cloud_memory.make_generate("llma", _DEFAULT_SEMANTIC)
    first = gen(prompt="p", llm_cfg={"model": "qwen3"}, system_prompt="s", log_ctx={"run_id": "run-1"})

    cloud_memory.confirm_cache(first)  # 透传入口
    hit = gen(prompt="p", llm_cfg={"model": "qwen3"}, system_prompt="s", log_ctx={"run_id": "run-1"})
    assert hit.cache_hit is True

    pending_gen = cloud_memory.make_generate("llma", _DEFAULT_SEMANTIC)
    pending_gen(prompt="q", llm_cfg={"model": "qwen3"}, system_prompt="s", log_ctx={"run_id": "run-2"})
    assert cloud_memory.cleanup_pending_cache() == 1  # 运行结束清理 pending
