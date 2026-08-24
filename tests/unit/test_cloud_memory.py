"""cloud_memory：会话缓存（3b 会话：裁剪/失败不入历史/同窗替换）+ 3a 无状态路径 + 成本护栏。

3a（llma）默认无会话（无上下文需求，可自由并发）；3b（llmb）保留会话历史。
"""
from __future__ import annotations

import pytest

from sbmachine import cloud_memory
from sbmachine.llm_shim import _ApiChatResult


@pytest.fixture(autouse=True)
def _clean_sessions():
    cloud_memory.clear_sessions()
    yield
    cloud_memory.clear_sessions()


def _fake_execute(calls: list, usage_tokens: int = 100):
    def fake(messages, llm_cfg, max_tokens=None, log_ctx=None, secret_scope=None, response_format=None):
        calls.append({
            "messages": messages,
            "llm_cfg": dict(llm_cfg),
            "max_tokens": max_tokens,
            "secret_scope": secret_scope,
            "response_format": response_format,
            "log_ctx": dict(log_ctx or {}),
        })
        return _ApiChatResult(
            '{"ok": true}',
            scope=secret_scope,
            source_run_id="x",
            request_payload={"messages": messages},
            log_ctx=log_ctx,
            raw_response={"choices": [{"message": {"content": '{"ok": true}'}}]},
            finish_reason="stop",
            http_status=200,
            usage={"total_tokens": usage_tokens},
        )
    return fake


def _log_ctx(run_id: str, round_no: int, scene: str) -> dict:
    return {"run_id": run_id, "round": f"round{round_no}", "scene": scene}


def _call(generate, run_id, round_no, scene, max_tokens=128, system=None):
    return generate("user-prompt", {"model": "m"}, system_prompt=system, max_tokens=max_tokens, log_ctx=_log_ctx(run_id, round_no, scene))


def _commit(result):
    cloud_memory.commit_round(result)


# ── 会话历史（业务验收后显式提交）──

def test_llma_defaults_to_stateless_no_session(monkeypatch):
    """3a（llma）默认无会话：不建会话、消息恒 [system,user]、commit 为 no-op。"""
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llma")
    result = _call(generate, "r1", 1, "w1")
    _commit(result)
    _call(generate, "r1", 1, "w2")
    assert cloud_memory._sessions == {}
    for call in calls:
        assert [m["role"] for m in call["messages"]] == ["system", "user"]


def test_llma_session_enabled_explicitly(monkeypatch):
    """显式 cloud_analyst_conversation_max_rounds>0 时 3a 才走会话路径（A/B 用）。"""
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llma", {"cloud_analyst_conversation_max_rounds": 4})
    _commit(_call(generate, "r1", 1, "w1"))
    _commit(_call(generate, "r1", 1, "w2"))
    _call(generate, "r1", 1, "w3")
    roles = [m["role"] for m in calls[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert len(cloud_memory._sessions) == 1


def test_messages_include_history_before_current(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llmb")
    _commit(_call(generate, "r1", 1, "w1"))
    _commit(_call(generate, "r1", 1, "w2"))
    _commit(_call(generate, "r1", 2, "w1", system="SYS"))
    messages = calls[-1]["messages"]
    assert messages[0] == {"role": "system", "content": "SYS"}
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[-1]["content"] == "user-prompt"


def test_max_rounds_trims_oldest(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llmb", {"cloud_conversation_max_rounds": 2})
    for i in range(4):
        _commit(_call(generate, "r1", 1, f"w{i}"))
    roles = [m["role"] for m in calls[-1]["messages"]]
    # 历史只保留最近 2 轮：system + 2×(user,assistant) + 当前 user
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]


def test_uncommitted_call_not_in_history(monkeypatch):
    """generate 只计 usage，不写历史：业务校验通过前任何调用都不入历史。"""
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llmb")
    _call(generate, "r1", 1, "w1")  # 不 commit（模拟 HTTP 200 但业务失败）
    _call(generate, "r1", 1, "w2")
    roles = [m["role"] for m in calls[-1]["messages"]]
    assert roles == ["system", "user"]  # 历史恒空
    conversation = next(iter(cloud_memory._sessions.values()))
    assert conversation.rounds == []


def test_failed_round_not_in_history(monkeypatch):
    calls = []

    def flaky(messages, llm_cfg, max_tokens=None, log_ctx=None, secret_scope=None, response_format=None):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return _ApiChatResult("ok", scope=secret_scope, source_run_id="x", request_payload={"messages": messages}, log_ctx=log_ctx, raw_response={"choices": [{"message": {"content": "ok"}}]}, finish_reason="stop", http_status=200, usage={"total_tokens": 100})

    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", flaky)
    generate = cloud_memory.make_generate("llmb")
    with pytest.raises(RuntimeError):
        _call(generate, "r1", 1, "w1")
    # w1 失败轮：即使 commit 也不入历史（rounds 为空，commit 无对应会话条目）
    conversation = next(iter(cloud_memory._sessions.values()))
    assert conversation.rounds == []
    _commit(_call(generate, "r1", 1, "w2"))
    # w2 commit 后入历史；再调用 w3 可见历史只含 w2
    _call(generate, "r1", 1, "w3")
    roles = [m["role"] for m in calls[-1]]
    assert roles == ["system", "user", "assistant", "user"]


def test_same_scene_retry_replaces_entry(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llmb")
    _commit(_call(generate, "r1", 1, "w1"))
    _commit(_call(generate, "r1", 1, "w1"))
    roles = [m["role"] for m in calls[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_disabled_conversation_no_history(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llmb", {"cloud_conversation_max_rounds": 0})
    for i in range(3):
        _commit(_call(generate, "r1", 1, f"w{i}"))
    roles = [m["role"] for m in calls[-1]["messages"]]
    assert roles == ["system", "user"]


def test_conversation_max_tokens_trims(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    # 每轮估算 token：user(11)//2 + assistant(12)//2 ≈ 11 > 4 → 入历史即被裁剪，历史恒空
    generate = cloud_memory.make_generate("llmb", {"cloud_conversation_max_tokens": 4})
    _commit(_call(generate, "r1", 1, "w1"))
    _commit(_call(generate, "r1", 1, "w2"))
    roles = [m["role"] for m in calls[-1]["messages"]]
    assert roles == ["system", "user"]
    conversation = next(iter(cloud_memory._sessions.values()))
    assert conversation.rounds == []
    assert conversation.max_tokens == 4


# ── 成本护栏 ──

def test_fixed_budget_silences_after_exceed(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls, usage_tokens=100))
    generate = cloud_memory.make_generate("llma", {"cloud_token_budget_enabled": True, "cloud_token_budget_per_match": 150})
    _call(generate, "r1", 1, "w1")
    _call(generate, "r1", 1, "w2")
    assert len(calls) == 2  # 前两窗正常调用
    silence = _call(generate, "r1", 1, "w3")
    assert len(calls) == 2  # 第三窗不再发请求
    assert '{"neutral": ""}' == str(silence)


def test_budget_default_disabled_never_silences(monkeypatch):
    """预算护栏默认关闭：不限额、永不 silence（预算已放开）。"""
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls, usage_tokens=100))
    generate = cloud_memory.make_generate("llma")
    for i in range(6):
        _call(generate, "r1", 1, f"w{i}")
    assert len(calls) == 6  # 全部真实调用，无 silence


def test_dynamic_budget_formula_and_silence(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls, usage_tokens=100))
    # max_rounds=2（回合数估计）× 首回合窗口数 2 × 均值 100 × factor 2 = 800
    generate = cloud_memory.make_generate("llmb", {"cloud_token_budget_enabled": True, "cloud_conversation_max_rounds": 2, "cloud_token_budget_factor": 2.0})
    for i in range(2):
        _call(generate, "r1", 1, f"w{i}")  # 首回合 2 窗：usage 100/200
    for i in range(7):
        _call(generate, "r1", 2, f"w{i}")  # 第二回合 7 窗：累计 300..900（>800 已超）
    assert len(calls) == 9
    silence = _call(generate, "r1", 3, "w0")  # 900 > 800 → silence
    assert len(calls) == 9
    assert '{"commentary": "", "felt_intensity": 0}' == str(silence)  # llmb 契约


def test_dynamic_budget_locks_after_windows_observed(monkeypatch):
    """并发下锁定必须延迟：首回合窗口未完成时过早锁定会低估预算 → 提前 silence。"""
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls, usage_tokens=50))
    generate = cloud_memory.make_generate("llmb", {"cloud_token_budget_enabled": True, "cloud_conversation_max_rounds": 4, "cloud_token_budget_factor": 2.0})
    for i in range(4):
        _call(generate, "r1", 1, f"w{i}")  # 首回合 4 窗：usage 50/100/150/200
    # 第二回合首窗：seen=5 < 2×2 → 未锁定，正常调用
    _call(generate, "r1", 2, "w0")
    assert len(calls) == 5
    # seen=6（≥2×3）且窗口数 4 → 预算 = 4 × 4 × 50 × 2 = 1600
    _call(generate, "r1", 2, "w1")
    conversation = next(iter(cloud_memory._sessions.values()))
    assert conversation._budget_locked
    assert conversation._budget_total == 1600


def test_dynamic_budget_concurrent_threads_safe(monkeypatch):
    """回合级并发（ThreadPool）下：历史/预算状态无竞态，全部窗口真实调用。"""
    import threading
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls, usage_tokens=50))
    generate = cloud_memory.make_generate("llmb", {"cloud_token_budget_enabled": True, "cloud_conversation_max_rounds": 4, "cloud_token_budget_factor": 2.0})

    def worker(i: int) -> None:
        _call(generate, "match1", i % 3, f"w{i}", max_tokens=64)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    conversation = cloud_memory._sessions["match1"]
    assert len(calls) == 12  # 全部真实调用，无竞态丢窗/误 silence
    assert conversation._seen_windows == 12
    assert conversation.usage_total == 600
    assert conversation._budget_locked


def test_session_creation_is_thread_safe(monkeypatch):
    """并发首调同一 run_id：_sessions_lock 保证只建一个会话，不丢调用。"""
    import threading
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llmb")

    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        barrier.wait()
        _call(generate, "match-c", 1, f"w{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sessions = [k for k in cloud_memory._sessions if k == "match-c"]
    assert len(sessions) == 1
    assert len(calls) == 8  # 无一被创建竞态吞掉


def test_fixed_budget_silence_after_third(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls, usage_tokens=100))
    generate = cloud_memory.make_generate("llma", {"cloud_token_budget_enabled": True, "cloud_token_budget_per_match": 250})
    _call(generate, "r1", 1, "w1")
    _call(generate, "r1", 1, "w2")
    _call(generate, "r1", 1, "w3")  # 累计 300 > 250
    result = _call(generate, "r1", 1, "w4")  # 超限 → silence
    assert '{"neutral": ""}' == str(result)
    assert len(calls) == 3


# ── 契约兼容 ──

def test_cloud_model_override(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llma", {"cloud_model": "deepseek-v4-flash-free"})
    _call(generate, "r1", 1, "w1")
    assert calls[-1]["llm_cfg"]["model"] == "deepseek-v4-flash-free"


def test_positional_system_prompt_like_phase3b(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llmb")
    generate("prompt", {"model": "m"}, "SYSTEM", max_tokens=64, log_ctx=_log_ctx("r1", 1, "w1"), response_format={"type": "json_object"})
    assert calls[-1]["messages"][0]["content"] == "SYSTEM"
    assert calls[-1]["response_format"] == {"type": "json_object"}
    assert calls[-1]["secret_scope"] == "llmb"


def test_llmb_silence_uses_commentary_contract(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_execute(calls))
    generate = cloud_memory.make_generate("llmb", {"cloud_token_budget_per_match": 0})
    _call(generate, "r1", 1, "w1")
    _call(generate, "r1", 1, "w2")
    # 预算 0=动态且单回合永不锁定 → 不超限；直接测 silence_result 的 llmb 契约
    conversation = next(iter(cloud_memory._sessions.values()))
    result = conversation.silence_result(_log_ctx("r1", 1, "w3"))
    assert str(result) == '{"commentary": "", "felt_intensity": 0}'
    assert result.scope == "llmb"
    assert len(calls) == 2
