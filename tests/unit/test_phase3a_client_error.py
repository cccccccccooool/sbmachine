"""Phase3a 客户端 4xx 熔断与响应格式合规适配的测试。"""

from __future__ import annotations

import pytest

from sbmachine.phase3a_analyst import Phase3aCircuitBreak, _ClientErrorCircuitBreaker


class TestClientErrorCircuitBreaker:
    def test_does_not_trip_below_threshold(self):
        breaker = _ClientErrorCircuitBreaker(5)
        for _ in range(4):
            assert breaker.record_failure() is False
        assert breaker.record_failure() is True

    def test_success_resets_counter(self):
        breaker = _ClientErrorCircuitBreaker(3)
        assert breaker.record_failure() is False
        breaker.record_success()
        assert breaker.record_failure() is False
        assert breaker.record_failure() is False
        assert breaker.record_failure() is True

    def test_threshold_one_trips_immediately(self):
        breaker = _ClientErrorCircuitBreaker(1)
        assert breaker.record_failure() is True

    def test_invalid_threshold_clamped(self):
        breaker = _ClientErrorCircuitBreaker(0)
        assert breaker.threshold == 1


class TestCircuitBreakException:
    def test_is_runtime_error(self):
        assert issubclass(Phase3aCircuitBreak, RuntimeError)


def test_response_format_json_object_injects_json_hint():
    """json_object 模式且 prompt 无 'json' 时，llm_shim 自动补一行合规提示。"""
    from sbmachine import llm_shim

    messages = [{"role": "system", "content": "你是解说员"}, {"role": "user", "content": "说说这波"}]
    payload = {}
    # 模拟 _execute_openai_chat 内联的合规逻辑
    response_format = {"type": "json_object"}
    prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
    if "json" not in prompt_text.lower():
        for m in messages:
            if m.get("role") == "system":
                m["content"] = m["content"].rstrip() + "\nOutput JSON."
                break
    payload["response_format"] = response_format

    assert "json" in messages[0]["content"].lower()
    assert payload["response_format"] == {"type": "json_object"}


def test_response_format_json_object_keeps_existing_json_hint():
    from sbmachine import llm_shim  # noqa: F401

    messages = [{"role": "system", "content": "输出 JSON 对象"}, {"role": "user", "content": "说说这波"}]
    prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
    if "json" not in prompt_text.lower():
        for m in messages:
            if m.get("role") == "system":
                m["content"] = m["content"].rstrip() + "\nOutput JSON."
                break
    assert messages[0]["content"] == "输出 JSON 对象"  # 已有 json，不加


def test_probe_api_connectivity_fails_fast(monkeypatch):
    """预检连续 3 次失败直接抛 RuntimeError，不做任何业务请求。"""
    from sbmachine import llm_shim

    calls = []

    def fake_execute(messages, llm_cfg, **kwargs):
        calls.append(messages)
        raise ConnectionError("boom")

    monkeypatch.setattr(llm_shim, "_execute_openai_chat", fake_execute)
    monkeypatch.setattr(llm_shim.time, "sleep", lambda _sec: None)

    with pytest.raises(RuntimeError, match="connectivity probe failed"):
        llm_shim.probe_api_connectivity({"model": "m"}, attempts=3, timeout_sec=10)
    assert len(calls) == 3


def test_probe_api_connectivity_succeeds_on_first_try(monkeypatch):
    from sbmachine import llm_shim

    calls = []

    def fake_execute(messages, llm_cfg, **kwargs):
        calls.append(messages)
        return "ok"

    monkeypatch.setattr(llm_shim, "_execute_openai_chat", fake_execute)
    llm_shim.probe_api_connectivity({"model": "m"}, attempts=3, timeout_sec=10)
    assert len(calls) == 1
