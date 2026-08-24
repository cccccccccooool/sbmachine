"""llm_shim 全局请求节流单测。"""

from __future__ import annotations

import time

from sbmachine import llm_shim


def _reset_throttle() -> None:
    with llm_shim._throttle_lock:
        llm_shim._last_request_ts = 0.0


def test_request_throttle_disabled_when_interval_zero(monkeypatch):
    _reset_throttle()
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(sec))
    llm_shim._request_throttle(0.0)
    llm_shim._request_throttle(None)
    llm_shim._request_throttle(-1.0)
    assert sleeps == []


def test_request_throttle_enforces_min_interval(monkeypatch):
    _reset_throttle()
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(sec))
    # 首次调用：单调时钟远大于 0，无需等待
    llm_shim._request_throttle(3.0)
    assert sleeps == []
    # 立即再调：距上次 < 3s，需补齐剩余间隔
    llm_shim._request_throttle(3.0)
    assert len(sleeps) == 1
    assert 0.0 <= sleeps[0] <= 3.0


def test_request_throttle_only_sleeps_remaining_gap(monkeypatch):
    _reset_throttle()
    sleeps: list[float] = []
    real_monotonic = time.monotonic
    current = [1_000_000.0]

    def fake_monotonic() -> float:
        return current[0]

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(sec))

    llm_shim._request_throttle(5.0)  # t=1000000，last=0 -> 不 sleep
    current[0] += 2.0  # 过 2s
    llm_shim._request_throttle(5.0)  # 距上次 2s < 5s -> sleep 3s
    assert sleeps == [3.0]
    current[0] += 5.0  # 过 5s
    llm_shim._request_throttle(5.0)  # 距上次 5s >= 5s -> 不 sleep
    assert sleeps == [3.0]
    assert real_monotonic is not None


def _capture_execute(llm_shim_module, monkeypatch):
    """返回 (captured_payloads, throttle_calls)；monkeypatch 后调用 _execute_openai_chat。"""
    captured = {"throttle_calls": 0}

    def fake_post(url, payload, headers, timeout, stream=None):
        return {"choices": [{"message": {"content": "ok"}}]}

    def fake_throttle(min_interval_sec):
        captured["throttle_calls"] += 1

    monkeypatch.setattr(llm_shim_module, "_post_openai_with_retry", fake_post)
    monkeypatch.setattr(llm_shim_module, "_dump_api_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_shim_module, "_request_throttle", fake_throttle)
    monkeypatch.setenv("AI6657_API_KEY", "test-key")
    return captured


def test_throttle_applies_only_to_loopback_endpoints(monkeypatch):
    from sbmachine import llm_shim

    # 云端官方 API（非回环）：节流不生效
    captured = _capture_execute(llm_shim, monkeypatch)
    monkeypatch.setenv("AI6657_BASE_URL", "https://api.deepseek.com/v1")
    llm_shim._execute_openai_chat([{"role": "user", "content": "hi"}], {"model": "m", "request_interval_sec": 1.0})
    assert captured["throttle_calls"] == 0

    # 本地 vLLM（回环）：节流生效
    monkeypatch.setenv("AI6657_BASE_URL", "http://127.0.0.1:8000/v1")
    llm_shim._execute_openai_chat([{"role": "user", "content": "hi"}], {"model": "m", "request_interval_sec": 1.0})
    assert captured["throttle_calls"] == 1


def test_throttle_skipped_for_loopback_when_interval_zero(monkeypatch):
    from sbmachine import llm_shim

    captured = _capture_execute(llm_shim, monkeypatch)
    monkeypatch.setenv("AI6657_BASE_URL", "http://127.0.0.1:8000/v1")
    llm_shim._execute_openai_chat([{"role": "user", "content": "hi"}], {"model": "m", "request_interval_sec": 0.0})
    assert captured["throttle_calls"] == 1  # 仍调用，但内部 <=0 直接返回
