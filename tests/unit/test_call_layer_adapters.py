"""调用层拆分（计划书 §2/§3.1）的适配器边界测试。

验证 cloud_adapter / local_adapter 的职责隔离：
- cloud_generate 只接受远端端点（loopback 拒绝），local_generate 只接受回环端点（远端拒绝）
- cloud 强制 SSE；local 强制非流式；两端均不注入任何 thinking 调配字段
- 薄入口 llm_shim._execute_openai_chat 按 loopback 分流到正确 adapter
"""
import json

import pytest
import requests

from sbmachine import llm_shim
from sbmachine import llm_protocol
from sbmachine import cloud_adapter
from sbmachine import local_adapter


def _patch_transport(monkeypatch, fake_post, *, dump_log=True):
    monkeypatch.setattr(llm_protocol, "_post_openai_with_retry", fake_post)
    monkeypatch.setattr(cloud_adapter, "_post_openai_with_retry", fake_post)
    monkeypatch.setattr(local_adapter, "_post_openai_with_retry", fake_post)
    if dump_log:
        monkeypatch.setattr(llm_protocol, "_dump_api_log", lambda *a, **k: None)
        monkeypatch.setattr(cloud_adapter, "_dump_api_log", lambda *a, **k: None)
        monkeypatch.setattr(local_adapter, "_dump_api_log", lambda *a, **k: None)


def test_cloud_generate_rejects_loopback_url(monkeypatch):
    _patch_transport(monkeypatch, lambda *a, **k: {"choices": []})
    monkeypatch.setenv("AI6657_CLOUD_BASE_URL", "http://127.0.0.1:8000/v1")
    with pytest.raises(ValueError, match="requires a remote endpoint"):
        cloud_adapter.cloud_generate([{"role": "user", "content": "hi"}], {"model": "qwen3"})


def test_local_generate_rejects_remote_url(monkeypatch):
    _patch_transport(monkeypatch, lambda *a, **k: {"choices": []})
    monkeypatch.setenv("AI6657_CLOUD_BASE_URL", "https://api.example.test/v1")
    with pytest.raises(ValueError, match="requires a loopback endpoint"):
        local_adapter.local_generate([{"role": "user", "content": "hi"}], {"model": "qwen3"})


def test_cloud_generate_forces_sse_without_thinking_controls(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout, **kwargs):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    _patch_transport(monkeypatch, fake_post)
    monkeypatch.setenv("AI6657_CLOUD_API_KEY", "test-key")
    monkeypatch.setenv("AI6657_CLOUD_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("AI6657_CLOUD_MODEL", "qwen3-flash")

    cloud_adapter.cloud_generate(
        [{"role": "user", "content": "hi"}],
        {"model": "qwen3-flash", "enable_thinking": False, "cloud_thinking_extra": {"thinking": {"type": "disabled"}}},
    )

    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert "thinking" not in captured
    assert "chat_template_kwargs" not in captured
    assert "enable_thinking" not in captured


def test_local_generate_forces_non_stream_without_thinking_controls(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout, **kwargs):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    _patch_transport(monkeypatch, fake_post)
    monkeypatch.setenv("AI6657_API_KEY", "test-key")
    monkeypatch.setenv("AI6657_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("AI6657_LLM_MODEL", "qwen3")

    local_adapter.local_generate(
        [{"role": "user", "content": "hi"}],
        {"model": "qwen3", "enable_thinking": False},
    )

    assert "stream" not in captured
    assert "chat_template_kwargs" not in captured
    assert "enable_thinking" not in captured


def test_execute_openai_chat_dispatches_to_cloud_for_remote(monkeypatch):
    cloud_calls = []
    local_calls = []

    def fake_cloud(messages, llm_config, **kwargs):
        cloud_calls.append(messages)
        return "cloud-ok"

    def fake_local(messages, llm_config, **kwargs):
        local_calls.append(messages)
        return "local-ok"

    monkeypatch.setattr(cloud_adapter, "cloud_generate", fake_cloud)
    monkeypatch.setattr(local_adapter, "local_generate", fake_local)
    monkeypatch.setenv("AI6657_CLOUD_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("AI6657_CLOUD_MODEL", "qwen3")

    result = llm_shim._execute_openai_chat([{"role": "user", "content": "hi"}], {})
    assert result == "cloud-ok"
    assert len(cloud_calls) == 1 and len(local_calls) == 0


def test_execute_openai_chat_dispatches_to_local_for_loopback(monkeypatch):
    cloud_calls = []
    local_calls = []

    def fake_cloud(messages, llm_config, **kwargs):
        cloud_calls.append(messages)
        return "cloud-ok"

    def fake_local(messages, llm_config, **kwargs):
        local_calls.append(messages)
        return "local-ok"

    monkeypatch.setattr(cloud_adapter, "cloud_generate", fake_cloud)
    monkeypatch.setattr(local_adapter, "local_generate", fake_local)
    monkeypatch.setenv("AI6657_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("AI6657_LLM_MODEL", "qwen3")

    result = llm_shim._execute_openai_chat([{"role": "user", "content": "hi"}], {})
    assert result == "local-ok"
    assert len(local_calls) == 1 and len(cloud_calls) == 0
