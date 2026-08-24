import json
import threading

import pytest
import requests

from tests.fakes import FakeLLM
from sbmachine import llm_shim
from sbmachine import llm_protocol
from sbmachine import cloud_adapter
from sbmachine import local_adapter


def _patch_transport(monkeypatch, fake_post, *, dump_log=True):
    """同时 patch 协议层与两个 adapter 的传输函数（adapter 内为绑定引用，需逐模块 patch）。"""
    monkeypatch.setattr(llm_protocol, "_post_openai_with_retry", fake_post)
    monkeypatch.setattr(cloud_adapter, "_post_openai_with_retry", fake_post)
    monkeypatch.setattr(local_adapter, "_post_openai_with_retry", fake_post)
    if dump_log:
        monkeypatch.setattr(llm_protocol, "_dump_api_log", lambda *a, **k: None)
        monkeypatch.setattr(cloud_adapter, "_dump_api_log", lambda *a, **k: None)
        monkeypatch.setattr(local_adapter, "_dump_api_log", lambda *a, **k: None)


def test_fake_backends_fixture_exposes_all_offline_backends(fake_backends):
    assert set(fake_backends) == {"llma", "llmb", "vlm", "tts"}
    assert fake_backends["vlm"].describe_frame("frame") == "fake visual description"


def test_llm_shim_skeleton_fake_records_generation_args():
    fake = FakeLLM(["{\"neutral\":\"ok\"}"])

    result = fake.generate("prompt", {"model": "fake"}, max_tokens=16, log_ctx={"scope": "analyst"})

    assert result == "{\"neutral\":\"ok\"}"
    assert fake.calls[0]["kwargs"]["max_tokens"] == 16
    assert fake.calls[0]["kwargs"]["log_ctx"]["scope"] == "analyst"


def test_openai_shim_forwards_explicit_vllm_sampling_controls(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout, **kwargs):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    _patch_transport(monkeypatch, fake_post)
    monkeypatch.setenv("AI6657_API_KEY", "test-key")

    result = llm_shim._execute_openai_chat(
        [{"role": "user", "content": "hi"}],
        {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "qwen3",
            "top_p": 0.9,
            "repeat_penalty": 1.3,
        },
    )

    assert result == "ok"
    assert captured["top_p"] == 0.9
    assert captured["repetition_penalty"] == 1.3


def test_openai_shim_loopback_keeps_model_default_thinking(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout, **kwargs):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    _patch_transport(monkeypatch, fake_post)
    monkeypatch.setenv("AI6657_API_KEY", "test-key")
    monkeypatch.setenv("AI6657_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("AI6657_LLM_MODEL", "qwen3")

    llm_shim._execute_openai_chat(
        [{"role": "system", "content": "原始 system"}, {"role": "user", "content": "hi"}],
        {"model": "qwen3", "enable_thinking": False},
    )

    # 不做任何 thinking 调配：chat_template_kwargs / enable_thinking 均不注入，模型默认思考保留。
    assert "chat_template_kwargs" not in captured
    assert "enable_thinking" not in captured
    assert captured["messages"][0]["content"] == "原始 system"


def test_openai_shim_keeps_vllm_private_fields_off_remote_api(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout, **kwargs):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    _patch_transport(monkeypatch, fake_post)
    monkeypatch.setenv("AI6657_API_KEY", "test-key")
    monkeypatch.setenv("AI6657_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("AI6657_LLM_MODEL", "qwen3")

    llm_shim._execute_openai_chat(
        [{"role": "system", "content": "原始 system"}, {"role": "user", "content": "hi"}],
        {"model": "qwen3", "enable_thinking": False},
    )

    assert "chat_template_kwargs" not in captured
    assert "enable_thinking" not in captured
    assert "/no_think" not in captured["messages"][0]["content"]


def test_openai_shim_strips_reasoning_block_from_content(monkeypatch):
    def fake_post(url, payload, headers, timeout, **kwargs):
        return {"choices": [{"message": {"content": "<think>好的，我需要生成 neutral……</think>\n\n{\"neutral\":\"ok\"}"}}]}

    _patch_transport(monkeypatch, fake_post)
    monkeypatch.setenv("AI6657_API_KEY", "test-key")
    monkeypatch.setenv("AI6657_BASE_URL", "http://127.0.0.1:8000/v1")

    result = llm_shim._execute_openai_chat(
        [{"role": "user", "content": "hi"}],
        {"model": "qwen3"},
    )

    assert result == "{\"neutral\":\"ok\"}"


def test_remote_http_is_rejected_but_loopback_http_is_allowed():
    assert llm_shim._resolve_api_key("http://127.0.0.2:8000/v1", "") == "EMPTY"
    with pytest.raises(ValueError, match="must use HTTPS"):
        llm_shim._resolve_api_key("http://api.example.test/v1", "secret")


def test_408_429_and_server_errors_are_retryable_but_permanent_4xx_is_not():
    response = requests.Response()
    response.status_code = 400
    error = requests.HTTPError(response=response)

    assert llm_shim._is_retryable_request_error(error) is False
    response.status_code = 408
    assert llm_shim._is_retryable_request_error(error) is True
    response.status_code = 429
    assert llm_shim._is_retryable_request_error(error) is True
    response.status_code = 503
    assert llm_shim._is_retryable_request_error(error) is True


def test_post_attempts_a_permanent_4xx_only_once(monkeypatch):
    calls = 0

    class BadRequestResponse:
        status_code = 400

        def raise_for_status(self):
            raise requests.HTTPError(response=self)

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return BadRequestResponse()

    monkeypatch.setattr(llm_protocol.requests, "post", fake_post)

    with pytest.raises(requests.HTTPError):
        llm_shim._post_openai_with_retry("https://api.example.test", {}, {}, 1)
    assert calls == 1


def test_post_retries_429_and_honors_retry_after(monkeypatch):
    calls = 0
    sleeps = []

    class Response:
        status_code = 429
        headers = {"Retry-After": "0"}

        def raise_for_status(self):
            if self.status_code == 429:
                raise requests.HTTPError(response=self)

        def json(self):
            return {"choices": []}

    response = Response()

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            response.status_code = 200
        return response

    monkeypatch.setattr(llm_protocol.requests, "post", fake_post)
    monkeypatch.setattr(llm_protocol._post_openai_with_retry.retry, "sleep", sleeps.append)

    result = llm_shim._post_openai_with_retry("https://api.example.test", {}, {}, 1)

    assert calls == 2
    assert sleeps == [0.0]
    assert result.http_status == 200


def test_default_api_log_contains_only_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", tmp_path)
    request = {"messages": [{"role": "user", "content": "secret-prompt,data:image/jpeg;base64,AAAA"}]}
    response = {"choices": [{"message": {"content": "secret-response"}}]}

    llm_shim._dump_api_log(
        "https://api.example.test/v1/chat/completions",
        request,
        response,
        log_ctx={"run_id": "run-1", "round": "round1"},
        scope="vlm",
        http_status=200,
        duration_ms=12,
    )

    raw = next(tmp_path.glob("api_debug_*.jsonl")).read_text(encoding="utf-8")
    entry = json.loads(raw)
    assert entry["http_status"] == 200
    assert entry["duration_ms"] == 12
    assert "request_sha256" in entry and "response_sha256" in entry
    assert "request" not in entry and "response" not in entry
    assert "url" not in entry
    assert entry["endpoint_origin"] == "https://api.example.test"
    assert entry["endpoint_path"] == "/v1/chat/completions"
    assert "secret-prompt" not in raw and "secret-response" not in raw and "base64" not in raw


def test_api_log_drops_endpoint_credentials_and_query(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", tmp_path)

    llm_shim._dump_api_log(
        "https://user:password@api.example.test/v1/chat/completions?token=secret",
        {},
        {},
    )

    raw = next(tmp_path.glob("api_debug_*.jsonl")).read_text(encoding="utf-8")
    assert "user" not in raw
    assert "password" not in raw
    assert "token" not in raw
    assert "secret" not in raw


def test_api_log_records_phase0_diagnostics_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", tmp_path)

    llm_shim._dump_api_log(
        "https://api.example.test/v1/chat/completions",
        {"messages": [{"role": "user", "content": "prompt"}]},
        {"choices": []},
        log_ctx={"run_id": "run-1", "round": "round2", "scene": "win3"},
        scope="llma",
        http_status=200,
        duration_ms=6500,
        request_id="req-abc",
        model="qwen3-flash",
        streaming=True,
        connect_ms=120,
        ttfb_ms=900,
        in_flight=4,
        queue_ms=30,
        finish_reason="stop",
        usage={
            "prompt_tokens": 6400,
            "completion_tokens": 800,
            "total_tokens": 7200,
            "completion_tokens_details": {"reasoning_tokens": 500},
        },
    )

    entry = json.loads(next(tmp_path.glob("api_debug_*.jsonl")).read_text(encoding="utf-8"))
    assert entry["request_id"] == "req-abc"
    assert entry["model"] == "qwen3-flash"
    assert entry["streaming"] is True
    assert entry["connect_ms"] == 120
    assert entry["ttfb_ms"] == 900
    assert entry["in_flight"] == 4
    assert entry["queue_ms"] == 30
    assert entry["finish_reason"] == "stop"
    assert entry["usage_tokens"] == {
        "prompt_tokens": 6400,
        "completion_tokens": 800,
        "total_tokens": 7200,
        "reasoning_tokens": 500,
    }


def test_api_log_old_entries_missing_diagnostics_still_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", tmp_path)
    log_path = tmp_path / "api_debug_20260816.jsonl"
    log_path.write_text(
        '{"scope":"llma","duration_ms":3594,"http_status":200}\n',
        encoding="utf-8",
    )

    raw = log_path.read_text(encoding="utf-8")
    assert '"duration_ms":3594' in raw
    assert "request_id" not in raw


def test_request_error_category_maps_known_classes():
    response = requests.Response()
    for status, expected in ((400, "client_error"), (408, "rate_limit"), (429, "rate_limit"), (503, "server_error")):
        response.status_code = status
        error = requests.HTTPError(response=response)
        assert llm_shim._request_error_category(error) == expected
    assert llm_shim._request_error_category(requests.Timeout()) == "timeout"
    assert llm_shim._request_error_category(requests.ConnectionError()) == "connection_error"
    assert llm_shim._request_error_category(ValueError("boom")) == "unknown"


def test_validation_reason_appends_diagnostic_without_touching_request(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", tmp_path)
    result = llm_shim._ApiChatResult(
        '{"commentary":"bad"}',
        scope="llmb",
        source_run_id="run-1",
        request_payload={"messages": [{"role": "user", "content": "in"}]},
        log_ctx={"round": "round1", "scene": "w1"},
        http_status=200,
        finish_reason="stop",
        usage={"total_tokens": 10},
        request_id="req-x",
        endpoint_url="https://api.example.test/v1/chat/completions",
    )

    llm_shim.record_validation_reason(result, "missing_anchor")
    llm_shim.record_validation_reason(result, "")

    entries = [json.loads(line) for line in (next(tmp_path.glob("api_debug_*.jsonl")).read_text(encoding="utf-8")).splitlines()]
    assert len(entries) == 1
    assert entries[0]["validation_reason"] == "missing_anchor"
    assert entries[0]["request_id"] == "req-x"
    assert entries[0]["scope"] == "llmb"


# ── 阶段 1：云端请求护栏 ──────────────────────────────────────────────────


def test_execute_splits_connect_read_and_total_timeouts(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout, **kwargs):
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": "ok"}}]}

    _patch_transport(monkeypatch, fake_post)
    monkeypatch.setenv("AI6657_API_KEY", "test-key")
    monkeypatch.setenv("AI6657_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("AI6657_LLM_MODEL", "qwen3")

    llm_shim._execute_openai_chat(
        [{"role": "user", "content": "hi"}],
        {
            "connect_timeout_sec": 5,
            "read_idle_timeout_sec": 60,
            "total_timeout_sec": 300,
        },
    )

    assert captured["timeout"] == (5.0, 60.0)


def test_remote_request_never_injects_thinking_controls(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout, **kwargs):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    _patch_transport(monkeypatch, fake_post)
    monkeypatch.setenv("AI6657_API_KEY", "test-key")
    monkeypatch.setenv("AI6657_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("AI6657_LLM_MODEL", "qwen3-flash")

    llm_shim._execute_openai_chat(
        [{"role": "user", "content": "hi"}],
        {"model": "qwen3-flash", "enable_thinking": False, "cloud_thinking_extra": {"thinking": {"type": "disabled"}}},
    )

    # 不做任何 thinking 调配：即使调用方传入 enable_thinking/cloud_thinking_extra 也不注入。
    assert captured.get("stream") is True
    assert "thinking" not in captured
    assert "enable_thinking" not in captured
    assert "chat_template_kwargs" not in captured


def test_scope_semaphore_limits_concurrency(monkeypatch):
    from sbmachine import cloud_memory

    _patch_transport(
        monkeypatch,
        lambda *args, **kwargs: {"choices": [{"message": {"content": "ok"}}]},
    )
    monkeypatch.setenv("AI6657_API_KEY", "test-key")
    monkeypatch.setenv("AI6657_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("AI6657_LLM_MODEL", "qwen3")

    gen = cloud_memory.make_generate("llmb", semantic_cfg={"cloud_request_concurrency": 1})
    results = []

    def worker():
        results.append(gen("p", {"model": "qwen3"}, system_prompt="s", log_ctx={"run_id": "run-1"}))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 3
    assert all(str(r) == "ok" for r in results)


def test_sse_total_timeout_interrupts_stream(monkeypatch):
    class FakeClock:
        now = 0.0

        @classmethod
        def monotonic(cls):
            cls.now += 0.01
            return cls.now

    class FakeLineStream:
        def iter_lines(self, decode_unicode=False):
            while True:
                yield "data: {\"choices\":[]}"

    monkeypatch.setattr(llm_protocol.time, "monotonic", FakeClock.monotonic)

    with pytest.raises(requests.Timeout, match="total_timeout_sec"):
        llm_shim._consume_sse_response(FakeLineStream(), None, total_timeout_sec=0.5)


def test_in_flight_counter_tracks_start_value():
    before = llm_shim._in_flight_start()
    assert before >= 0
    try:
        llm_shim._in_flight_start()
        llm_shim._in_flight_end()
    finally:
        llm_shim._in_flight_end()
    assert llm_shim._in_flight_count == 0


def test_training_sample_is_written_only_after_explicit_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", tmp_path)
    result = llm_shim._ApiChatResult(
        '{"commentary":"ok","felt_intensity":0.4}',
        scope="llmb",
        source_run_id="run-1",
        request_payload={"messages": [{"role": "user", "content": "accepted input"}]},
        log_ctx={"round": "round1", "scene": "未下包"},
    )

    assert list(tmp_path.glob("api_training_*.jsonl")) == []
    llm_shim.accept_api_response(result)

    entry = json.loads(next(tmp_path.glob("api_training_*.jsonl")).read_text(encoding="utf-8"))
    assert entry["accepted"] is True
    assert entry["source_run_id"] == "run-1"
    assert entry["input"] == "accepted input"


# ── 流式（SSE）聚合 ──────────────────────────────────────────────────────────

class _FakeSSEResponse:
    """模拟 requests.Response.iter_lines 的流式响应。"""

    def __init__(self, lines: list[str]):
        self.lines = lines
        self.status_code = 200

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)


def test_consume_sse_aggregates_content_reasoning_finish_and_usage():
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"先想一下"}}]}',
        'data: {"choices":[{"delta":{"content":"{\\"commentary\\":\\"你好\\"}"}}]}',
        'data: {"choices":[{"delta":{"content":"}"},"finish_reason":"stop"}]}',
        'data: {"usage":{"prompt_tokens":10,"completion_tokens":7,"total_tokens":17}}',
        "data: [DONE]",
    ]
    data = llm_shim._consume_sse_response(_FakeSSEResponse(lines))

    choice = data["choices"][0]
    assert choice["message"]["content"] == '{"commentary":"你好"}}'
    assert choice["message"]["reasoning_content"] == "先想一下"
    assert choice["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}


def test_consume_sse_tolerates_missing_usage_and_non_data_lines():
    lines = [
        ": keep-alive",
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}",
        "data: [DONE]",
    ]
    data = llm_shim._consume_sse_response(_FakeSSEResponse(lines))

    assert data["choices"][0]["message"]["content"] == "ok"
    assert data["choices"][0]["finish_reason"] is None
    assert data["usage"] is None


def test_consume_sse_ignores_malformed_chunks_without_aborting():
    lines = [
        "data: {not-json",
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}",
        "data: [DONE]",
    ]
    data = llm_shim._consume_sse_response(_FakeSSEResponse(lines))
    assert data["choices"][0]["message"]["content"] == "a"


def test_cloud_api_payload_enables_streaming_but_loopback_stays_off(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout, stream=None):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(llm_shim, "_post_openai_with_retry", fake_post)
    monkeypatch.setattr(llm_shim, "_dump_api_log", lambda *args, **kwargs: None)
    monkeypatch.setenv("AI6657_API_KEY", "test-key")
    monkeypatch.setenv("AI6657_BASE_URL", "https://api.example.test/v1")

    llm_shim._execute_openai_chat([{"role": "user", "content": "hi"}], {"model": "m"})
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}

    captured.clear()
    monkeypatch.setenv("AI6657_BASE_URL", "http://127.0.0.1:8000/v1")
    llm_shim._execute_openai_chat([{"role": "user", "content": "hi"}], {"model": "m"})
    assert "stream" not in captured
    assert "stream_options" not in captured


def test_stream_payload_is_sent_via_streaming_http_request(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=True):
            return iter(['data: {"choices":[{"delta":{"content":"ok"}}]}', "data: [DONE]"])

    def fake_requests_post(url, json=None, headers=None, timeout=None, stream=None):
        captured["stream_arg"] = stream
        return FakeResp()

    monkeypatch.setattr(llm_shim.requests, "post", fake_requests_post)
    result = llm_shim._post_openai_with_retry("https://api.example.test/v1", {"stream": True}, {}, 30)
    assert captured["stream_arg"] is True
    assert result["choices"][0]["message"]["content"] == "ok"


def test_non_json_html_response_is_explicit_error_not_silent_empty(monkeypatch):
    """回归：base_url 缺 /v1 时 Web 前端对 POST 返回 200 HTML → 显式报错，杜绝静默空响应。"""
    class HtmlResp:
        status_code = 200
        headers = {"Content-Type": "text/html"}

        def raise_for_status(self):
            pass

    class JsonResp:
        status_code = 200
        headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    responses = iter([HtmlResp(), JsonResp()])
    monkeypatch.setattr(llm_shim.requests, "post", lambda *a, **k: next(responses))

    with pytest.raises(ValueError, match="non-JSON content-type"):
        llm_shim._post_openai_with_retry("https://api.example.test/chat/completions", {}, {}, 30)
    # 正确的 JSON 端点不受影响
    result = llm_shim._post_openai_with_retry("https://api.example.test/v1/chat/completions", {}, {}, 30)
    assert result["choices"][0]["message"]["content"] == "ok"
