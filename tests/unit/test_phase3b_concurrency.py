"""阶段 2：LLM-B 有界并发（确定性准备 → 并发请求 → 顺序验收/提交）的单元测试。

覆盖：
- style_concurrent_scenes=1 完全复现串行行为（请求顺序、验收顺序、retry_count）。
- 并发 >1 时请求真正重叠、验收/发布顺序仍按窗口时间顺序。
- 回合末兜底重试：主循环已有效重试则跳过，除非可恢复基础设施错误。
- LLM-B 云端会话默认关闭（cloud_conversation_max_rounds 缺省 0）。
- 诊断字段 dispatch_order/completion_order 只写诊断 JSONL。
"""
import json
import threading
import time

from sbmachine import cloud_memory, llm_shim, phase3b_style
from sbmachine.phase3b_style import _is_recoverable_infra_failure, run_phase3b

from tests.unit.test_phase3b_response import _api_style_response, _phase3b_paths, _scene

_LONG_TEXT = "超" * 40
_OK_TEXT = "[平述]事实完整，语气细节已经充分补足。"


def _scenes_with_ids():
    scenes = []
    for i, t in enumerate([0.0, 0.6, 1.2]):
        scene = _scene(t, t + 0.5, f"场景{i + 1}")
        scene["window_id"] = f"w{i + 1:02d}"
        scenes.append(scene)
    return scenes


def _config_with(text_extra: str) -> str:
    return f"llm:\n  backend: vllm\nsemantic:\n  style_backend: vllm\n{text_extra}"


def _run(tmp_path, monkeypatch, scenes, generate, config_extra=""):
    rounds_path, neutral_path, config_path = _phase3b_paths(tmp_path, scenes)
    config_path.write_text(_config_with(config_extra), encoding="utf-8")
    from sbmachine import llmb_api

    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(llmb_api, "generate", generate)
    return run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
    )


def test_concurrency_one_reproduces_serial_request_and_acceptance_order(tmp_path, monkeypatch):
    """并发=1 时逐窗串行：每窗 attempt0 失败后立即重试，再进入下一窗。"""
    calls = []

    def generate(*args, **kwargs):
        scene = kwargs["log_ctx"]["scene"]
        calls.append(scene)
        if calls.count(scene) == 1:
            return _api_style_response("[平述]" + _LONG_TEXT, "long-" + scene)
        return _api_style_response(_OK_TEXT, "ok-" + scene)

    manifest = _run(tmp_path, monkeypatch, _scenes_with_ids(), generate)

    round_item = manifest["rounds"][0]
    assert [w["retry_count"] for w in round_item["window_results"]] == [1, 1, 1]
    assert [s["style_status"] for s in round_item["scenes"]] == ["retry_success"] * 3
    assert [s["window_id"] for s in round_item["scenes"]] == ["w01", "w02", "w03"]
    # 逐窗交替（串行复现）：每窗 attempt0 + 重试完成后才进入下一窗。
    assert calls == ["w01", "w01", "w02", "w02", "w03", "w03"]


def test_under_budget_retries_with_length_window_feedback(tmp_path, monkeypatch):
    prompts = []

    def generate(prompt, *args, **kwargs):
        prompts.append(json.loads(prompt))
        if len(prompts) == 1:
            return _api_style_response("[平述]太短", "short")
        return _api_style_response(_OK_TEXT, "expanded")

    manifest = _run(tmp_path, monkeypatch, _scenes_with_ids()[:1], generate)

    result = manifest["rounds"][0]["window_results"][0]
    assert result["style_status"] == "retry_success"
    assert result["retry_count"] == 1
    assert prompts[1]["retry_feedback"]["failure_reason"] == "under_budget"
    assert "0.8~1.2" in prompts[1]["retry_feedback"]["instruction"]


def test_under_budget_exhaustion_publishes_last_short_draft(tmp_path, monkeypatch):
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs["log_ctx"]["scene"])
        return _api_style_response("[平述]太短", f"short-{len(calls)}")

    manifest = _run(tmp_path, monkeypatch, _scenes_with_ids()[:1], generate)

    round_item = manifest["rounds"][0]
    result = round_item["window_results"][0]
    assert round_item["status"] == "ok"
    assert round_item["scenes"][0]["text"] == "太短"
    assert result["style_status"] == "retry_success"
    assert result["retry_count"] == 2
    assert calls == ["w01", "w01", "w01"]


def test_concurrent_requests_actually_overlap(tmp_path, monkeypatch):
    """并发=2 时两个窗口的请求真实重叠（同时 in-flight）。"""
    lock = threading.Lock()
    state = {"inflight": 0, "max_inflight": 0}

    def generate(*args, **kwargs):
        with lock:
            state["inflight"] += 1
            state["max_inflight"] = max(state["max_inflight"], state["inflight"])
        time.sleep(0.15)
        with lock:
            state["inflight"] -= 1
        return _api_style_response(_OK_TEXT, "ok")

    _run(tmp_path, monkeypatch, _scenes_with_ids()[:2], generate, config_extra="  style_concurrent_scenes: 2\n")

    assert state["max_inflight"] >= 2


def test_concurrent_acceptance_follows_window_order_regardless_of_completion(tmp_path, monkeypatch):
    """并发=2 时第二个窗口先完成，验收/发布顺序仍按窗口时间顺序。"""
    barrier = threading.Barrier(2, timeout=10)
    completed = []

    def generate(*args, **kwargs):
        scene = kwargs["log_ctx"]["scene"]
        if scene in ("w01", "w02"):
            barrier.wait(timeout=10)
            time.sleep(0.05)
        completed.append(scene)
        return _api_style_response(_OK_TEXT, "ok-" + scene)

    manifest = _run(tmp_path, monkeypatch, _scenes_with_ids(), generate, config_extra="  style_concurrent_scenes: 2\n")

    round_item = manifest["rounds"][0]
    assert [s["window_id"] for s in round_item["scenes"]] == ["w01", "w02", "w03"]
    assert [s["style_status"] for s in round_item["scenes"]] == ["ok"] * 3
    assert [w["published_scene_index"] for w in round_item["window_results"]] == [0, 1, 2]


def test_round_end_backstop_skipped_after_effective_main_retry(tmp_path, monkeypatch):
    """主循环已做有效重试（retry_count=2）且失败类别为业务类 → 回合末兜底跳过。"""
    calls = []

    def generate(*args, **kwargs):
        scene = kwargs["log_ctx"]["scene"]
        calls.append(scene)
        return _api_style_response("[平述]" + _LONG_TEXT, "bad-" + scene)

    manifest = _run(tmp_path, monkeypatch, _scenes_with_ids()[:2], generate)

    results = manifest["rounds"][0]["window_results"]
    assert results[0]["style_status"] == "style_failed"
    assert results[0]["retry_count"] == 2
    assert results[1]["style_status"] == "style_failed"
    assert results[1]["retry_count"] == 2
    # 每窗主循环 3 次尝试后放弃；无任何回合末兜底调用。
    assert calls == ["w01", "w01", "w01", "w02", "w02", "w02"]


def test_round_end_backstop_compensates_recoverable_infra_error(tmp_path, monkeypatch):
    """主循环重试耗尽但最后失败为可恢复基础设施错误 → 兜底仍补偿一次。"""
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs["log_ctx"]["scene"])
        if len(calls) <= 3:
            raise ConnectionError("connection refused")
        return _api_style_response(_OK_TEXT, "w01-ok")

    manifest = _run(tmp_path, monkeypatch, _scenes_with_ids()[:1], generate)

    result = manifest["rounds"][0]["window_results"][0]
    assert result["style_status"] == "retry_success"
    assert result["retry_count"] == 2
    assert len(calls) == 4


def test_round_end_backstop_still_runs_without_main_retry(tmp_path, monkeypatch):
    """主循环一次重试都没做（retry_count=0）→ 兜底照常执行。"""
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs["log_ctx"]["scene"])
        if len(calls) == 1:
            return _api_style_response("[平述]" + _LONG_TEXT, "w01-bad")
        return _api_style_response(_OK_TEXT, "w01-ok")

    manifest = _run(
        tmp_path, monkeypatch, _scenes_with_ids()[:1], generate,
        config_extra="  style_max_retries: 0\n",
    )

    result = manifest["rounds"][0]["window_results"][0]
    assert result["style_status"] == "retry_success"
    assert result["retry_count"] == 0
    assert calls == ["w01", "w01"]


def test_llmb_cloud_session_off_by_default(tmp_path, monkeypatch):
    """api 后端下 make_generate 收到 cloud_conversation_max_rounds=0（缺省无会话）。"""
    rounds_path, neutral_path, config_path = _phase3b_paths(tmp_path, [_scene(0.0, 2.0)])
    config_path.write_text("llm:\n  backend: api\nsemantic:\n  style_backend: api\n", encoding="utf-8")
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    captured = {}

    def fake_make_generate(secret_scope, semantic_cfg=None):
        captured["scope"] = secret_scope
        captured["semantic_cfg"] = dict(semantic_cfg or {})
        return lambda *args, **kwargs: _api_style_response(_OK_TEXT, "ok")

    monkeypatch.setattr(cloud_memory, "make_generate", fake_make_generate)

    manifest = run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
    )

    assert manifest["rounds"][0]["status"] == "ok"
    assert captured["scope"] == "llmb"
    assert captured["semantic_cfg"].get("cloud_conversation_max_rounds") == 0
    assert captured["semantic_cfg"].get("cloud_conversation_max_tokens") == 0


def test_llmb_real_cloud_path_creates_no_session(tmp_path, monkeypatch):
    """api 后端真实 cloud_memory 路径：缺省配置不创建会话，commit_round 为 no-op。"""
    from sbmachine import llm_shim

    rounds_path, neutral_path, config_path = _phase3b_paths(tmp_path, [_scene(0.0, 2.0)])
    config_path.write_text("llm:\n  backend: api\nsemantic:\n  style_backend: api\n", encoding="utf-8")
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    cloud_memory.clear_sessions()

    def fake_chat(messages, cfg, **kwargs):
        return llm_shim._ApiChatResult(
            json.dumps({"commentary": "[平述]事实完整，语气细节已经充分补足。", "felt_intensity": 0.2}, ensure_ascii=False),
            scope="llmb",
            source_run_id="run-x",
            request_payload={"messages": messages},
            log_ctx=kwargs.get("log_ctx"),
        )

    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", fake_chat)

    manifest = run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
    )

    assert manifest["rounds"][0]["status"] == "ok"
    assert cloud_memory._sessions == {}


def test_llmb_cloud_session_still_available_when_explicitly_configured(tmp_path, monkeypatch):
    """显式配置 cloud_conversation_max_rounds>0 时，make_generate 仍收到会话参数。"""
    rounds_path, neutral_path, config_path = _phase3b_paths(tmp_path, [_scene(0.0, 2.0)])
    config_path.write_text(
        "llm:\n  backend: api\nsemantic:\n  style_backend: api\n  cloud_conversation_max_rounds: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    captured = {}

    def fake_make_generate(secret_scope, semantic_cfg=None):
        captured["semantic_cfg"] = dict(semantic_cfg or {})
        return lambda *args, **kwargs: _api_style_response(_OK_TEXT, "ok")

    monkeypatch.setattr(cloud_memory, "make_generate", fake_make_generate)

    run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
    )

    assert captured["semantic_cfg"].get("cloud_conversation_max_rounds") == 2


def test_dispatch_and_completion_order_written_to_diagnostics(tmp_path, monkeypatch):
    """诊断 JSONL 带 dispatch_order/completion_order；completion_order 按验收顺序单调。"""
    calls = []

    def generate(*args, **kwargs):
        scene = kwargs["log_ctx"]["scene"]
        calls.append(scene)
        if calls.count(scene) == 1:
            return _api_style_response("[平述]" + _LONG_TEXT, "long-" + scene)
        return _api_style_response(_OK_TEXT, "ok-" + scene)

    manifest = _run(tmp_path, monkeypatch, _scenes_with_ids()[:2], generate, config_extra="  style_concurrent_scenes: 2\n")

    run_id = manifest["source_neutral_run_id"]
    diag = tmp_path / "diagnostics" / "phase3b" / f"{run_id}_diagnostics.jsonl"
    entries = [json.loads(line) for line in diag.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 4
    assert {e["dispatch_order"] for e in entries} == {1, 2, 3, 4}
    completion_orders = [e["completion_order"] for e in entries]
    assert completion_orders == [1, 2, 3, 4]
    # 按验收顺序（窗口时间顺序）写诊断：w01 两条在前。
    assert entries[0]["window_id"] == "w01"
    assert entries[0]["attempt"] == 0
    assert entries[1]["window_id"] == "w01"
    assert entries[1]["attempt"] == 1
    assert entries[2]["window_id"] == "w02"
    assert entries[3]["window_id"] == "w02"


def test_infra_failure_classification():
    """基础设施错误分类辅助：http/transport/rate_limit 可恢复，业务类不可恢复。"""
    assert _is_recoverable_infra_failure({"retry_category": "connection_error"})
    assert _is_recoverable_infra_failure({"retry_category": "rate_limit"})
    assert _is_recoverable_infra_failure({"retry_category": "timeout"})
    assert _is_recoverable_infra_failure({"http_status": 429})
    assert _is_recoverable_infra_failure({"http_status": 503})
    assert _is_recoverable_infra_failure({"error": "requests.ConnectionError"})
    assert not _is_recoverable_infra_failure({})
    assert not _is_recoverable_infra_failure({"retry_category": "client_error"})
    assert not _is_recoverable_infra_failure({"http_status": 400})
    assert not _is_recoverable_infra_failure({"error": "ValueError"})
