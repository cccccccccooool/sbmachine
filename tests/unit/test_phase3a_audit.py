"""S5 (§10) 审计产物、窗口统计与训练样本门禁测试。"""
import json
from pathlib import Path

import pytest

from sbmachine import llm_shim
from sbmachine import llm_protocol
from sbmachine.phase3a_audit import (
    AUDIT_ARTIFACT_KIND,
    AUDIT_CONTRACT_VERSION,
    build_audit_artifact,
    build_window_statistics,
    compute_source_hashes,
    read_audit_artifact,
    sha256_file,
)
from sbmachine.phase3a_analyst import _format_window_statistics, run_phase3a


# ── 审计产物结构（§10.2） ──


def test_build_audit_artifact_has_required_top_level_fields():
    llma_rounds = [{"round_no": 1, "windows": []}]
    result_rounds = [{"round_no": 1, "scenes": []}]
    artifact = build_audit_artifact(llma_rounds, result_rounds, "run-abc", {})

    assert artifact["artifact_kind"] == AUDIT_ARTIFACT_KIND
    assert artifact["contract_version"] == AUDIT_CONTRACT_VERSION
    assert artifact["run_id"] == "run-abc"
    assert artifact["source_hashes"] == {}
    assert artifact["rounds"] == [{"round_no": 1, "windows": []}]


def test_build_audit_artifact_merges_projection_with_window_identity():
    llma_rounds = [{
        "round_no": 1,
        "windows": [
            {
                "main_topic": {"kind": "kill", "summary": "A击杀B"},
                "selected_actions": [{"type": "kill_topic", "attacker": "A"}],
                "player_state": "首次快照：JDC（CT，100血，M4，中路）；REZ已阵亡",
                "character_limit": 74,
            },
            {
                "main_topic": {"kind": "silence", "summary": ""},
                "selected_actions": [],
                "rule_state": {"kind": "snapshot", "teams": {}},
                "character_limit": 50,
            },
        ],
    }]
    result_rounds = [{
        "round_no": 1,
        "scenes": [
            {"window_id": "r001_w01", "t_start": 0.0, "t_end": 5.0, "scene": "setup"},
            {"window_id": "r001_w02", "t_start": 5.0, "t_end": 10.0, "scene": "未下包"},
        ],
    }]
    artifact = build_audit_artifact(llma_rounds, result_rounds, "run-1", {})

    windows = artifact["rounds"][0]["windows"]
    assert len(windows) == 2

    w0 = windows[0]
    assert w0["window_index"] == 1
    assert w0["window_id"] == "r001_w01"
    assert w0["t_start"] == 0.0
    assert w0["t_end"] == 5.0
    assert w0["scene"] == "setup"
    assert w0["character_limit"] == 74
    assert "character_limit" not in w0["projection"]
    assert w0["projection"]["main_topic"] == {"kind": "kill", "summary": "A击杀B"}
    assert w0["projection"]["player_state"] == "首次快照：JDC（CT，100血，M4，中路）；REZ已阵亡"

    w1 = windows[1]
    assert w1["window_index"] == 2
    assert w1["window_id"] == "r001_w02"
    assert w1["character_limit"] == 50
    assert "character_limit" not in w1["projection"]
    assert w1["projection"]["rule_state"]["kind"] == "snapshot"


def test_build_audit_artifact_fallback_window_id_when_scene_missing():
    llma_rounds = [{"round_no": 3, "windows": [{"main_topic": {}, "character_limit": 10}]}]
    result_rounds = [{"round_no": 3, "scenes": []}]
    artifact = build_audit_artifact(llma_rounds, result_rounds, "run-x", {})
    w = artifact["rounds"][0]["windows"][0]
    assert w["window_id"] == "r003_w01"


# ── 窗口统计不变量（§10.5 Gate 5） ──


def _scene(round_no, idx, *, status, source, neutral="", scene_label="setup"):
    return {
        "window_id": f"r{round_no:03d}_w{idx:02d}",
        "t_start": float(idx),
        "t_end": float(idx + 1),
        "scene": scene_label,
        "neutral": neutral,
        "neutral_source": source,
        "generation_status": status,
    }


def _result_round(round_no, scenes, *, fallback=0):
    status_counts: dict[str, int] = {}
    for s in scenes:
        st = str(s.get("generation_status", "unknown"))
        status_counts[st] = status_counts.get(st, 0) + 1
    return {
        "round_no": round_no,
        "scenes": scenes,
        "generation_status_counts": status_counts,
        "fallback_windows": fallback,
        "analyst_failed": False,
    }


def test_window_statistics_gate5_invariants_all_success():
    scenes_r1 = [
        _scene(1, 1, status="success", source="llm", neutral="A击杀B"),
        _scene(1, 2, status="success", source="intentional_empty"),
    ]
    scenes_r2 = [
        _scene(2, 1, status="success", source="llm", neutral="C投出烟雾"),
        _scene(2, 2, status="success", source="llm", neutral="D完成双杀"),
        _scene(2, 3, status="success", source="intentional_empty"),
    ]
    rounds = [_result_round(1, scenes_r1), _result_round(2, scenes_r2)]
    stats = build_window_statistics(rounds)

    assert stats["rounds_total"] == 2
    assert stats["windows_total"] == 5
    assert stats["model_calls"] == 3  # 3 llm success
    assert stats["intentional_silence"] == 2
    # Gate 5: model_calls + intentional_silence = windows_total
    assert stats["model_calls"] + stats["intentional_silence"] == stats["windows_total"]
    # Gate 5: windows_total = sum(generation_status_counts)
    assert sum(stats["generation_status_counts"].values()) == stats["windows_total"]
    assert stats["fallback_windows"] == 0
    assert stats["publishable"] is True


def test_window_statistics_gate5_invariants_with_failures():
    scenes = [
        _scene(1, 1, status="success", source="llm", neutral="有效稿"),
        _scene(1, 2, status="contract_error", source=""),
        _scene(1, 3, status="truncated", source=""),
        _scene(1, 4, status="success", source="intentional_empty"),
    ]
    rounds = [_result_round(1, scenes)]
    stats = build_window_statistics(rounds)

    assert stats["windows_total"] == 4
    assert stats["model_calls"] == 3  # 1 llm + 2 failures = 3 model calls
    assert stats["intentional_silence"] == 1
    assert stats["model_calls"] + stats["intentional_silence"] == stats["windows_total"]
    assert sum(stats["generation_status_counts"].values()) == stats["windows_total"]
    assert stats["generation_status_counts"]["success"] == 2
    assert stats["generation_status_counts"]["contract_error"] == 1
    assert stats["generation_status_counts"]["truncated"] == 1
    assert stats["fallback_windows"] == 0
    assert stats["publishable"] is False  # has non-success windows


def test_window_statistics_lists_all_known_statuses():
    rounds = [_result_round(1, [])]
    stats = build_window_statistics(rounds)
    for status in ("success", "transport_error", "http_error",
                   "response_error", "truncated", "parse_error", "contract_error"):
        assert status in stats["generation_status_counts"]


def test_console_window_statistics_lists_new_and_unknown_failures():
    line = _format_window_statistics({
        "windows_total": 7,
        "model_calls": 7,
        "intentional_silence": 0,
        "generation_status_counts": {
            "success": 1,
            "projection_budget_error": 2,
            "unexpected_fact": 1,
            "required_fact_missing": 1,
            "side_mismatch": 1,
            "semantic_contract_error": 1,
            "future_contract_error": 2,
        },
        "unrecoverable_count": 4,
        "fallback_windows": 0,
        "publishable": False,
    })
    for status in (
        "projection_budget_error=2", "unexpected_fact=1",
        "required_fact_missing=1", "side_mismatch=1",
        "semantic_contract_error=1", "future_contract_error=2",
        "unrecoverable=4", "publishable=False",
    ):
        assert status in line


def test_window_statistics_empty_rounds():
    stats = build_window_statistics([])
    assert stats["rounds_total"] == 0
    assert stats["windows_total"] == 0
    assert stats["model_calls"] == 0
    assert stats["intentional_silence"] == 0
    assert stats["publishable"] is True


# ── 向后兼容 reader（§10.4） ──


def test_read_audit_artifact_version3(tmp_path):
    artifact = {
        "artifact_kind": AUDIT_ARTIFACT_KIND,
        "contract_version": AUDIT_CONTRACT_VERSION,
        "run_id": "run-v3",
        "source_hashes": {"rounds_with_yolo.json": "sha256:abc"},
        "rounds": [{"round_no": 1, "windows": []}],
    }
    path = tmp_path / "llma_input.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    result = read_audit_artifact(path)
    assert result["contract_version"] == 3
    assert result["run_id"] == "run-v3"


def test_read_audit_artifact_legacy_version1(tmp_path):
    legacy = {"rounds": [{"round_no": 1, "windows": [{"main_topic": {}}]}]}
    path = tmp_path / "llma_input.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    result = read_audit_artifact(path)
    assert result["contract_version"] == 1
    assert result["rounds"] == legacy["rounds"]


# ── 源文件哈希 ──


def test_compute_source_hashes_hashes_existing_files(tmp_path):
    rounds = tmp_path / "rounds_with_yolo.json"
    rounds.write_text('{"rounds":[]}', encoding="utf-8")
    semantic = tmp_path / "semantic_frames.json"
    semantic.write_text('[]', encoding="utf-8")
    sys_prompt = tmp_path / "analyst_system.txt"
    sys_prompt.write_text("system prompt", encoding="utf-8")
    round_prompt = tmp_path / "analyst_round.txt"
    round_prompt.write_text("round prompt", encoding="utf-8")

    hashes = compute_source_hashes(rounds, semantic, sys_prompt, round_prompt)
    assert hashes["rounds_with_yolo.json"] == sha256_file(rounds)
    assert hashes["semantic_frames.json"] == sha256_file(semantic)
    assert hashes["analyst_system.txt"] == sha256_file(sys_prompt)
    assert hashes["analyst_round.txt"] == sha256_file(round_prompt)
    assert all(v.startswith("sha256:") for v in hashes.values())


def test_compute_source_hashes_skips_missing_files(tmp_path):
    hashes = compute_source_hashes(
        tmp_path / "missing.json",
        tmp_path / "missing_semantic.json",
        tmp_path / "missing_sys.txt",
        tmp_path / "missing_round.txt",
    )
    assert hashes == {}


# ── 集成：run_phase3a 产出审计产物与窗口统计 ──


def _frame(t: float, events: dict | None = None, *, phase: str = "in_round") -> dict:
    data = {
        "when": {"video_time": t, "relative_sec": t, "phase": phase},
        "who": {"view": "player", "pov_player": "p1"},
    }
    if events:
        data["events"] = events
    return data


def _round_record(round_no: int, frames: list[dict]) -> dict:
    start = float(frames[0]["when"]["video_time"])
    end = float(frames[-1]["when"]["video_time"])
    return {
        "round_no": round_no,
        "start_sec": start,
        "end_sec": end,
        "score_before": {"ct": 0, "t": 0},
        "score_after": {"ct": 0, "t": 1},
        "demo_round_hint": round_no,
        "_phase2_yolo": {
            "key_frames": [
                {
                    "time_sec": frame["when"]["video_time"],
                    "gate_reason": "test",
                    "background_info": frame,
                    "has_frame": True,
                }
                for frame in frames
            ],
        },
    }


def test_run_phase3a_writes_audit_artifact_with_contract_version_3(tmp_path, monkeypatch):
    frames = [_frame(0.0), _frame(5.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47"}]}), _frame(10.0)]
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps({"video_path": "test.mp4", "map_name": "de_test", "rounds": [_round_record(1, frames)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    semantic_frames = [_frame(0.0), _frame(5.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47", "tick": 512}]}), _frame(10.0)]
    semantic_path.write_text(json.dumps([{"round_no": 1, "frames": semantic_frames}], ensure_ascii=False), encoding="utf-8")
    config_path.write_text(
        f"""
llm:
  backend: vllm
semantic:
  analyst_backend: vllm
  analyst_model: qwen3
  analyst_output_max_tokens: 256
  analyst_concurrent_rounds: 1
  window_max_sec: 10
  window_min_sec: 3
paths:
  rounds_with_yolo_semantic_json: "{semantic_path.as_posix()}"
debug:
  phase3: false
""",
        encoding="utf-8",
    )
    training_dir = tmp_path / "logs"
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", training_dir)

    def fake_generate(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs):
        return llm_shim._ApiChatResult(
            json.dumps({"neutral": _contract_neutral_from_prompt(prompt)}, ensure_ascii=False),
            scope="llma",
            source_run_id="run-test",
            request_payload={"messages": [{"role": "user", "content": prompt}]},
            log_ctx=log_ctx,
        )

    import sbmachine.llma_api as llma_api
    monkeypatch.setattr(llma_api, "generate", fake_generate)

    run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    artifact = json.loads((tmp_path / "llma_input.json").read_text(encoding="utf-8"))
    assert artifact["artifact_kind"] == "phase3a_llm_input"
    assert artifact["contract_version"] == 3
    assert isinstance(artifact["run_id"], str) and artifact["run_id"]
    assert "rounds_with_yolo.json" in artifact["source_hashes"]
    assert "semantic_frames.json" in artifact["source_hashes"]
    assert "analyst_system.txt" in artifact["source_hashes"]
    assert "analyst_round.txt" in artifact["source_hashes"]

    windows = artifact["rounds"][0]["windows"]
    assert len(windows) >= 1
    w = windows[0]
    assert "window_index" in w
    assert "window_id" in w
    assert "t_start" in w
    assert "t_end" in w
    assert "scene" in w
    assert "character_limit" in w
    assert "projection" in w
    assert "character_limit" not in w["projection"]
    assert "commentary_plan" not in w["projection"]


def test_run_phase3a_window_stats_satisfy_gate5_invariants(tmp_path, monkeypatch):
    frames = [_frame(0.0), _frame(5.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47"}]}), _frame(10.0)]
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps({"video_path": "test.mp4", "map_name": "de_test", "rounds": [_round_record(1, frames)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    semantic_frames = [_frame(0.0), _frame(5.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47", "tick": 512}]}), _frame(10.0)]
    semantic_path.write_text(json.dumps([{"round_no": 1, "frames": semantic_frames}], ensure_ascii=False), encoding="utf-8")
    config_path.write_text(
        f"""
llm:
  backend: vllm
semantic:
  analyst_backend: vllm
  analyst_model: qwen3
  analyst_output_max_tokens: 256
  analyst_concurrent_rounds: 1
  window_max_sec: 10
  window_min_sec: 3
paths:
  rounds_with_yolo_semantic_json: "{semantic_path.as_posix()}"
debug:
  phase3: false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", tmp_path / "logs")

    def fake_generate(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs):
        return llm_shim._ApiChatResult(
            json.dumps({"neutral": _contract_neutral_from_prompt(prompt)}, ensure_ascii=False),
            scope="llma",
            source_run_id="run-stats",
            request_payload={"messages": [{"role": "user", "content": prompt}]},
            log_ctx=log_ctx,
        )

    import sbmachine.llma_api as llma_api
    monkeypatch.setattr(llma_api, "generate", fake_generate)

    run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    from sbmachine.phase3a_analyst import _DIAGNOSTICS_DIR, _DIAGNOSTICS_RUN_ID
    stats_path = _DIAGNOSTICS_DIR / f"{_DIAGNOSTICS_RUN_ID}_window_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    # Gate 5 invariants
    assert stats["model_calls"] + stats["intentional_silence"] == stats["windows_total"]
    assert sum(stats["generation_status_counts"].values()) == stats["windows_total"]
    assert stats["fallback_windows"] == 0
    assert stats["publishable"] is True


# ── 训练样本门禁（§10.2/§10.3/§10.5） ──


def test_training_samples_not_accepted_when_manifest_not_publishable(tmp_path, monkeypatch):
    """单窗口解析成功但整体产物未通过门禁时，不提交训练样本。"""
    frames = [
        _frame(0.0),
        _frame(2.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47"}]}),
        _frame(5.0, {"kills": [{"attacker": "C", "victim": "D", "weapon": "AK-47"}]}),
        _frame(10.0),
    ]
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps({"video_path": "test.mp4", "map_name": "de_test", "rounds": [_round_record(1, frames)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    semantic_frames = [
        _frame(0.0),
        _frame(2.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47", "tick": 256}]}),
        _frame(5.0, {"kills": [{"attacker": "C", "victim": "D", "weapon": "AK-47", "tick": 512}]}),
        _frame(10.0),
    ]
    semantic_path.write_text(json.dumps([{"round_no": 1, "frames": semantic_frames}], ensure_ascii=False), encoding="utf-8")
    config_path.write_text(
        f"""
llm:
  backend: vllm
semantic:
  analyst_backend: vllm
  analyst_model: qwen3
  analyst_output_max_tokens: 256
  analyst_concurrent_rounds: 1
  window_max_sec: 10
  window_min_sec: 3
paths:
  rounds_with_yolo_semantic_json: "{semantic_path.as_posix()}"
debug:
  phase3: false
""",
        encoding="utf-8",
    )
    training_dir = tmp_path / "logs"
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", training_dir)

    # 两个窗口都返回截断的 JSON（truncated）。恢复未启用时直接失败，
    # 归并阶段由规则层 fallback（neutral_source=rule）兜底为 success：
    # fallback 文本不进入 accepted_samples，因此不产生任何训练样本。

    def fake_generate(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs):
        return llm_shim._ApiChatResult(
            '{"neutral":"未完成',
            scope="llma",
            source_run_id="run-gate",
            request_payload={"messages": [{"role": "user", "content": prompt}]},
            log_ctx=log_ctx,
            finish_reason="length",
            raw_response={"choices": [{"message": {"content": '{"neutral":"未完成'}, "finish_reason": "length"}]},
        )

    import sbmachine.llma_api as llma_api
    monkeypatch.setattr(llma_api, "generate", fake_generate)
    # 规则层兜底失效（模拟规则层无 summary 可兜底的窗口）：truncated 保持失败，
    # 产物不可发布，训练样本不得提交。
    import sbmachine.phase3a_analyst as phase3a_analyst
    monkeypatch.setattr(phase3a_analyst, "fallback_neutral", lambda plan: "")

    run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    # 窗口均未产生 llm 源成功样本，不应提交任何训练样本
    assert list(training_dir.glob("api_training_*.jsonl")) == []


def test_training_samples_accepted_when_manifest_publishable(tmp_path, monkeypatch):
    """整体产物通过门禁时，才提交训练样本。"""
    frames = [_frame(0.0), _frame(5.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47"}]}), _frame(10.0)]
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps({"video_path": "test.mp4", "map_name": "de_test", "rounds": [_round_record(1, frames)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    semantic_frames = [_frame(0.0), _frame(5.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47", "tick": 512}]}), _frame(10.0)]
    semantic_path.write_text(json.dumps([{"round_no": 1, "frames": semantic_frames}], ensure_ascii=False), encoding="utf-8")
    config_path.write_text(
        f"""
llm:
  backend: vllm
semantic:
  analyst_backend: vllm
  analyst_model: qwen3
  analyst_output_max_tokens: 256
  analyst_concurrent_rounds: 1
  window_max_sec: 10
  window_min_sec: 3
paths:
  rounds_with_yolo_semantic_json: "{semantic_path.as_posix()}"
debug:
  phase3: false
""",
        encoding="utf-8",
    )
    training_dir = tmp_path / "logs"
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", training_dir)

    def fake_generate(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs):
        return llm_shim._ApiChatResult(
            json.dumps({"neutral": _contract_neutral_from_prompt(prompt)}, ensure_ascii=False),
            scope="llma",
            source_run_id="run-ok",
            request_payload={"messages": [{"role": "user", "content": prompt}]},
            log_ctx=log_ctx,
        )

    import sbmachine.llma_api as llma_api
    monkeypatch.setattr(llma_api, "generate", fake_generate)

    run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    training_lines = next(training_dir.glob("api_training_*.jsonl")).read_text(encoding="utf-8").splitlines()
    assert len(training_lines) >= 1
def _contract_neutral_from_prompt(prompt: str) -> str:
    projection = json.loads(prompt.strip().splitlines()[-1])
    return "，".join(
        fact["canonical_text"]
        for fact in projection["required_facts"]
        if fact.get("required") is True
    )
