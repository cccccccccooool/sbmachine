import json

from sbmachine import phase3a_cloud_runner
from sbmachine.neutral_contract import validate_neutral_manifest
from sbmachine.phase3b_style import run_phase3b


def _round(round_no: int, start: float) -> dict:
    return {
        "round_no": round_no, "start_sec": start, "end_sec": start + 10.0,
        "score_before": {"ct": 0, "t": round_no - 1}, "score_after": {"ct": 0, "t": round_no},
        "_phase2_yolo": {"key_frames": [
            {"time_sec": start, "gate_reason": "test", "has_frame": True, "background_info": {"when": {"video_time": start, "phase": "in_round", "round_no": round_no}, "where": {"players": [{"name": "A", "side": "T", "hp": 100, "weapon": "AK", "callout": "Ramp"}]}, "events": {}}},
            {"time_sec": start + 3, "gate_reason": "test", "has_frame": True, "background_info": {"when": {"video_time": start + 3, "phase": "in_round", "round_no": round_no}, "events": {"kills": [{"tick": round_no * 10, "attacker": "A", "victim": "B", "weapon": "AK"}]}}},
        ]},
    }


def test_cloud_runner_calls_once_per_round_and_writes_phase3b_compatible_manifest(tmp_path, monkeypatch):
    rounds_path = tmp_path / "rounds_with_yolo.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(json.dumps({"video_path": "x.mp4", "map_name": "de_test", "rounds": [_round(1, 0), _round(2, 10)]}), encoding="utf-8")
    config_path.write_text("llm:\n  base_url: https://example.test/v1\nsemantic:\n  analyst_model: cloud\n", encoding="utf-8")
    calls = []

    def fake_generate(prompt, llm_cfg, *, system_prompt, max_tokens, log_ctx):
        calls.append(prompt)
        assert max_tokens == 2048
        payload = json.loads(prompt.splitlines()[1])
        canonical = payload["windows"][0]["required_facts"][0]["canonical_text"]
        return json.dumps({"window_id": "window-1", "neutral": canonical + "。"})

    monkeypatch.setattr(phase3a_cloud_runner, "generate_cloud_round", fake_generate)
    manifest = phase3a_cloud_runner.run_cloud_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    assert len(calls) == 2
    assert all("roster" not in prompt and "timeline" not in prompt and "where" not in prompt for prompt in calls)
    assert all("window-1" in prompt for prompt in calls)
    assert manifest["phase3a_mode"] == "cloud_round_timeline"
    assert all(len(round_data["scenes"]) == 1 for round_data in manifest["rounds"])
    assert all("fact_anchors" in round_data["scenes"][0] for round_data in manifest["rounds"])
    validate_neutral_manifest(json.loads(output_path.read_text(encoding="utf-8")), rounds_path)
    styled = run_phase3b(neutral_path=output_path, rounds_path=rounds_path, output_rounds_path=tmp_path / "commentary_rounds.json", commentary_path=tmp_path / "commentary.json", config_path=config_path, dry_run=True)
    assert len(styled["rounds"]) == 2


def test_cloud_dry_run_returns_report_without_writing_publish_neutral(tmp_path):
    """云端 dry-run 不调用模型，也不得写入正式 neutral 路径。"""
    rounds_path = tmp_path / "rounds_with_yolo.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps({"video_path": "x.mp4", "map_name": "de_test", "rounds": [_round(1, 0)]}),
        encoding="utf-8",
    )
    config_path.write_text("llm:\n  base_url: https://example.test/v1\nsemantic:\n  analyst_model: cloud\n", encoding="utf-8")

    report = phase3a_cloud_runner.run_cloud_phase3a(
        rounds_path=rounds_path,
        output_path=output_path,
        config_path=config_path,
        dry_run=True,
    )

    assert report == {
        "mode": "phase3a_dry_run",
        "writes_performed": False,
        "publish_path": None,
        "rounds": 1,
        "windows": 1,
        "fallback_windows": 0,
    }
    assert not output_path.exists()
