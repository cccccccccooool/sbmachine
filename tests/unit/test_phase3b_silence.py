import json

import pytest

from sbmachine.neutral_contract import CLOUD_PHASE3A_MODE, new_manifest_metadata
from sbmachine.preflight import PublishContractError, validate_commentary_publishable
from sbmachine.phase3b_style import run_phase3b


def test_phase3b_keeps_intentional_silence_out_of_tts_artifacts(tmp_path):
    rounds_path = tmp_path / "rounds_with_yolo.json"
    rounds_path.write_text(json.dumps({
        "video_path": "match.mp4",
        "map_name": "de_test",
        "rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 10.0}],
    }), encoding="utf-8")
    neutral_path = tmp_path / "rounds_with_neutral.json"
    neutral_path.write_text(json.dumps({
        **new_manifest_metadata(rounds_path),
        "rounds": [{
            "round_no": 1,
            "avg_hype": 0.0,
            "analyst_failed": False,
            "scenes": [{
                "t_start": 0.0,
                "t_end": 10.0,
                "scene": "默认场景",
                "commentary_plan": {},
                "fact_anchors": {"players": [], "teams": [], "numbers": [], "events": [], "results": [], "locations": [], "weapons": []},
                "neutral": "",
                "neutral_source": "intentional_empty",
                "generation_status": "success",
            }],
        }],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  backend: vllm\nsemantic:\n  style_backend: vllm\n", encoding="utf-8")

    manifest = run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
    )

    item = manifest["rounds"][0]
    assert item["status"] == "silent"
    assert item["commentary_text"] == ""
    assert item["emotion_segments"] == []
    assert item["window_results"][0]["style_status"] == "skipped_intentional_empty"
    validate_commentary_publishable(tmp_path / "commentary.json", neutral_path)


def test_phase3b_output_does_not_carry_phase2_yolo(tmp_path):
    rounds_path = tmp_path / "rounds_with_yolo.json"
    rounds_path.write_text(json.dumps({
        "video_path": "match.mp4",
        "map_name": "de_test",
        "rounds": [{
            "round_no": 1,
            "start_sec": 0.0,
            "end_sec": 10.0,
            "_phase2_yolo": {
                "key_frames": [{
                    "time_sec": 1.0,
                    "gate_reason": "demo_only",
                    "has_frame": False,
                    "ui_regions": [{"label": "pov_name", "x1": 0.41, "y1": 0.88}],
                    "background_info": {"infernos_active": [{"hull_x": [657.25]}]},
                }],
            },
        }],
    }), encoding="utf-8")
    neutral_path = tmp_path / "rounds_with_neutral.json"
    neutral_path.write_text(json.dumps({
        **new_manifest_metadata(rounds_path),
        "rounds": [{
            "round_no": 1,
            "avg_hype": 0.0,
            "analyst_failed": False,
            "scenes": [{
                "t_start": 0.0,
                "t_end": 10.0,
                "scene": "默认场景",
                "commentary_plan": {},
                "fact_anchors": {"players": [], "teams": [], "numbers": [], "events": [], "results": [], "locations": [], "weapons": []},
                "neutral": "",
                "neutral_source": "intentional_empty",
                "generation_status": "success",
            }],
        }],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  backend: vllm\nsemantic:\n  style_backend: vllm\n", encoding="utf-8")
    output_rounds_path = tmp_path / "rounds_with_commentary.json"

    run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=output_rounds_path,
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
    )

    saved_round = json.loads(output_rounds_path.read_text(encoding="utf-8"))["rounds"][0]
    assert "_phase2_yolo" not in saved_round
    assert "_phase3_semantic" in saved_round


def test_preflight_recomputes_neutral_identity_and_scene_budget_audit(tmp_path, monkeypatch):
    from tests.unit.test_phase3b_response import _api_style_response, _phase3b_paths, _scene
    from sbmachine import llmb_api, phase3b_style

    rounds_path, neutral_path, config_path = _phase3b_paths(tmp_path, [_scene(0.0, 2.0)])
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(llmb_api, "generate", lambda *args, **kwargs: _api_style_response("[平述]事实完整，语气细节已经充分补足。", "style-ok"))
    commentary_path = tmp_path / "commentary.json"
    run_phase3b(neutral_path=neutral_path, rounds_path=rounds_path, output_rounds_path=tmp_path / "rounds_with_commentary.json", commentary_path=commentary_path, config_path=config_path)
    validate_commentary_publishable(commentary_path, neutral_path)

    payload = json.loads(commentary_path.read_text(encoding="utf-8"))
    payload["rounds"][0]["scenes"][0]["output_chars"] += 1
    commentary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PublishContractError, match="budget audit"):
        validate_commentary_publishable(commentary_path, neutral_path)


def _run_zero_scene_phase3b(tmp_path, *, phase3a_mode: str, analyst_failed: bool = False):
    rounds_path = tmp_path / "rounds_with_yolo.json"
    rounds_path.write_text(json.dumps({
        "video_path": "match.mp4",
        "map_name": "de_test",
        "rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 10.0}],
    }), encoding="utf-8")
    neutral_path = tmp_path / "rounds_with_neutral.json"
    neutral_path.write_text(json.dumps({
        **new_manifest_metadata(rounds_path),
        "phase3a_mode": phase3a_mode,
        "rounds": [{
            "round_no": 1,
            "avg_hype": 0.0,
            "analyst_failed": analyst_failed,
            "scenes": [],
        }],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  backend: vllm\nsemantic:\n  style_backend: vllm\n", encoding="utf-8")
    commentary_path = tmp_path / "commentary.json"
    manifest = run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=commentary_path,
        config_path=config_path,
    )
    return manifest, commentary_path


def test_zero_window_round_is_partial_not_vacuously_silent(tmp_path):
    manifest, commentary_path = _run_zero_scene_phase3b(
        tmp_path,
        phase3a_mode=CLOUD_PHASE3A_MODE,
    )

    item = manifest["rounds"][0]
    assert item["status"] == "partial"
    assert item["commentary_text"] == ""
    assert item["emotion_segments"] == []
    with pytest.raises(PublishContractError, match="partial"):
        validate_commentary_publishable(commentary_path)


def test_local_zero_scene_remains_non_publishable(tmp_path):
    manifest, commentary_path = _run_zero_scene_phase3b(
        tmp_path,
        phase3a_mode="llma_slicer_then_llma_analyze",
    )

    assert manifest["rounds"][0]["status"] == "partial"
    with pytest.raises(PublishContractError, match="partial"):
        validate_commentary_publishable(commentary_path)


def test_cloud_zero_scene_with_analyst_failure_is_rejected_before_phase3b(tmp_path):
    with pytest.raises(PublishContractError, match="analyst failure"):
        _run_zero_scene_phase3b(
            tmp_path,
            phase3a_mode=CLOUD_PHASE3A_MODE,
            analyst_failed=True,
        )

def test_phase3b_rejects_fallback_neutral_before_style_model_call(tmp_path, monkeypatch):
    """Phase3b-only 入口必须在构造 LLMB 请求前拒绝 fallback neutral。"""
    rounds_path = tmp_path / "rounds_with_yolo.json"
    rounds_path.write_text(json.dumps({
        "video_path": "match.mp4",
        "map_name": "de_test",
        "rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 10.0}],
    }), encoding="utf-8")
    neutral_path = tmp_path / "rounds_with_neutral.json"
    neutral_path.write_text(json.dumps({
        **new_manifest_metadata(rounds_path),
        "rounds": [{
            "round_no": 1,
            "avg_hype": 0.0,
            "analyst_failed": False,
            "scenes": [{
                "t_start": 0.0,
                "t_end": 10.0,
                "scene": "test_scene",
                "commentary_plan": {"main_topic": {"kind": "retake", "summary": "C4已安装"}},
                "fact_anchors": {"players": [], "teams": [], "numbers": [], "events": ["bomb_planted"], "results": [], "locations": [], "weapons": []},
                "neutral": "C4已安装",
                "neutral_source": "fallback",
                "generation_status": "success",
            }],
        }],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  backend: vllm\nsemantic:\n  style_backend: vllm\n", encoding="utf-8")
    calls = []
    from sbmachine import llmb_api

    def forbidden_generate(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("LLMB must not be called for fallback neutral")

    monkeypatch.setattr(llmb_api, "generate", forbidden_generate)
    with pytest.raises(PublishContractError, match="rule fallback neutral is not publishable"):
        run_phase3b(
            neutral_path=neutral_path,
            rounds_path=rounds_path,
            output_rounds_path=tmp_path / "rounds_with_commentary.json",
            commentary_path=tmp_path / "commentary.json",
            config_path=config_path,
        )
    assert calls == []
