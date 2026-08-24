import json
import pytest

from sbmachine import llm_protocol, phase3a_analyst
from sbmachine.phase3a_analyst import run_phase3a
from sbmachine.phase3b_style import run_phase3b
from sbmachine.scene_context import build_scene_contexts, classify_frame_scene


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


def _prompt_payload(prompt: str) -> dict:
    start = prompt.rfind("\n{")
    assert start >= 0, prompt
    return json.loads(prompt[start + 1:])


def _labels_for_windows(windows: list[tuple[float, float]], frames: list[dict]) -> list[str]:
    labels = []
    for index, (lo, hi) in enumerate(windows):
        is_last = index == len(windows) - 1
        frame = next((
            item for item in frames
            if lo <= item["when"]["video_time"]
            and (item["when"]["video_time"] <= hi if is_last else item["when"]["video_time"] < hi)
        ), None)
        labels.append(classify_frame_scene(frame) if frame else "默认场景")
    return labels

def test_classify_scene_uses_rules_without_llm():
    frames = [
        _frame(0.0, phase="pre_round"),
        _frame(5.0, {"smokes_active": [{"thrower": "A"}]}),
        _frame(10.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47"}]}),
        _frame(15.0, {"c4": {"planted": True}}),
        _frame(20.0, phase="post_round"),
    ]
    windows = [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 18.0), (18.0, 22.0)]

    assert _labels_for_windows(windows, frames) == ["准备", "未下包", "未下包", "炸弹", "收尾"]


def test_run_phase3a_calls_llm_once_per_non_silence_window_and_keeps_all_scenes(tmp_path, monkeypatch):
    frames = [
        _frame(0.0),
        _frame(1.0),
        _frame(2.0),
        _frame(3.0),
        _frame(4.0),
        _frame(5.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47"}]}),
        _frame(6.0),
        _frame(7.0),
        _frame(8.0),
        _frame(9.0),
        _frame(10.0, {"c4": {"planted": True}}),
        _frame(11.0),
        _frame(12.0),
    ]
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps(
            {
                "video_path": "test.mp4",
                "map_name": "de_test",
                "rounds": [_round_record(1, frames)],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    semantic_frames = [_frame(0.0), _frame(4.0), _frame(8.0), _frame(12.0)]
    semantic_frames[0]["where"] = {"players": [{"name": "raw_player", "side": "T", "hp": 100, "weapon": "AK", "callout": "Ramp"}]}
    semantic_frames[2]["events"] = {"kills": [{"attacker": "semantic_source", "victim": "B", "weapon": "AK-47", "tick": 512}]}
    semantic_path.write_text(
        json.dumps([{"round_no": 1, "frames": semantic_frames}], ensure_ascii=False),
        encoding="utf-8",
    )
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
demo:
  parsed_dir: missing-demo-dir
debug:
  phase3: false
""",
        encoding="utf-8",
    )

    expected_contexts = build_scene_contexts(semantic_frames, 0.0, 12.0, window_max_sec=10, window_min_sec=3)
    expected_windows = [(window.t_start, window.t_end) for window in expected_contexts]
    prompts: list[str] = []
    training_dir = tmp_path / "logs"
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", training_dir)

    def fake_generate(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs):
        prompts.append(prompt)
        assert max_tokens == 256
        assert list(training_dir.glob("api_training_*.jsonl")) == []
        projection = json.loads(prompt.strip().splitlines()[-1])
        if "player_state" in projection:
            assert projection["player_state"].startswith(("首次快照：", "更新："))
        assert "hp_total" not in prompt
        neutral = "，".join(
            fact["canonical_text"]
            for fact in projection["required_facts"]
            if fact.get("required") is True
        )
        return llm_protocol._ApiChatResult(
            json.dumps({"neutral": neutral}, ensure_ascii=False),
            scope="llma",
            source_run_id=f"run-{len(prompts)}",
            request_payload={"messages": [{"role": "user", "content": prompt}]},
            log_ctx=log_ctx,
        )

    import sbmachine.llma_api as llma_api

    monkeypatch.setattr(llma_api, "generate", fake_generate)

    manifest = run_phase3a(
        rounds_path=rounds_path,
        output_path=output_path,
        config_path=config_path,
    )

    scenes = manifest["rounds"][0]["scenes"]
    assert len(scenes) == len(expected_windows)
    non_silence = [
        scene
        for scene in scenes
        if scene["commentary_plan"]["main_topic"]["kind"] != "silence"
    ]
    intentional_empty = [
        scene for scene in scenes if scene["neutral_source"] == "intentional_empty"
    ]
    assert len(prompts) == len(non_silence)
    assert all(scene["neutral"] for scene in non_silence)
    assert all(scene["neutral"] == "" for scene in intentional_empty)
    assert [scene["scene"] for scene in scenes] == [window.scene for window in expected_contexts]
    assert all("vlm_descs" not in prompt and '"what"' not in prompt for prompt in prompts)
    assert all("pov_player" not in prompt and "context_frames" not in prompt and "state_block" not in prompt for prompt in prompts)
    assert all("hp_total" not in prompt for prompt in prompts)
    archived_input = json.loads((tmp_path / "llma_input.json").read_text(encoding="utf-8"))
    archived_text = json.dumps(archived_input, ensure_ascii=False)
    assert "pov_player" not in archived_text
    assert "context_frames" not in archived_text
    assert "state_block" not in archived_text
    assert "hp_total" not in archived_text
    assert "commentary_plan" not in archived_text
    training_lines = next(training_dir.glob("api_training_*.jsonl")).read_text(encoding="utf-8").splitlines()
    assert len(training_lines) == len(non_silence)

    styled = run_phase3b(
        neutral_path=output_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
        dry_run=True,
    )
    assert styled["rounds"][0]["round_no"] == 1

    for training_log in training_dir.glob("api_training_*.jsonl"):
        training_log.unlink()
    monkeypatch.setattr(
        phase3a_analyst,
        "write_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("final write failed")),
    )
    with pytest.raises(OSError, match="final write failed"):
        run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)
    assert list(training_dir.glob("api_training_*.jsonl")) == []


def test_phase3a_dry_run_returns_report_without_writing_publish_artifacts(tmp_path):
    """dry-run 可检查规则窗口，但不得产生可发布 neutral 或 llma_input。"""
    frames = [_frame(0.0), _frame(10.0)]
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps(
            {"video_path": "test.mp4", "map_name": "de_test", "rounds": [_round_record(1, frames)]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    semantic_path.write_text(
        json.dumps([{"round_no": 1, "frames": frames}], ensure_ascii=False),
        encoding="utf-8",
    )
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

    report = run_phase3a(
        rounds_path=rounds_path,
        output_path=output_path,
        config_path=config_path,
        dry_run=True,
    )

    assert report["mode"] == "phase3a_dry_run"
    assert report["writes_performed"] is False
    assert report["publish_path"] is None
    assert report["rounds"] == 1
    assert report["windows"] >= 1
    assert report["fallback_windows"] == 0
    assert not output_path.exists()
    assert not (tmp_path / "llma_input.json").exists()
