import json

from sbmachine.phase3a_analyst import run_phase3a


def _frame(t, events=None):
    data = {"when": {"video_time": t, "relative_sec": t, "phase": "in_round"}, "who": {"view": "player", "pov_player": "p1"}}
    if events:
        data["events"] = events
    return data


def _record(frames):
    return {"round_no": 1, "start_sec": 6.0, "end_sec": 10.0, "score_before": {"ct": 0, "t": 0}, "score_after": {"ct": 0, "t": 1}, "demo_round_hint": 14, "_phase2_yolo": {"key_frames": [{"time_sec": f["when"]["video_time"], "gate_reason": "test", "background_info": f, "has_frame": True} for f in frames]}}


def _payload(prompt):
    return json.loads(prompt[prompt.rfind("\n{") + 1:])


def test_run_phase3a_uses_demo_stem_and_hint_and_warns_on_video_round_key_conflict(tmp_path, monkeypatch, capsys):
    frames = [_frame(6.0, {"kills": [{"attacker": "p1", "victim": "p2", "weapon": "ak47", "tick": 100}]}), _frame(10.0)]
    rounds_path, semantic_path, output_path, config_path = (tmp_path / name for name in ("rounds.json", "semantic.json", "output.json", "config.yaml"))
    demo_path = tmp_path / "gamerlegion-vs-big-m2-ancient.dem"
    notes_path = tmp_path / "database" / "match_notes" / "gamerlegion-vs-big-m2-ancient.json"
    notes_path.parent.mkdir(parents=True)
    notes_path.write_text(json.dumps({"rounds": {"14": {"note": "正确笔记", "tactic_id": None}, "1": {"note": "错误视频局笔记", "tactic_id": None}}}, ensure_ascii=False), encoding="utf-8")
    demo_path.write_text("", encoding="utf-8")
    rounds_path.write_text(json.dumps({"video_path": "test.mp4", "map_name": "de_test", "rounds": [_record(frames)]}), encoding="utf-8")
    semantic_path.write_text(json.dumps([{"round_no": 1, "frames": frames}]), encoding="utf-8")
    config_path.write_text(f'''llm:\n  backend: vllm\nsemantic:\n  analyst_backend: vllm\n  analyst_model: qwen3\n  analyst_concurrent_rounds: 1\npaths:\n  rounds_with_yolo_semantic_json: "{semantic_path.as_posix()}"\n  demo: "{demo_path.as_posix()}"\n''', encoding="utf-8")
    import sbmachine.llma_api as llma_api
    import sbmachine.manual_notes as manual_notes
    monkeypatch.setattr(manual_notes, "_PROJECT_ROOT", tmp_path)
    prompts = []

    def fake_generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        projection = json.loads(prompt.strip().splitlines()[-1])
        neutral = "，".join(
            fact["canonical_text"]
            for fact in projection["required_facts"]
            if fact.get("required") is True
        )
        return json.dumps({"neutral": neutral}, ensure_ascii=False)

    monkeypatch.setattr(llma_api, "generate", fake_generate)
    run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    assert "正确笔记" not in prompts[0]
    assert "错误视频局笔记" not in prompts[0]
    assert "人工注" not in prompts[0]
