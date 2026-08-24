from types import SimpleNamespace

from sbmachine.phase3a_payload import _normalize_planning_frames, _semantic_payload, _slim_frame_for_prompt


def test_planning_payload_keeps_coordinates_but_prompt_payload_drops_them():
    frame = SimpleNamespace(
        background_info={
            "where": {"players": [{"name": "p1", "side": "T", "x": 10, "y": 20, "z": 30}]},
            "when": {"video_time": 1.0, "phase": "in_round"},
        },
        has_frame=False,
    )
    round_record = SimpleNamespace(
        round_no=1,
        start_sec=0.0,
        end_sec=10.0,
        demo_round_hint=1,
        phase2_yolo=SimpleNamespace(key_frames=[frame]),
    )

    planning_frame = _semantic_payload(round_record)["keyframes"][0]
    assert planning_frame["where"]["players"][0]["x"] == 10
    assert planning_frame["where"]["players"][0]["z"] == 30
    prompt_frame = _slim_frame_for_prompt(planning_frame)
    assert "players" not in prompt_frame
    assert "events" not in prompt_frame

def test_payload_marks_same_victim_after_first_death_as_corpse_shoot():
    frames = [
        {"events": {"kills": [{"tick": 1, "attacker": "A", "victim": "B"}]}},
        {"events": {"kills": [{"tick": 2, "attacker": "C", "victim": "B"}]}},
    ]

    cleaned = _normalize_planning_frames(frames)

    assert "is_corpse_shoot" not in cleaned[0]["events"]["kills"][0]
    assert cleaned[1]["events"]["kills"][0]["is_corpse_shoot"] is True


def test_payload_uses_external_semantic_frames_when_provided():
    round_record = SimpleNamespace(
        round_no=1,
        start_sec=0.0,
        end_sec=10.0,
        demo_round_hint=1,
        phase2_yolo=SimpleNamespace(key_frames=[]),
    )

    payload = _semantic_payload(
        round_record,
        external_frames=[
            {"when": {"video_time": 3.0, "phase": "in_round"}, "events": {"kills": []}}
        ],
    )

    assert len(payload["keyframes"]) == 1
    assert payload["keyframes"][0]["when"]["video_time"] == 3.0
    assert payload["keyframes"][0]["has_frame"] is True
