import pytest

from sbmachine.neutral_contract import CLOUD_PHASE3A_MODE, new_manifest_metadata, validate_neutral_manifest

_EMPTY_ANCHORS = {key: [] for key in ("players", "teams", "numbers", "events", "results", "locations", "weapons")}


def test_neutral_manifest_requires_current_mode_and_source_fingerprint(tmp_path):
    rounds = tmp_path / "rounds_with_yolo.json"
    rounds.write_text('{"rounds": []}', encoding="utf-8")
    payload = {**new_manifest_metadata(rounds), "rounds": []}
    assert payload["schema_version"] == 3
    assert validate_neutral_manifest(payload, rounds) is payload


def test_neutral_manifest_accepts_cloud_round_timeline_mode(tmp_path):
    rounds = tmp_path / "rounds_with_yolo.json"
    rounds.write_text('{"rounds": []}', encoding="utf-8")
    payload = {**new_manifest_metadata(rounds), "phase3a_mode": CLOUD_PHASE3A_MODE, "rounds": []}
    assert validate_neutral_manifest(payload, rounds) is payload


def test_neutral_manifest_rejects_legacy_or_cross_input_artifacts(tmp_path):
    rounds = tmp_path / "rounds_with_yolo.json"
    rounds.write_text('{"rounds": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        validate_neutral_manifest({"rounds": []}, rounds)
    payload = {**new_manifest_metadata(rounds), "rounds": []}
    rounds.write_text('{"rounds": [{"round_no": 1}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_neutral_manifest(payload, rounds)

def test_phase3b_rejects_legacy_neutral_before_model_call(tmp_path):
    from sbmachine.phase3b_style import run_phase3b

    rounds = tmp_path / "rounds_with_yolo.json"
    rounds.write_text('{"video_path":"x.mp4","map_name":"de_test","rounds":[]}', encoding="utf-8")
    neutral = tmp_path / "rounds_with_neutral.json"
    neutral.write_text('{"rounds":[]}', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  backend: vllm\nsemantic:\n  style_backend: vllm\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        run_phase3b(
            neutral_path=neutral,
            rounds_path=rounds,
            output_rounds_path=tmp_path / "rounds_with_commentary.json",
            commentary_path=tmp_path / "commentary.json",
            config_path=config,
        )


@pytest.mark.parametrize(
    ("scene", "message"),
    [
        ({"t_start": 0.0, "t_end": 1.0, "scene": "默认场景", "neutral": "x", "fact_anchors": _EMPTY_ANCHORS}, "commentary_plan"),
        ({"t_start": 1.0, "t_end": 1.0, "scene": "默认场景", "commentary_plan": {}, "neutral": "x", "fact_anchors": _EMPTY_ANCHORS}, "t_start < t_end"),
        ({"t_start": 0.0, "t_end": 1.0, "scene": "", "commentary_plan": {}, "neutral": "x", "fact_anchors": _EMPTY_ANCHORS}, "non-empty"),
        ({"t_start": 0.0, "t_end": 1.0, "scene": "默认场景", "commentary_plan": [], "neutral": "x", "fact_anchors": _EMPTY_ANCHORS}, "must be an object"),
        ({"t_start": 0.0, "t_end": 1.0, "scene": "默认场景", "commentary_plan": {}, "neutral": 1, "fact_anchors": _EMPTY_ANCHORS}, "must be a string"),
    ],
)
def test_neutral_manifest_validates_runtime_scene_fields(tmp_path, scene, message):
    rounds = tmp_path / "rounds_with_yolo.json"
    rounds.write_text('{"rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 2.0}]}', encoding="utf-8")
    payload = {
        **new_manifest_metadata(rounds),
        "rounds": [{"round_no": 1, "scenes": [scene]}],
    }

    with pytest.raises(ValueError, match=message):
        validate_neutral_manifest(payload, rounds)


def _valid_scene(**overrides):
    scene = {
        "t_start": 0.0,
        "t_end": 1.0,
        "scene": "default",
        "commentary_plan": {},
        "neutral": "x",
        "fact_anchors": {key: [] for key in _EMPTY_ANCHORS},
        "hype": 0.5,
        "char_budget": 20,
    }
    scene.update(overrides)
    return scene


def test_neutral_rounds_are_unique_and_match_source_one_to_one(tmp_path):
    rounds = tmp_path / "rounds_with_yolo.json"
    rounds.write_text(
        '{"rounds": ['
        '{"round_no": 1, "start_sec": 0.0, "end_sec": 2.0},'
        '{"round_no": 2, "start_sec": 2.0, "end_sec": 4.0}'
        ']}',
        encoding="utf-8",
    )
    metadata = new_manifest_metadata(rounds)

    duplicate = {
        **metadata,
        "rounds": [
            {"round_no": 1, "scenes": [_valid_scene()]},
            {"round_no": 1, "scenes": [_valid_scene()]},
        ],
    }
    with pytest.raises(ValueError, match="duplicated"):
        validate_neutral_manifest(duplicate, rounds)

    missing = {**metadata, "rounds": [{"round_no": 1, "scenes": [_valid_scene()]}]}
    with pytest.raises(ValueError, match="one-to-one"):
        validate_neutral_manifest(missing, rounds)


@pytest.mark.parametrize(
    ("scene", "message"),
    [
        (_valid_scene(t_end=2.1), "source round range"),
        (_valid_scene(hype="high"), "hype"),
        (_valid_scene(hype=True), "hype"),
        (_valid_scene(char_budget=20.5), "char_budget"),
        (_valid_scene(char_budget=True), "char_budget"),
    ],
)
def test_neutral_scene_range_and_numeric_types_are_strict(tmp_path, scene, message):
    rounds = tmp_path / "rounds_with_yolo.json"
    rounds.write_text(
        '{"rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 2.0}]}',
        encoding="utf-8",
    )
    payload = {
        **new_manifest_metadata(rounds),
        "rounds": [{"round_no": 1, "scenes": [scene]}],
    }

    with pytest.raises(ValueError, match=message):
        validate_neutral_manifest(payload, rounds)
