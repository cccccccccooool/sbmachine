from tests.contracts import validate_final_manifest, validate_final_voice_task


def test_phase4_fixture_matches_contract(load_fixture):
    payload = load_fixture("phase4/rounds_final.json")

    assert validate_final_manifest(payload) == []
    item = payload["assemble_manifest"]["rounds"][0]
    assert item["aligned"] is True
    assert item["segments"] == [
        {
            "source_start_sec": 2.0,
            "source_end_sec": 5.0,
            "relative_start_sec": 2.0,
            "audio_duration_sec": 1.5,
            "audio_path": "rounds/round_001_scene_001.wav",
        }
    ]


def _v4_final_scene(payload, index):
    return payload["rounds_final"]["rounds"][0]["scenes"][index]


def test_phase4_final_voice_task_fixture_matches_contract(load_fixture):
    payload = load_fixture("phase4/final_voice_task.json")

    assert validate_final_voice_task(payload) == []


def test_phase4_final_rejects_selected_variant_not_in_task_candidates(load_fixture):
    payload = load_fixture("phase4/final_voice_task.json")
    commentary = load_fixture("phase3/commentary_v3.json")
    _v4_final_scene(payload, 0)["selected_variant_id"] = "bogus_variant"

    assert any(
        "selected_variant_id is not among" in error
        for error in validate_final_voice_task(payload, commentary=commentary)
    )


def test_phase4_final_rejects_fit_without_audio_fields(load_fixture):
    payload = load_fixture("phase4/final_voice_task.json")
    del _v4_final_scene(payload, 0)["audio_start_tick"]

    assert any("audio_start_tick" in error for error in validate_final_voice_task(payload))


def test_phase4_final_rejects_audio_end_before_audio_start(load_fixture):
    payload = load_fixture("phase4/final_voice_task.json")
    scene = _v4_final_scene(payload, 0)
    scene["audio_end_tick"] = 300

    assert any("audio_end_tick >= audio_start_tick" in error for error in validate_final_voice_task(payload))


def test_phase4_final_rejects_missing_fit_state(load_fixture):
    payload = load_fixture("phase4/final_voice_task.json")
    del _v4_final_scene(payload, 1)["fit_state"]

    assert any("fit_state" in error for error in validate_final_voice_task(payload))
