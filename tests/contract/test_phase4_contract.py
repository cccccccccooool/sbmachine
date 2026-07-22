from tests.contracts import validate_final_manifest


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
