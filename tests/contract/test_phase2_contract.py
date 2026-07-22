from tests.contracts import validate_vision_timeline


def test_phase2_fixture_matches_contract(load_fixture):
    payload = load_fixture("phase2/rounds_with_yolo.json")

    assert validate_vision_timeline(payload) == []
