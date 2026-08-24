import copy

from tests.contracts import validate_neutral, validate_neutral_v4


def _v4_active_scene(payload):
    return next(scene for scene in payload["rounds"][0]["scenes"] if scene.get("neutral"))


def test_phase3a_fixture_matches_contract(load_fixture):
    payload = load_fixture("phase3/rounds_with_neutral.json")

    assert validate_neutral(payload) == []


def test_phase3a_contract_allows_intentional_silence(load_fixture):
    payload = load_fixture("phase3/rounds_with_neutral.json")
    payload["rounds"][0]["scenes"][0]["neutral"] = ""

    assert validate_neutral(payload) == []


def test_phase3a_v4_fixture_matches_contract(load_fixture):
    payload = load_fixture("phase3/neutral_v4.json")

    assert validate_neutral_v4(payload) == []


def test_phase3a_v4_rejects_missing_rule_capsule(load_fixture):
    payload = load_fixture("phase3/neutral_v4.json")
    del _v4_active_scene(payload)["rule_capsule"]

    assert any("rule_capsule" in error for error in validate_neutral_v4(payload))


def test_phase3a_v4_rejects_missing_fact_catalog(load_fixture):
    payload = load_fixture("phase3/neutral_v4.json")
    del _v4_active_scene(payload)["fact_catalog"]

    assert any("fact_catalog" in error for error in validate_neutral_v4(payload))


def test_phase3a_v4_rejects_invalid_fact_id_format(load_fixture):
    payload = load_fixture("phase3/neutral_v4.json")
    _v4_active_scene(payload)["fact_catalog"][0]["fact_id"] = "fact:v1:r001_w03:kill:00360:zzzzzzzz"

    assert any("invalid format" in error for error in validate_neutral_v4(payload))


def test_phase3a_v4_rejects_required_fact_id_missing_from_catalog(load_fixture):
    payload = load_fixture("phase3/neutral_v4.json")
    scene = _v4_active_scene(payload)
    scene["required_fact_ids"].append("fact:v1:r001_w03:kill:00360:deadbeef")

    assert any("missing from fact_catalog" in error for error in validate_neutral_v4(payload))


def test_phase3a_v4_rejects_render_slot_tick_out_of_bounds(load_fixture):
    payload = load_fixture("phase3/neutral_v4.json")
    render_slot = _v4_active_scene(payload)["render_slot"]
    render_slot["start_tick"] = 500
    render_slot["end_tick"] = 700

    assert any("start_tick" in error for error in validate_neutral_v4(payload))


def test_phase3a_v4_rejects_missing_speech_budget(load_fixture):
    payload = load_fixture("phase3/neutral_v4.json")
    del _v4_active_scene(payload)["speech_budget"]

    assert any("speech_budget" in error for error in validate_neutral_v4(payload))


def test_phase3a_v4_rejects_missing_neutral_renderer(load_fixture):
    payload = load_fixture("phase3/neutral_v4.json")
    del _v4_active_scene(payload)["neutral_renderer"]

    assert any("neutral_renderer" in error for error in validate_neutral_v4(payload))
