from tests.contracts import validate_commentary, validate_commentary_v3


def _v3_task(payload, voice_task_id):
    return next(task for task in payload["voice_tasks"] if task["voice_task_id"] == voice_task_id)


def test_phase3b_fixture_matches_contract(load_fixture):
    payload = load_fixture("phase3/commentary.json")

    assert validate_commentary(payload) == []


def test_phase3b_rejects_removed_emotion():
    payload = {
        "rounds": [
            {
                "commentary_text": "残局",
                "emotion_segments": [{"emotion": "紧张", "text": "残局"}],
            }
        ]
    }

    assert validate_commentary(payload) == [
        "phase3b.rounds[0].emotion_segments[0].emotion has unsupported value"
    ]

def test_phase3b_allows_intentional_silence():
    payload = {
        "rounds": [
            {"commentary_text": "", "emotion_segments": []},
        ]
    }

    assert validate_commentary(payload) == []


def test_phase3b_v3_fixture_matches_contract(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")

    assert validate_commentary_v3(payload) == []


def test_phase3b_v3_amber_task_is_valid(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")

    amber = _v3_task(payload, "r001_w03")
    assert amber["risk_class"] == "amber"
    assert [candidate["variant_id"] for candidate in amber["candidates"]] == ["primary", "compact", "capsule"]


def test_phase3b_v3_green_task_is_valid(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")

    green = _v3_task(payload, "r001_w02")
    assert green["risk_class"] == "green"
    assert [candidate["variant_id"] for candidate in green["candidates"]] == ["primary", "capsule"]


def test_phase3b_v3_red_task_is_valid(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")
    amber = _v3_task(payload, "r001_w03")
    amber["risk_class"] = "red"
    amber["selection_order"] = ["compact", "capsule"]
    amber["candidates"] = [candidate for candidate in amber["candidates"] if candidate["variant_id"] != "primary"]

    assert validate_commentary_v3(payload) == []


def test_phase3b_v3_rejects_preserved_fact_ids_not_covering_required(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")
    candidate = _v3_task(payload, "r001_w03")["candidates"][0]
    candidate["preserved_fact_ids"] = candidate["preserved_fact_ids"][:1]

    assert any("preserved_fact_ids must cover" in error for error in validate_commentary_v3(payload))


def test_phase3b_v3_rejects_green_with_compact(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")
    green = _v3_task(payload, "r001_w02")
    compact = _v3_task(payload, "r001_w03")["candidates"][1]
    green["candidates"].append(compact)
    green["selection_order"].append("compact")

    assert any("green risk_class" in error for error in validate_commentary_v3(payload))


def test_phase3b_v3_rejects_red_with_primary(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")
    amber = _v3_task(payload, "r001_w03")
    amber["risk_class"] = "red"

    assert any("red risk_class" in error for error in validate_commentary_v3(payload))


def test_phase3b_v3_rejects_more_than_three_candidates(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")
    task = _v3_task(payload, "r001_w03")
    task["candidates"].append(dict(task["candidates"][0]))

    assert any("must not exceed 3" in error for error in validate_commentary_v3(payload))


def test_phase3b_v3_rejects_missing_voice_task_id(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")
    del _v3_task(payload, "r001_w03")["voice_task_id"]

    assert any("voice_task_id" in error for error in validate_commentary_v3(payload))


def test_phase3b_v3_rejects_missing_source_neutral_sha256(load_fixture):
    payload = load_fixture("phase3/commentary_v3.json")
    del payload["source_neutral_sha256"]

    assert any("source_neutral_sha256" in error for error in validate_commentary_v3(payload))
