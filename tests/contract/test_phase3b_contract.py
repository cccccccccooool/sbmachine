from tests.contracts import validate_commentary


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
