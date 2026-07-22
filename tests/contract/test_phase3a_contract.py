from tests.contracts import validate_neutral


def test_phase3a_fixture_matches_contract(load_fixture):
    payload = load_fixture("phase3/rounds_with_neutral.json")

    assert validate_neutral(payload) == []


def test_phase3a_contract_allows_intentional_silence(load_fixture):
    payload = load_fixture("phase3/rounds_with_neutral.json")
    payload["rounds"][0]["scenes"][0]["neutral"] = ""

    assert validate_neutral(payload) == []
