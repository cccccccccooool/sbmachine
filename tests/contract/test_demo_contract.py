from tests.contracts import validate_demo_artifacts
from tools.demo.demo_manifest import validate_demo_manifest


def test_demo_fixture_matches_contract(load_fixture):
    payload = {
        "rounds": load_fixture("demo/rounds.json"),
        "kills": load_fixture("demo/kills.json"),
        "grenades": load_fixture("demo/grenades.json"),
        "roster": load_fixture("demo/roster.json"),
    }

    assert validate_demo_artifacts(payload) == []


def test_demo_fixture_has_a_complete_integrity_manifest(fixtures_dir):
    manifest = validate_demo_manifest(fixtures_dir / "demo")

    assert manifest["status"] == "complete"
    assert manifest["files"]["damages.json"]["rows"] == 0
