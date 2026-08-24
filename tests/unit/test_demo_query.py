import shutil

import pytest

from sbmachine.demo_query import DemoQuery
from tests.contracts import validate_demo_artifacts
from tools.demo.demo_manifest import DemoManifestError


def test_demo_query_skeleton_fixture_has_tick_boundaries(load_fixture):
    payload = {
        "rounds": load_fixture("demo/rounds.json"),
        "kills": load_fixture("demo/kills.json"),
        "roster": load_fixture("demo/roster.json"),
    }

    assert validate_demo_artifacts(payload) == []
    assert payload["rounds"][0]["freeze_end_tick"] > payload["rounds"][0]["start_tick"]


def test_demo_query_accepts_manifest_declared_zero_events(fixtures_dir):
    demo = DemoQuery.load(fixtures_dir / "demo")

    assert demo.damages == []
    assert demo.manifest["files"]["damages.json"]["rows"] == 0


def test_demo_query_rejects_a_hash_mismatch(fixtures_dir, tmp_path):
    parsed_dir = tmp_path / "demo"
    shutil.copytree(fixtures_dir / "demo", parsed_dir)
    (parsed_dir / "kills.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(DemoManifestError, match="hash mismatch"):
        DemoQuery.load(parsed_dir)


def test_demo_query_rejects_a_missing_manifest(tmp_path):
    with pytest.raises(DemoManifestError, match="missing demo_manifest.json"):
        DemoQuery.load(tmp_path)


def test_state_at_stays_in_round_and_honors_max_distance(fixtures_dir):
    pytest.importorskip("pandas")
    demo = DemoQuery.load(fixtures_dir / "demo")

    assert {row["name"] for row in demo.state_at(650, round_no=1, max_distance_ticks=64)} == {
        "alpha",
        "bravo",
    }
    assert demo.state_at(800, round_no=1, max_distance_ticks=64) == []
    assert demo.state_at(8639, round_no=1, max_distance_ticks=64) == []


def test_one_character_ocr_cannot_bind_a_player(fixtures_dir):
    demo = DemoQuery.load(fixtures_dir / "demo")

    assert demo.match_player("a").steamid == ""
    assert demo.match_player("al").steamid == "1"


def test_round_lookup_never_falls_back_to_list_position(tmp_path):
    demo = DemoQuery(tmp_path)
    demo.rounds = [{"round_no": 10}]

    with pytest.raises(IndexError, match="round_no not found"):
        demo.round_by_no(1)


def test_round_by_no_returns_first_match_on_duplicate_round_no(tmp_path):
    demo = DemoQuery(tmp_path)
    first = {"round_no": 5, "tag": "first"}
    second = {"round_no": 5, "tag": "second"}
    demo.rounds = [first, second]

    assert demo.round_by_no(5) is first


def test_round_by_no_rebuilds_when_rounds_replaced(tmp_path):
    demo = DemoQuery(tmp_path)
    demo.rounds = [{"round_no": 3, "tag": "old"}]
    assert demo.round_by_no(3)["tag"] == "old"

    # 替换底层列表后，旧索引必须失效并重建
    demo.rounds = [{"round_no": 7, "tag": "new"}]
    assert demo.round_by_no(7)["tag"] == "new"
    with pytest.raises(IndexError, match="round_no not found"):
        demo.round_by_no(3)
