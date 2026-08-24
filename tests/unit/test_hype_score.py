from tests.contracts import validate_neutral

from sbmachine.hype_score import _scene_hype, _scene_scream_eligible, compute_hype


def test_hype_score_skeleton_requires_scene_hype_shape(load_fixture):
    payload = load_fixture("phase3/rounds_with_neutral.json")

    assert validate_neutral(payload) == []
    assert payload["rounds"][0]["scenes"][1]["hype"] > payload["rounds"][0]["scenes"][0]["hype"]


def _beat(t: float, events: dict | None = None) -> dict:
    return {"when": {"video_time": t, "round_no": 1}, "events": events or {}}


def test_hype_does_not_leak_future_kills_into_quiet_frames():
    beats = [
        _beat(0.0),
        _beat(5.0),
        _beat(10.0, {"kills": [{"attacker": "A", "victim": "B"}]}),
    ]

    scores = compute_hype(beats)

    assert scores[:2] == [0.0, 0.0]
    assert scores[2] > 0.0


def test_hype_cross_attacker_kills_gain_exchange_priority():
    beats = [
        _beat(0.0, {"kills": [{"tick": 1, "attacker": "A", "victim": "X"}]}),
        _beat(1.0, {"kills": [{"tick": 2, "attacker": "B", "victim": "Y"}]}),
        _beat(2.0, {"kills": [{"tick": 3, "attacker": "C", "victim": "Z"}]}),
    ]

    assert compute_hype(beats) == [0.4, 0.72, 0.9]


def test_hype_ignores_repeated_or_corpse_kill_events():
    repeated = {"tick": 1, "attacker": "A", "victim": "X"}
    beats = [
        _beat(0.0, {"kills": [repeated]}),
        _beat(1.0, {"kills": [repeated]}),
        _beat(2.0, {"kills": [{"tick": 2, "attacker": "A", "victim": "X", "is_corpse_shoot": True}]}),
    ]

    assert compute_hype(beats)[-1] < 0.4

def test_scene_hype_uses_peak_hard_fact_instead_of_diluting_single_kill():
    beats = [_beat(0.0), _beat(1.0), _beat(2.0)]
    scores = [0.0, 0.4, 0.2]

    assert _scene_hype(beats, scores, 0.0, 3.0) == 0.4


def test_scene_scream_requires_rapid_three_kill_hard_fact():
    beats = [
        _beat(1.0, {"kills": [{"attacker": "A", "victim": "B"}]}),
        _beat(4.0, {"kills": [{"attacker": "A", "victim": "C"}]}),
        _beat(7.0, {"kills": [{"attacker": "A", "victim": "D"}]}),
    ]

    assert _scene_scream_eligible(beats, 7.0, 8.0) is True
    assert _scene_scream_eligible(beats[:2], 4.0, 5.0) is False


def _beat_with_players(t: float, players: list, events: dict | None = None) -> dict:
    return {
        "when": {"video_time": t, "round_no": 1},
        "where": {"players": players},
        "events": events or {},
    }


def test_hype_clutch_adds_score_when_man_disadvantage_post_plant():
    players = [
        {"name": "t1", "side": "T", "hp": 80},
        {"name": "t2", "side": "T", "hp": 40},
        {"name": "ct1", "side": "CT", "hp": 100},
    ]
    beats = [
        _beat_with_players(
            5.0,
            players,
            {"c4": {"planted": True, "plant_tick": 100, "begin_defuse_tick": 0}},
        ),
    ]

    scores = compute_hype(beats)

    assert scores[0] == 0.6  # 2v1 残局 + 下包，clutch 分

def test_hype_no_clutch_when_even_or_overwhelming_numbers():
    even = [
        {"name": "t1", "side": "T", "hp": 80},
        {"name": "t2", "side": "T", "hp": 80},
        {"name": "t3", "side": "T", "hp": 80},
        {"name": "ct1", "side": "CT", "hp": 100},
        {"name": "ct2", "side": "CT", "hp": 100},
        {"name": "ct3", "side": "CT", "hp": 100},
    ]
    beats = [
        _beat_with_players(5.0, even, {"c4": {"planted": True, "plant_tick": 100, "begin_defuse_tick": 0}}),
    ]

    # 3v3 均匀，无残局加分；但下包 base 0.4 仍在
    assert compute_hype(beats)[0] == 0.4
