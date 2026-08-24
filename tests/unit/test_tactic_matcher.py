import json

from sbmachine.tactic_book import compile_tactic_book, load_tactic_book
from sbmachine.tactic_matcher import match_window


def _frame(time, players, utilities=None):
    return {
        "when": {"video_time": time, "relative_sec": time, "phase": "in_round"},
        "where": {"players": players},
        "events": {"utilities": utilities or []},
    }


def _player(name, side, callout):
    return {"name": name, "side": side, "callout": callout, "hp": 100}


def _fake_a_rule():
    return {
        "version": 1,
        "map": "de_test",
        "tactics": [
            {
                "id": "fake_a_hit_b",
                "label": "假爆A真打B",
                "hint": "A小一人道具牵制，B区三人集结。",
                "side": "T",
                "when": [
                    {"kind": "zone_count", "side": "T", "zone": {"callouts_any": ["A_Short"]}, "count": [1, 1]},
                    {"kind": "zone_count", "side": "T", "zone": {"callouts_any": ["B_Main"]}, "count": [3, 5]},
                    {
                        "kind": "event_count",
                        "event": "utility_throw",
                        "actor_side": "T",
                        "actor_zone": {"callouts_any": ["A_Short"]},
                        "types_any": ["Smoke Grenade", "Flashbang"],
                        "window_sec": 6,
                        "count": [2, None],
                    },
                ],
                "priority": 10,
            }
        ],
    }


def test_match_window_labels_fake_a_when_a_lurker_throws_and_three_t_are_in_b():
    book = compile_tactic_book("de_test", _fake_a_rule())
    players = [
        _player("a_lurker", "T", "A_Short"),
        _player("b1", "T", "B_Main"),
        _player("b2", "T", "B_Main"),
        _player("b3", "T", "B_Main"),
    ]
    frames = [
        _frame(10.0, players, [{"_event": "throw", "type": "Smoke Grenade", "thrower": "a_lurker", "throw_tick": 10}]),
        _frame(12.0, players, [{"_event": "throw", "type": "Flashbang", "thrower": "a_lurker", "throw_tick": 12}]),
    ]

    match = match_window(book, frames, context_frames=frames, scene="未下包")

    assert match is not None
    assert match.rule_id == "fake_a_hit_b"
    assert match.label == "假爆A真打B"
    assert match.hint == "A小一人道具牵制，B区三人集结。"
    assert match.matched_at == 12.0
    assert match.to_prompt_payload() == {
        "rule_id": "fake_a_hit_b",
        "label": "\u5047\u7206A\u771f\u6253B",
        "hint": "A小一人道具牵制，B区三人集结。",
        "matched_at": 12.0,
    }


def test_match_window_labels_t_mid_four_without_action_requirement():
    book = compile_tactic_book(
        "de_test",
        {
            "version": 1,
            "map": "de_test",
            "tactics": [
                {
                    "id": "t_mid_stack_retake",
                    "label": "中路摆谱中期反清",
                    "side": "T",
                    "when": [
                        {
                            "kind": "zone_count",
                            "side": "T",
                            "zone": {"callouts_any": ["Mid", "TopMid", "BottomMid"]},
                            "count": [4, 5],
                        }
                    ],
                    "priority": 5,
                }
            ],
        },
    )
    frames = [_frame(20.0, [_player(f"t{i}", "T", "Mid") for i in range(4)])]

    match = match_window(book, frames, context_frames=frames, scene="未下包")

    assert match is not None
    assert match.label == "中路摆谱中期反清"
    assert match.hint == "中路摆谱中期反清"


def test_match_window_does_not_count_utility_from_future_frame():
    book = compile_tactic_book("de_test", _fake_a_rule())
    players = [
        _player("a_lurker", "T", "A_Short"),
        _player("b1", "T", "B_Main"),
        _player("b2", "T", "B_Main"),
        _player("b3", "T", "B_Main"),
    ]
    ownership = [_frame(10.0, players, [{"_event": "throw", "type": "Smoke Grenade", "thrower": "a_lurker", "throw_tick": 10}])]
    future = _frame(11.0, players, [{"_event": "throw", "type": "Flashbang", "thrower": "a_lurker", "throw_tick": 11}])

    assert match_window(book, ownership, context_frames=ownership + [future], scene="未下包") is None


def test_match_window_does_not_count_the_same_utility_throw_in_two_frames_twice():
    book = compile_tactic_book("de_test", _fake_a_rule())
    players = [
        _player("a_lurker", "T", "A_Short"),
        _player("b1", "T", "B_Main"),
        _player("b2", "T", "B_Main"),
        _player("b3", "T", "B_Main"),
    ]
    repeated_throw = {
        "_event": "throw",
        "entity_id": 42,
        "throw_tick": 120,
        "thrower": "a_lurker",
        "type": "Smoke Grenade",
    }
    frames = [_frame(10.0, players, [repeated_throw]), _frame(11.0, players, [repeated_throw])]

    assert match_window(book, frames, context_frames=frames, scene="unscoped") is None

def test_match_window_fails_closed_for_malformed_frame_shapes():
    book = compile_tactic_book("de_test", _fake_a_rule())
    malformed_frames = [
        None,
        {"when": [], "where": {}, "events": {}},
        {"when": {}, "where": [], "events": {}},
        {"when": {}, "where": {"players": "not-a-list"}, "events": []},
    ]

    assert match_window(book, malformed_frames, context_frames=malformed_frames, scene="unscoped") is None


def test_match_window_fails_closed_for_missing_or_non_string_callouts():
    source = _fake_a_rule()
    source["tactics"][0]["when"] = [
        {"kind": "zone_count", "side": "T", "zone": {"callouts_any": ["A_Short"]}, "count": [0, 0]}
    ]
    book = compile_tactic_book("de_test", source)
    frames = [
        _frame(10.0, [{"name": "missing", "side": "T", "hp": 100}]),
        _frame(11.0, [{"name": "invalid", "side": "T", "hp": 100, "callout": {}}]),
    ]

    assert match_window(book, frames, context_frames=frames, scene="unscoped") is None

def test_match_window_returns_silence_for_same_priority_matches():
    source = _fake_a_rule()
    duplicate = dict(source["tactics"][0])
    duplicate["id"] = "same_priority"
    source["tactics"].append(duplicate)
    book = compile_tactic_book("de_test", source)
    players = [
        _player("a_lurker", "T", "A_Short"),
        _player("b1", "T", "B_Main"),
        _player("b2", "T", "B_Main"),
        _player("b3", "T", "B_Main"),
    ]
    frames = [
        _frame(10.0, players, [{"_event": "throw", "type": "Smoke Grenade", "thrower": "a_lurker", "throw_tick": 10}]),
        _frame(12.0, players, [{"_event": "throw", "type": "Flashbang", "thrower": "a_lurker", "throw_tick": 12}]),
    ]

    assert match_window(book, frames, context_frames=frames, scene="未下包") is None


def test_match_window_silences_only_while_a_rule_remains_active():
    book = compile_tactic_book("de_test", _fake_a_rule())
    players = [
        _player("a_lurker", "T", "A_Short"),
        _player("b1", "T", "B_Main"),
        _player("b2", "T", "B_Main"),
        _player("b3", "T", "B_Main"),
    ]
    frames = [
        _frame(10.0, players, [{"_event": "throw", "type": "Smoke Grenade", "thrower": "a_lurker", "throw_tick": 10}]),
        _frame(12.0, players, [{"_event": "throw", "type": "Flashbang", "thrower": "a_lurker", "throw_tick": 12}]),
    ]
    active_rule_ids = set()

    first = match_window(book, frames, context_frames=frames, scene="未下包", active_rule_ids=active_rule_ids)
    assert first is not None
    assert active_rule_ids == {"fake_a_hit_b"}
    assert match_window(book, frames, context_frames=frames, scene="未下包", active_rule_ids=active_rule_ids) is None

    inactive = [_frame(14.0, players[:2])]
    assert match_window(book, inactive, context_frames=inactive, scene="未下包", active_rule_ids=active_rule_ids) is None
    assert active_rule_ids == set()

    again = match_window(book, frames, context_frames=frames, scene="未下包", active_rule_ids=active_rule_ids)
    assert again is not None


def test_compile_tactic_book_fails_closed_for_unknown_top_level_fields():
    source = _fake_a_rule()
    source["unexpected"] = True

    assert compile_tactic_book("de_test", source).tactics == ()


def test_load_tactic_book_returns_empty_for_missing_and_invalid_sources(tmp_path):
    assert load_tactic_book("de_missing", database_root=tmp_path).tactics == ()

    tactics_dir = tmp_path / "tactics"
    tactics_dir.mkdir()
    (tactics_dir / "de_bad.json").write_text(json.dumps({"version": 99, "tactics": []}), encoding="utf-8")

    assert load_tactic_book("de_bad", database_root=tmp_path).tactics == ()
