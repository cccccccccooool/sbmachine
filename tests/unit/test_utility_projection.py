from sbmachine.scene_context import extract_actions
from sbmachine.utility_projection import project_grenade, project_grenades


def test_projection_prefers_parser_stable_id_and_emits_one_throw():
    row = {
        "stable_event_id": "1|100|steam|Smoke Grenade|1",
        "round_no": 1,
        "thrower": "A",
        "type": "Smoke Grenade",
        "throw_tick": 100,
        "det_tick": 500,
        "dest_x": 10,
        "dest_y": 20,
        "dest_z": 30,
    }

    event = project_grenade(row)

    assert event["_event"] == "throw"
    assert event["stable_event_id"] == row["stable_event_id"]
    assert event["event_id"] == f"utility_throw:{row['stable_event_id']}"
    assert event["kind"] == "smoke"
    assert event["landing_xyz"] == [10, 20, 30]


def test_projection_uses_smoke_start_tick_instead_of_late_raw_detonation():
    grenades = [
        {
            "stable_event_id": "1|100|steam|Smoke Grenade|1",
            "round_no": 1,
            "thrower": "A",
            "type": "Smoke Grenade",
            "throw_tick": 100,
            "det_tick": 900,
            "dest_x": 1000.0,
            "dest_y": 2000.0,
        }
    ]
    smokes = [
        {
            "round_no": 1,
            "thrower": "A",
            "start_tick": 220,
            "pos_x": 1000.1,
            "pos_y": 1999.9,
        }
    ]

    projected = project_grenades(grenades, smokes=smokes)

    assert len(projected) == 1
    assert projected[0]["effect_tick"] == 220
    assert projected[0]["effect_source"] == "smokes.json"


def test_legacy_projection_ids_are_deterministic_and_collision_free():
    row = {"round_no": 1, "thrower": "A", "type": "Flashbang", "throw_tick": 100}

    first = project_grenades([row, dict(row)])
    second = project_grenades([row, dict(row)])

    assert [item["stable_event_id"] for item in first] == [
        item["stable_event_id"] for item in second
    ]
    assert first[0]["stable_event_id"] != first[1]["stable_event_id"]


def test_projected_smoke_suppresses_duplicate_active_effect_action():
    frames = [
        {
            "when": {"video_time": 3.0, "tick_rate": 64},
            "who": {"pov_player": "A"},
            "where": {"players": []},
            "events": {
                "utilities": [
                    {
                        "_event": "throw",
                        "stable_event_id": "1|100|steam|Smoke Grenade|1",
                        "type": "Smoke Grenade",
                        "kind": "smoke",
                        "thrower": "A",
                        "throw_tick": 100,
                        "effect_tick": 220,
                    }
                ],
                "smokes_active": [
                    {"entity_id": 9, "thrower": "A", "start_tick": 220}
                ],
            },
        }
    ]

    actions = extract_actions(frames, 0.0, 4.0)

    assert [action["type"] for action in actions] == ["utility_throw"]

