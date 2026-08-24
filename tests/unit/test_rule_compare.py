from sbmachine.commentary_planner import PlannerState, plan_window
from sbmachine.rule_compare import compare_kill, compare_round_context, pov_role
from sbmachine.scene_context import SceneWindow


def test_air_noscope_uses_direct_kill_snapshot_fields():
    result = compare_kill(
        {
            "event_tick": 100,
            "attacker": "A",
            "victim": "B",
            "weapon": "AWP",
            "killer_airborne": True,
            "no_scope": True,
        }
    )

    assert result["primary"] == "air_noscope"
    assert result["tags"][:3] == ["air_noscope", "jump_kill", "no_scope"]


def test_rules_fail_closed_when_incremental_fields_are_absent():
    result = compare_kill(
        {"event_tick": 100, "attacker": "A", "victim": "B", "weapon": "AK-47"}
    )

    assert result == {"tags": [], "primary": None, "confidence": 0.0, "evidence": {}}


def test_cross_tick_evidence_detects_movement_flick_and_single_fire():
    snapshots = [
        {"tick": 80, "event_tick": 100, "name": "A", "x": 0, "y": 0, "yaw": 0},
        {"tick": 96, "event_tick": 100, "name": "A", "x": 80, "y": 0, "yaw": 10},
    ]
    fires = [{"tick": 99, "shooter": "A", "weapon": "AK-47"}]

    result = compare_kill(
        {
            "event_tick": 100,
            "attacker": "A",
            "victim": "B",
            "weapon": "AK-47",
            "killer_yaw": 70,
        },
        snapshots=snapshots,
        fires=fires,
        tick_rate=64,
    )

    assert {"moving_kill", "flick_shot", "one_tap"}.issubset(result["tags"])


def test_pov_role_never_promotes_missing_pov_to_protagonist():
    kill = {"attacker": "A", "victim": "B"}

    assert pov_role(kill, "A") == "killer"
    assert pov_role(kill, "B") == "victim"
    assert pov_role(kill, "C") == "observer"
    assert pov_role(kill, "") == "unavailable"


def test_round_context_uses_score_before_and_exact_spend_without_calling_it_equipment():
    players = [
        {"name": f"ct{i}", "side": "CT", "money_spent_this_round": 1000}
        for i in range(3)
    ] + [
        {"name": f"t{i}", "side": "T", "money_spent_this_round": 5000}
        for i in range(3)
    ]
    context = compare_round_context(
        [
            {
                "when": {"video_time": 1.0, "score_before": {"ct": 12, "t": 11}},
                "where": {"players": players},
            }
        ]
    )

    assert context["score_before"] == {"ct": 12, "t": 11}
    assert context["team_money_spent"] == {"t": 15000, "ct": 3000}
    assert context["lower_spend_side"] == "CT"
    assert context["tags"] == ["ct_match_point", "spend_gap"]


def test_planner_keeps_victim_pov_as_sentence_protagonist():
    frame = {
        "when": {"video_time": 5.0, "relative_sec": 5.0, "tick_rate": 64},
        "who": {"pov_player": "B", "view": "player"},
        "where": {
            "players": [
                {"name": "A", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 0},
                {"name": "B", "side": "T", "hp": 0, "x": 100, "y": 0, "z": 0},
                {"name": "C", "side": "T", "hp": 100, "x": 200, "y": 0, "z": 0},
            ]
        },
        "events": {
            "kills": [
                {
                    "tick": 320,
                    "attacker": "A",
                    "victim": "B",
                    "weapon": "P2000",
                    "attacker_blind": True,
                }
            ]
        },
    }
    window = SceneWindow(4.0, 6.0, "未下包", 4.0, 6.0)

    plan = plan_window("de_missing", window, [frame], [frame], [frame], PlannerState())

    assert plan["main_topic"]["summary"].startswith("B被")
    assert plan["selected_actions"][0]["pov_role"] == "victim"
    assert plan["event_ledger"][0]["pov_relation"] == "on_pov"
