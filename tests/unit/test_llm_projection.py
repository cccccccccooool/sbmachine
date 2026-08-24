import json

from sbmachine.llm_projection import (
    _project_main_topic,
    build_rule_state_delta,
    merge_required_fact_anchors,
)
from sbmachine.commentary_planner import _utility_summary
from sbmachine.common import count_spoken_chars
from sbmachine.phase3a_cloud_payload import build_cloud_round_payload
from sbmachine.phase3a_prompt import _build_window_prompt


def _unsafe_plan() -> dict:
    return {
        "main_topic": {"kind": "position", "summary": "A is holding Ramp"},
        "selected_actions": [{
            "type": "utility_throw",
            "utility": "Smoke",
            "thrower": "A",
            "actor_zone": "Ramp",
            "callout": "B_Main",
            "throw_position": {"x": 1, "y": 2, "z": 3},
            "destination": {"x": 4, "y": 5, "z": 6},
            "event_time": 12.0,
            "evidence": {"raw": "B_Main"},
        }],
        "tactic_hint": {
            "rule_id": "fake_a_hit_b",
            "label": "fake A hit B",
            "hint": "a verified diversion",
            "matched_at": 12.0,
            "evidence": {"zone": "Ramp"},
        },
        "spatial": {"anchor": {"callout": "Ramp"}},
        "ownership": {"t_start": 10.0, "t_end": 13.0},
        "read_only_context": {"t_start": 4.0, "t_end": 13.0},
    }


def test_local_prompt_reprojects_rule_plan_and_rejects_raw_state_and_frames():
    prompt = _build_window_prompt(
        {
            "commentary_plan": _unsafe_plan(),
            "context_frames": [{"where": {"players": [{"callout": "Ramp"}]}}],
            "state_block": "state change: A moved from Ramp to B_Main",
            "rule_state": {"kind": "snapshot", "teams": {
                "T": {"alive_count": 4, "hp_total": 320},
                "CT": {"alive_count": 5, "hp_total": 500},
            }, "changed_teams": ["T"]},
        },
        t_start=10.0,
        t_end=13.0,
        scene="unplanted",
        state_block="state change: A moved from Ramp to B_Main",
    )

    assert "Ramp" not in prompt
    assert "B_Main" not in prompt
    assert "context_frames" not in prompt
    assert "state_block" not in prompt
    assert '"spatial"' not in prompt
    assert '"ownership"' not in prompt
    assert '"read_only_context"' not in prompt
    assert '"evidence"' not in prompt
    assert '"t_start"' not in prompt
    assert '"t_end"' not in prompt
    prompt_payload = json.loads(prompt[prompt.rfind("\n{") + 1:])
    assert prompt_payload["projection_version"] == 2
    assert prompt_payload["rule_state"]["kind"] == "snapshot"
    assert set(prompt_payload["rule_state"]["teams"]) == {"T", "CT"}
    assert prompt_payload["rule_state"]["changed_teams"] == ["T"]
    assert prompt_payload["required_facts"]


def test_rule_state_delta_is_team_aggregate_and_only_emits_changes():
    first = [{
        "where": {"players": [
            {"name": "t1", "side": "T", "hp": 100, "callout": "Ramp", "weapon": "AK"},
            {"name": "t2", "side": "T", "hp": 80, "callout": "B_Main", "weapon": "AK"},
            {"name": "ct1", "side": "CT", "hp": 90, "callout": "Mid", "weapon": "M4"},
            {"name": "ct2", "side": "CT", "hp": 0, "callout": "A", "weapon": "M4"},
        ]},
    }]
    reported: dict[str, dict[str, int]] = {}

    assert build_rule_state_delta(first, reported) == {
        "kind": "snapshot",
        "teams": {
            "T": {"alive_count": 2},
            "CT": {"alive_count": 1},
        },
        "changed_teams": ["T", "CT"],
    }


def test_rule_state_delta_ignores_hp_change_in_changed_teams():
    frame_a = [{"where": {"players": [
        {"name": "t1", "side": "T", "hp": 100},
        {"name": "ct1", "side": "CT", "hp": 100},
    ]}}]
    frame_b = [{"where": {"players": [
        {"name": "t1", "side": "T", "hp": 50},
        {"name": "ct1", "side": "CT", "hp": 100},
    ]}}]
    reported: dict[str, dict[str, int]] = {}
    build_rule_state_delta(frame_a, reported)
    second = build_rule_state_delta(frame_b, reported)
    assert second["changed_teams"] == []
    assert second["teams"]["T"] == {"alive_count": 1}
    assert "hp_total" not in str(second)


def test_projection_excludes_team_hp_total():
    plan = _unsafe_plan()
    plan["rule_state"] = {"kind": "snapshot", "teams": {
        "T": {"alive_count": 4, "hp_total": 320},
        "CT": {"alive_count": 5, "hp_total": 500},
    }, "changed_teams": ["T"]}
    prompt = _build_window_prompt(plan)
    assert "hp_total" not in prompt
    prompt_payload = json.loads(prompt[prompt.rfind("\n{") + 1:])
    assert prompt_payload["rule_state"]["teams"]["T"] == {"alive_count": 4}
    assert prompt_payload["rule_state"]["teams"]["CT"] == {"alive_count": 5}


def test_projection_excludes_hp_total_from_result_state():
    plan = {
        "main_topic": {"kind": "exchange", "summary": "exchange"},
        "selected_actions": [{
            "type": "exchange_topic", "kill_count": 2,
            "result_state": {
                "T": {"alive_count": 2, "hp_total": 180},
                "CT": {"alive_count": 1, "hp_total": 90},
            },
        }],
        "required_facts": [{
            "canonical_text": "双方连续交换2次击杀", "required": True,
        }],
    }
    prompt = _build_window_prompt(plan)
    assert "hp_total" not in prompt
    prompt_payload = json.loads(prompt[prompt.rfind("\n{") + 1:])
    assert prompt_payload["selected_actions"][0]["result_state"] == {
        "T": {"alive_count": 2}, "CT": {"alive_count": 1},
    }


def test_projection_passes_player_state_when_non_empty():
    plan = _unsafe_plan()
    player_state = "首次快照：JDC（CT，100血，M4，中路）；REZ已阵亡"
    prompt = _build_window_prompt({**plan, "player_state": player_state})
    prompt_payload = json.loads(prompt[prompt.rfind("\n{") + 1:])
    assert prompt_payload["player_state"] == player_state


def test_projection_omits_player_state_when_empty_or_oversized():
    plan = _unsafe_plan()
    for candidate in (plan, {**plan, "player_state": "   "}, {**plan, "player_state": "x" * 1001}):
        prompt = _build_window_prompt(candidate)
        prompt_payload = json.loads(prompt[prompt.rfind("\n{") + 1:])
        assert "player_state" not in prompt_payload


def test_utility_summary_uses_chinese_names_and_budgeted_abbreviations():
    assert _utility_summary({
        "type": "utility_throw", "thrower": "faveN", "utility": "Incendiary Grenade",
    }, 14) == "faveN投出燃烧弹"
    assert _utility_summary({
        "type": "utility_throw", "thrower": "hypex", "utility": "Smoke Grenade",
    }, 19) == "hypex投出烟雾弹"
    compact = _utility_summary({
        "type": "utility_throw", "thrower": "gr1ks", "utility": "Smoke Grenade",
    }, 9)
    # 英文词按 1 字计：gr1ks(1) + 投出烟雾弹(5) = 6 <= 9，选自然长句
    assert compact == "gr1ks投出烟雾弹"
    assert count_spoken_chars(compact) <= 9


def test_utility_summary_returns_shortest_candidate_when_all_exceed_budget():
    short = _utility_summary({
        "type": "utility_throw", "thrower": "VeryLongPlayerName", "utility": "Incendiary Grenade",
    }, 6)
    # VeryLongPlayerName 按 1 字计：1 + 投出燃烧弹(5) = 6 <= 6，不再强制降级
    assert short == "VeryLongPlayerName投出燃烧弹"
    assert count_spoken_chars(short) <= 6


def test_project_main_topic_utility_uses_chinese_names():
    actions = [
        {"type": "utility_throw", "thrower": "faveN", "utility": "Incendiary Grenade"},
    ]
    topic = _project_main_topic(
        {"kind": "utility", "summary": "faveN投出燃烧弹"},
        tactic_hint=None,
        selected_actions=actions,
    )
    assert topic["kind"] == "utility"
    assert topic["summary"] == "faveN投出燃烧弹"



def test_cloud_payload_excludes_raw_state_timeline_and_rule_audit_details():
    raw = _unsafe_plan()
    payload, safe_windows = build_cloud_round_payload(
        round_no=7,
        map_name="de_test",
        windows=[{
            "t_start": 10.0,
            "t_end": 13.0,
            "scene": "unplanted",
            "main_topic": raw["main_topic"],
            "selected_actions": raw["selected_actions"],
            "tactic_hint": raw["tactic_hint"],
            "state_block": "state change: A moved from Ramp to B_Main",
            "rule_state": {"kind": "snapshot", "teams": {
                "T": {"alive_count": 4, "hp_total": 320},
                "CT": {"alive_count": 5, "hp_total": 500},
            }, "changed_teams": ["T"]},
            "context_frames": [{"where": {"players": [{"callout": "Ramp"}]}}],
        }],
    )

    public = str(payload)
    assert "Ramp" not in public
    assert "B_Main" not in public
    assert "state_block" not in public
    assert "context_frames" not in public
    assert "timeline" not in public
    assert '"t_start"' not in public
    assert '"t_end"' not in public
    assert '"evidence"' not in public
    public_window = payload["windows"][0]
    assert public_window["main_topic"]["kind"] == "position"
    assert public_window["main_topic"]["summary"] == public_window["required_facts"][0]["canonical_text"]
    assert payload["windows"][0]["selected_actions"] == [{"type": "utility_throw", "utility": "Smoke", "thrower": "A"}]
    assert payload["contract_version"] == 5
    assert payload["windows"][0]["projection_version"] == 2
    assert set(payload["windows"][0]["rule_state"]["teams"]) == {"T", "CT"}
    assert safe_windows["window-1"]["t_start"] == 10.0


def test_required_fact_anchors_are_merged_from_projection_only():
    merged = merge_required_fact_anchors([
        {"anchors": {"players": ["A"], "teams": ["T"], "numbers": [2], "events": ["kill"]}},
        {"anchors": {"players": ["A", "B"], "teams": ["CT"], "numbers": [2, 1], "results": ["lead"]}},
    ])
    assert merged["players"] == ["A", "B"]
    assert merged["teams"] == ["T", "CT"]
    assert merged["numbers"] == [2, 1]
    assert merged["events"] == ["kill"]
    assert merged["results"] == ["lead"]
