from sbmachine.commentary_planner import PlannerState
from sbmachine.phase3a_prompt import _build_window_prompt
from sbmachine.scene_context import SceneWindow
from sbmachine.tactic_book import compile_tactic_book
from sbmachine.tactic_projection import build_window_rule_projection


def _player(name: str, side: str, callout: str) -> dict:
    return {"name": name, "side": side, "hp": 100, "callout": callout, "weapon": "AK-47"}


def _frame(t: float, players: list[dict], utilities: list[dict] | None = None) -> dict:
    return {
        "when": {"video_time": t, "relative_sec": t, "phase": "in_round"},
        "who": {"pov_player": "a_lurker", "view": "player"},
        "where": {"players": players},
        "events": {"utilities": utilities or []},
    }


def _book() -> object:
    return compile_tactic_book("de_test", {
        "version": 1,
        "map": "de_test",
        "tactics": [
            {
                "id": "fake_a_hit_b", "label": "假爆A真打B",
                "hint": "A小一人道具牵制，B区三人集结。", "side": "T",
                "when": [
                    {"kind": "zone_count", "side": "T", "zone": {"callouts_any": ["A_Short"]}, "count": [1, 1]},
                    {"kind": "zone_count", "side": "T", "zone": {"callouts_any": ["B_Main"]}, "count": [3, 5]},
                    {"kind": "event_count", "event": "utility_throw", "actor_side": "T", "actor_zone": {"callouts_any": ["A_Short"]}, "types_any": ["Smoke Grenade", "Flashbang"], "window_sec": 6, "count": [2, None]},
                ], "priority": 10,
            },
            {
                "id": "t_mid_stack_retake", "label": "中路摆谱中期反清", "side": "T",
                "when": [{"kind": "zone_count", "side": "T", "zone": {"callouts_any": ["Mid", "TopMid", "BottomMid"]}, "count": [4, 5]}],
                "priority": 5,
            },
        ],
    })


def test_projection_without_match_keeps_legacy_planner_shape():
    frames = [_frame(1.0, [_player("t1", "T", "Long")])]
    projection = build_window_rule_projection(
        "de_test", SceneWindow(1, 2, "未下包", 1, 2), frames, frames, frames,
        PlannerState(), _book(),
    )

    assert projection.tactic_hint is None
    assert "tactic_hint" not in projection.plan
    assert projection.plan["main_topic"]["kind"] in {"position", "silence"}


def test_projection_injects_each_fixture_only_in_matching_window_and_prompt_hides_evidence():
    fake_players = [
        _player("a_lurker", "T", "A_Short"),
        _player("b1", "T", "B_Main"), _player("b2", "T", "B_Main"), _player("b3", "T", "B_Main"),
    ]
    frames = [
        _frame(10.0, fake_players, [{"_event": "throw", "entity_id": 1, "throw_tick": 10, "thrower": "a_lurker", "type": "Smoke Grenade"}]),
        _frame(12.0, fake_players, [{"_event": "throw", "entity_id": 2, "throw_tick": 12, "thrower": "a_lurker", "type": "Flashbang"}]),
    ]
    projection = build_window_rule_projection(
        "de_test", SceneWindow(10, 13, "未下包", 10, 13), frames, frames, frames,
        PlannerState(), _book(),
    )
    prompt = _build_window_prompt(
        {"round_no": 1, "t_start": 10, "t_end": 13, "scene": "未下包", "commentary_plan": projection.plan, "context_frames": []},
        t_start=10, t_end=13, scene="未下包", state_block="状态变化：a_lurker 转移A_Short",
    )

    assert projection.tactic_hint == {"rule_id": "fake_a_hit_b", "label": "假爆A真打B", "hint": "A小一人道具牵制，B区三人集结。", "matched_at": 12.0}
    # 计划书 phase3a-minimal-rule-coverage-expansion（§4.1）：纯道具窗不再因无交火静默，
    # utility（rank 4）优先于 tactic（rank 5）作主话题；tactic_hint 仍注入供下游参考。
    assert projection.plan["main_topic"]["kind"] == "utility"
    assert projection.plan["tactic_hint"]["label"] == "假爆A真打B"
    # S3: tactic hint 在 JSON 投影内，不再作为独立文本块。
    assert "假爆A真打B" in prompt
    assert "A小一人道具牵制，B区三人集结" in prompt
    assert "where.players" not in prompt and '"event_ledger"' not in prompt and "evidence" not in prompt
    # 主话题为 utility：投影携带 utility 事件且 tactic_hint 仍随 JSON 进入 prompt。
    assert "utility" in prompt
    assert '"x"' not in prompt and '"y"' not in prompt and '"z"' not in prompt


def test_mid_stack_fixture_promotes_tactic_without_action():
    frames = [_frame(20.0, [_player(f"t{i}", "T", "Mid") for i in range(4)])]
    projection = build_window_rule_projection(
        "de_test", SceneWindow(20, 21, "未下包", 20, 21), frames, frames, frames,
        PlannerState(), _book(),
    )

    assert projection.plan["tactic_hint"]["label"] == "中路摆谱中期反清"
    assert projection.plan["main_topic"] == {
        "kind": "tactic",
        "summary": "中路摆谱中期反清",
        "priority_class": "verified_tactic",
    }
