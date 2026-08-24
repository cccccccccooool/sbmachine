from sbmachine.commentary_planner import (
    PlannerState,
    build_atomic_fact_units,
    fallback_neutral,
    plan_window,
)
from sbmachine.phase2_ocr import OcrConsensus
from sbmachine.scene_context import SceneWindow, extract_actions
from sbmachine.spatial_context import resolve_spatial_context
from sbmachine.time_align import RoundTimeAlign


def _player(name, side, x, y=0):
    return {
        "name": name,
        "side": side,
        "hp": 100,
        "x": x,
        "y": y,
        "z": 0,
        "callout": "A" if x < 500 else "B",
        "weapon": "AK-47",
        "ammo": 30,
    }


def _frame(t, players, events=None, *, pov="p1", view="player"):
    return {
        "when": {"video_time": t, "relative_sec": t, "phase": "in_round"},
        "who": {"pov_player": pov, "view": view},
        "where": {"players": players},
        "events": events or {},
    }


def test_planner_keeps_off_pov_kill_in_ledger_and_ranks_it_above_utility():
    players = [
        _player("p1", "T", 0),
        _player("p2", "T", 100),
        _player("r1", "CT", 4000),
        _player("r2", "CT", 4100),
    ]
    events = {
        "kills": [{"tick": 20, "attacker": "r1", "victim": "r2", "weapon": "AK-47"}],
        "utilities": [
            {
                "_event": "throw",
                "entity_id": 3,
                "throw_tick": 21,
                "thrower": "p1",
                "type": "Smoke",
            }
        ],
    }
    frames = [_frame(6, players, events), _frame(7, players)]
    window = SceneWindow(6, 8, "未下包", 6, 8)
    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())

    assert plan["scene"] == "未下包"
    assert plan["version"] == 2
    assert plan["main_topic"]["kind"] == "kill"
    assert plan["selected_actions"][0]["type"] == "kill_topic"
    assert plan["event_ledger"][0]["pov_relation"] == "off_pov"
    assert plan["event_ledger"][0]["locality_verified"] is False
    assert fallback_neutral(plan) == plan["main_topic"]["summary"]


def test_active_utility_is_one_action_not_one_per_frame():
    players = [_player("p1", "T", 0)]
    frames = [
        _frame(
            6,
            players,
            {"smokes_active": [{"entity_id": 9, "start_tick": 99, "thrower": "p1"}]},
        ),
        _frame(
            7,
            players,
            {"smokes_active": [{"entity_id": 9, "start_tick": 99, "thrower": "p1"}]},
        ),
    ]
    actions = extract_actions(frames, 6, 8)
    assert [item["type"] for item in actions] == ["utility_effect"]


def test_director_fallback_uses_stable_opposite_side_isolated_player():
    players = [
        _player("ct_near_a", "CT", 0),
        _player("ct_near_b", "CT", 100),
        _player("ct_lurk", "CT", 1500),
        _player("t1", "T", 20),
    ]
    frames = [
        _frame(6, players, pov="", view="director"),
        _frame(7, players, pov="", view="director"),
    ]
    spatial = resolve_spatial_context("de_missing", "未下包", frames, [])
    assert spatial["anchor"]["name"] == "ct_lurk"
    assert spatial["anchor_source"] == "isolated_opposite"


def test_director_without_reliable_evidence_has_no_anchor():
    players = [
        _player("ct1", "CT", 0),
        _player("ct2", "CT", 100),
        _player("t1", "T", 1000),
        _player("t2", "T", 1100),
    ]
    frames = [_frame(6, players, pov="", view="director")]
    spatial = resolve_spatial_context("de_missing", "未下包", frames, [])

    assert spatial["anchor"] is None
    assert spatial["anchor_source"] == "none"
    assert spatial["nearby"] == {"teammates": [], "enemies": []}


def test_ocr_consensus_and_alignment_warnings_are_compact():
    consensus = OcrConsensus(window=3, min_confidence=0.5)
    consensus.update({"raw_text": "playerA", "confidence": 0.9})
    result = consensus.update({"raw_text": "pIayerA", "confidence": 0.55})
    assert result["raw_text"] == "playerA"
    assert result["consensus"] is True

    consensus.update({"raw_text": "", "confidence": 0.0})
    consensus.update({"raw_text": "", "confidence": 0.0})
    expired = consensus.update({"raw_text": "", "confidence": 0.0})
    assert expired["raw_text"] == ""

    align = RoundTimeAlign({"start_tick": 0}, 64)
    align.add_warning("same")
    align.add_warning("same")
    assert align.take_new_warnings() == ["same"]
    assert align.take_new_warnings() == []


def test_bomb_fallback_contains_only_confirmed_fact_and_focus():
    players = [_player("ct1", "CT", 0), _player("t1", "T", 1000)]
    events = {"c4": {"planted": True, "plant_tick": 100}}
    frames = [_frame(20, players, events, pov="ct1")]
    window = SceneWindow(20, 24, "炸弹", 20, 24)

    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())
    text = fallback_neutral(plan)

    assert text == "C4已安装"
    assert "路线" not in text
    assert "分析" not in text
    assert "规则" not in text


def test_extract_actions_last_window_can_own_exact_endpoint():
    players = [_player("p1", "T", 0), _player("ct1", "CT", 100)]
    frames = [
        _frame(
            8.0, players, {"kills": [{"tick": 80, "attacker": "p1", "victim": "ct1"}]}
        )
    ]

    assert extract_actions(frames, 6.0, 8.0) == []
    assert [
        action["type"] for action in extract_actions(frames, 6.0, 8.0, include_end=True)
    ] == ["kill"]


def test_persistent_plant_tick_is_owned_by_only_one_window():
    players = [_player("p1", "T", 0), _player("ct1", "CT", 100)]
    planted = {"c4": {"planted": True, "plant_tick": 150}}
    frames = [
        {
            **_frame(1.0, players),
            "when": {
                "video_time": 1.0,
                "relative_sec": 1.0,
                "phase": "in_round",
                "tick": 100,
            },
        },
        {
            **_frame(2.0, players, planted),
            "when": {
                "video_time": 2.0,
                "relative_sec": 2.0,
                "phase": "in_round",
                "tick": 200,
            },
        },
        {
            **_frame(3.0, players, planted),
            "when": {
                "video_time": 3.0,
                "relative_sec": 3.0,
                "phase": "in_round",
                "tick": 300,
            },
        },
    ]

    actions = extract_actions(frames, 0.0, 4.0)
    assert [
        action["event_id"] for action in actions if action["type"] == "bomb_planted"
    ] == ["bomb_planted:150"]
    assert extract_actions(frames[2:], 3.0, 4.0) == []


def test_terminal_keeps_only_the_final_related_kill():
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
        _player("t2", "T", 20),
        _player("ct2", "CT", 120),
    ]
    frames = [
        _frame(
            6.0, players, {"kills": [{"tick": 60, "attacker": "t1", "victim": "ct1"}]}
        ),
        _frame(
            7.0,
            players,
            {
                "kills": [{"tick": 70, "attacker": "ct2", "victim": "t2"}],
                "c4": {"bomb_exploded_tick": 70},
            },
        ),
    ]
    window = SceneWindow(6.0, 8.0, "炸弹", 6.0, 8.0)

    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())

    assert plan["main_topic"]["priority_class"] == "terminal"
    assert plan["scene_override"] == {"scene": "收尾", "reason": "bomb_exploded"}
    kill_rows = [row for row in plan["event_ledger"] if row["type"] == "kill"]
    assert kill_rows
    assert len(plan["selected_actions"]) == 2
    assert plan["selected_actions"][1]["type"] == "kill_topic"
    assert sum("suppressed_reason" not in row for row in kill_rows) == 1


def test_cross_attacker_kills_select_exchange_topic_with_result_state():
    players = [
        _player("t1", "T", 0),
        _player("t2", "T", 20),
        _player("ct1", "CT", 100),
        _player("ct2", "CT", 120),
    ]
    frames = [
        _frame(
            6.0, players, {"kills": [{"tick": 60, "attacker": "t1", "victim": "ct1"}]}
        ),
        _frame(
            7.0, players, {"kills": [{"tick": 70, "attacker": "ct2", "victim": "t2"}]}
        ),
        _frame(
            8.0, players, {"kills": [{"tick": 80, "attacker": "t1", "victim": "ct2"}]}
        ),
    ]
    window = SceneWindow(6.0, 9.0, "未下包", 6.0, 9.0)

    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())

    assert plan["main_topic"]["kind"] == "exchange"
    assert plan["main_topic"]["priority_class"] == "critical_exchange"
    assert plan["selected_actions"][0]["type"] == "exchange_topic"
    assert len(plan["selected_actions"][0]["event_ids"]) == 3
    assert set(plan["selected_actions"][0]["result_state"]) == {"T", "CT"}


def _utility_event(thrower, u_type, tick):
    return {
        "_event": "throw",
        "entity_id": tick,
        "throw_tick": tick,
        "thrower": thrower,
        "type": u_type,
    }


def test_plain_utility_without_fight_context_becomes_topic():
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
    ]
    frames = [
        _frame(6.0, players, {"utilities": [_utility_event("t1", "Smoke", 61)]}),
    ]
    window = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)

    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())

    assert plan["main_topic"]["kind"] == "utility"
    assert plan["main_topic"]["summary"] == "t1投出烟雾弹"
    assert plan["selected_actions"][0]["type"] == "utility_throw"


def test_effective_flash_still_narrated_without_fight_context():
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
    ]
    frames = [
        _frame(
            6.0,
            players,
            {"flashes": [{"tick": 62, "attacker": "t1", "victim": "ct1", "duration_s": 3.0}]},
        ),
    ]
    window = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)

    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())

    assert plan["main_topic"]["kind"] == "utility"


def test_utility_quota_limits_plain_utility_windows_per_round():
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
    ]
    state = PlannerState()
    kinds = []
    for i, t in enumerate((6.0, 16.0, 26.0)):
        # 给窗口加一个 kill，使 fight context 成立，smoke 才可进候选；
        # 但 kill rank 3 < utility rank 4，kill 始终优先，utility 不会成为主话题。
        frames = [
            _frame(
                t,
                players,
                {
                    "kills": [{"tick": int(t * 10) + 1, "attacker": "t1", "victim": "ct1", "weapon": "AK-47"}],
                    "utilities": [_utility_event("t1", "Smoke", int(t * 10) + 2)],
                },
            ),
        ]
        window = SceneWindow(t, t + 2.0, "未下包", t, t + 2.0)
        plan = plan_window("de_missing", window, frames, frames, frames, state)
        kinds.append(plan["main_topic"]["kind"])
    # 前两窗击杀未被解说 -> kill 胜出；第三窗击杀已解说 -> smoke 凭 fight context 上位
    assert kinds == ["kill", "kill", "utility"]


def test_utility_beyond_soft_quota_can_still_win_empty_window():
    # 前两窗无击杀仅 smoke：均成为 utility，软配额 2 次全部消耗。
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
    ]
    state = PlannerState()
    for t in (6.0, 16.0):
        frames = [
            _frame(t, players, {"utilities": [_utility_event("t1", "Smoke", int(t * 10) + 2)]})
        ]
        window = SceneWindow(t, t + 2.0, "未下包", t, t + 2.0)
        plan = plan_window("de_missing", window, frames, frames, frames, state)
        assert plan["main_topic"]["kind"] == "utility"
    # 第 3 窗无击杀仅新 smoke：已超软配额（rank 降 1），但空窗没有更高话题，
    # 排序降级只影响相对优先级，不再直接静默。
    frames3 = [_frame(26.0, players, {"utilities": [_utility_event("t1", "Smoke", 262)]})]
    window3 = SceneWindow(26.0, 28.0, "未下包", 26.0, 28.0)
    plan3 = plan_window("de_missing", window3, frames3, frames3, frames3, state)
    assert plan3["main_topic"]["kind"] == "utility"
    assert plan3["main_topic"]["summary"] == "t1投出烟雾弹"


def test_state_topic_emitted_on_alive_change():
    players5 = [
        _player("t1", "T", 0),
        _player("t2", "T", 20),
        _player("t3", "T", 40),
        _player("ct1", "CT", 100),
        _player("ct2", "CT", 120),
    ]
    players32 = [
        _player("t1", "T", 0),
        _player("t2", "T", 20),
        _player("t3", "T", 40),
        _player("ct1", "CT", 100),
    ]
    before = [_frame(6.0, players5)]
    now = [_frame(8.0, players32)]
    window = SceneWindow(7.0, 9.0, "未下包", 6.0, 9.0)
    frames = before + now

    plan = plan_window("de_missing", window, before, frames, frames, PlannerState())

    assert plan["main_topic"]["kind"] == "state"
    assert plan["main_topic"]["summary"] == "T方3人、CT方1人"


def test_kill_streak_summary_includes_round_kill_count():
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
        _player("ct2", "CT", 120),
    ]
    frames = [
        _frame(6.0, players, {"kills": [{"tick": 61, "attacker": "t1", "victim": "ct1", "weapon": "AK-47"}]}),
    ]
    window = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    state = PlannerState()
    state.attacker_round_kills["t1"] = 1  # 前窗口已 1 杀

    plan = plan_window("de_missing", window, frames, frames, frames, state)

    assert plan["main_topic"]["kind"] == "kill"
    assert "连杀2杀" in plan["main_topic"]["summary"]


def test_fight_context_improves_utility_strength_not_eligibility():
    """§7.1 用例2：fight context 只提高排序强度，不改变资格；超软配额降级后空窗仍可胜出。"""
    players = [
        _player("P1", "T", 0),
        _player("ct1", "CT", 100),
    ]
    state = PlannerState()
    for t in (6.0, 16.0):
        frames = [
            _frame(t, players, {"utilities": [_utility_event("P1", "Smoke", int(t * 10) + 2)]})
        ]
        window = SceneWindow(t, t + 2.0, "未下包", t, t + 2.0)
        assert plan_window("de_missing", window, frames, frames, frames, state)[
            "main_topic"
        ]["kind"] == "utility"
    # 超配额窗口：utility(rank5, strength0.5) 与 position(rank5, strength0.4)
    # 同 rank 竞争，强度只影响相对顺序，双方都仍具备资格 → utility 胜出。
    frames3 = [
        _frame(26.0, players, {"utilities": [_utility_event("P1", "Smoke", 262)]})
    ]
    window3 = SceneWindow(26.0, 28.0, "未下包", 26.0, 28.0)
    plan3 = plan_window(
        "de_missing", window3, frames3, frames3, frames3, state
    )
    assert plan3["main_topic"]["kind"] == "utility"


def test_same_utility_event_not_narrated_twice():
    """§7.1 用例4：同一 utility event 已真正播报后，后续窗口不再重复。"""
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
    ]
    state = PlannerState()
    frames1 = [_frame(6.0, players, {"utilities": [_utility_event("t1", "Smoke", 61)]})]
    window1 = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan1 = plan_window("de_missing", window1, frames1, frames1, frames1, state)
    assert plan1["main_topic"]["kind"] == "utility"
    # 同一 tick=61 的 throw 再次出现在后窗：narrated 已含 → 无候选 → 静默。
    frames2 = [_frame(9.0, players, {"utilities": [_utility_event("t1", "Smoke", 61)]})]
    window2 = SceneWindow(9.0, 11.0, "未下包", 9.0, 11.0)
    plan2 = plan_window("de_missing", window2, frames2, frames2, frames2, state)
    assert plan2["main_topic"]["kind"] == "silence"


def _three_kill_events(t0):
    return [
        {"tick": t0 + 1, "attacker": "t1", "victim": "ct1", "weapon": "AK-47"},
        {"tick": t0 + 2, "attacker": "ct2", "victim": "t2", "weapon": "AK-47"},
        {"tick": t0 + 3, "attacker": "t1", "victim": "ct2", "weapon": "AK-47"},
    ]


def test_plant_related_to_exchange_is_narrated_in_the_same_window():
    """§7.1 用例6：plant 与关键交换同窗被压后，后续 planted 窗补报一次（选 bomb_planted）。"""
    players = [
        _player("t1", "T", 0),
        _player("t2", "T", 20),
        _player("ct1", "CT", 100),
        _player("ct2", "CT", 120),
    ]
    state = PlannerState()
    events1 = {
        "kills": _three_kill_events(61),
        "c4": {"planted": True, "plant_tick": 65},
    }
    frames1 = [_frame(6.0, players, events1)]
    window1 = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan1 = plan_window("de_missing", window1, frames1, frames1, frames1, state)
    assert plan1["main_topic"]["kind"] == "exchange"
    assert plan1["main_topic"]["summary"] != "C4已安装"
    assert [action["type"] for action in plan1["selected_actions"]] == [
        "exchange_topic", "bomb_planted"
    ]

    frames2 = [_frame(9.0, players, {"c4": {"planted": True, "plant_tick": 65}})]
    window2 = SceneWindow(9.0, 11.0, "未下包", 9.0, 11.0)
    plan2 = plan_window("de_missing", window2, frames2, frames2, frames2, state)
    assert plan2["main_topic"]["kind"] == "silence"


def test_narrated_plant_not_caught_up_again():
    """§7.1 用例7：plant 已真正播报后不再补报。"""
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
    ]
    state = PlannerState()
    frames1 = [_frame(6.0, players, {"c4": {"planted": True, "plant_tick": 65}})]
    window1 = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan1 = plan_window("de_missing", window1, frames1, frames1, frames1, state)
    assert plan1["main_topic"]["kind"] == "retake"

    frames2 = [_frame(9.0, players, {"c4": {"planted": True, "plant_tick": 65}})]
    window2 = SceneWindow(9.0, 11.0, "未下包", 9.0, 11.0)
    plan2 = plan_window("de_missing", window2, frames2, frames2, frames2, state)
    assert plan2["main_topic"]["kind"] == "silence"


def test_future_plant_tick_not_broadcast_early():
    """§7.1 用例8：P2 提前携带未来 plant_tick 时不得补报，击杀照常胜出。"""
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
    ]
    frame = _frame(
        6.0,
        players,
        {
            "kills": [{"tick": 61, "attacker": "t1", "victim": "ct1", "weapon": "AK-47"}],
            "c4": {"planted": True, "plant_tick": 500},
        },
    )
    frame["when"].update({"tick": 300})  # 当前帧 tick 300 < 500
    window = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan = plan_window("de_missing", window, [frame], [frame], [frame], PlannerState())
    assert plan["main_topic"]["kind"] == "kill"
    assert "C4已安装" not in plan["main_topic"]["summary"]


def test_defuse_start_can_be_caught_up_once():
    """§7.1 用例9：已发生但未播的 defuse start 在后续窗口补报一次。"""
    players = [
        _player("t1", "T", 0),
        _player("ct1", "CT", 100),
    ]
    state = PlannerState()
    # 窗1：plant 先被播报（双帧让 transition 归属到本窗的采样帧）。
    pre1 = _frame(5.0, players)
    pre1["when"].update({"tick": 55})
    plant_frame = _frame(6.0, players, {"c4": {"planted": True, "plant_tick": 60}})
    plant_frame["when"].update({"tick": 62})
    frames1 = [pre1, plant_frame]
    window1 = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan1 = plan_window(
        "de_missing", window1, frames1, frames1, frames1, state
    )
    assert plan1["main_topic"]["kind"] == "retake"
    assert plan1["main_topic"]["summary"] == "C4已安装"

    # 窗2：当前 tick 已超过 begin_defuse_tick=70，补报 defuse 一次（plant 已播则跳过）。
    pre2 = _frame(8.0, players)
    pre2["when"].update({"tick": 68})
    defuse_frame = _frame(
        9.0,
        players,
        {"c4": {"planted": True, "plant_tick": 60, "begin_defuse_tick": 70}},
    )
    defuse_frame["when"].update({"tick": 75})
    frames2 = [pre2, defuse_frame]
    window2 = SceneWindow(9.0, 11.0, "未下包", 9.0, 11.0)
    plan2 = plan_window(
        "de_missing", window2, frames2, frames2, frames2, state
    )
    assert plan2["main_topic"]["kind"] == "retake"
    assert plan2["main_topic"]["summary"] == "CT开始拆弹"
    assert plan2["selected_actions"][0]["type"] == "defuse_started"


def test_position_with_callout_when_reliable_pov_anchor():
    """§7.1 用例10：无 reviewed map、锚点携带 callout、无更高话题时选择 position。"""
    players = [_player("P1", "T", 400)]
    frames = [_frame(6.0, players, pov="P1")]
    window = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())
    assert plan["main_topic"]["kind"] == "position"
    assert plan["main_topic"]["summary"] == "P1（T）位于A"
    assert plan["selected_actions"] == []


def test_same_player_callout_selected_once_then_new_callout_allowed():
    """§7.1 用例11：同一“选手 + callout”回合内只选一次；移动到新 callout 后可再次选择。"""
    players_at_a = [_player("P1", "T", 400)]
    players_at_b = [_player("P1", "T", 600)]
    state = PlannerState()
    window1 = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan1 = plan_window(
        "de_missing", window1, [_frame(6.0, players_at_a, pov="P1")],
        [_frame(6.0, players_at_a, pov="P1")], [_frame(6.0, players_at_a, pov="P1")], state,
    )
    assert plan1["main_topic"]["kind"] == "position"
    window2 = SceneWindow(9.0, 11.0, "未下包", 9.0, 11.0)
    plan2 = plan_window(
        "de_missing", window2, [_frame(9.0, players_at_a, pov="P1")],
        [_frame(9.0, players_at_a, pov="P1")], [_frame(9.0, players_at_a, pov="P1")], state,
    )
    assert plan2["main_topic"]["kind"] == "silence"  # 同“选手+callout”已播
    window3 = SceneWindow(12.0, 14.0, "未下包", 12.0, 14.0)
    plan3 = plan_window(
        "de_missing", window3, [_frame(12.0, players_at_b, pov="P1")],
        [_frame(12.0, players_at_b, pov="P1")], [_frame(12.0, players_at_b, pov="P1")], state,
    )
    assert plan3["main_topic"]["kind"] == "position"
    assert plan3["main_topic"]["summary"] == "P1（T）位于B"


def test_position_suppressed_by_kill_does_not_consume_dedup():
    """§7.1 用例12：callout 候选被高话题压过时不提前消耗去重。"""
    players = [
        _player("P1", "T", 400),
        _player("ct1", "CT", 1000),
    ]
    state = PlannerState()
    frames1 = [
        _frame(6.0, players, {"kills": [{"tick": 61, "attacker": "P1", "victim": "ct1", "weapon": "AK-47"}]}, pov="P1")
    ]
    window1 = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan1 = plan_window("de_missing", window1, frames1, frames1, frames1, state)
    assert plan1["main_topic"]["kind"] == "kill"
    frames2 = [_frame(9.0, players, pov="P1")]
    window2 = SceneWindow(9.0, 11.0, "未下包", 9.0, 11.0)
    plan2 = plan_window("de_missing", window2, frames2, frames2, frames2, state)
    assert plan2["main_topic"]["kind"] == "position"  # 去重未被消耗，仍可播


def test_missing_callout_stays_silent_no_focus_sentence():
    """§7.1 用例13：缺 callout 时保持静默，不输出“是当前关注对象”这类空话。"""
    p = {"name": "P1", "side": "T", "hp": 100, "x": 400, "y": 0, "z": 0,
         "weapon": "AK-47", "ammo": 30}  # 无 callout 字段
    frames = [_frame(6.0, [p], pov="P1")]
    window = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())
    assert plan["main_topic"]["kind"] == "silence"
    assert "重点关注" not in plan["main_topic"]["summary"]
    assert (plan["spatial"].get("anchor") or {}).get("callout") is None


def test_position_required_facts_match_summary():
    """§7.1 用例14：position required fact 的人物/阵营/地点与摘要一致。"""
    players = [_player("P1", "T", 400)]
    frames = [_frame(6.0, players, pov="P1")]
    window = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())
    fact = plan["required_facts"][0]
    assert fact["fact_id"] == "topic:position"
    assert fact["canonical_text"] == "P1（T）位于A"
    assert fact["anchors"]["players"] == ["P1"]
    assert fact["anchors"]["teams"] == ["T"]
    assert fact["anchors"]["locations"] == ["A"]


def test_reviewed_graph_position_still_works(monkeypatch):
    """§7.1 用例15：已有 reviewed_graph 功能不回退（保留附近队友/敌人与 callout_zh）。"""
    fake_spatial = {
        "map_template_available": True,
        "map_precision": "reviewed_graph",
        "anchor": {"name": "P1", "side": "T", "callout": "A", "callout_zh": "正门", "weapon": "AK-47"},
        "anchor_source": "pov",
        "anchor_confidence": 1.0,
        "nearby": {
            "teammates": [{"name": "P2", "callout": "A", "callout_zh": "正门", "distance_units": 200, "weapon": "AK-47", "relation": "same_callout"}],
            "enemies": [],
        },
        "local_actions": [],
    }
    import sbmachine.commentary_planner as planner_module
    monkeypatch.setattr(planner_module, "resolve_spatial_context", lambda *a, **k: dict(fake_spatial))
    players = [_player("P1", "T", 400), _player("P2", "T", 350)]
    frames = [_frame(6.0, players, pov="P1")]
    window = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)
    plan = plan_window("de_missing", window, frames, frames, frames, PlannerState())
    assert plan["main_topic"]["kind"] == "position"
    assert plan["main_topic"]["summary"] == "P1（T）位于正门，附近有队友P2"


def test_tactical_phase_selects_two_independent_utility_facts_and_fact_units():
    players = [_player("t1", "T", 0), _player("t2", "T", 100), _player("ct1", "CT", 1000)]
    frames = [_frame(12.0, players, {
        "utilities": [
            _utility_event("t1", "Smoke", 121),
            _utility_event("t2", "Smoke", 122),
        ],
    })]
    plan = plan_window(
        "de_missing", SceneWindow(12.0, 14.0, "未下包", 12.0, 14.0),
        frames, frames, frames, PlannerState(), char_budget=40,
    )

    assert plan["main_topic"]["kind"] == "utility"
    assert len(plan["selected_actions"]) == 2
    assert [fact["fact_id"] for fact in plan["required_facts"]] == [
        "topic:utility", "topic:utility:2",
    ]
    assert fallback_neutral(plan) == "，".join(
        fact["canonical_text"] for fact in plan["required_facts"]
    )
    units = build_atomic_fact_units("r001_w01", plan)
    assert len(units["fact_units"]) == len(units["required_fact_ids"]) == 2


def test_opening_utilities_use_tactical_pair_and_over_budget_supporter_fails_closed():
    players = [_player("t1", "T", 0), _player("t2", "T", 100), _player("ct1", "CT", 1000)]
    events = {"utilities": [_utility_event("t1", "Smoke", 61), _utility_event("t2", "Smoke", 62)]}
    frames = [_frame(6.0, players, events)]
    window = SceneWindow(6.0, 8.0, "未下包", 6.0, 8.0)

    opening = plan_window("de_missing", window, frames, frames, frames, PlannerState(), char_budget=40)
    too_small = plan_window(
        "de_missing", SceneWindow(12.0, 14.0, "未下包", 12.0, 14.0),
        [_frame(12.0, players, events)], [_frame(12.0, players, events)],
        [_frame(12.0, players, events)], PlannerState(), char_budget=6,
    )

    assert len(opening["required_facts"]) == 2
    assert len(too_small["required_facts"]) == 1
    assert "projection_budget_error" not in too_small


def test_c4_and_kill_are_selected_once_and_all_selected_ids_are_narrated():
    players = [_player("t1", "T", 0), _player("ct1", "CT", 1000)]
    frames = [_frame(20.0, players, {
        "kills": [{"tick": 201, "attacker": "t1", "victim": "ct1"}],
        "c4": {"planted": True, "plant_tick": 200},
    })]
    state = PlannerState()
    plan = plan_window(
        "de_missing", SceneWindow(20.0, 22.0, "炸弹", 20.0, 22.0),
        frames, frames, frames, state, char_budget=40,
    )

    assert [action["type"] for action in plan["selected_actions"]] == [
        "bomb_planted", "kill_topic"
    ]
    selected_ids = {
        event_id
        for action in plan["selected_actions"]
        for event_id in (action.get("event_ids") or [action.get("event_id")])
        if event_id
    }
    assert selected_ids <= state.narrated_event_ids
    assert len(plan["required_facts"]) == 2


def test_opening_window_does_not_generate_setup_topic_from_economy_snapshot():
    players = [_player("t1", "T", 0), _player("ct1", "CT", 1000)]
    for player in players:
        player.update({"weapon": "Glock-18", "armor": 0, "helmet": False, "money_spent_this_round": 4700})
    frames = [_frame(5.0, players)]
    plan = plan_window(
        "de_missing", SceneWindow(5.0, 7.0, "未下包", 5.0, 7.0),
        frames, frames, frames, PlannerState(), char_budget=40,
    )
    assert plan["main_topic"]["kind"] != "setup"
    assert "本回合投入" not in plan["main_topic"]["summary"]
