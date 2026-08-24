import pytest

from sbmachine.commentary_planner import PlannerState, plan_window
from sbmachine.scene_context import SceneWindow


def test_planner_rejects_legacy_manual_note_argument():
    window = SceneWindow(6, 8, "未下包", 6, 8)

    with pytest.raises(TypeError):
        plan_window("de_missing", window, [], [], [], PlannerState(), manual_note="旧逐局笔记")


def test_planner_without_legacy_note_keeps_silence_fallback():
    window = SceneWindow(6, 8, "未下包", 6, 8)

    plan = plan_window("de_missing", window, [], [], [], PlannerState())

    assert plan["main_topic"] == {
        "kind": "silence",
        "summary": "",
        "priority_class": "silence",
    }
    assert "tactic_hint" not in plan
