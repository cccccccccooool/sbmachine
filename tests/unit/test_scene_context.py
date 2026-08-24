from sbmachine.scene_context import (
    SceneWindow,
    _effective_min_sec,
    _merge_short_windows,
    build_scene_contexts,
)


def _frame(t, events=None, phase="in_round"):
    return {"when": {"video_time": t, "relative_sec": t, "phase": phase}, "events": events or {}}


def test_scene_contexts_are_ordered_and_cover_round():
    frames = [_frame(0), _frame(5, {"kills": [{"attacker": "a", "victim": "b"}]}), _frame(12)]
    windows = build_scene_contexts(frames, 0.0, 12.0, window_max_sec=6.0, window_min_sec=3.0)
    assert windows[0].t_start == 0.0
    assert windows[-1].t_end == 12.0
    assert all(window.t_start < window.t_end for window in windows)
    assert all(windows[index].t_end <= windows[index + 1].t_start for index in range(len(windows) - 1))


def test_kill_action_does_not_replace_the_scene():
    frames = [_frame(6, {"smokes_active": [{}]}), _frame(8, {"kills": [{"attacker": "a", "victim": "b"}]})]
    windows = build_scene_contexts(frames, 6.0, 10.0)
    assert [window.scene for window in windows] == ["未下包"]


def test_effective_min_sec_uses_worst_speech_rate():
    sec = _effective_min_sec(3.0)
    assert sec >= 3.0
    assert isinstance(sec, float)


def test_merge_short_first_window_forward():
    w1 = SceneWindow(0, 2, "准备", 0, 2)       # 2s < min
    w2 = SceneWindow(2, 10, "未下包", 2, 10)
    result = _merge_short_windows([w1, w2], effective_min_sec=3.0)
    assert len(result) == 1
    assert result[0].t_start == 0.0
    assert result[0].t_end == 10.0
    assert result[0].scene == "未下包"  # 未下包 > 准备


def test_merge_short_middle_window_backward():
    w1 = SceneWindow(0, 10, "未下包", 0, 10)
    w2 = SceneWindow(10, 12, "准备", 10, 12)     # 2s < min
    w3 = SceneWindow(12, 22, "未下包", 12, 22)
    result = _merge_short_windows([w1, w2, w3], effective_min_sec=3.0)
    assert len(result) == 2
    assert result[0].t_start == 0.0
    assert result[0].t_end == 12.0           # w1 absorbed w2
    assert result[0].scene == "未下包"         # 未下包 > 准备
    assert result[1].t_start == 12.0
    assert result[1].t_end == 22.0


def test_merge_short_last_window_backward():
    w1 = SceneWindow(0, 10, "未下包", 0, 10)
    w2 = SceneWindow(10, 12, "准备", 10, 12)     # 2s < min
    result = _merge_short_windows([w1, w2], effective_min_sec=3.0)
    assert len(result) == 1
    assert result[0].t_start == 0.0
    assert result[0].t_end == 12.0


def test_merge_short_window_takes_higher_priority_scene():
    w1 = SceneWindow(0, 2, "未下包", 0, 2)       # 2s, 优先级1
    w2 = SceneWindow(2, 10, "炸弹", 2, 10)        # 优先级2
    result = _merge_short_windows([w1, w2], effective_min_sec=3.0)
    assert len(result) == 1
    assert result[0].scene == "炸弹"              # 炸弹 > 未下包


def test_merge_short_window_reverse_priority():
    w1 = SceneWindow(0, 8, "炸弹", 0, 8)
    w2 = SceneWindow(8, 10, "未下包", 8, 10)      # 2s, 优先级1
    result = _merge_short_windows([w1, w2], effective_min_sec=3.0)
    assert len(result) == 1
    assert result[0].scene == "炸弹"               # 炸弹 > 未下包, w1 场景保留


def test_merge_does_nothing_when_all_windows_above_min():
    w1 = SceneWindow(0, 10, "未下包", 0, 10)
    w2 = SceneWindow(10, 20, "炸弹", 10, 20)
    result = _merge_short_windows([w1, w2], effective_min_sec=3.0)
    assert len(result) == 2
    assert result[0].scene == "未下包"
    assert result[1].scene == "炸弹"


def test_merge_single_window_kept():
    w1 = SceneWindow(0, 2, "未下包", 0, 2)         # 2s < min, 但只有一个窗口
    result = _merge_short_windows([w1], effective_min_sec=3.0)
    assert len(result) == 1
    assert result[0].t_start == 0.0
    assert result[0].t_end == 2.0


def test_merge_empty_list():
    result = _merge_short_windows([], effective_min_sec=3.0)
    assert result == []


def test_end_to_end_short_window_merged_in_full_pipeline():
    """模拟 r009 场景：短窗 + 场景切换 → 合并后窗口数减少"""
    frames = [
        _frame(0, {"c4": {"planted": True}}),
        _frame(2),          # 2s: 场景从炸弹切到未下包（planted 变 false）
        _frame(5),
        _frame(12),
    ]
    windows = build_scene_contexts(frames, 0.0, 12.0, window_max_sec=10.0, window_min_sec=3.0)
    assert len(windows) >= 1
    for w in windows:
        assert w.t_end - w.t_start >= 3.0