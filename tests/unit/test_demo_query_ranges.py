"""锁定 DemoQuery 事件区间查询的行为契约。

这些测试直接构造内存对象（不走文件加载），覆盖乱序输入、tick 开闭区间边界、
active_at 的 start/end 边界与缺失字段处理。它们保护后续把线性扫描换成 bisect
索引时返回的元素集合与顺序保持逐字节不变。
"""
from __future__ import annotations

from pathlib import Path

from sbmachine.demo_query import DemoQuery


def _query() -> DemoQuery:
    return DemoQuery(Path("."))


def test_kills_between_uses_open_lower_closed_upper_bound_and_keeps_source_order():
    demo = _query()
    # 故意乱序，且包含正好落在 lo 与 hi 上的 tick
    demo.kills = [
        {"tick": 300, "attacker": "d"},
        {"tick": 100, "attacker": "a"},
        {"tick": 200, "attacker": "b"},
        {"tick": 100, "attacker": "a2"},
        {"tick": 400, "attacker": "e"},
    ]
    # 区间 (100, 300]：排除 tick==100（开区间下界），包含 tick==300（闭区间上界）
    result = demo.kills_between(100, 300)
    assert result == [
        {"tick": 300, "attacker": "d"},
        {"tick": 200, "attacker": "b"},
    ]


def test_kills_between_normalizes_swapped_arguments():
    demo = _query()
    demo.kills = [{"tick": 150, "attacker": "a"}, {"tick": 250, "attacker": "b"}]
    # 参数顺序被 sorted 归一为 (100, 300]，两条都落在区间内且保持原始顺序
    assert demo.kills_between(300, 100) == [
        {"tick": 150, "attacker": "a"},
        {"tick": 250, "attacker": "b"},
    ]


def test_kills_between_missing_tick_defaults_to_minus_one_and_is_excluded():
    demo = _query()
    demo.kills = [{"attacker": "no_tick"}, {"tick": 50, "attacker": "a"}]
    # 缺失 tick 视为 -1，永远落在区间外
    assert demo.kills_between(0, 100) == [{"tick": 50, "attacker": "a"}]


def test_damages_and_flashes_between_share_the_same_boundary_rule():
    demo = _query()
    demo.damages = [{"tick": 10}, {"tick": 20}, {"tick": 30}]
    demo.flashes = [{"tick": 30}, {"tick": 10}, {"tick": 20}]
    assert demo.damages_between(10, 30) == [{"tick": 20}, {"tick": 30}]
    assert demo.flashes_between(10, 30) == [{"tick": 30}, {"tick": 20}]


def test_utilities_between_tags_throw_and_detonate_with_closed_interval():
    demo = _query()
    demo.grenades = [
        {"throw_tick": 100, "det_tick": 500, "grenade_type": "smoke"},
        {"throw_tick": 800, "det_tick": 900, "grenade_type": "he"},
    ]
    # utilities_between 用闭区间 [lo, hi]，throw 优先于 det
    result = demo.utilities_between(100, 550)
    assert result == [
        {"throw_tick": 100, "det_tick": 500, "grenade_type": "smoke", "_event": "throw"},
    ]
    # 只命中 det_tick 时标记 detonate
    result2 = demo.utilities_between(450, 550)
    assert result2 == [
        {"throw_tick": 100, "det_tick": 500, "grenade_type": "smoke", "_event": "detonate"},
    ]


def test_utilities_between_prefers_throw_when_both_ticks_in_range():
    demo = _query()
    demo.grenades = [{"throw_tick": 100, "det_tick": 120, "grenade_type": "flash"}]
    result = demo.utilities_between(50, 200)
    assert result == [{"throw_tick": 100, "det_tick": 120, "grenade_type": "flash", "_event": "throw"}]


def test_smokes_active_at_includes_start_and_end_boundary():
    demo = _query()
    demo.smokes = [
        {"start_tick": 100, "end_tick": 300, "id": "s1"},
        {"start_tick": 400, "end_tick": None, "id": "s2_no_end"},
        {"start_tick": None, "end_tick": 999, "id": "s3_no_start"},
    ]
    # start 边界包含
    assert demo.smokes_active_at(100) == [{"start_tick": 100, "end_tick": 300, "id": "s1"}]
    # end 边界包含
    assert demo.smokes_active_at(300) == [{"start_tick": 100, "end_tick": 300, "id": "s1"}]
    # 超出 end、尚未到下一颗烟的 start：无活跃烟雾
    assert demo.smokes_active_at(301) == []
    # end 为 None 表示持续生效
    assert demo.smokes_active_at(100000) == [{"start_tick": 400, "end_tick": None, "id": "s2_no_end"}]
    # start 为 None 的记录永远被跳过
    assert all(s.get("id") != "s3_no_start" for s in demo.smokes_active_at(500))


def test_infernos_active_at_matches_smoke_boundary_semantics():
    demo = _query()
    demo.infernos = [
        {"start_tick": 200, "end_tick": 400, "id": "i1"},
        {"start_tick": 500, "end_tick": None, "id": "i2"},
    ]
    assert demo.infernos_active_at(200) == [{"start_tick": 200, "end_tick": 400, "id": "i1"}]
    assert demo.infernos_active_at(400) == [{"start_tick": 200, "end_tick": 400, "id": "i1"}]
    assert demo.infernos_active_at(401) == []
    assert demo.infernos_active_at(500) == [{"start_tick": 500, "end_tick": None, "id": "i2"}]


def test_smokes_active_at_returns_all_overlapping_intervals_in_source_order():
    demo = _query()
    # 故意乱序、区间相互重叠，覆盖同一 tick=250 的应全部命中，且保持原始列表顺序
    demo.smokes = [
        {"start_tick": 200, "end_tick": 260, "id": "late_start"},
        {"start_tick": 100, "end_tick": 300, "id": "wide"},
        {"start_tick": 900, "end_tick": 999, "id": "far"},
        {"start_tick": 240, "end_tick": None, "id": "open_end"},
    ]
    assert demo.smokes_active_at(250) == [
        {"start_tick": 200, "end_tick": 260, "id": "late_start"},
        {"start_tick": 100, "end_tick": 300, "id": "wide"},
        {"start_tick": 240, "end_tick": None, "id": "open_end"},
    ]
    # 只落在 wide 区间内
    assert demo.smokes_active_at(120) == [{"start_tick": 100, "end_tick": 300, "id": "wide"}]
    # tick=700 时，仅 end_tick 为 None 的 open_end 仍持续生效
    assert demo.smokes_active_at(700) == [{"start_tick": 240, "end_tick": None, "id": "open_end"}]


def test_infernos_active_at_returns_overlapping_intervals_in_source_order():
    demo = _query()
    demo.infernos = [
        {"start_tick": 500, "end_tick": 800, "id": "b"},
        {"start_tick": 100, "end_tick": 600, "id": "a"},
    ]
    assert demo.infernos_active_at(550) == [
        {"start_tick": 500, "end_tick": 800, "id": "b"},
        {"start_tick": 100, "end_tick": 600, "id": "a"},
    ]


def test_range_queries_on_empty_lists_return_empty():
    demo = _query()
    assert demo.kills_between(0, 100) == []
    assert demo.damages_between(0, 100) == []
    assert demo.flashes_between(0, 100) == []
    assert demo.utilities_between(0, 100) == []
    assert demo.smokes_active_at(50) == []
    assert demo.infernos_active_at(50) == []


def test_active_at_matches_naive_scan_over_random_intervals():
    """随机对拍：区间树结果必须与朴素线性扫描逐条一致（集合与顺序）。"""
    import random

    def naive(events: list[dict], tick: int) -> list[dict]:
        result = []
        for event in events:
            start = event.get("start_tick")
            end = event.get("end_tick")
            if start is None:
                continue
            if int(start) <= int(tick) and (end is None or int(tick) <= int(end)):
                result.append(event)
        return result

    rng = random.Random(20260715)
    for _ in range(50):
        events = []
        for index in range(rng.randint(0, 40)):
            start = rng.randint(0, 200)
            roll = rng.random()
            if roll < 0.15:
                end = None
            elif roll < 0.25:
                end = start  # 零长度区间
            else:
                end = start + rng.randint(0, 80)
            events.append({"start_tick": start, "end_tick": end, "id": index})
        # 掺入几条缺失 start_tick 的记录，必须被跳过
        for _ in range(rng.randint(0, 3)):
            events.insert(rng.randint(0, len(events)), {"end_tick": rng.randint(0, 200), "id": "no_start"})

        demo = _query()
        demo.smokes = events
        for tick in range(-5, 205, 3):
            assert demo.smokes_active_at(tick) == naive(events, tick)
