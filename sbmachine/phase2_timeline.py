"""Phase 2 timeline construction helpers."""
from __future__ import annotations


def build_timeline(
    start_sec: float,
    end_sec: float,
    demo: "DemoQuery",
    round_meta: dict,
    align: "RoundTimeAlign",
    *,
    demo_interval_sec: float = 1.0,
    vlm_interval_sec: float = 3.0,
    dense_pre_sec: float = 3.0,
    dense_post_sec: float = 1.5,
    dense_fps: float = 2.0,
    round_no: int = 0,
) -> list[tuple[float, bool]]:
    """统一时间轴构造。

    返回 [(video_time, is_vlm), ...] 严格有序。
    is_vlm=True → 视觉行(解码 + VLM);False → 背景行(仅 demo 查询,不解码)。
    """
    snap = 0.1   # 亚秒对齐网格

    def _snap(t: float) -> float:
        return round(round(t / snap) * snap, 6)

    # 1s 网格
    grid: set[float] = set()
    t = start_sec
    while t <= end_sec + 1e-6:
        grid.add(_snap(t))
        t += demo_interval_sec

    # 3s VLM 节奏(对齐到整秒)
    vlm_times: set[float] = set()
    t = start_sec
    while t <= end_sec + 1e-6:
        snapped = _snap(t)
        if snapped in grid:
            vlm_times.add(snapped)
        t += vlm_interval_sec

    # 事件窗加密:击杀 + 炸弹
    event_ticks: list[int] = []
    kills = demo.kills_in_round(round_no)
    for k in kills:
        tk = k.get("tick")
        if tk is not None:
            event_ticks.append(int(tk))
    for key in ("bomb_planted_tick", "bomb_exploded_tick", "bomb_defused_tick"):
        v = round_meta.get(key)
        if v is not None:
            event_ticks.append(int(v))

    dense_step = 1.0 / max(dense_fps, 0.1)
    for tk in event_ticks:
        center = align.to_video_time(tk)
        t = center - dense_pre_sec
        while t <= center + dense_post_sec + 1e-6:
            snapped = _snap(t)
            if start_sec - 0.05 <= snapped <= end_sec + 0.05:
                grid.add(snapped)
                vlm_times.add(snapped)
            t += dense_step

    timeline = sorted(grid)
    return [(t, t in vlm_times) for t in timeline]

