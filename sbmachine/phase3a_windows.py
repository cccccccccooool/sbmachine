"""Phase 3a deterministic window helpers."""
from __future__ import annotations

from sbmachine.phase3a_payload import _CHARS_PER_TOKEN, _dumps_compact, _frame_is_event


def build_scene_windows(
    beats: list[dict],
    start_sec: float,
    end_sec: float,
    window_max_sec: float = 10.0,
    window_min_sec: float = 3.0,
) -> list[tuple[float, float]]:
    """从 demo 事件锚点确定性切窗。

    算法：
    1. 收集所有击杀/植弹事件的 video_time 作为锚点。
    2. 锚点之间若间距 > window_max_sec，等距插补窗。
    3. 若相邻边界间距 < window_min_sec，合并到相邻窗。
    4. 返回有序、无缝覆盖 [start_sec, end_sec] 的窗口列表。
    5. beats 为空返回 [(start_sec, end_sec)]（单窗兜底）。
    """
    anchor_times: list[float] = []
    for beat in beats:
        t = float((beat.get("when") or {}).get("video_time", 0))
        ev = beat.get("events") or {}
        has_kill = any(not k.get("is_corpse_shoot") for k in (ev.get("kills") or []))
        has_bomb = bool((ev.get("c4") or {}).get("planted"))
        if has_kill or has_bomb:
            anchor_times.append(t)

    anchors: list[float] = sorted(set(
        t for t in anchor_times if start_sec <= t <= end_sec
    ))

    if not anchors:
        return [(start_sec, end_sec)]

    boundaries: list[float] = [start_sec]
    for a in anchors:
        if a > boundaries[-1]:   # 去重：锚点与前边界相同时跳过，避免零长窗
            boundaries.append(a)
    if end_sec > boundaries[-1]:
        boundaries.append(end_sec)

    # 等距插补过长间距
    expanded: list[float] = [boundaries[0]]
    for i in range(1, len(boundaries)):
        lo, hi = boundaries[i - 1], boundaries[i]
        gap = hi - lo
        if gap > window_max_sec:
            n_insert = int(gap / window_max_sec)
            step = gap / (n_insert + 1)
            for j in range(1, n_insert + 1):
                expanded.append(round(lo + j * step, 3))
        expanded.append(hi)

    # 合并过短窗（< window_min_sec）：把过短边界并入后一侧，不删边界
    merged: list[float] = [expanded[0]]
    for i in range(1, len(expanded)):
        gap = expanded[i] - merged[-1]
        is_last = (i == len(expanded) - 1)
        if gap < window_min_sec and not is_last:
            # 跳过该边界（并入后段），保证下一个更大间距消化掉当前短段
            continue
        merged.append(expanded[i])

    # 钳制 [start_sec, end_sec]，截掉超界边界，确保末边界 == end_sec
    merged = [b for b in merged if start_sec <= b <= end_sec]
    if not merged or merged[-1] != end_sec:
        merged.append(end_sec)

    windows = [(merged[i], merged[i + 1]) for i in range(len(merged) - 1)
               if merged[i] < merged[i + 1]]    # 滤掉零长/逆序对（防御）
    return windows if windows else [(start_sec, end_sec)]


def _is_cut_candidate(frames: list[dict], i: int) -> bool:
    """帧 i 是否可作段边界：phase 切换 / 空窗帧；绝不在含 kills/damages/c4 的事件帧上落刀。"""
    if _frame_is_event(frames[i]):
        return False
    prev_phase = (frames[i - 1].get("when", {}) or {}).get("phase")
    cur_phase = (frames[i].get("when", {}) or {}).get("phase")
    if cur_phase != prev_phase:
        return True
    return not frames[i].get("events")


def _segment_windows(frames: list[dict], budget: int, overlap: int) -> list[dict]:
    """贪心把帧切成 K 个归属窗。超 budget 时回退到最近候选切点落刀。
    返回 [{"lo":float,"hi":float,"frames":[...含前向 overlap 帧作上下文...]}]，
    lo/hi 为不含重叠的归属时间区间（按 when.video_time），相邻窗 [lo,hi) 无缝无叠。"""
    n = len(frames)
    if n == 0:
        return []

    def vtime(idx: int) -> float:
        return float((frames[idx].get("when", {}) or {}).get("video_time", 0.0))

    # 1) 贪心确定每段起始索引
    starts = [0]
    acc = 0.0
    last_cand: int | None = None
    i = 0
    while i < n:
        acc += len(_dumps_compact(frames[i])) / _CHARS_PER_TOKEN
        if acc > budget and i > starts[-1]:
            cut = last_cand if (last_cand is not None and last_cand > starts[-1]) else i
            starts.append(cut)
            last_cand = None
            acc = 0.0
            i = cut          # 从切点重新累计
            continue
        if i > starts[-1] and _is_cut_candidate(frames, i):
            last_cand = i
        i += 1

    # 2) 起始索引 → 归属窗（lo/hi/frames）
    bounds = starts + [n]
    windows = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        if s >= e:
            continue
        lo = vtime(s)
        hi = vtime(e) if e < n else float("inf")   # 末段 hi=inf，吃到回合结束
        ctx = max(0, s - overlap)
        windows.append({"lo": lo, "hi": hi, "frames": frames[ctx:e]})
    return windows

