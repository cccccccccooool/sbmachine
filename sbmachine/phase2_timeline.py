"""第二阶段：视觉时间轴构建辅助函数。"""
from __future__ import annotations


def build_timeline(
    start_sec: float,
    end_sec: float,
    *,
    interval_sec: float = 1.0,
) -> list[tuple[float, bool]]:
    """为一个粗粒度视频片段构建固定采样率的视觉时间轴。

    每个采样点按顺序解码并送入 HUD YOLO；DEM 事件不会改变采样密度。
    """
    interval = max(0.1, float(interval_sec))
    if end_sec < start_sec:
        return []
    times = []
    time_sec = float(start_sec)
    while time_sec <= end_sec + 1e-6:
        times.append((round(time_sec, 3), True))
        time_sec += interval
    return times

