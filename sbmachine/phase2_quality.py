"""第二阶段：感知质量告警，仅提示问题、绝不阻断 DEM 输出。"""
from __future__ import annotations


def coalesce_yolo_gaps(times: list[float], *, max_gap_sec: float) -> list[dict]:
    """把连续无 YOLO 检测的采样帧合并成显式的时间区间。"""
    ordered = sorted({round(float(value), 3) for value in times})
    if not ordered:
        return []
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= max(0.0, float(max_gap_sec)):
            groups[-1].append(value)
        else:
            groups.append([value])
    return [
        {
            "type": "yolo_no_detection",
            "start_sec": group[0],
            "end_sec": group[-1],
            "sample_count": len(group),
            "message": "YOLO 未检测到 UI；该时段继续输出 DEM 时间轴事实",
        }
        for group in groups
    ]
