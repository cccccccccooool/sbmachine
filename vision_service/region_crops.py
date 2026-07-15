"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：图像区域裁剪与遮罩工具函数。

启动方式：被 sbmachine/phase2_yolo.py 和 sbmachine/phase2_yolo_gate.py 导入调用。
输入数据流：视频帧（numpy 数组）和区域坐标（像素/归一化框）。
输出数据流：返回裁剪后的图像区域或遮罩后的图像。
用法用途：提供 region_type/screen_side_for_label/clip_box/box_from_norm/crop_frame/mask_regions/write_image 等函数，
用于 UI 区域分类、坐标裁剪、画面遮罩等操作。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


PLAYER_HUD_REGION_LABELS = {
    "left_player_hud_group",
    "right_player_hud_group",
    "player_hud_group_unknown",
}

KILLFEED_REGION_LABELS = {"killfeed", "killfeed_area"}
TIMER_REGION_LABELS = {"timer", "timer_area", "round_timer"}
C4_REGION_LABELS = {"c4", "c4_area", "c4_status"}
MINIMAP_REGION_LABELS = {"minimap", "radar", "minimap_area"}
TOP_HUD_REGION_LABELS = {"top_hud", "top_ui", "score", "score_area"}
POV_NAME_REGION_LABELS = {"pov_name", "pov_name_area", "pov_player_bar", "pov_marker_bar"}
SUPPORTED_UI_REGION_LABELS = (
    PLAYER_HUD_REGION_LABELS
    | KILLFEED_REGION_LABELS
    | TIMER_REGION_LABELS
    | C4_REGION_LABELS
    | MINIMAP_REGION_LABELS
    | TOP_HUD_REGION_LABELS
    | POV_NAME_REGION_LABELS
)


def region_type(label: str) -> str:
    """根据 YOLO 标签返回区域类型字符串。被 phase2_yolo_gate.py 的 structure_background() 调用。

    Parameters
    ----------
    label : str
        YOLO 检测标签（如 "timer"、"killfeed"、"pov_name" 等）。

    Returns
    -------
    str
        区域类型（player_hud_group/killfeed/timer/c4/minimap/top_hud/pov_name/unknown）。
    """
    text = str(label or "").strip()
    if text in PLAYER_HUD_REGION_LABELS:
        return "player_hud_group"
    if text in KILLFEED_REGION_LABELS:
        return "killfeed"
    if text in TIMER_REGION_LABELS:
        return "timer"
    if text in C4_REGION_LABELS:
        return "c4"
    if text in MINIMAP_REGION_LABELS:
        return "minimap"
    if text in TOP_HUD_REGION_LABELS:
        return "top_hud"
    if text in POV_NAME_REGION_LABELS:
        return "pov_name"
    return "unknown"


def screen_side_for_label(label: str) -> str:
    """根据标签前缀返回屏幕方位（left/right/unknown）。被 phase2_yolo_gate.py 调用。

    Parameters
    ----------
    label : str
        YOLO 检测标签。

    Returns
    -------
    str
        "left"、"right" 或 "unknown"。
    """
    text = str(label or "").strip()
    if text.startswith("left_"):
        return "left"
    if text.startswith("right_"):
        return "right"
    return "unknown"


def clip_box(box: Iterable[float], width: int, height: int, *, padding: int = 0) -> list[int]:
    """将浮点坐标框裁剪到画面边界内。被 crop_frame() 和 mask_regions() 调用。

    Parameters
    ----------
    box : Iterable[float]
        四元组 [x1, y1, x2, y2]。
    width : int
        画面宽度。
    height : int
        画面高度。
    padding : int
        向外扩展的像素数。

    Returns
    -------
    list[int]
        裁剪后的整数坐标 [x1, y1, x2, y2]。
    """
    values = list(box)
    if len(values) != 4:
        raise ValueError(f"box must contain 4 numbers, got: {values}")
    x1, y1, x2, y2 = [float(v) for v in values]
    x1 = max(0, min(width, int(round(x1 - padding))))
    y1 = max(0, min(height, int(round(y1 - padding))))
    x2 = max(0, min(width, int(round(x2 + padding))))
    y2 = max(0, min(height, int(round(y2 + padding))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid clipped box: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]


def box_from_norm(box: Iterable[float], width: int, height: int) -> list[float]:
    """将归一化坐标 [0,1] 转为像素坐标。被 phase2_yolo.py 的 _resolve_ocr_box() 调用。

    Parameters
    ----------
    box : Iterable[float]
        归一化四元组 [x1, y1, x2, y2]，值域 [0, 1]。
    width : int
        画面宽度。
    height : int
        画面高度。

    Returns
    -------
    list[float]
        像素坐标 [x1, y1, x2, y2]。
    """
    values = list(box)
    if len(values) != 4:
        raise ValueError(f"box must contain 4 numbers, got: {values}")
    x1, y1, x2, y2 = [float(v) for v in values]
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def crop_frame(frame: Any, box: Iterable[float], *, padding: int = 0) -> Any:
    """从帧中裁剪指定区域。被 phase2_yolo.py 的 read_ocr_text() 和 DebugWriter.crop_image() 调用。

    Parameters
    ----------
    frame : Any
        视频帧（numpy 数组）。
    box : Iterable[float]
        坐标框 [x1, y1, x2, y2]。
    padding : int
        向外扩展的像素数。

    Returns
    -------
    Any
        裁剪后的图像区域（numpy 数组的 copy）。
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height, padding=padding)
    return frame[y1:y2, x1:x2].copy()


def mask_regions(frame: Any, regions: Iterable[dict], *, padding: int = 0, color: tuple[int, int, int] = (0, 0, 0)) -> Any:
    """将指定区域用纯色遮罩。被调试工具调用。

    Parameters
    ----------
    frame : Any
        视频帧。
    regions : Iterable[dict]
        区域列表，每个含 "box" 键。
    padding : int
        向外扩展的像素数。
    color : tuple[int, int, int]
        遮罩颜色，默认黑色 (0,0,0)。

    Returns
    -------
    Any
        遮罩后的帧（numpy 数组的 copy）。
    """
    masked = frame.copy()
    height, width = masked.shape[:2]
    for region in regions:
        try:
            x1, y1, x2, y2 = clip_box(region.get("box", []), width, height, padding=padding)
        except ValueError:
            continue
        masked[y1:y2, x1:x2] = color
    return masked


def write_image(path: Path, image: Any) -> None:
    """将图像写入磁盘。被 tools/ 下的标注工具调用。

    Parameters
    ----------
    path : Path
        输出路径（自动创建父目录）。
    image : Any
        图像数组。
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")
