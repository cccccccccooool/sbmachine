"""Phase 2 OCR and region helpers."""
from __future__ import annotations

import re

from vision_service.region_crops import box_from_norm, crop_frame


_RAPID_OCR_ENGINE = None
_RAPID_OCR_IMPORT_FAILED = False


def _get_rapid_ocr():
    """返回单个共享的 RapidOCR 引擎,仅在首次调用时延迟构建。"""
    global _RAPID_OCR_ENGINE, _RAPID_OCR_IMPORT_FAILED
    if _RAPID_OCR_ENGINE is None and not _RAPID_OCR_IMPORT_FAILED:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _RAPID_OCR_ENGINE = RapidOCR()
        except Exception:
            _RAPID_OCR_IMPORT_FAILED = True
    return _RAPID_OCR_ENGINE


def read_ocr_text(frame, region: dict, *, padding: int = 0) -> dict:
    crop = crop_frame(frame, region.get("box", []), padding=padding)
    raw_text = ""
    ocr_engine = _get_rapid_ocr()
    if ocr_engine is None:
        return {"region": region, "raw_text": "", "engine": "unavailable:ImportError"}
    engine = "rapidocr_onnxruntime"
    try:
        result, _ = ocr_engine(crop)
        if result:
            raw_text = " ".join(str(item[1]) for item in result if len(item) >= 2)
    except Exception as exc:
        engine = f"unavailable:{type(exc).__name__}"
    return {"region": region, "raw_text": raw_text.strip(), "engine": engine}


def _regions_by_type(background: dict, names: set[str]) -> list[dict]:
    out = []
    for region in background.get("regions", []) or []:
        label = str(region.get("label", "")).lower()
        rtype = str(region.get("type", "")).lower()
        if label in names or rtype in names:
            out.append(region)
    return out


def _first_pov_region(background: dict) -> dict | None:
    regions = _regions_by_type(background, {"pov_name", "pov_name_area", "pov_player_bar", "pov_marker_bar"})
    if regions:
        return max(regions, key=lambda item: float(item.get("confidence", 0.0)))
    return None


def _first_timer_region(background: dict) -> dict | None:
    regions = _regions_by_type(background, {"timer", "timer_area", "round_timer"})
    if regions:
        return max(regions, key=lambda item: float(item.get("confidence", 0.0)))
    return None


def _first_score_region(background: dict) -> dict | None:
    regions = _regions_by_type(background, {"score", "score_area", "top_hud_score", "top_hud"})
    if regions:
        return max(regions, key=lambda r: float(r.get("confidence", 0.0)))
    return None


def _resolve_ocr_box(
    yolo_region: dict | None,
    fixed_cfg: dict,
    frame_shape,
    yolo_source_name: str = "yolo",
) -> tuple[dict | None, str]:
    """返回 (region_dict, source_label)。

    优先级:YOLO 检测到的区域 → 配置文件中固定的归一化 ROI → (None, "no_region")。
    """
    if yolo_region is not None:
        return yolo_region, yolo_source_name
    if fixed_cfg.get("enabled", False) and fixed_cfg.get("box"):
        box_str = str(fixed_cfg.get("box", "")).strip()
        if box_str:
            try:
                parts = [float(x.strip()) for x in box_str.split(",")]
                if len(parts) == 4:
                    h, w = frame_shape[:2]
                    pixel_box = box_from_norm(parts, w, h)
                    return {"box": pixel_box, "label": "fixed_roi", "confidence": 1.0}, "fixed_roi"
            except (ValueError, IndexError):
                pass
    return None, "no_region"


def _detect_pov_ocr(
    frame,
    yolo_background: dict | None,
    pov_ocr_config: dict,
    crop_padding: int,
) -> tuple[dict, str, dict | None]:
    """POV 玩家名称 OCR。"""
    yolo_region = _first_pov_region(yolo_background or {})
    region, source = _resolve_ocr_box(yolo_region, pov_ocr_config, frame.shape, "yolo_pov_region")
    if region:
        return read_ocr_text(frame, region, padding=crop_padding), source, region
    return {"raw_text": "", "engine": f"no_region:{source}", "region": None}, source, None


def _detect_score_ocr(
    frame,
    yolo_background: dict | None,
    score_ocr_config: dict,
    crop_padding: int,
) -> dict:
    """来自顶部 HUD 的回合比分 OCR。

    首选路径:YOLO 检测 'score' / 'top_hud_score' 类别 → 裁剪图像 → OCR。
    备用路径:配置文件中固定的归一化 ROI(在 YOLO 重新训练前的过渡方案)。

    返回 {"ct": int|None, "t": int|None, "raw": str, "source": str}。
    """
    yolo_region = _first_score_region(yolo_background or {})
    region, source = _resolve_ocr_box(yolo_region, score_ocr_config, frame.shape, "yolo_score_region")
    if not region:
        return {"ct": None, "t": None, "raw": "", "source": source}
    ocr = read_ocr_text(frame, region, padding=crop_padding)
    raw = str(ocr.get("raw_text", ""))
    m = re.search(r"(\d+)\s*[:\-]\s*(\d+)", raw)
    if m:
        return {"ct": int(m.group(1)), "t": int(m.group(2)), "raw": raw, "source": source}
    return {"ct": None, "t": None, "raw": raw, "source": source}


def _maskable_regions(regions: list[dict]) -> list[dict]:
    skip = {"timer", "timer_area", "round_timer", "c4", "c4_area", "c4_status", "killfeed", "killfeed_area"}
    return [
        region
        for region in regions
        if str(region.get("label", "")).lower() not in skip
        and str(region.get("type", "")).lower() not in {"timer", "c4", "killfeed"}
    ]

