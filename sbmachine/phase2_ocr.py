"""第二阶段：OCR 识别、候选筛选与短窗口共识（consensus）的辅助函数。"""
from __future__ import annotations

from collections import deque
import re

from vision_service.region_crops import box_from_norm, crop_frame


_RAPID_OCR_ENGINE = None
_RAPID_OCR_IMPORT_FAILED = False


def _get_rapid_ocr():
    """惰性加载 RapidOCR 引擎；导入失败后缓存标记，避免反复重试。"""
    global _RAPID_OCR_ENGINE, _RAPID_OCR_IMPORT_FAILED
    if _RAPID_OCR_ENGINE is None and not _RAPID_OCR_IMPORT_FAILED:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID_OCR_ENGINE = RapidOCR()
        except Exception:
            _RAPID_OCR_IMPORT_FAILED = True
    return _RAPID_OCR_ENGINE


def _variants(crop) -> list[tuple[str, object]]:
    """保留原始 ROI，并追加若干保守的放大 / 对比度增强变体。"""
    variants = [("original", crop)]
    try:
        import cv2
        enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY) if len(enlarged.shape) == 3 else enlarged
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _, threshold = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.extend((("upscaled_gray", gray), ("clahe", clahe), ("threshold", threshold)))
    except Exception:
        pass
    return variants


def _join_ocr_lines(result: object) -> tuple[str, float]:
    """把 OCR 结果的多行文本拼成一行，并返回其平均置信度。"""
    texts: list[str] = []
    scores: list[float] = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = str(item[1] or "").strip()
        if not text:
            continue
        texts.append(text)
        try:
            scores.append(float(item[2]))
        except (IndexError, TypeError, ValueError):
            scores.append(0.5)
    return " ".join(texts).strip(), round(sum(scores) / len(scores), 3) if scores else 0.0


def read_ocr_text(
    frame,
    region: dict,
    *,
    padding: int = 0,
    accept_pattern: str = "",
    early_confidence: float | None = None,
) -> dict:
    """识别单个 OCR 区域，一旦得到调用方可用的结果就提前停止。

    识别失败或不确定的裁剪仍会尝试每一种增强变体。计时器/比分调用方可在
    文本格式合法时停止，POV 调用方则可在置信度达标时停止。
    """
    crop = crop_frame(frame, region.get("box", []), padding=padding)
    ocr_engine = _get_rapid_ocr()
    if ocr_engine is None:
        return {"region": region, "raw_text": "", "confidence": 0.0, "engine": "unavailable:ImportError"}
    candidates: list[dict] = []
    accepted: dict | None = None
    engine = "rapidocr_onnxruntime"
    for variant_name, image in _variants(crop):
        try:
            result, _ = ocr_engine(image)
            text, confidence = _join_ocr_lines(result)
            if text:
                candidate = {"text": text, "confidence": confidence, "variant": variant_name}
                candidates.append(candidate)
                pattern_accepted = bool(accept_pattern and re.search(accept_pattern, text))
                confidence_accepted = early_confidence is not None and confidence >= float(early_confidence)
                if pattern_accepted or confidence_accepted:
                    accepted = candidate
                    break
        except Exception as exc:
            engine = f"unavailable:{type(exc).__name__}"
    best = accepted or max(candidates, key=lambda item: (item["confidence"], len(item["text"])), default={"text": "", "confidence": 0.0, "variant": "none"})
    return {
        "region": region, "raw_text": best["text"], "confidence": best["confidence"],
        "variant": best["variant"], "candidates": candidates, "engine": engine,
    }


class OcrConsensus:
    """在短时间窗内稳定 HUD 文本，但绝不编造 OCR 未曾读到的值。"""
    def __init__(self, window: int = 3, min_confidence: float = 0.35) -> None:
        self._items: deque[tuple[str, float]] = deque(maxlen=max(1, window))
        self.min_confidence = min_confidence

    @staticmethod
    def _key(text: object) -> str:
        return re.sub(r"\s+", "", str(text or "")).casefold()

    def update(self, result: dict) -> dict:
        text = str(result.get("raw_text") or "").strip()
        confidence = float(result.get("confidence") or 0.0)
        # 每一帧都推进窗口。空白或低置信度的 OCR 因此会让旧 POV 过期，
        # 而不是永远保留一个陈旧的玩家。
        if text and confidence >= self.min_confidence:
            self._items.append((text, confidence))
        else:
            self._items.append(("", 0.0))
        weights: dict[str, float] = {}
        render: dict[str, str] = {}
        for observed, score in self._items:
            key = self._key(observed)
            if not key:
                continue
            weights[key] = weights.get(key, 0.0) + score
            render.setdefault(key, observed)
        if not weights:
            return result
        winner = max(weights, key=weights.get)
        stable = render[winner]
        if winner != self._key(text):
            result = dict(result)
            result["raw_text_raw"] = text
            result["raw_text"] = stable
            result["consensus"] = True
        return result


def _regions_by_type(background: dict, names: set[str]) -> list[dict]:
    """按 label 或 type 命中指定名称集合，筛出对应的 HUD 区域。"""
    return [
        region for region in background.get("regions", []) or []
        if str(region.get("label", "")).lower() in names or str(region.get("type", "")).lower() in names
    ]


def _first_pov_region(background: dict) -> dict | None:
    regions = _regions_by_type(background, {"pov_name", "pov_name_area", "pov_player_bar", "pov_marker_bar"})
    return max(regions, key=lambda item: float(item.get("confidence", 0.0))) if regions else None


def _first_timer_region(background: dict) -> dict | None:
    regions = _regions_by_type(background, {"timer", "timer_area", "round_timer"})
    return max(regions, key=lambda item: float(item.get("confidence", 0.0))) if regions else None


def _first_score_region(background: dict) -> dict | None:
    regions = _regions_by_type(background, {"score", "score_area", "top_hud_score", "top_hud"})
    return max(regions, key=lambda item: float(item.get("confidence", 0.0))) if regions else None


def _resolve_ocr_box(yolo_region: dict | None, fixed_cfg: dict, frame_shape, yolo_source_name: str = "yolo") -> tuple[dict | None, str]:
    if yolo_region is not None:
        return yolo_region, yolo_source_name
    if fixed_cfg.get("enabled", False) and fixed_cfg.get("box"):
        try:
            parts = [float(x.strip()) for x in str(fixed_cfg["box"]).split(",")]
            if len(parts) == 4:
                h, w = frame_shape[:2]
                return {"box": box_from_norm(parts, w, h), "label": "fixed_roi", "confidence": 1.0}, "fixed_roi"
        except (TypeError, ValueError, IndexError):
            pass
    return None, "no_region"


def _detect_pov_ocr(frame, yolo_background: dict | None, pov_ocr_config: dict, crop_padding: int) -> tuple[dict, str, dict | None]:
    yolo_region = _first_pov_region(yolo_background or {})
    if not yolo_region:
        return {"raw_text": "", "confidence": 0.0, "engine": "no_region:yolo_pov_region", "region": None}, "no_yolo_pov_region", None
    white_ratio = pov_white_text_ratio(frame, yolo_region, pov_ocr_config, crop_padding)
    if white_ratio < float(pov_ocr_config.get("white_ratio_threshold", 0.01)):
        return {
            "raw_text": "", "confidence": 0.0, "engine": "skipped:pov_white_gate",
            "region": yolo_region, "white_ratio": white_ratio,
        }, "yolo_pov_white_gate", yolo_region
    region, source = yolo_region, "yolo_pov_region"
    result = read_ocr_text(
        frame,
        region,
        padding=crop_padding,
        early_confidence=float(pov_ocr_config.get("early_accept_confidence", 0.75)),
    )
    result["white_ratio"] = white_ratio
    if float(result.get("confidence") or 0.0) < float(pov_ocr_config.get("min_confidence", 0.35)):
        result["raw_text_raw"] = result.get("raw_text", "")
        result["raw_text"] = ""
        result["low_confidence"] = True
    return result, source, region


def pov_white_text_ratio(frame, region: dict, pov_ocr_config: dict, crop_padding: int) -> float:
    """估计 YOLO 定位到的 POV 名条内是否含有白色玩家名文字。"""
    try:
        import cv2

        crop = crop_frame(frame, region.get("box", []), padding=crop_padding)
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white = (hsv[:, :, 1] <= int(pov_ocr_config.get("white_saturation_max", 60))) & (
            hsv[:, :, 2] >= int(pov_ocr_config.get("white_value_min", 220))
        )
        return round(float(white.mean()), 4)
    except Exception:
        return 0.0


def _detect_score_ocr(frame, yolo_background: dict | None, score_ocr_config: dict, crop_padding: int) -> dict:
    yolo_region = _first_score_region(yolo_background or {})
    region, source = _resolve_ocr_box(yolo_region, score_ocr_config, frame.shape, "yolo_score_region")
    if not region:
        return {"ct": None, "t": None, "raw": "", "source": source, "confidence": 0.0}
    ocr = read_ocr_text(
        frame,
        region,
        padding=crop_padding,
        accept_pattern=r"\d+\s*[:\-]\s*\d+",
    )
    raw = str(ocr.get("raw_text", ""))
    match = re.search(r"(\d+)\s*[:\-]\s*(\d+)", raw)
    return {
        "ct": int(match.group(1)) if match else None, "t": int(match.group(2)) if match else None,
        "raw": raw, "source": source, "confidence": ocr.get("confidence", 0.0),
    }


def _maskable_regions(regions: list[dict]) -> list[dict]:
    """OCR/路由消费完裁剪后，返回所有可供遮罩的 HUD 区域（带 box 的）。"""
    return [region for region in regions if isinstance(region, dict) and region.get("box")]
