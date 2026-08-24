"""第二阶段：OCR 识别、候选筛选与短窗口共识（consensus）的辅助函数。"""
from __future__ import annotations

from collections import deque
import math
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
    min_accept_confidence: float = 0.0,
) -> dict:
    """识别单个 OCR 区域，一旦得到调用方可用的结果就提前停止。

    识别失败或不确定的裁剪仍会尝试每一种增强变体。计时器/比分调用方可在
    文本格式合法时停止，POV 调用方则可在置信度达标时停止。
    """
    crop = crop_frame(frame, region.get("box", []), padding=padding)
    ocr_engine = _get_rapid_ocr()
    if ocr_engine is None:
        return {
            "region": region, "raw_text": "", "confidence": 0.0,
            "engine": "unavailable:ImportError", "variant_inference_calls": 0,
        }
    candidates: list[dict] = []
    accepted: dict | None = None
    engine = "rapidocr_onnxruntime"
    variant_calls = 0
    for variant_name, image in _variants(crop):
        try:
            variant_calls += 1
            result, _ = ocr_engine(image)
            text, confidence = _join_ocr_lines(result)
            if text:
                candidate = {"text": text, "confidence": confidence, "variant": variant_name}
                candidates.append(candidate)
                pattern_accepted = bool(
                    accept_pattern
                    and confidence >= float(min_accept_confidence)
                    and re.search(accept_pattern, text)
                )
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
        "variant_inference_calls": variant_calls,
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
    regions = _regions_by_type(background, {"score", "score_area"})
    return max(regions, key=lambda item: float(item.get("confidence", 0.0))) if regions else None


def _score_regions(background: dict) -> list[dict]:
    """Return only single-side score boxes; top_hud is never a score candidate."""
    regions = _regions_by_type(background, {"score", "score_area"})
    return sorted(
        regions,
        key=lambda item: (
            (float(item.get("box", [0, 0, 0, 0])[0]) + float(item.get("box", [0, 0, 0, 0])[2])) / 2.0,
            -float(item.get("confidence", 0.0)),
        ),
    )


def parse_timer_observation(
    raw_text: object,
    *,
    video_time: float,
    confidence: float = 0.0,
    roi_source: str = "yolo_timer_region",
    variant: str = "none",
    min_confidence: float = 0.35,
) -> dict:
    """Parse OCR text into an uncommitted timer observation with one failure class."""
    raw = str(raw_text or "").strip()
    base = {
        "kind": "timer", "video_time": round(float(video_time), 3), "raw_text": raw,
        "normalized": "", "timer_sec": None, "ocr_confidence": float(confidence or 0.0),
        "roi_source": str(roi_source), "variant": str(variant or "none"),
        "parse_status": "ocr_empty" if not raw else "parse_rejected",
        "alignment_status": "pending", "value": "",
    }
    if not raw:
        return base
    normalized = re.sub(r"\s+", "", raw.replace("\uFF1A", ":"))
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", normalized)
    if not match:
        base["normalized"] = normalized
        return base
    minutes, seconds = int(match.group(1)), int(match.group(2))
    timer_sec = minutes * 60 + seconds
    base["normalized"] = f"{minutes}:{seconds:02d}"
    if seconds > 59 or not 0 <= timer_sec <= 115:
        return base
    base["timer_sec"] = timer_sec
    base["parse_status"] = "parsed"
    base["value"] = base["normalized"]
    if float(confidence or 0.0) < float(min_confidence):
        base["alignment_status"] = "state_rejected"
    return base


def unsampled_timer_observation(video_time: float, status: str, *, source: str = "") -> dict:
    """Create an explicit timer status when OCR was not called."""
    if status not in {"not_scheduled", "no_region", "budget_exhausted"}:
        raise ValueError(f"invalid timer observation status: {status}")
    return {
        "kind": "timer", "video_time": round(float(video_time), 3), "raw_text": "",
        "normalized": "", "timer_sec": None, "ocr_confidence": 0.0,
        "roi_source": source or status, "variant": "none", "parse_status": status,
        "alignment_status": status, "value": "", "variant_inference_calls": 0,
    }


def _normalize_score_digit(raw_text: object) -> str:
    text = re.sub(r"\s+", "", str(raw_text or ""))
    return "1" if text in {"I", "i", "L", "l", "+", "/", "\uFF0F"} else text


def parse_score_side(raw_text: object, *, confidence: float, roi_source: str, max_value: int = 30) -> dict:
    """Strictly parse one score box without extracting digits from longer text."""
    raw = str(raw_text or "").strip()
    normalized = _normalize_score_digit(raw)
    parsed = bool(re.fullmatch(r"\d{1,2}", normalized))
    value = int(normalized) if parsed else None
    if value is not None and not 0 <= value <= int(max_value):
        value, parsed = None, False
    return {
        "value": value, "raw_text": raw, "normalized": normalized,
        "ocr_confidence": float(confidence or 0.0), "roi_source": roi_source,
        "parse_status": "parsed" if parsed else ("ocr_empty" if not raw else "parse_rejected"),
    }


def normalize_score_observation(observation: object, *, video_time: float = 0.0) -> dict:
    """Normalize live and legacy score debug payloads without promoting them to facts."""
    if not isinstance(observation, dict):
        return {
            "kind": "score_observation",
            "video_time": round(float(video_time), 3),
            "left": None,
            "right": None,
            "pair_status": "not_scheduled",
            "observation_status": "not_scheduled",
            "ct": None,
            "t": None,
            "source": "not_scheduled",
            "confidence": 0.0,
            "variant_inference_calls": 0,
        }
    result = dict(observation)
    is_live_observation = (
        result.get("kind") == "score_observation"
        or "left" in result
        or "right" in result
    )
    if not is_live_observation and ("ct" in result or "t" in result):
        result.update({
            "kind": "score_observation",
            "video_time": round(float(result.get("video_time", video_time)), 3),
            "left": None,
            "right": None,
            "pair_status": "legacy_unverified",
            "observation_status": "legacy_unverified",
            "legacy_ct": result.get("ct"),
            "legacy_t": result.get("t"),
        })
    else:
        result.setdefault("kind", "score_observation")
        result.setdefault("video_time", round(float(video_time), 3))
        result.setdefault("left", None)
        result.setdefault("right", None)
        result.setdefault("pair_status", "incomplete")
        result.setdefault("observation_status", "observed")
    result["ct"] = None
    result["t"] = None
    result.setdefault("source", "score_observation")
    result.setdefault("confidence", 0.0)
    result.setdefault("variant_inference_calls", 0)
    return result


class ScorePairConsensus:
    """Build short-window consensus without mapping screen sides to CT/T."""

    def __init__(self, window: int = 5, min_votes: int | None = None) -> None:
        self.window = max(1, int(window))
        self.min_votes = int(min_votes) if min_votes is not None else max(1, math.ceil(self.window * 0.6))
        self._items: deque[dict] = deque(maxlen=self.window)

    def update(self, observation: dict) -> dict:
        self._items.append(dict(observation))
        usable = [
            item for item in self._items
            if isinstance(item.get("left"), dict)
            and isinstance(item.get("right"), dict)
            and item["left"].get("value") is not None
            and item["right"].get("value") is not None
        ]
        result = dict(observation)
        if len(self._items) < self.window:
            result["pair_status"] = (
                "incomplete" if observation.get("pair_status") == "incomplete" else "pending_consensus"
            )
            return result
        weights: dict[tuple[int, int], float] = {}
        votes: dict[tuple[int, int], int] = {}
        for item in usable:
            pair = (int(item["left"]["value"]), int(item["right"]["value"]))
            confidence = (
                float(item["left"].get("ocr_confidence", 0.0))
                + float(item["right"].get("ocr_confidence", 0.0))
            ) / 2.0
            weights[pair] = weights.get(pair, 0.0) + confidence
            votes[pair] = votes.get(pair, 0) + 1
        if not votes:
            result["pair_status"] = "incomplete"
            return result
        ranked = sorted(votes, key=lambda pair: (votes[pair], weights[pair], pair), reverse=True)
        winner = ranked[0]
        tied = len(ranked) > 1 and votes[ranked[1]] == votes[winner] and weights[ranked[1]] == weights[winner]
        if votes[winner] < self.min_votes or tied:
            result["pair_status"] = "conflict"
            return result
        selected = next(
            item for item in reversed(usable)
            if int(item["left"]["value"]) == winner[0] and int(item["right"]["value"]) == winner[1]
        )
        result.update({
            "left": dict(selected["left"]), "right": dict(selected["right"]),
            "pair_status": "accepted_for_alignment",
            "confidence": round(weights[winner] / votes[winner], 3),
        })
        return result


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


def _detect_score_ocr(
    frame,
    yolo_background: dict | None,
    score_ocr_config: dict,
    crop_padding: int,
    *,
    video_time: float = 0.0,
) -> dict:
    """Read left and right score boxes independently; never fill a missing side."""
    regions = _score_regions(yolo_background or {})
    if not regions:
        return {
            "kind": "score_observation", "video_time": round(float(video_time), 3),
            "left": None, "right": None, "pair_status": "incomplete", "ct": None, "t": None,
            "source": "no_region:yolo_score_pair", "confidence": 0.0, "variant_inference_calls": 0,
        }
    if len(regions) >= 2:
        selected_regions = {"left": regions[0], "right": regions[-1]}
    else:
        box = regions[0].get("box", [0, 0, 0, 0])
        center = (float(box[0]) + float(box[2])) / 2.0
        side = "left" if center < float(frame.shape[1]) / 2.0 else "right"
        selected_regions = {side: regions[0]}
    max_value = int(score_ocr_config.get("max_observed_value", 30))
    min_confidence = float(score_ocr_config.get("min_confidence", 0.35))
    parsed_sides: dict[str, dict | None] = {"left": None, "right": None}
    variant_calls = 0
    for side, region in selected_regions.items():
        ocr = read_ocr_text(
            frame,
            region,
            padding=crop_padding,
            accept_pattern=r"^\s*(?:\d{1,2}|[IiLl+\uFF0F/])\s*$",
            min_accept_confidence=min_confidence,
        )
        variant_calls += int(ocr.get("variant_inference_calls", 0))
        parsed = parse_score_side(
            ocr.get("raw_text", ""),
            confidence=float(ocr.get("confidence", 0.0)),
            roi_source=f"yolo_score_{side}",
            max_value=max_value,
        )
        if parsed["ocr_confidence"] < min_confidence and parsed["parse_status"] == "parsed":
            parsed["parse_status"], parsed["value"] = "state_rejected", None
        parsed_sides[side] = parsed
    complete = all(isinstance(side, dict) and side["value"] is not None for side in parsed_sides.values())
    observed = [side for side in parsed_sides.values() if isinstance(side, dict)]
    confidence = sum(side["ocr_confidence"] for side in observed) / len(observed)
    return {
        "kind": "score_observation", "video_time": round(float(video_time), 3),
        "left": parsed_sides["left"], "right": parsed_sides["right"],
        "pair_status": "pending_consensus" if complete else "incomplete",
        "observation_status": "observed",
        "ct": None, "t": None, "source": "yolo_score_pair", "confidence": round(confidence, 3),
        "variant_inference_calls": variant_calls,
        "_regions": dict(selected_regions),
    }


def _maskable_regions(regions: list[dict]) -> list[dict]:
    """OCR/路由消费完裁剪后，返回所有可供遮罩的 HUD 区域（带 box 的）。"""
    return [region for region in regions if isinstance(region, dict) and region.get("box")]
