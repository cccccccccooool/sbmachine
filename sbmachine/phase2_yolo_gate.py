"""第二阶段（YOLO UI 路由门槛）。利用 YOLO 模型检测画面中的 UI 区域，并将各个区域坐标分发给 OCR、VLM 遮罩等不同感知模块。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vision_service.region_crops import region_type, screen_side_for_label


@dataclass
class GateDecision:
    should_describe: bool
    reason: str
    hint: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    background: dict = field(default_factory=dict)


class YoloGate:
    """
    第二阶段 UI 区域路由器。
    """

    def __init__(self, config: dict) -> None:
        self.config = config or {}
        self.model_path = str(self.config.get("model_path", "")).strip()
        if not self.model_path:
            raise ValueError("Phase 2 UI YOLO requires vision.yolo.model_path")

        resolved_path = Path(self.model_path)
        if not resolved_path.is_absolute():
            from sbmachine.common import PROJECT_ROOT
            resolved_path = PROJECT_ROOT / resolved_path

        if not resolved_path.exists():
            container_fallback = Path("/opt/models_workspace/yolo_cs2.pt")
            if container_fallback.exists():
                resolved_path = container_fallback

        self.model_path = str(resolved_path)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Missing ultralytics. Install it before running Phase 2 UI YOLO.") from exc
        self.model = YOLO(self.model_path)
        self.model.to("cpu")  # force CPU to avoid competing with VLM for GPU VRAM
        self.conf_threshold = float(self.config.get("conf_threshold", 0.35))
        self.prompt_labels = self.config.get("prompt_labels", {})
        self.skip_labels = set(self.config.get("skip_labels", []))
        self.pov_name_labels = {"pov_name", "pov_name_area", "pov_player_bar", "pov_marker_bar"}
        self.ocr_labels = {"timer", "timer_area", "round_timer"}
        self.c4_labels = {"c4", "c4_area", "c4_status"}
        self.coordinate_only_labels = {"minimap", "radar", "minimap_area", "top_hud", "top_ui", "score", "score_area"}

    def decide(self, frame: Any) -> GateDecision:
        if self._is_near_white(frame):
            return GateDecision(False, "flash_or_white_frame", "skip near-white frame", ["flash"], 1.0)

        result = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        names = getattr(result, "names", {}) or {}
        tags: list[str] = []
        detections: list[dict] = []
        confidence = 0.0
        for box in getattr(result, "boxes", []) or []:
            cls_id = int(box.cls[0])
            label = str(names.get(cls_id, cls_id))
            score = float(box.conf[0])
            if score < self.conf_threshold:
                continue
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({"label": label, "confidence": score, "box": xyxy})
            tags.append(f"{label}({score:.2f})")
            confidence = max(confidence, score)

        labels = {tag.split("(", 1)[0] for tag in tags}
        background = self.structure_background(detections)
        if labels & self.skip_labels:
            return GateDecision(False, "ui_yolo_skip_label", "skip label from UI YOLO", tags, confidence, background)
        if not tags:
            return GateDecision(False, "no_ui_yolo_signal", "no UI YOLO region detected", [], 0.0, background)

        hints = [self.prompt_labels.get(label, "") for label in sorted(labels)]
        hint = " / ".join(item for item in hints if item) or "YOLO located UI regions; route crops to local/OCR/C4 readers. Global VLM sees UI-masked scene only."
        return GateDecision(True, "ui_yolo_signal", hint, tags, confidence, background)

    def _is_near_white(self, frame: Any) -> bool:
        threshold = float(self.config.get("white_frame_mean_threshold", 245))
        try:
            return bool(frame.mean() >= threshold)
        except Exception:
            return False

    def structure_background(self, detections: list[dict]) -> dict:
        regions = []
        player_hud_groups = []
        ocr_regions = []
        c4_regions = []
        coordinate_only_regions = []
        loose = []
        for det in detections:
            label = det["label"]
            rtype = region_type(label)
            if rtype == "unknown":
                loose.append(det)
                continue
            region = {
                "label": label,
                "type": rtype,
                "box": det["box"],
                "confidence": det["confidence"],
                "screen_side": screen_side_for_label(label),
            }
            regions.append(region)
            if label in self.pov_name_labels:
                region["send_to_ocr"] = True
                ocr_regions.append(region)
            elif label in self.ocr_labels:
                region["send_to_ocr"] = True
                ocr_regions.append(region)
            elif label in self.c4_labels:
                region["send_to_c4_detector"] = True
                c4_regions.append(region)
            elif label in self.coordinate_only_labels:
                coordinate_only_regions.append(region)
            if rtype == "player_hud_group":
                player_hud_groups.append(region)

        return {
            "type": "yolo_ui_region_router",
            "regions": regions,
            "player_hud_groups": player_hud_groups,
            "ocr_regions": ocr_regions,
            "c4_regions": c4_regions,
            "coordinate_only_regions": coordinate_only_regions,
            "detections": detections,
            "loose_detections": loose,
            "role_resolution": {
                "status": "region_router_only",
                "rule": "YOLO only returns coordinates. timer and pov_player_bar go to OCR; all other UI boxes except timer/c4 are masked before VLM.",
            },
            "reserved_future": {
                "c4": "C4 visual boxes are retained as coordinates but are not masked and not used for plant detection."
            },
        }


YoloUiDetector = YoloGate
