"""Phase 2 OCR alignment, budget, and unsupported quality summaries."""
from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class OcrBudget:
    """Count OCR calls by ROI; enhanced variants are tracked separately."""

    baseline_roi_calls: int
    normal_ratio: float = 1.0
    degraded_extra_ratio: float = 0.2
    timer_roi_calls: int = 0
    score_left_roi_calls: int = 0
    score_right_roi_calls: int = 0
    variant_inference_calls: int = 0
    budget_exhausted: bool = False

    @property
    def actual_roi_calls(self) -> int:
        return self.timer_roi_calls + self.score_left_roi_calls + self.score_right_roi_calls

    @property
    def normal_limit(self) -> int:
        ratio = min(1.0, max(0.0, float(self.normal_ratio)))
        return max(0, int(self.baseline_roi_calls * ratio))

    @property
    def hard_limit(self) -> int:
        extra_ratio = min(0.2, max(0.0, float(self.degraded_extra_ratio)))
        return max(self.normal_limit, int(self.baseline_roi_calls * (1.0 + extra_ratio)))

    def can_consume(self, roi_calls: int, *, degraded: bool = False) -> bool:
        limit = self.hard_limit if degraded else self.normal_limit
        allowed = self.actual_roi_calls + int(roi_calls) <= limit
        if not allowed:
            self.budget_exhausted = True
        return allowed

    def consume(
        self,
        kind: str,
        roi_calls: int,
        *,
        variant_calls: int = 0,
        score_sides: tuple[str, ...] | None = None,
    ) -> None:
        if kind == "timer":
            self.timer_roi_calls += int(roi_calls)
        elif kind == "score":
            if score_sides is None:
                self.score_left_roi_calls += int(roi_calls > 0)
                self.score_right_roi_calls += int(roi_calls > 1)
            else:
                self.score_left_roi_calls += int("left" in score_sides)
                self.score_right_roi_calls += int("right" in score_sides)
        else:
            raise ValueError(f"unknown OCR budget kind: {kind}")
        self.variant_inference_calls += int(variant_calls)

    def summary(self) -> dict:
        change = (
            ((self.actual_roi_calls - self.baseline_roi_calls) / self.baseline_roi_calls) * 100.0
            if self.baseline_roi_calls else 0.0
        )
        return {
            "baseline_roi_calls": self.baseline_roi_calls,
            "actual_roi_calls": self.actual_roi_calls,
            "timer_roi_calls": self.timer_roi_calls,
            "score_left_roi_calls": self.score_left_roi_calls,
            "score_right_roi_calls": self.score_right_roi_calls,
            "variant_inference_calls": self.variant_inference_calls,
            "change_percent": round(change, 1),
            "normal_limit": self.normal_limit,
            "hard_limit": self.hard_limit,
            "budget_exhausted": self.budget_exhausted,
        }


def timer_status_counts(observations: list[dict]) -> dict:
    counts = {key: 0 for key in (
        "not_scheduled", "no_region", "ocr_empty", "parse_rejected",
        "state_rejected", "accepted", "budget_exhausted",
    )}
    for item in observations:
        alignment_status = str(item.get("alignment_status", ""))
        parse_status = str(item.get("parse_status", ""))
        status = alignment_status if alignment_status in {"state_rejected", "accepted", "budget_exhausted"} else parse_status
        if status in counts:
            counts[status] += 1
    return counts


def build_alignment_warning(
    result: dict,
    timer_observations: list[dict],
    score_observations: list[dict],
    budget_summary: dict,
) -> dict:
    """Build the Phase2 alignment summary in the existing warning channel."""
    return {
        "type": "ocr_alignment",
        "status": result.get("status", "alignment_unresolved"),
        "confidence": result.get("alignment_confidence", "unknown"),
        "timer": timer_status_counts(timer_observations),
        "score": {
            "pair_accepted": sum(item.get("pair_status") == "accepted_for_alignment" for item in score_observations),
            "pair_incomplete": sum(item.get("pair_status") == "incomplete" for item in score_observations),
            "score_fact_support": result.get("score_fact_support", "unsupported"),
            "alignment_evidence": result.get("score_evidence", "none"),
        },
        "c4_evidence": result.get("c4_evidence", "unsupported"),
        "c4_support_reason": result.get("c4_support_reason", "unknown"),
        "c4_candidate_onsets": result.get("c4_candidate_onsets", 0),
        "c4_residual_sec": result.get("c4_residual_sec"),
        "mapping": {
            "slope_tick_per_sec": result.get("slope_tick_per_sec"),
            "offset_tick": result.get("offset_tick"),
            "residual_median_sec": result.get("residual_median_sec"),
            "state_history": result.get("state_history", []),
        },
        "ocr_calls": dict(budget_summary),
    }
