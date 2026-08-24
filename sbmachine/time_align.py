"""回合内 video-time 到 demo-tick 的对齐器，附带精简诊断信息。"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field


def parse_timer_seconds(timer: str) -> float | None:
    text = str(timer or "").strip()
    if not text or ":" not in text:
        return None
    left, right = text.split(":", 1)
    try:
        minutes, seconds = int(left), int(right[:2])
    except ValueError:
        return None
    return float(minutes * 60 + seconds) if 0 <= seconds < 60 else None


class C4VisualTracker:
    """Reduce existing YOLO C4 boxes to a stable onset without creating C4 facts."""

    def __init__(self, window: int = 3, min_present: int = 2) -> None:
        self.window = max(1, int(window))
        self.min_present = max(1, min(int(min_present), self.window))
        self._recent: deque[bool] = deque(maxlen=self.window)
        self._stable_present = False

    def update(self, video_time: float, regions: list[dict] | None) -> dict:
        supported = regions is not None
        items = list(regions or [])
        present = bool(items)
        self._recent.append(present)
        stable = len(self._recent) == self.window and sum(self._recent) >= self.min_present
        transition = "candidate_onset" if stable and not self._stable_present else "none"
        if stable:
            self._stable_present = True
        elif len(self._recent) == self.window and not any(self._recent):
            self._stable_present = False
        return {
            "kind": "c4_visual",
            "video_time": round(float(video_time), 3),
            "state": "present" if present else "absent",
            "confidence": max((float(item.get("confidence", 0.0)) for item in items), default=0.0),
            "source": (
                "existing_yolo_c4_region" if items
                else "existing_yolo_c4_region_absent" if supported
                else "unsupported"
            ),
            "source_supported": supported,
            "transition": transition,
        }


@dataclass
class RoundTimeAlign:
    round_meta: dict
    tick_rate: float
    anchor_tolerance_sec: float = 2.0
    max_anchor_error_sec: float = 2.0
    lock_min_samples: int = 3
    residual_tolerance_sec: float = 2.0
    max_drift_ratio: float = 0.05
    reset_low_sec: float = 10.0
    reset_high_sec: float = 100.0
    reset_confirm_samples: int = 3
    offsets: list[float] = field(default_factory=list)
    frozen_offset: float | None = None
    provisional_offset: float | None = None
    warnings: list[str] = field(default_factory=list)
    state: str = "UNANCHORED"
    slope_tick_per_sec: float | None = None
    offset_tick: float | None = None
    observations: list[dict] = field(default_factory=list)
    result: dict | None = None
    _last_timer_sec: float | None = field(default=None, repr=False, compare=False)
    _pending_anchor: tuple[float, float] | None = field(default=None, repr=False, compare=False)
    _reported_warning_count: int = field(default=0, repr=False, compare=False)

    @property
    def is_frozen(self) -> bool:
        return self.frozen_offset is not None or self.state == "LOCKED"

    @property
    def is_locked(self) -> bool:
        return self.state == "LOCKED" and self.slope_tick_per_sec is not None and self.offset_tick is not None

    def observe_timer(self, observation: dict) -> None:
        """Collect a timer observation without mutating the committed mapping."""
        self.observations.append(observation)

    @staticmethod
    def _median(values: list[float]) -> float:
        return float(statistics.median(values)) if values else 0.0

    def _timer_tick(self, timer_sec: float) -> float:
        freeze_end_tick = int(self.round_meta.get("freeze_end_tick", self.round_meta.get("start_tick", 0)))
        return float(freeze_end_tick) + (115.0 - float(timer_sec)) * self.tick_rate

    def _confirmed_reset_times(self, candidates: list[dict]) -> list[float]:
        resets: list[float] = []
        confirm = max(2, int(self.reset_confirm_samples))
        for index in range(1, len(candidates)):
            previous = float(candidates[index - 1]["timer_sec"])
            current = float(candidates[index]["timer_sec"])
            if previous > self.reset_low_sec or current < self.reset_high_sec:
                continue
            if index > 1:
                before_previous = float(candidates[index - 2]["timer_sec"])
                if before_previous > previous + 5.0:
                    continue
            cluster = candidates[index:index + confirm]
            if len(cluster) < confirm:
                continue
            valid = True
            for left, right in zip(cluster, cluster[1:]):
                dt = float(right["video_time"]) - float(left["video_time"])
                timer_drop = float(left["timer_sec"]) - float(right["timer_sec"])
                if dt <= 0 or timer_drop < -1.0 or timer_drop > dt + 2.0:
                    valid = False
                    break
            if valid:
                resets.append(float(candidates[index]["video_time"]))
        return sorted(set(resets))

    def _fit_cluster(self, candidates: list[dict]) -> tuple[list[dict], float, float, float]:
        """Filter robust offsets, then fit bounded drift around the demo tick rate."""
        if not candidates:
            return [], self.tick_rate, 0.0, float("inf")
        offsets = [self._timer_tick(item["timer_sec"]) - float(item["video_time"]) * self.tick_rate for item in candidates]
        center = self._median(offsets)
        accepted = [
            item for item, offset in zip(candidates, offsets)
            if abs(offset - center) / self.tick_rate <= self.residual_tolerance_sec
        ]
        if len(accepted) < self.lock_min_samples:
            return accepted, self.tick_rate, center, float("inf")
        xs = [float(item["video_time"]) for item in accepted]
        ys = [self._timer_tick(item["timer_sec"]) for item in accepted]
        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        denominator = sum((value - mean_x) ** 2 for value in xs)
        fitted_slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
            if denominator > 1e-9 else self.tick_rate
        )
        if abs(fitted_slope - self.tick_rate) / self.tick_rate > self.max_drift_ratio:
            return [], self.tick_rate, center, float("inf")
        fitted_offset = self._median([y - fitted_slope * x for x, y in zip(xs, ys)])
        residuals = [abs((fitted_slope * x + fitted_offset) - y) / self.tick_rate for x, y in zip(xs, ys)]
        residual = self._median(residuals)
        accepted = [item for item, value in zip(accepted, residuals) if value <= self.residual_tolerance_sec]
        return accepted, float(fitted_slope), float(fitted_offset), residual

    def solve(
        self,
        *,
        source_start_sec: float,
        source_end_sec: float,
        score_observations: list[dict] | None = None,
        c4_observations: list[dict] | None = None,
    ) -> dict:
        """Commit one mapping from the segment; unresolved input exposes no formal tick."""
        history = ["UNANCHORED"]
        parsed_total = sum(item.get("parse_status") == "parsed" for item in self.observations)
        candidates = sorted(
            [
                item for item in self.observations
                if item.get("parse_status") == "parsed"
                and item.get("alignment_status") != "state_rejected"
                and item.get("timer_sec") is not None
            ],
            key=lambda item: float(item["video_time"]),
        )
        history.append("ACQUIRING")
        reset_times = self._confirmed_reset_times(candidates)
        effective_start = float(source_start_sec)
        unsupported_reason = ""
        if len(reset_times) > 1:
            unsupported_reason = "unsupported_multi_round_segment"
        elif reset_times:
            effective_start = reset_times[0]
            candidates = [item for item in candidates if float(item["video_time"]) >= effective_start]

        reanchored = False
        if len(candidates) >= self.lock_min_samples * 2:
            offsets = [
                self._timer_tick(item["timer_sec"]) - float(item["video_time"]) * self.tick_rate
                for item in candidates
            ]
            first_center = self._median(offsets[:self.lock_min_samples])
            last_center = self._median(offsets[-self.lock_min_samples:])
            if abs(last_center - first_center) / self.tick_rate > self.residual_tolerance_sec * 2.0:
                start = len(candidates) - 1
                while start > 0 and abs(offsets[start - 1] - last_center) / self.tick_rate <= self.residual_tolerance_sec:
                    start -= 1
                if len(candidates) - start >= self.lock_min_samples:
                    candidates = candidates[start:]
                    reanchored = True

        accepted, slope, offset, residual = self._fit_cluster(candidates)
        if len(accepted) >= self.lock_min_samples and not unsupported_reason:
            accepted_ids = {id(item) for item in accepted}
            for item in self.observations:
                if id(item) in accepted_ids:
                    item["alignment_status"] = "accepted"
                elif item.get("parse_status") == "parsed" and item.get("alignment_status") == "pending":
                    item["alignment_status"] = "state_rejected"
            self.slope_tick_per_sec = slope
            self.offset_tick = offset
            self.frozen_offset = offset
            self.state = "LOCKED"
            if reanchored:
                history.extend(["DEGRADED", "LOST", "REANCHORING"])
            history.append("LOCKED")
        else:
            for item in self.observations:
                if item.get("parse_status") == "parsed" and item.get("alignment_status") == "pending":
                    item["alignment_status"] = "state_rejected"
            self.state = "LOST"
            history.extend(["DEGRADED", "LOST"])
            unsupported_reason = unsupported_reason or "alignment_unresolved"

        score_items = score_observations or []
        score_accepted = sum(item.get("pair_status") == "accepted_for_alignment" for item in score_items)
        score_incomplete = sum(item.get("pair_status") == "incomplete" for item in score_items)
        accepted_score = next(
            (item for item in reversed(score_items) if item.get("pair_status") == "accepted_for_alignment"),
            None,
        )
        score_evidence = "none"
        if accepted_score is not None:
            observed_round = (
                int(accepted_score["left"]["value"])
                + int(accepted_score["right"]["value"])
                + 1
            )
            score_evidence = (
                "round_total_consistent"
                if observed_round == int(self.round_meta.get("round_no", 0))
                else "round_total_conflict"
            )
        score_fact_support = (
            "supported"
            if all(key in self.round_meta for key in ("score_before", "score_after"))
            else "unsupported"
        )
        c4_items = c4_observations or []
        onsets = [item for item in c4_items if item.get("transition") == "candidate_onset"]
        onset = onsets[0] if onsets else None
        c4_source_supported = any(item.get("source_supported") for item in c4_items)
        if not c4_source_supported:
            c4_evidence = "unsupported"
            c4_support_reason = "c4_regions_not_available"
        elif self.round_meta.get("bomb_planted_tick") is None and onset is not None:
            c4_evidence = "unsupported"
            c4_support_reason = "stable_onset_in_nonplant_round"
        elif self.round_meta.get("bomb_planted_tick") is None:
            c4_evidence = "not_applicable"
            c4_support_reason = "demo_round_has_no_plant"
        else:
            c4_evidence = "no_onset"
            c4_support_reason = "no_stable_visual_onset"
        c4_residual = None
        if onset is not None and self.is_locked and self.round_meta.get("bomb_planted_tick") is not None:
            plant_tick = float(self.round_meta["bomb_planted_tick"])
            predicted_video = (plant_tick - float(self.offset_tick)) / float(self.slope_tick_per_sec)
            c4_residual = abs(predicted_video - float(onset["video_time"]))
            c4_evidence = "consistent" if c4_residual <= self.anchor_tolerance_sec else "conflict"
            c4_support_reason = "residual_within_tolerance" if c4_evidence == "consistent" else "residual_conflict"
        alignment_confidence = (
            "unresolved"
            if not self.is_locked
            else "degraded"
            if score_evidence == "round_total_conflict" or c4_evidence == "conflict"
            else "high"
        )

        foreign_tail = None
        if effective_start > float(source_start_sec):
            foreign_tail = {
                "start_sec": round(float(source_start_sec), 3),
                "end_sec": round(effective_start - 1.0, 3),
            }
        self.result = {
            "status": "locked" if self.is_locked else unsupported_reason,
            "demo_round_no": int(self.round_meta.get("round_no", 0)),
            "source_start_sec": round(float(source_start_sec), 3),
            "effective_start_sec": round(effective_start, 3),
            "effective_end_sec": round(float(source_end_sec), 3),
            "slope_tick_per_sec": round(float(slope), 6) if self.is_locked else None,
            "offset_tick": round(float(offset), 3) if self.is_locked else None,
            "residual_median_sec": round(float(residual), 3) if residual != float("inf") else None,
            "timer_accepted": len(accepted),
            "timer_rejected": max(0, parsed_total - len(accepted)),
            "score_evidence": score_evidence,
            "score_pair_accepted": score_accepted,
            "score_pair_incomplete": score_incomplete,
            "score_fact_support": score_fact_support,
            "c4_evidence": c4_evidence,
            "c4_support_reason": c4_support_reason,
            "c4_candidate_onsets": len(onsets),
            "c4_residual_sec": round(c4_residual, 3) if c4_residual is not None else None,
            "alignment_confidence": alignment_confidence,
            "foreign_tail": foreign_tail,
            "state_history": history,
            "confirmed_resets": [round(value, 3) for value in reset_times],
        }
        return dict(self.result)

    def add_warning(self, text: str) -> None:
        """诊断信息去重：重复的 OCR 失败不应让每一行都膨胀。"""
        if text not in self.warnings:
            self.warnings.append(text)

    def take_new_warnings(self) -> list[str]:
        new = self.warnings[self._reported_warning_count:]
        self._reported_warning_count = len(self.warnings)
        return list(new)

    def add_anchor(self, video_time: float, timer_str: str) -> int | None:
        timer_sec = parse_timer_seconds(timer_str)
        if timer_sec is None:
            return None
        if not 0.0 <= timer_sec <= 115.0:
            self.add_warning(f"skip out-of-range anchor timer={timer_str} ({timer_sec:.0f}s)")
            return None
        if self._last_timer_sec is not None and timer_sec > self._last_timer_sec + 3.0:
            self.add_warning(f"skip non-monotone anchor timer={timer_str} ({timer_sec:.0f}s > prev {self._last_timer_sec:.0f}s + 3)")
            return None
        relative_sec = 115.0 - timer_sec
        freeze_end_tick = int(self.round_meta.get("freeze_end_tick", self.round_meta.get("start_tick", 0)))
        tick = int(round(freeze_end_tick + relative_sec * self.tick_rate))
        offset = tick - float(video_time) * self.tick_rate
        if self.offsets and abs(offset - statistics.median(self.offsets)) > self.max_anchor_error_sec * self.tick_rate:
            self.add_warning(f"drop outlier anchor timer={timer_str}")
            return tick

        if not self.offsets and self.provisional_offset is not None:
            first_tolerance_sec = max(self.max_anchor_error_sec, self.anchor_tolerance_sec * 2.0)
            deviation_sec = abs(offset - self.provisional_offset) / self.tick_rate
            if deviation_sec > first_tolerance_sec:
                self.add_warning(
                    f"drop first anchor timer={timer_str}: provisional offset deviation={deviation_sec:.1f}s"
                )
                return tick
        elif not self.offsets:
            if self._pending_anchor is None:
                self._pending_anchor = (offset, timer_sec)
                return tick
            pending_offset, pending_timer_sec = self._pending_anchor
            if timer_sec > pending_timer_sec + 3.0:
                self.add_warning(
                    f"skip non-monotone anchor timer={timer_str} ({timer_sec:.0f}s > pending {pending_timer_sec:.0f}s + 3)"
                )
                return tick
            deviation_sec = abs(offset - pending_offset) / self.tick_rate
            if deviation_sec > self.max_anchor_error_sec:
                self.add_warning(f"first anchor not confirmed; deviation={deviation_sec:.1f}s")
                self._pending_anchor = (offset, timer_sec)
                return tick
            self.offsets.append(pending_offset)
            self._pending_anchor = None

        self._last_timer_sec = timer_sec
        self.offsets.append(offset)
        return tick

    def freeze(self, video_time_event: float, event_tick: int | None = None) -> None:
        if event_tick is None:
            event_tick = self.round_meta.get("bomb_planted_tick")
        current_offset = self._current_offset()
        if current_offset is None:
            if event_tick is None:
                return
            current_offset = float(event_tick) - float(video_time_event) * self.tick_rate
        if event_tick is not None:
            expected_video_time = (float(event_tick) - current_offset) / self.tick_rate
            if abs(expected_video_time - float(video_time_event)) > self.anchor_tolerance_sec:
                self.add_warning(f"event freeze mismatch video={video_time_event:.3f} demo={expected_video_time:.3f} tick={event_tick}")
        self.frozen_offset = float(current_offset)

    def to_tick(self, video_time: float) -> int:
        slope = float(self.slope_tick_per_sec) if self.is_locked else self.tick_rate
        return int(round(float(video_time) * slope + self._effective_offset()))

    def to_video_time(self, tick: int) -> float:
        slope = float(self.slope_tick_per_sec) if self.is_locked else self.tick_rate
        return (float(tick) - self._effective_offset()) / slope

    def relative_sec_for_tick(self, tick: int) -> float:
        freeze_end_tick = int(self.round_meta.get("freeze_end_tick", self.round_meta.get("start_tick", 0)))
        return (int(tick) - freeze_end_tick) / self.tick_rate

    def _current_offset(self) -> float | None:
        if self.offsets:
            return float(statistics.median(self.offsets))
        return self.provisional_offset

    def _effective_offset(self) -> float:
        if self.is_locked and self.offset_tick is not None:
            return float(self.offset_tick)
        if self.frozen_offset is not None:
            return float(self.frozen_offset)
        current = self._current_offset()
        if current is not None:
            return float(current)
        return float(self.round_meta.get("freeze_end_tick", self.round_meta.get("start_tick", 0)))
