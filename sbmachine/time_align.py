"""回合内 video-time 到 demo-tick 的对齐器，附带精简诊断信息。"""
from __future__ import annotations

import statistics
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


@dataclass
class RoundTimeAlign:
    round_meta: dict
    tick_rate: float
    anchor_tolerance_sec: float = 2.0
    max_anchor_error_sec: float = 2.0
    offsets: list[float] = field(default_factory=list)
    frozen_offset: float | None = None
    provisional_offset: float | None = None
    warnings: list[str] = field(default_factory=list)
    _last_timer_sec: float | None = field(default=None, repr=False, compare=False)
    _pending_anchor: tuple[float, float] | None = field(default=None, repr=False, compare=False)
    _reported_warning_count: int = field(default=0, repr=False, compare=False)

    @property
    def is_frozen(self) -> bool:
        return self.frozen_offset is not None

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
        return int(round(float(video_time) * self.tick_rate + self._effective_offset()))

    def to_video_time(self, tick: int) -> float:
        return (float(tick) - self._effective_offset()) / self.tick_rate

    def relative_sec_for_tick(self, tick: int) -> float:
        freeze_end_tick = int(self.round_meta.get("freeze_end_tick", self.round_meta.get("start_tick", 0)))
        return (int(tick) - freeze_end_tick) / self.tick_rate

    def _current_offset(self) -> float | None:
        if self.offsets:
            return float(statistics.median(self.offsets))
        return self.provisional_offset

    def _effective_offset(self) -> float:
        if self.frozen_offset is not None:
            return float(self.frozen_offset)
        current = self._current_offset()
        if current is not None:
            return float(current)
        return float(self.round_meta.get("freeze_end_tick", self.round_meta.get("start_tick", 0)))
