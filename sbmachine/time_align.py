"""回合内时间对齐器。负责通过比分/计时器OCR锚点，计算视频时间与对局 tick 之间的偏移量，支持单调性检验、异常值剔除和基于炸弹事件的对齐锁定。"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


def parse_timer_seconds(timer: str) -> float | None:
    """将计时器字符串 (M:SS) 解析为秒数。"""
    text = str(timer or "").strip().replace(":", ":")
    if not text or ":" not in text:
        return None
    left, right = text.split(":", 1)
    try:
        minutes = int(left)
        seconds = int(right[:2])
    except ValueError:
        return None
    if not 0 <= seconds < 60:
        return None
    return float(minutes * 60 + seconds)


@dataclass
class RoundTimeAlign:
    round_meta: dict
    tick_rate: float
    anchor_tolerance_sec: float = 2.0
    max_anchor_error_sec: float = 2.0
    offsets: list[float] = field(default_factory=list)
    frozen_offset: float | None = None
    warnings: list[str] = field(default_factory=list)
    # 内部单调性 / 首个锚点状态
    _last_timer_sec: float | None = field(default=None, repr=False, compare=False)
    _first_anchor_done: bool = field(default=False, repr=False, compare=False)

    @property
    def is_frozen(self) -> bool:
        return self.frozen_offset is not None

    def add_anchor(self, video_time: float, timer_str: str) -> int | None:
        timer_sec = parse_timer_seconds(timer_str)
        if timer_sec is None:
            return None
        # 门槛限制:只接受有效的回合内范围 [0, 115] 的值。
        # 此范围之外的值来自于冻结/购买阶段的倒计时或回合结束后的画面。
        if not (0.0 <= timer_sec <= 115.0):
            self.warnings.append(f"skip out-of-range anchor timer={timer_str} ({timer_sec:.0f}s)")
            return None
        # 单调性保护:CS 计时器必须严格递减;若向上跳转超过 3 秒,则意味着
        # 这是一个新的冻结阶段或 OCR 误读 -- 拒绝以避免污染偏移量池。
        if self._last_timer_sec is not None and timer_sec > self._last_timer_sec + 3.0:
            self.warnings.append(
                f"skip non-monotone anchor timer={timer_str} ({timer_sec:.0f}s > prev {self._last_timer_sec:.0f}s + 3)"
            )
            return None
        self._last_timer_sec = timer_sec

        relative_sec = 115.0 - timer_sec
        freeze_end_tick = int(self.round_meta.get("freeze_end_tick", self.round_meta.get("start_tick", 0)))
        tick = int(round(freeze_end_tick + relative_sec * self.tick_rate))
        offset = tick - float(video_time) * self.tick_rate
        if self.offsets:
            median = statistics.median(self.offsets)
            if abs(offset - median) > self.max_anchor_error_sec * self.tick_rate:
                self.warnings.append(f"drop outlier anchor timer={timer_str} video_time={video_time:.3f}")
                return tick
        # 首个临近开局的锚点:验证与 demo 中 freeze_end_tick 的一致性。
        if not self._first_anchor_done and timer_sec >= 110.0 and freeze_end_tick > 0:
            estimated_freeze_vt = float(video_time) - relative_sec
            demo_freeze_vt = (float(freeze_end_tick) - offset) / self.tick_rate
            deviation = abs(estimated_freeze_vt - demo_freeze_vt)
            if deviation > self.anchor_tolerance_sec * 2:
                self.warnings.append(
                    f"first anchor: freeze_end at video_time≈{estimated_freeze_vt:.1f}s, "
                    f"demo says {demo_freeze_vt:.1f}s, deviation={deviation:.1f}s"
                )
        self._first_anchor_done = True
        self.offsets.append(offset)
        return tick

    def freeze(self, video_time_event: float, event_tick: int | None = None) -> None:
        """用任意已知事件(安放/爆炸/拆弹)冻结 offset。"""
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
                self.warnings.append(
                    f"event freeze mismatch video={video_time_event:.3f} demo={expected_video_time:.3f} tick={event_tick}"
                )
        self.frozen_offset = float(current_offset)

    def to_tick(self, video_time: float) -> int:
        offset = self.frozen_offset
        if offset is None:
            offset = self._current_offset()
        if offset is None:
            offset = float(self.round_meta.get("freeze_end_tick", self.round_meta.get("start_tick", 0)))
        return int(round(float(video_time) * self.tick_rate + offset))

    def to_video_time(self, tick: int) -> float:
        offset = self.frozen_offset
        if offset is None:
            offset = self._current_offset() or 0.0
        return (float(tick) - offset) / self.tick_rate

    def relative_sec_for_tick(self, tick: int) -> float:
        freeze_end_tick = int(self.round_meta.get("freeze_end_tick", self.round_meta.get("start_tick", 0)))
        return (int(tick) - freeze_end_tick) / self.tick_rate

    def _current_offset(self) -> float | None:
        if not self.offsets:
            return None
        return float(statistics.median(self.offsets))
