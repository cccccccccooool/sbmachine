"""Exact media timeline to PTS/sample conversions for Phase 4.

The module deliberately keeps the clock independent from render ticks.  All
endpoint calculations use :class:`fractions.Fraction` and one centralized
half-even rounding implementation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from numbers import Rational
from typing import Any
import math


RationalLike = int | float | str | Decimal | Fraction


def _as_fraction(value: Any, *, field: str) -> Fraction:
    """Convert a public numeric input without importing binary float noise."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be a finite rational number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{field} must be finite")
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return Fraction(str(value))
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field} must not be empty")
        try:
            return Fraction(text)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{field} must be a rational number, got {value!r}") from exc
    if isinstance(value, Rational):
        try:
            return Fraction(value.numerator, value.denominator)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{field} must be a rational number, got {value!r}") from exc
    raise ValueError(f"{field} must be a rational number, got {type(value).__name__}")


def _as_integer(value: Any, *, field: str) -> int:
    fraction = _as_fraction(value, field=field)
    if fraction.denominator != 1:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return fraction.numerator


def _as_positive_integer(value: Any, *, field: str) -> int:
    result = _as_integer(value, field=field)
    if result <= 0:
        raise ValueError(f"{field} must be positive, got {result}")
    return result


def _float_value(value: Fraction, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is outside the supported floating-point range")
    return result


def parse_time_base(value: RationalLike) -> Fraction:
    """Parse a positive FFmpeg time base such as ``1/90000`` exactly."""
    result = _as_fraction(value, field="time_base")
    if result <= 0:
        raise ValueError(f"time_base must be positive, got {value!r}")
    return result


def format_rational(value: RationalLike) -> str:
    """Return the canonical ``numerator/denominator`` representation."""
    fraction = _as_fraction(value, field="value")
    return f"{fraction.numerator}/{fraction.denominator}"


def round_half_even(value: RationalLike, denominator: RationalLike | None = None) -> int:
    """Round a rational value to an integer using ties-to-even.

    ``denominator`` is accepted for callers that naturally have a numerator
    and denominator pair: ``round_half_even(5, 2)`` is ``2``.
    """
    if denominator is None:
        fraction = _as_fraction(value, field="value")
    else:
        numerator_fraction = _as_fraction(value, field="numerator")
        denominator_fraction = _as_fraction(denominator, field="denominator")
        if denominator_fraction == 0:
            raise ValueError("denominator must not be zero")
        fraction = numerator_fraction / denominator_fraction

    numerator, denominator_value = fraction.numerator, fraction.denominator
    quotient, remainder = divmod(numerator, denominator_value)
    doubled_remainder = remainder * 2
    if doubled_remainder < denominator_value:
        return quotient
    if doubled_remainder > denominator_value:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def round_half_even_fraction(value: RationalLike) -> int:
    """Named alias for callers that want to make the rational policy explicit."""
    return round_half_even(value)


def round_fraction_half_even(value: RationalLike) -> int:
    """Compatibility alias for the centralized half-even implementation."""
    return round_half_even(value)


def seconds_to_pts(
    timeline_sec: RationalLike,
    time_base: RationalLike,
    stream_start_pts: int = 0,
    *,
    timeline_origin_sec: RationalLike = 0,
) -> int:
    """Map a timeline second to a stream PTS using exact endpoint arithmetic."""
    time_base_fraction = parse_time_base(time_base)
    timeline = _as_fraction(timeline_sec, field="timeline_sec")
    origin = _as_fraction(timeline_origin_sec, field="timeline_origin_sec")
    start_pts = _as_integer(stream_start_pts, field="stream_start_pts")
    return start_pts + round_half_even((timeline - origin) / time_base_fraction)


def seconds_to_sample(
    timeline_sec: RationalLike,
    sample_rate: int,
    *,
    timeline_origin_sec: RationalLike = 0,
) -> int:
    """Map a timeline second to an integer PCM sample endpoint."""
    rate = _as_positive_integer(sample_rate, field="sample_rate")
    timeline = _as_fraction(timeline_sec, field="timeline_sec")
    origin = _as_fraction(timeline_origin_sec, field="timeline_origin_sec")
    return round_half_even((timeline - origin) * rate)


def seconds_to_samples(
    timeline_sec: RationalLike,
    sample_rate: int,
    *,
    timeline_origin_sec: RationalLike = 0,
) -> int:
    """Plural alias for :func:`seconds_to_sample`."""
    return seconds_to_sample(
        timeline_sec,
        sample_rate,
        timeline_origin_sec=timeline_origin_sec,
    )


def seconds_to_pts_and_sample(
    timeline_sec: RationalLike,
    *,
    time_base: RationalLike,
    stream_start_pts: int,
    sample_rate: int,
    timeline_origin_sec: RationalLike = 0,
) -> dict[str, int]:
    """Return both independent endpoint coordinates for one timeline second."""
    return {
        "pts": seconds_to_pts(
            timeline_sec,
            time_base,
            stream_start_pts,
            timeline_origin_sec=timeline_origin_sec,
        ),
        "sample": seconds_to_sample(
            timeline_sec,
            sample_rate,
            timeline_origin_sec=timeline_origin_sec,
        ),
    }


class MediaClockError(ValueError):
    """Base error for invalid or unavailable clock inputs."""


class MediaClockNotCheckedError(MediaClockError):
    """Raised when a probe cannot supply the facts needed for clock mapping."""

    status = "not_checked"

    def __init__(self, message: str, *, missing_fields: list[str] | None = None):
        self.media_sync_status = "not_checked"
        self.missing_fields = tuple(missing_fields or ())
        super().__init__(
            f"{message} (media_sync_status=not_checked)"
            if "media_sync_status=not_checked" not in message
            else message
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "media_sync_status": "not_checked",
            "media_sync_reason": str(self),
            "missing_fields": list(self.missing_fields),
        }


@dataclass(frozen=True, slots=True)
class SlotMapping:
    """Attribute form of a clock slot map; ``to_dict`` is JSON-ready."""

    clock_map_version: int
    timeline_id: str | None
    timeline_origin_sec: float
    slot_start_sec: float
    slot_end_sec: float
    expected_start_pts: int
    expected_end_pts: int
    decoded_start_pts: int | None
    decoded_end_pts: int | None
    sample_rate: int
    timeline_start_sample: int
    timeline_end_sample: int
    rounding_policy: str
    slot_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clock_map_version": self.clock_map_version,
            "timeline_id": self.timeline_id,
            "timeline_origin_sec": self.timeline_origin_sec,
            "slot_start_sec": self.slot_start_sec,
            "slot_end_sec": self.slot_end_sec,
            "expected_start_pts": self.expected_start_pts,
            "expected_end_pts": self.expected_end_pts,
            "decoded_start_pts": self.decoded_start_pts,
            "decoded_end_pts": self.decoded_end_pts,
            "sample_rate": self.sample_rate,
            "timeline_start_sample": self.timeline_start_sample,
            "timeline_end_sample": self.timeline_end_sample,
            "rounding_policy": self.rounding_policy,
            "slot_id": self.slot_id,
        }

    as_dict = to_dict

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


ClockSlotMapping = SlotMapping


def _slot_field(slot: Mapping[str, Any] | object, names: tuple[str, ...]) -> Any:
    if isinstance(slot, Mapping):
        for name in names:
            if name in slot:
                return slot[name]
        return None
    for name in names:
        if hasattr(slot, name):
            return getattr(slot, name)
    return None


@dataclass(frozen=True, slots=True)
class MediaClockAdapter:
    """Single source of truth for seconds to PTS/sample endpoint mapping."""

    time_base: RationalLike
    stream_start_pts: int
    sample_rate: int
    timeline_origin_sec: RationalLike = 0
    timeline_id: str | None = None
    stream_start_time_sec: RationalLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_base", parse_time_base(self.time_base))
        object.__setattr__(
            self,
            "stream_start_pts",
            _as_integer(self.stream_start_pts, field="stream_start_pts"),
        )
        object.__setattr__(
            self,
            "sample_rate",
            _as_positive_integer(self.sample_rate, field="sample_rate"),
        )
        object.__setattr__(
            self,
            "timeline_origin_sec",
            _as_fraction(self.timeline_origin_sec, field="timeline_origin_sec"),
        )
        if self.stream_start_time_sec is not None:
            object.__setattr__(
                self,
                "stream_start_time_sec",
                _as_fraction(self.stream_start_time_sec, field="stream_start_time_sec"),
            )
        if self.timeline_id is not None and not isinstance(self.timeline_id, str):
            raise ValueError("timeline_id must be a string or None")

    @classmethod
    def from_probe(
        cls,
        probe: Mapping[str, Any],
        *,
        sample_rate: int,
        timeline_origin_sec: RationalLike = 0,
        timeline_id: str | None = None,
    ) -> "MediaClockAdapter":
        """Construct an adapter from the normalized result of ``probe_media``."""
        if not isinstance(probe, Mapping):
            raise MediaClockNotCheckedError("media probe result must be a mapping")
        missing = [
            field
            for field in ("time_base", "stream_start_pts")
            if probe.get(field) is None
        ]
        if missing:
            raise MediaClockNotCheckedError(
                "media probe lacks clock facts: " + ", ".join(missing),
                missing_fields=missing,
            )
        effective_timeline_id = timeline_id if timeline_id is not None else probe.get("timeline_id")
        return cls(
            time_base=probe["time_base"],
            stream_start_pts=probe["stream_start_pts"],
            sample_rate=sample_rate,
            timeline_origin_sec=timeline_origin_sec,
            timeline_id=effective_timeline_id,
            stream_start_time_sec=probe.get("stream_start_time_sec"),
        )

    @property
    def time_base_str(self) -> str:
        return format_rational(self.time_base)

    def expected_pts(self, timeline_sec: RationalLike) -> int:
        return seconds_to_pts(
            timeline_sec,
            self.time_base,
            self.stream_start_pts,
            timeline_origin_sec=self.timeline_origin_sec,
        )

    seconds_to_pts = expected_pts

    def timeline_sample(self, timeline_sec: RationalLike) -> int:
        return seconds_to_sample(
            timeline_sec,
            self.sample_rate,
            timeline_origin_sec=self.timeline_origin_sec,
        )

    seconds_to_sample = timeline_sample
    seconds_to_samples = timeline_sample

    def map_time(self, timeline_sec: RationalLike) -> dict[str, int]:
        """Map one second independently to the video and PCM coordinates."""
        return {
            "pts": self.expected_pts(timeline_sec),
            "sample": self.timeline_sample(timeline_sec),
        }

    def _extract_slot(
        self,
        slot_start_sec: RationalLike | Mapping[str, Any] | object | None,
        slot_end_sec: RationalLike | None,
        slot: Mapping[str, Any] | object | None,
    ) -> tuple[Fraction, Fraction, str | None]:
        if slot is None and slot_start_sec is not None and not isinstance(
            slot_start_sec, (int, float, str, Decimal, Fraction)
        ):
            slot = slot_start_sec
            slot_start_sec = None
        if slot is not None:
            if slot_start_sec is not None or slot_end_sec is not None:
                raise ValueError("provide either slot or slot_start_sec/slot_end_sec")
            start_value = _slot_field(slot, ("start_sec", "slot_start_sec", "t_start"))
            end_value = _slot_field(slot, ("end_sec", "slot_end_sec", "t_end"))
            slot_id_value = _slot_field(slot, ("slot_id", "window_id", "id"))
        else:
            start_value = slot_start_sec
            end_value = slot_end_sec
            slot_id_value = None
        if start_value is None or end_value is None:
            raise ValueError("slot start_sec and end_sec are required")
        start = _as_fraction(start_value, field="slot_start_sec")
        end = _as_fraction(end_value, field="slot_end_sec")
        if start >= end:
            raise ValueError("slot_start_sec must be less than slot_end_sec")
        if slot_id_value is not None and not isinstance(slot_id_value, str):
            raise ValueError("slot_id must be a string or None")
        return start, end, slot_id_value

    @staticmethod
    def _boundary_value(boundary: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in boundary:
                return boundary[name]
        return None

    def map_slot(
        self,
        slot_start_sec: RationalLike | Mapping[str, Any] | object | None = None,
        slot_end_sec: RationalLike | None = None,
        *,
        slot: Mapping[str, Any] | object | None = None,
        decoded_start_pts: int | None = None,
        decoded_end_pts: int | None = None,
        actual_start_pts: int | None = None,
        actual_end_pts: int | None = None,
        boundary: Mapping[str, Any] | None = None,
        slot_id: str | None = None,
    ) -> dict[str, Any]:
        """Map a seconds slot and optionally attach actual decoded boundaries.

        ``decoded_*`` values remain ``None`` until a targeted frame probe has
        supplied them.  They are never copied from the expected PTS values.
        """
        start, end, inferred_slot_id = self._extract_slot(slot_start_sec, slot_end_sec, slot)
        if slot_id is None:
            slot_id = inferred_slot_id
        if slot_id is not None and not isinstance(slot_id, str):
            raise ValueError("slot_id must be a string or None")

        if boundary is not None:
            if not isinstance(boundary, Mapping):
                raise ValueError("boundary must be a mapping")
            if decoded_start_pts is None:
                decoded_start_pts = self._boundary_value(
                    boundary, "decoded_start_pts", "actual_start_pts", "start_pts"
                )
            if decoded_end_pts is None:
                decoded_end_pts = self._boundary_value(
                    boundary, "decoded_end_pts", "actual_end_pts", "end_pts"
                )
        if decoded_start_pts is None:
            decoded_start_pts = actual_start_pts
        if decoded_end_pts is None:
            decoded_end_pts = actual_end_pts
        if (decoded_start_pts is None) != (decoded_end_pts is None):
            raise ValueError("decoded_start_pts and decoded_end_pts must be supplied together")
        if decoded_start_pts is not None:
            decoded_start_pts = _as_integer(decoded_start_pts, field="decoded_start_pts")
            decoded_end_pts = _as_integer(decoded_end_pts, field="decoded_end_pts")
            if decoded_start_pts > decoded_end_pts:
                raise ValueError("decoded_start_pts must be <= decoded_end_pts")

        mapping = SlotMapping(
            clock_map_version=1,
            timeline_id=self.timeline_id,
            timeline_origin_sec=_float_value(self.timeline_origin_sec, field="timeline_origin_sec"),
            slot_start_sec=_float_value(start, field="slot_start_sec"),
            slot_end_sec=_float_value(end, field="slot_end_sec"),
            expected_start_pts=self.expected_pts(start),
            expected_end_pts=self.expected_pts(end),
            decoded_start_pts=decoded_start_pts,
            decoded_end_pts=decoded_end_pts,
            sample_rate=self.sample_rate,
            timeline_start_sample=self.timeline_sample(start),
            timeline_end_sample=self.timeline_sample(end),
            rounding_policy="half_even_endpoints",
            slot_id=slot_id,
        )
        return mapping.to_dict()

    def map_interval(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Alias for :meth:`map_slot` used by interval-oriented callers."""
        return self.map_slot(*args, **kwargs)

    slot_mapping = map_slot
    map_slot_to_pts = map_slot


__all__ = [
    "ClockSlotMapping",
    "MediaClockAdapter",
    "MediaClockError",
    "MediaClockNotCheckedError",
    "RationalLike",
    "SlotMapping",
    "format_rational",
    "parse_time_base",
    "round_fraction_half_even",
    "round_half_even",
    "round_half_even_fraction",
    "seconds_to_pts",
    "seconds_to_pts_and_sample",
    "seconds_to_sample",
    "seconds_to_samples",
]
