"""Structured FFprobe facts and bounded decoded-frame boundary probing."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os
import subprocess

from sbmachine.media_clock import (
    RationalLike,
    format_rational,
    parse_time_base,
    round_half_even,
    seconds_to_pts,
)


PathLike = str | os.PathLike[str]


class MediaProbeError(RuntimeError):
    """Fail-closed media probing error with machine-readable not_checked data."""

    status = "not_checked"

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.media_sync_status = "not_checked"
        self.details = dict(details or {})
        suffix = f" (media_sync_status=not_checked; reason={reason_code})"
        super().__init__(message if suffix in message else message + suffix)

    def as_not_checked(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "media_sync_status": "not_checked",
            "media_sync_reason": self.reason_code,
            "media_sync_error": str(self),
        }
        result.update(self.details)
        return result

    to_dict = as_not_checked


def not_checked_status(
    reason: str,
    *,
    missing_fields: Sequence[str] = (),
    error: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an explicit status payload without implying media sync success."""
    result: dict[str, Any] = {
        "media_sync_status": "not_checked",
        "media_sync_reason": reason,
        "missing_fields": list(missing_fields),
    }
    if error:
        result["media_sync_error"] = error
    if details:
        result.update(details)
    return result


def sha256_file(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a media file as lowercase hexadecimal SHA-256."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executable_prefix(ffprobe_bin: PathLike | Sequence[str]) -> list[str]:
    if isinstance(ffprobe_bin, (str, os.PathLike)):
        return [str(ffprobe_bin)]
    if isinstance(ffprobe_bin, Sequence) and not isinstance(ffprobe_bin, (bytes, bytearray)):
        prefix = [str(part) for part in ffprobe_bin]
        if prefix:
            return prefix
    raise ValueError("ffprobe_bin must be an executable name/path or a non-empty command sequence")


def _text_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_ffprobe_json(
    path: Path,
    args: Sequence[str],
    *,
    ffprobe_bin: PathLike | Sequence[str],
) -> dict[str, Any]:
    command = _executable_prefix(ffprobe_bin) + [str(arg) for arg in args] + [str(path)]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MediaProbeError(
            f"ffprobe executable is unavailable: {ffprobe_bin!r}; cannot inspect {path}",
            reason_code="ffprobe_missing",
            details={"path": str(path), "ffprobe": str(ffprobe_bin)},
        ) from exc
    except PermissionError as exc:
        raise MediaProbeError(
            f"ffprobe executable is not executable: {ffprobe_bin!r}",
            reason_code="ffprobe_unavailable",
            details={"path": str(path), "ffprobe": str(ffprobe_bin)},
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = _text_output(exc.stderr).strip()
        detail = f": {stderr}" if stderr else ""
        raise MediaProbeError(
            f"ffprobe could not inspect {path}{detail}",
            reason_code="ffprobe_failed",
            details={
                "path": str(path),
                "ffprobe": str(ffprobe_bin),
                "returncode": exc.returncode,
                "stderr": stderr,
            },
        ) from exc
    except OSError as exc:
        raise MediaProbeError(
            f"failed to execute ffprobe for {path}: {exc}",
            reason_code="ffprobe_unavailable",
            details={"path": str(path), "ffprobe": str(ffprobe_bin)},
        ) from exc

    if getattr(completed, "returncode", 0) not in (0, None):
        stderr = _text_output(getattr(completed, "stderr", "")).strip()
        raise MediaProbeError(
            f"ffprobe returned exit code {completed.returncode} for {path}"
            + (f": {stderr}" if stderr else ""),
            reason_code="ffprobe_failed",
            details={
                "path": str(path),
                "ffprobe": str(ffprobe_bin),
                "returncode": completed.returncode,
                "stderr": stderr,
            },
        )

    stdout = _text_output(getattr(completed, "stdout", ""))
    if not stdout.strip():
        raise MediaProbeError(
            f"ffprobe returned empty JSON for {path}",
            reason_code="ffprobe_empty_output",
            details={"path": str(path), "ffprobe": str(ffprobe_bin)},
        )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(
            f"ffprobe returned invalid JSON for {path}: {exc.msg}",
            reason_code="ffprobe_invalid_json",
            details={"path": str(path), "ffprobe": str(ffprobe_bin)},
        ) from exc
    if not isinstance(payload, dict):
        raise MediaProbeError(
            f"ffprobe JSON for {path} must be an object",
            reason_code="ffprobe_invalid_payload",
            details={"path": str(path), "ffprobe": str(ffprobe_bin)},
        )
    return payload


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return None
    try:
        candidate = Fraction(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return candidate.numerator if candidate.denominator == 1 else None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _optional_fraction(value: Any, *, positive: bool = False) -> Fraction | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL", "0/0"}:
        return None
    try:
        result = parse_time_base(text)
    except ValueError:
        return None
    if positive and result <= 0:
        return None
    return result


def _canonical_optional_fraction(value: Any) -> str | None:
    fraction = _optional_fraction(value)
    return format_rational(fraction) if fraction is not None else None


def _parse_explicit_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes"}:
            return True
        if text in {"0", "false", "no"}:
            return False
    return None


def _infer_variable_frame_rate(stream: Mapping[str, Any]) -> tuple[bool | None, str]:
    explicit = _parse_explicit_bool(stream.get("variable_frame_rate"))
    if explicit is not None:
        return explicit, "stream_flag"
    average = _optional_fraction(stream.get("avg_frame_rate"))
    real = _optional_fraction(stream.get("r_frame_rate"))
    if average is None or real is None:
        return None, "insufficient_rate_data"
    if average != real:
        return True, "avg_frame_rate_differs_from_r_frame_rate"
    return False, "avg_frame_rate_equals_r_frame_rate"


def _first_positive_duration(*values: Any) -> float | None:
    for value in values:
        candidate = _optional_float(value)
        if candidate is not None and candidate > 0:
            return candidate
    return None


def _stream_fact(stream: Mapping[str, Any]) -> dict[str, Any]:
    start_pts = _optional_int(stream.get("start_pts"))
    time_base = _optional_fraction(stream.get("time_base"), positive=True)
    start_time = _optional_float(stream.get("start_time"))
    if start_time is None and start_pts is not None and time_base is not None:
        start_time = float(Fraction(start_pts) * time_base)
    vfr, vfr_reason = _infer_variable_frame_rate(stream)
    return {
        "index": _optional_int(stream.get("index")),
        "codec_type": _optional_text(stream.get("codec_type")),
        "codec_name": _optional_text(stream.get("codec_name")),
        "time_base": format_rational(time_base) if time_base is not None else None,
        "start_pts": start_pts,
        "start_time_sec": start_time,
        "duration_sec": _first_positive_duration(stream.get("duration")),
        "avg_frame_rate": _optional_text(stream.get("avg_frame_rate")),
        "r_frame_rate": _optional_text(stream.get("r_frame_rate")),
        "variable_frame_rate": vfr,
        "vfr_detection": vfr_reason,
        "width": _optional_int(stream.get("width")),
        "height": _optional_int(stream.get("height")),
    }


def _not_checked_probe_error(
    path: Path,
    result: Mapping[str, Any],
    missing_fields: Sequence[str],
    *,
    strict: bool,
) -> dict[str, Any]:
    reason = "missing_required_media_facts"
    if not strict:
        return not_checked_status(
            reason,
            missing_fields=missing_fields,
            details={"probe": dict(result)},
        )
    raise MediaProbeError(
        f"media probe for {path} is incomplete; missing {', '.join(missing_fields)}",
        reason_code=reason,
        details={"path": str(path), "missing_fields": list(missing_fields), "probe": dict(result)},
    )


def probe_media(
    path: PathLike,
    *,
    ffprobe_bin: PathLike | Sequence[str] = "ffprobe",
    compute_hash: bool = True,
    include_hash: bool | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Return normalized media facts from ffprobe.

    Basic probing never claims media synchronization ``pass``.  The returned
    object keeps ``media_sync_status=not_checked`` until a bounded decoded
    frame probe and downstream gate have supplied actual boundary evidence.
    With ``strict=True``, missing clock facts or duration raise
    :class:`MediaProbeError` instead of returning the partial object.
    """
    if include_hash is not None:
        compute_hash = bool(include_hash)
    media_path = Path(path)
    payload = _run_ffprobe_json(
        media_path,
        [
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ],
        ffprobe_bin=ffprobe_bin,
    )

    raw_streams = payload.get("streams", [])
    if raw_streams is None:
        raw_streams = []
    if not isinstance(raw_streams, list) or any(not isinstance(item, dict) for item in raw_streams):
        raise MediaProbeError(
            f"ffprobe streams for {media_path} are not a JSON array of objects",
            reason_code="ffprobe_invalid_streams",
            details={"path": str(media_path)},
        )
    streams = [dict(item) for item in raw_streams]
    stream_facts = [_stream_fact(item) for item in streams]
    video_position, video_stream = next(
        (
            (position, stream)
            for position, stream in enumerate(streams)
            if stream.get("codec_type") == "video"
        ),
        (None, None),
    )
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]

    format_payload = payload.get("format")
    if not isinstance(format_payload, dict):
        format_payload = {}
    video_index = _optional_int(video_stream.get("index")) if video_stream is not None else None
    time_base_fraction = (
        _optional_fraction(video_stream.get("time_base"), positive=True)
        if video_stream is not None
        else None
    )
    start_pts = _optional_int(video_stream.get("start_pts")) if video_stream is not None else None
    start_time_sec = _optional_float(video_stream.get("start_time")) if video_stream is not None else None
    if start_time_sec is None and start_pts is not None and time_base_fraction is not None:
        start_time_sec = float(Fraction(start_pts) * time_base_fraction)

    duration_sec = _first_positive_duration(
        video_stream.get("duration") if video_stream is not None else None,
        format_payload.get("duration"),
    )
    if duration_sec is None and video_stream is not None and time_base_fraction is not None:
        duration_ts = _optional_int(video_stream.get("duration_ts"))
        if duration_ts is not None and duration_ts > 0:
            duration_sec = float(Fraction(duration_ts) * time_base_fraction)

    avg_frame_rate = _optional_text(video_stream.get("avg_frame_rate")) if video_stream else None
    r_frame_rate = _optional_text(video_stream.get("r_frame_rate")) if video_stream else None
    variable_frame_rate, vfr_reason = (
        _infer_variable_frame_rate(video_stream) if video_stream is not None else (None, "no_video_stream")
    )

    source_sha256: str | None = None
    hash_error: str | None = None
    if compute_hash:
        try:
            source_sha256 = sha256_file(media_path)
        except OSError as exc:
            hash_error = f"cannot hash media file {media_path}: {exc}"

    missing_fields: list[str] = []
    if video_stream is None:
        missing_fields.append("video_stream")
    if video_index is None:
        missing_fields.append("video_stream_index")
    if duration_sec is None:
        missing_fields.append("duration_sec")
    if time_base_fraction is None:
        missing_fields.append("time_base")
    if start_pts is None:
        missing_fields.append("stream_start_pts")
    if compute_hash and source_sha256 is None:
        missing_fields.append("source_sha256")

    result: dict[str, Any] = {
        "probe_schema_version": 1,
        "source_sha256": source_sha256,
        "source_hash_algorithm": "sha256",
        "hash_status": "checked" if source_sha256 is not None else "not_checked",
        "hash_error": hash_error,
        "has_audio": bool(audio_streams),
        "streams": streams,
        "stream_facts": stream_facts,
        "format": dict(format_payload),
        "duration_sec": duration_sec,
        "video_stream_index": video_index,
        "time_base": format_rational(time_base_fraction) if time_base_fraction is not None else None,
        "stream_start_pts": start_pts,
        "stream_start_time_sec": start_time_sec,
        "avg_frame_rate": avg_frame_rate,
        "r_frame_rate": r_frame_rate,
        "variable_frame_rate": variable_frame_rate,
        "vfr_detection": vfr_reason,
        "width": _optional_int(video_stream.get("width")) if video_stream is not None else None,
        "height": _optional_int(video_stream.get("height")) if video_stream is not None else None,
        "probe_status": "checked" if not missing_fields else "not_checked",
        "media_sync_status": "not_checked",
        "media_sync_reason": "boundary_frame_pts_not_checked",
    }
    if missing_fields:
        _not_checked_probe_error(media_path, result, missing_fields, strict=strict)
        result["not_checked"] = {
            "reason": "missing_required_media_facts",
            "missing_fields": missing_fields,
        }
    elif hash_error:
        result["not_checked"] = {
            "reason": "source_hash_unavailable",
            "missing_fields": ["source_sha256"],
        }
    return result


def _format_seconds(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return str(float(value))


def _frame_pts(frame: Mapping[str, Any], time_base: Fraction | None) -> tuple[int | None, bool]:
    for key in ("best_effort_timestamp", "pts", "pkt_pts"):
        value = _optional_int(frame.get(key))
        if value is not None:
            return value, key == "best_effort_timestamp"
    if time_base is not None:
        for key in ("best_effort_timestamp_time", "pts_time", "pkt_pts_time"):
            timestamp = _optional_float(frame.get(key))
            if timestamp is not None:
                return round_half_even(Fraction(str(timestamp)) / time_base), key == "best_effort_timestamp_time"
    return None, False


def _frame_time(frame: Mapping[str, Any], pts: int | None, time_base: Fraction | None) -> float | None:
    for key in ("best_effort_timestamp_time", "pts_time", "pkt_pts_time"):
        timestamp = _optional_float(frame.get(key))
        if timestamp is not None:
            return timestamp
    if pts is not None and time_base is not None:
        return float(Fraction(pts) * time_base)
    return None


def _frame_candidate(
    frame: Mapping[str, Any],
    *,
    position: int,
    time_base: Fraction | None,
) -> dict[str, Any] | None:
    pts, best_effort = _frame_pts(frame, time_base)
    timestamp = _frame_time(frame, pts, time_base)
    if pts is None and timestamp is None:
        return None
    return {
        "pts": pts,
        "time_sec": timestamp,
        "frame_position": position,
        "best_effort_timestamp": best_effort,
    }


def _choose_boundary_frame(
    candidates: list[dict[str, Any]],
    *,
    target_pts: int | None,
    target_time_sec: Fraction,
    is_start: bool,
) -> dict[str, Any]:
    if target_pts is not None:
        return min(
            candidates,
            key=lambda item: (
                abs(item["pts"] - target_pts) if item["pts"] is not None else math.inf,
                item["pts"] if item["pts"] is not None else math.inf,
            ),
        )
    with_time = [item for item in candidates if item["time_sec"] is not None]
    if with_time:
        target = float(target_time_sec)
        return min(
            with_time,
            key=lambda item: (
                abs(item["time_sec"] - target),
                item["time_sec"] if is_start else -item["time_sec"],
            ),
        )
    return candidates[0] if is_start else candidates[-1]


def _boundary_frame_payload(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pts": frame.get("pts"),
        "time_sec": frame.get("time_sec"),
        "frame_position": frame.get("frame_position"),
        "best_effort_timestamp": frame.get("best_effort_timestamp", False),
    }


def probe_frame_boundaries(
    path: PathLike,
    start_sec: RationalLike | None = None,
    end_sec: RationalLike | None = None,
    *,
    target_start_sec: RationalLike | None = None,
    target_end_sec: RationalLike | None = None,
    probe: Mapping[str, Any] | None = None,
    time_base: RationalLike | None = None,
    stream_start_pts: int | None = None,
    stream_start_time_sec: RationalLike | None = None,
    timeline_origin_sec: RationalLike = 0,
    video_stream_index: int | None = None,
    ffprobe_bin: PathLike | Sequence[str] = "ffprobe",
    strict: bool = False,
) -> dict[str, Any]:
    """Read only the requested interval and return nearest decoded frame PTS.

    The interval is passed to ffprobe via ``-read_intervals``.  If clock facts
    are available, nearest-frame selection is done in integer PTS space;
    otherwise it falls back to decoded timestamp seconds.  No full-media frame
    scan is performed by this function.
    """
    if start_sec is None:
        start_sec = target_start_sec
    elif target_start_sec is not None:
        raise ValueError("provide either start_sec or target_start_sec")
    if end_sec is None:
        end_sec = target_end_sec
    elif target_end_sec is not None:
        raise ValueError("provide either end_sec or target_end_sec")
    if start_sec is None or end_sec is None:
        raise ValueError("start_sec and end_sec are required")

    start = Fraction(str(start_sec)) if isinstance(start_sec, float) else Fraction(start_sec)
    end = Fraction(str(end_sec)) if isinstance(end_sec, float) else Fraction(end_sec)
    if start >= end:
        raise ValueError("start_sec must be less than end_sec")
    origin = Fraction(str(timeline_origin_sec)) if isinstance(timeline_origin_sec, float) else Fraction(timeline_origin_sec)

    if probe is not None:
        if not isinstance(probe, Mapping):
            raise ValueError("probe must be a mapping")
        if time_base is None:
            time_base = probe.get("time_base")
        if stream_start_pts is None:
            stream_start_pts = probe.get("stream_start_pts")
        if stream_start_time_sec is None:
            stream_start_time_sec = probe.get("stream_start_time_sec")
        if video_stream_index is None:
            video_stream_index = probe.get("video_stream_index")

    time_base_fraction: Fraction | None = None
    if time_base is not None:
        try:
            time_base_fraction = parse_time_base(time_base)
        except ValueError as exc:
            if strict:
                raise MediaProbeError(
                    f"invalid time_base for boundary probe of {path}: {time_base!r}",
                    reason_code="invalid_time_base",
                    details={"path": str(path), "time_base": time_base},
                ) from exc
    start_pts = _optional_int(stream_start_pts)
    stream_start_time = None
    if stream_start_time_sec is not None:
        stream_start_time = Fraction(str(stream_start_time_sec)) if isinstance(stream_start_time_sec, float) else Fraction(stream_start_time_sec)
    if stream_start_time is None and start_pts is not None and time_base_fraction is not None:
        stream_start_time = Fraction(start_pts) * time_base_fraction

    relative_start = start - origin
    relative_end = end - origin
    query_start = relative_start + (stream_start_time or Fraction(0))
    query_end = relative_end + (stream_start_time or Fraction(0))
    read_interval = f"{_format_seconds(query_start)}%{_format_seconds(query_end)}"
    payload = _run_ffprobe_json(
        Path(path),
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            read_interval,
            "-show_frames",
            "-show_entries",
            "frame=stream_index,best_effort_timestamp,best_effort_timestamp_time,pts,pts_time,pkt_pts,pkt_pts_time",
            "-print_format",
            "json",
        ],
        ffprobe_bin=ffprobe_bin,
    )
    raw_frames = payload.get("frames", [])
    if raw_frames is None:
        raw_frames = []
    if not isinstance(raw_frames, list) or any(not isinstance(item, dict) for item in raw_frames):
        raise MediaProbeError(
            f"ffprobe frames for {path} are not a JSON array of objects",
            reason_code="ffprobe_invalid_frames",
            details={"path": str(path), "read_interval": read_interval},
        )

    candidates: list[dict[str, Any]] = []
    for position, frame in enumerate(raw_frames):
        if video_stream_index is not None:
            frame_stream_index = _optional_int(frame.get("stream_index"))
            if frame_stream_index is not None and frame_stream_index != video_stream_index:
                continue
        candidate = _frame_candidate(frame, position=position, time_base=time_base_fraction)
        if candidate is not None:
            candidates.append(candidate)

    expected_start_pts = None
    expected_end_pts = None
    if time_base_fraction is not None and start_pts is not None:
        expected_start_pts = seconds_to_pts(
            start,
            time_base_fraction,
            start_pts,
            timeline_origin_sec=origin,
        )
        expected_end_pts = seconds_to_pts(
            end,
            time_base_fraction,
            start_pts,
            timeline_origin_sec=origin,
        )

    result: dict[str, Any] = {
        "boundary_probe_schema_version": 1,
        "target_start_sec": float(start),
        "target_end_sec": float(end),
        "query_start_sec": float(query_start),
        "query_end_sec": float(query_end),
        "read_interval": read_interval,
        "time_base": format_rational(time_base_fraction) if time_base_fraction is not None else None,
        "stream_start_pts": start_pts,
        "expected_start_pts": expected_start_pts,
        "expected_end_pts": expected_end_pts,
        "frames_considered": len(candidates),
        "selection_policy": "nearest_actual_decoded_frame",
        "media_sync_status": "not_checked",
    }
    if not candidates:
        result.update(
            not_checked_status(
                "no_decoded_frame_pts_in_target_interval",
                missing_fields=("decoded_start_pts", "decoded_end_pts"),
            )
        )
        result["boundary_probe_status"] = "not_checked"
        if strict:
            raise MediaProbeError(
                f"ffprobe returned no decoded frame PTS in {read_interval} for {path}",
                reason_code="no_boundary_frames",
                details={"path": str(path), "read_interval": read_interval, "probe": result},
            )
        return result

    start_frame = _choose_boundary_frame(
        candidates,
        target_pts=expected_start_pts,
        target_time_sec=query_start,
        is_start=True,
    )
    end_frame = _choose_boundary_frame(
        candidates,
        target_pts=expected_end_pts,
        target_time_sec=query_end,
        is_start=False,
    )
    start_payload = _boundary_frame_payload(start_frame)
    end_payload = _boundary_frame_payload(end_frame)
    result.update(
        {
            "boundary_probe_status": "checked",
            "decoded_start_pts": start_payload["pts"],
            "decoded_end_pts": end_payload["pts"],
            "decoded_start_time_sec": start_payload["time_sec"],
            "decoded_end_time_sec": end_payload["time_sec"],
            "start_frame": start_payload,
            "end_frame": end_payload,
        }
    )
    return result


def read_boundary_frame_pts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`probe_frame_boundaries`."""
    return probe_frame_boundaries(*args, **kwargs)


def probe_boundary_frame_pts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`probe_frame_boundaries`."""
    return probe_frame_boundaries(*args, **kwargs)


def probe_frame_pts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`probe_frame_boundaries`."""
    return probe_frame_boundaries(*args, **kwargs)


__all__ = [
    "MediaProbeError",
    "PathLike",
    "not_checked_status",
    "probe_boundary_frame_pts",
    "probe_frame_boundaries",
    "probe_frame_pts",
    "probe_media",
    "read_boundary_frame_pts",
    "sha256_file",
]
