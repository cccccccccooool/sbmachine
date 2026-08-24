"""Phase 4 strict media helpers: decoded clips, sidecars, muxing, and audits.

The legacy Phase 4 modules intentionally keep their current behaviour.  This
module is an independent strict path: it never treats a legacy clip as a
verified clip, and it never reports a media pass when ffmpeg/ffprobe or the
required PTS metadata is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping


CLIP_CONTRACT_VERSION = 1
STRICT_DECODE = "strict_decode"
BOUNDARY_VERIFIED = "verified"
PASS = "pass"
FAIL = "fail"
NOT_CHECKED = "not_checked"

_REQUIRED_SIDECAR_FIELDS = (
    "clip_contract_version",
    "source_sha256",
    "requested_start_sec",
    "requested_end_sec",
    "requested_start_pts",
    "requested_end_pts",
    "decoded_start_pts",
    "decoded_end_pts",
    "cut_mode",
    "boundary_status",
    "clip_sha256",
)


def _audit_result(status: str, reason: str | None = None, **fields: Any) -> dict[str, Any]:
    """Build the small, JSON-serialisable result contract used by this module."""
    result: dict[str, Any] = {
        "status": status,
        "ok": status == PASS,
        "media_sync_status": status,
        "media_sync_reason": reason,
        "reason": reason,
    }
    result.update(fields)
    return result


def _record_check(
    checks: dict[str, str],
    details: dict[str, str],
    name: str,
    status: str,
    detail: str | None = None,
) -> None:
    checks[name] = status
    if detail:
        details[name] = detail


def _status_for_checks(checks: Mapping[str, str]) -> str:
    if any(status == FAIL for status in checks.values()):
        return FAIL
    if any(status == NOT_CHECKED for status in checks.values()):
        return NOT_CHECKED
    return PASS


def _reason_for_checks(checks: Mapping[str, str], details: Mapping[str, str]) -> str | None:
    for status in (FAIL, NOT_CHECKED):
        for name, check_status in checks.items():
            if check_status == status:
                return details.get(name, name)
    return None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _validate_interval(start_sec: Any, end_sec: Any) -> tuple[float, float]:
    start = _finite_float(start_sec)
    end = _finite_float(end_sec)
    if start is None or end is None:
        raise ValueError("clip boundaries must be finite numbers")
    if start < 0.0 or end <= start:
        raise ValueError(f"clip boundaries must satisfy 0 <= start < end, got {start!r}, {end!r}")
    return start, end


def _validate_tolerance(value: Any, name: str) -> float:
    tolerance = _finite_float(value)
    if tolerance is None or tolerance < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return tolerance


def _parse_time_base(value: Any) -> Fraction | None:
    if value is None or value == "" or value == "N/A":
        return None
    try:
        parsed = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if parsed > 0 else None


def _parse_pts(value: Any) -> int | None:
    if value is None or value == "" or value == "N/A" or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        try:
            rational = Fraction(text)
        except (ValueError, ZeroDivisionError):
            rational = None
        if rational is not None and rational.denominator == 1:
            return rational.numerator
        try:
            parsed = float(text)
        except ValueError:
            return None
        return int(parsed) if math.isfinite(parsed) and parsed.is_integer() else None


def _as_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric boundary")
    if isinstance(value, int):
        return Fraction(value)
    parsed = _finite_float(value)
    if parsed is None:
        raise ValueError(f"cannot convert {value!r} to a finite rational number")
    return Fraction(str(value))


def round_half_even(value: Any, denominator: Any | None = None) -> int:
    """Round a rational value with the W4/W5 half-even endpoint policy."""
    fraction = _as_fraction(value)
    if denominator is not None:
        divisor = _as_fraction(denominator)
        if divisor == 0:
            raise ValueError("denominator must not be zero")
        fraction /= divisor
    numerator, denominator_value = fraction.as_integer_ratio()
    quotient, remainder = divmod(numerator, denominator_value)
    doubled = remainder * 2
    if doubled < denominator_value:
        return quotient
    if doubled > denominator_value:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def _digest_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        return None
    return text


def sha256_file(path: Path) -> str:
    """Return the raw SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_cache_fingerprint(source_sha256: str, start_sec: float, end_sec: float) -> str:
    """Return the legacy-compatible source-content plus interval cache key."""
    start, end = _validate_interval(start_sec, end_sec)
    source_identity = str(source_sha256).strip()
    if not source_identity:
        raise ValueError("source_sha256 must not be empty")
    identity = f"{source_identity}\0{start:.17g}\0{end:.17g}"
    return hashlib.sha256(identity.encode("ascii")).hexdigest()


def clip_cache_path(cache_dir: Path, fingerprint: str, suffix: str = ".mp4") -> Path:
    """Return the canonical strict clip path for a cache fingerprint."""
    if not fingerprint or any(char not in "0123456789abcdef" for char in fingerprint.lower()):
        raise ValueError("fingerprint must be a non-empty hexadecimal string")
    normalized_suffix = suffix if str(suffix).startswith(".") else f".{suffix}"
    return Path(cache_dir) / f"clip_{fingerprint}{normalized_suffix}"


def clip_sidecar_path(clip_path: Path, sidecar_path: Path | None = None) -> Path:
    """Return the default ``clip.mp4.json`` sidecar path."""
    if sidecar_path is not None:
        return Path(sidecar_path)
    return Path(f"{Path(clip_path)}.json")


def _sidecar_candidates(path: Path) -> list[Path]:
    media_path = Path(path)
    if media_path.suffix.lower() == ".json":
        return [media_path]
    candidates = [clip_sidecar_path(media_path)]
    legacy_candidate = media_path.with_suffix(".json")
    if legacy_candidate not in candidates:
        candidates.append(legacy_candidate)
    return candidates


def read_clip_sidecar(path: Path, sidecar_path: Path | None = None) -> dict[str, Any]:
    """Read a clip sidecar, accepting either a sidecar path or its clip path."""
    candidates = [clip_sidecar_path(path, sidecar_path)] if sidecar_path is not None else _sidecar_candidates(Path(path))
    target = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not target.is_file():
        raise FileNotFoundError(f"clip sidecar not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid clip sidecar JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"clip sidecar must be a JSON object: {target}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def write_clip_sidecar(
    clip_path: Path,
    sidecar: Mapping[str, Any],
    sidecar_path: Path | None = None,
) -> Path:
    """Validate and atomically write a strict clip sidecar."""
    payload = dict(sidecar)
    validation = validate_clip_sidecar(payload)
    if validation["status"] != PASS:
        raise ValueError(f"invalid clip sidecar: {validation['reason']}")
    return _atomic_write_json(clip_sidecar_path(clip_path, sidecar_path), payload)


def validate_clip_sidecar(
    sidecar: Mapping[str, Any] | Path,
    *,
    source_path: Path | None = None,
    source_sha256: str | None = None,
    clip_path: Path | None = None,
    requested_start_sec: float | None = None,
    requested_end_sec: float | None = None,
    expected_start_pts: int | None = None,
    expected_end_pts: int | None = None,
    observed_start_pts: int | None = None,
    observed_end_pts: int | None = None,
    time_base: str | Fraction | None = None,
    max_frame_boundary_error_sec: float | None = None,
    require_source: bool = False,
    require_clip: bool = False,
) -> dict[str, Any]:
    """Validate sidecar structure and any caller-supplied external identity.

    Structural validation is deterministic and does not need ffprobe.  The
    optional source/clip arguments add content checks; ``require_source`` and
    ``require_clip`` make absent external checks explicit ``not_checked``
    results instead of allowing a cache hit to look verified.
    """
    sidecar_path_value: str | None = None
    if isinstance(sidecar, (str, Path)):
        try:
            payload = read_clip_sidecar(Path(sidecar))
            sidecar_path_value = str(next(candidate for candidate in _sidecar_candidates(Path(sidecar)) if candidate.is_file()))
        except (FileNotFoundError, ValueError) as exc:
            status = FAIL
            return _audit_result(status, "sidecar_missing_or_invalid", error=str(exc), sidecar_path=sidecar_path_value)
    elif isinstance(sidecar, Mapping):
        payload = dict(sidecar)
    else:
        return _audit_result(FAIL, "sidecar_must_be_mapping_or_path")

    checks: dict[str, str] = {}
    details: dict[str, str] = {}
    missing = [key for key in _REQUIRED_SIDECAR_FIELDS if key not in payload]
    if missing:
        _record_check(checks, details, "required_fields", FAIL, f"missing fields: {', '.join(missing)}")
    else:
        _record_check(checks, details, "required_fields", PASS)

    if payload.get("clip_contract_version") == CLIP_CONTRACT_VERSION:
        _record_check(checks, details, "contract_version", PASS)
    else:
        _record_check(checks, details, "contract_version", FAIL, "unsupported clip_contract_version")

    if payload.get("cut_mode") == STRICT_DECODE:
        _record_check(checks, details, "cut_mode", PASS)
    else:
        _record_check(checks, details, "cut_mode", FAIL, "strict cache requires cut_mode=strict_decode")

    if payload.get("boundary_status") == BOUNDARY_VERIFIED:
        _record_check(checks, details, "boundary_status", PASS)
    else:
        _record_check(checks, details, "boundary_status", FAIL, "boundary_status is not verified")

    source_digest = _digest_text(payload.get("source_sha256"))
    clip_digest = _digest_text(payload.get("clip_sha256"))
    _record_check(
        checks,
        details,
        "source_sha256",
        PASS if source_digest is not None else FAIL,
        "source_sha256 must be a SHA-256 digest" if source_digest is None else None,
    )
    _record_check(
        checks,
        details,
        "clip_sha256",
        PASS if clip_digest is not None else FAIL,
        "clip_sha256 must be a SHA-256 digest" if clip_digest is None else None,
    )

    start = _finite_float(payload.get("requested_start_sec"))
    end = _finite_float(payload.get("requested_end_sec"))
    if start is None or end is None or start < 0.0 or end <= start:
        _record_check(checks, details, "requested_interval", FAIL, "requested interval is invalid")
    else:
        _record_check(checks, details, "requested_interval", PASS)

    requested_start_pts_value = _parse_pts(payload.get("requested_start_pts"))
    requested_end_pts_value = _parse_pts(payload.get("requested_end_pts"))
    decoded_start_pts_value = _parse_pts(payload.get("decoded_start_pts"))
    decoded_end_pts_value = _parse_pts(payload.get("decoded_end_pts"))
    pts_values = (
        requested_start_pts_value,
        requested_end_pts_value,
        decoded_start_pts_value,
        decoded_end_pts_value,
    )
    if any(value is None for value in pts_values):
        _record_check(checks, details, "boundary_pts", FAIL, "requested and decoded PTS must be integers")
    elif decoded_end_pts_value < decoded_start_pts_value:
        _record_check(checks, details, "boundary_pts", FAIL, "decoded PTS are not monotone")
    else:
        _record_check(checks, details, "boundary_pts", PASS)

    if start is not None and end is not None and start >= 0.0 and end > start:
        fingerprint = payload.get("cache_fingerprint")
        if fingerprint is None:
            _record_check(checks, details, "cache_fingerprint", PASS)
        else:
            expected_fingerprint = clip_cache_fingerprint(payload["source_sha256"], start, end)
            _record_check(
                checks,
                details,
                "cache_fingerprint",
                PASS if _digest_text(fingerprint) == expected_fingerprint else FAIL,
                "cache_fingerprint does not match source and interval",
            )

    if requested_start_sec is not None or requested_end_sec is not None:
        try:
            expected_start, expected_end = _validate_interval(requested_start_sec, requested_end_sec)
        except ValueError as exc:
            _record_check(checks, details, "requested_interval_match", FAIL, str(exc))
        else:
            requested_match = (
                start is not None
                and end is not None
                and math.isclose(start, expected_start, rel_tol=0.0, abs_tol=1e-9)
                and math.isclose(end, expected_end, rel_tol=0.0, abs_tol=1e-9)
            )
            _record_check(checks, details, "requested_interval_match", PASS if requested_match else FAIL, "sidecar interval does not match request")

    if source_sha256 is not None:
        expected_source_digest = _digest_text(source_sha256)
        source_match = expected_source_digest is not None and source_digest == expected_source_digest
        _record_check(checks, details, "source_identity_match", PASS if source_match else FAIL, "sidecar source hash does not match request")

    if source_path is not None:
        source_path = Path(source_path)
        if not source_path.is_file():
            _record_check(checks, details, "source_file", FAIL, f"source file does not exist: {source_path}")
        else:
            try:
                actual_source_digest = sha256_file(source_path)
            except OSError as exc:
                _record_check(checks, details, "source_file", FAIL, f"cannot hash source: {exc}")
            else:
                _record_check(
                    checks,
                    details,
                    "source_file",
                    PASS if source_digest == actual_source_digest else FAIL,
                    "sidecar source hash does not match source file",
                )
                if source_sha256 is not None:
                    _record_check(
                        checks,
                        details,
                        "source_request_file_match",
                        PASS if _digest_text(source_sha256) == actual_source_digest else FAIL,
                        "requested source hash does not match source file",
                    )
    elif require_source:
        _record_check(checks, details, "source_file", NOT_CHECKED, "source file was not supplied")

    if clip_path is not None:
        clip_path = Path(clip_path)
        if not clip_path.is_file():
            _record_check(checks, details, "clip_file", FAIL, f"clip file does not exist: {clip_path}")
        else:
            try:
                actual_clip_digest = sha256_file(clip_path)
            except OSError as exc:
                _record_check(checks, details, "clip_file", FAIL, f"cannot hash clip: {exc}")
            else:
                _record_check(
                    checks,
                    details,
                    "clip_file",
                    PASS if clip_digest == actual_clip_digest else FAIL,
                    "sidecar clip hash does not match clip file",
                )
    elif require_clip:
        _record_check(checks, details, "clip_file", NOT_CHECKED, "clip file was not supplied")

    if expected_start_pts is not None or expected_end_pts is not None:
        expected_start_pts_value = _parse_pts(expected_start_pts)
        expected_end_pts_value = _parse_pts(expected_end_pts)
        pts_match = (
            expected_start_pts_value is not None
            and expected_end_pts_value is not None
            and requested_start_pts_value == expected_start_pts_value
            and requested_end_pts_value == expected_end_pts_value
        )
        _record_check(checks, details, "requested_pts_match", PASS if pts_match else FAIL, "sidecar requested PTS do not match probe")

    if observed_start_pts is not None or observed_end_pts is not None:
        observed_start_pts_value = _parse_pts(observed_start_pts)
        observed_end_pts_value = _parse_pts(observed_end_pts)
        observed_match = (
            observed_start_pts_value is not None
            and observed_end_pts_value is not None
            and decoded_start_pts_value == observed_start_pts_value
            and decoded_end_pts_value == observed_end_pts_value
        )
        _record_check(checks, details, "decoded_pts_match", PASS if observed_match else FAIL, "sidecar decoded PTS do not match probe")

    if time_base is not None and all(value is not None for value in pts_values):
        parsed_time_base = _parse_time_base(time_base)
        if parsed_time_base is None:
            _record_check(checks, details, "boundary_error", NOT_CHECKED, "time_base is unavailable or invalid")
        else:
            start_error = abs(decoded_start_pts_value - requested_start_pts_value) * float(parsed_time_base)
            end_error = abs(decoded_end_pts_value - requested_end_pts_value) * float(parsed_time_base)
            payload["boundary_error_start_sec"] = payload.get("boundary_error_start_sec", start_error)
            payload["boundary_error_end_sec"] = payload.get("boundary_error_end_sec", end_error)
            if max_frame_boundary_error_sec is None:
                _record_check(checks, details, "boundary_error", PASS)
            else:
                tolerance = _validate_tolerance(max_frame_boundary_error_sec, "max_frame_boundary_error_sec")
                within_tolerance = start_error <= tolerance and end_error <= tolerance
                _record_check(
                    checks,
                    details,
                    "boundary_error",
                    PASS if within_tolerance else FAIL,
                    f"decoded boundary error exceeds {tolerance:.9f}s",
                )

    status = _status_for_checks(checks)
    return _audit_result(
        status,
        _reason_for_checks(checks, details),
        checks=checks,
        check_details=details,
        sidecar=dict(payload),
        sidecar_path=sidecar_path_value,
    )


def validate_clip_cache_entry(
    clip_path: Path,
    source_path: Path,
    start_sec: float,
    end_sec: float,
    *,
    sidecar_path: Path | None = None,
    expected_start_pts: int | None = None,
    expected_end_pts: int | None = None,
    observed_start_pts: int | None = None,
    observed_end_pts: int | None = None,
    time_base: str | Fraction | None = None,
    max_frame_boundary_error_sec: float = 0.05,
) -> dict[str, Any]:
    """Validate a strict cache entry, including source and clip content hashes."""
    clip_path = Path(clip_path)
    source_path = Path(source_path)
    if not clip_path.is_file():
        return _audit_result(FAIL, "clip_missing", clip_path=str(clip_path))
    if not source_path.is_file():
        return _audit_result(FAIL, "source_missing", source_path=str(source_path))
    try:
        source_digest = sha256_file(source_path)
    except OSError as exc:
        return _audit_result(FAIL, "source_hash_failed", error=str(exc), source_path=str(source_path))
    try:
        payload = read_clip_sidecar(clip_path, sidecar_path)
    except (FileNotFoundError, ValueError) as exc:
        return _audit_result(
            FAIL,
            "sidecar_missing_or_invalid",
            error=str(exc),
            clip_path=str(clip_path),
            sidecar_path=str(clip_sidecar_path(clip_path, sidecar_path)),
        )
    result = validate_clip_sidecar(
        payload,
        source_path=source_path,
        source_sha256=source_digest,
        clip_path=clip_path,
        requested_start_sec=start_sec,
        requested_end_sec=end_sec,
        expected_start_pts=expected_start_pts,
        expected_end_pts=expected_end_pts,
        observed_start_pts=observed_start_pts,
        observed_end_pts=observed_end_pts,
        time_base=time_base,
        max_frame_boundary_error_sec=max_frame_boundary_error_sec,
        require_source=True,
        require_clip=True,
    )
    result.update({
        "clip_path": str(clip_path),
        "source_path": str(source_path),
        "sidecar_path": str(clip_sidecar_path(clip_path, sidecar_path)),
        "fingerprint": clip_cache_fingerprint(source_digest, start_sec, end_sec),
    })
    return result


def _resolve_tool(tool: str | Path) -> str | None:
    requested = str(tool)
    if Path(requested).is_file():
        return requested
    return shutil.which(requested)


def _tool_missing_result(
    missing_tools: list[str],
    *,
    command: list[str] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return _audit_result(
        NOT_CHECKED,
        "required_media_tool_missing",
        missing_tools=missing_tools,
        command=command,
        **fields,
    )


def _run_media_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return _audit_result(NOT_CHECKED, "media_tool_not_found", error=str(exc), command=command)
    except OSError as exc:
        return _audit_result(FAIL, "media_tool_execution_error", error=str(exc), command=command)

    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    returncode = getattr(completed, "returncode", 0)
    if returncode != 0:
        return _audit_result(
            FAIL,
            "media_command_failed",
            error=stderr.strip() or f"process exited with code {returncode}",
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            command=command,
        )
    return _audit_result(PASS, None, returncode=returncode, stdout=stdout, stderr=stderr, command=command)


def _parse_probe_payload(command_result: Mapping[str, Any], path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    stdout = str(command_result.get("stdout") or "")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, _audit_result(FAIL, "ffprobe_invalid_json", error=str(exc), path=str(path), probe=command_result)
    if not isinstance(payload, dict):
        return None, _audit_result(FAIL, "ffprobe_payload_not_object", path=str(path), probe=command_result)
    return payload, None


def probe_media(path: Path, *, ffprobe_bin: str | Path = "ffprobe") -> dict[str, Any]:
    """Run structured ffprobe and expose basic media data plus strict PTS metadata."""
    path = Path(path)
    if not path.is_file():
        return _audit_result(FAIL, "media_missing", path=str(path))
    tool = _resolve_tool(ffprobe_bin)
    command = [
        tool or str(ffprobe_bin),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    if tool is None:
        return _tool_missing_result(["ffprobe"], command=command, path=str(path))

    command_result = _run_media_command(command)
    if command_result["status"] != PASS:
        command_result.update({"path": str(path)})
        return command_result
    payload, parse_error = _parse_probe_payload(command_result, path)
    if parse_error is not None:
        return parse_error
    assert payload is not None

    streams = payload.get("streams")
    if not isinstance(streams, list):
        return _audit_result(FAIL, "ffprobe_streams_missing", path=str(path), probe=payload, command=command)
    video_stream = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), None)
    if video_stream is None:
        return _audit_result(FAIL, "video_stream_missing", path=str(path), probe=payload, command=command)

    format_payload = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    raw_duration = video_stream.get("duration") or format_payload.get("duration")
    duration_sec = _finite_float(raw_duration)
    if duration_sec is None or duration_sec <= 0.0:
        return _audit_result(
            FAIL,
            "video_duration_missing_or_invalid",
            path=str(path),
            duration_sec=duration_sec,
            probe=payload,
            command=command,
        )

    audio_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"]
    audio_duration_sec: float | None = None
    if audio_streams:
        audio_duration_sec = _finite_float(audio_streams[0].get("duration"))
        if audio_duration_sec is None:
            audio_duration_sec = _finite_float(format_payload.get("duration"))

    time_base_value = video_stream.get("time_base")
    time_base = _parse_time_base(time_base_value)
    stream_start_pts = _parse_pts(video_stream.get("start_pts"))
    stream_start_time_sec = _finite_float(video_stream.get("start_time"))
    if stream_start_time_sec is None and stream_start_pts is not None and time_base is not None:
        stream_start_time_sec = float(Fraction(stream_start_pts) * time_base)
    strict_metadata_missing: list[str] = []
    if time_base is None:
        strict_metadata_missing.append("time_base")
    if stream_start_pts is None:
        strict_metadata_missing.append("start_pts")

    status = PASS if not strict_metadata_missing else NOT_CHECKED
    reason = None if status == PASS else "strict_pts_metadata_missing"
    return _audit_result(
        status,
        reason,
        path=str(path),
        duration_sec=duration_sec,
        audio_duration_sec=audio_duration_sec,
        has_audio=bool(audio_streams),
        video_stream=video_stream,
        audio_streams=audio_streams,
        stream_index=video_stream.get("index"),
        time_base=str(time_base_value) if time_base is not None else None,
        stream_start_pts=stream_start_pts,
        stream_start_time_sec=stream_start_time_sec,
        missing_strict_metadata=strict_metadata_missing,
        basic_status=PASS,
        probe=payload,
        command=command,
    )


def _expected_pts(
    target_sec: float,
    *,
    stream_start_pts: int,
    stream_start_time_sec: float | None = None,
    time_base: str | Fraction,
) -> int:
    parsed_time_base = _parse_time_base(time_base)
    if parsed_time_base is None:
        raise ValueError("time_base is unavailable or invalid")
    # The pipeline's business timeline is zero-based.  stream_start_time_sec
    # remains diagnostic metadata; it must not shift the authoritative mapping.
    from sbmachine.media_clock import seconds_to_pts

    return seconds_to_pts(target_sec, parsed_time_base, int(stream_start_pts), timeline_origin_sec=0)


def expected_boundary_pts(
    start_sec: float,
    end_sec: float,
    *,
    stream_start_pts: int,
    stream_start_time_sec: float | None = None,
    time_base: str | Fraction,
) -> tuple[int, int]:
    """Map source seconds to PTS using exact rational time-base arithmetic."""
    start, end = _validate_interval(start_sec, end_sec)
    return (
        _expected_pts(
            start,
            stream_start_pts=stream_start_pts,
            stream_start_time_sec=stream_start_time_sec,
            time_base=time_base,
        ),
        _expected_pts(
            end,
            stream_start_pts=stream_start_pts,
            stream_start_time_sec=stream_start_time_sec,
            time_base=time_base,
        ),
    )


def _frame_pts(frame: Mapping[str, Any]) -> int | None:
    best_effort = _parse_pts(frame.get("best_effort_timestamp"))
    return best_effort if best_effort is not None else _parse_pts(frame.get("pts"))


def probe_boundary_pts(
    path: Path,
    start_sec: float,
    end_sec: float,
    *,
    expected_start_pts: int | None = None,
    expected_end_pts: int | None = None,
    time_base: str | Fraction | None = None,
    stream_start_time_sec: float | None = None,
    max_frame_boundary_error_sec: float = 0.05,
    ffprobe_bin: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Probe only the requested video interval and select nearest actual PTS."""
    path = Path(path)
    try:
        start, end = _validate_interval(start_sec, end_sec)
        tolerance = _validate_tolerance(max_frame_boundary_error_sec, "max_frame_boundary_error_sec")
    except ValueError as exc:
        return _audit_result(FAIL, "invalid_boundary_request", error=str(exc), path=str(path))
    parsed_time_base = _parse_time_base(time_base)
    if expected_start_pts is None or expected_end_pts is None or parsed_time_base is None:
        return _audit_result(
            NOT_CHECKED,
            "boundary_pts_metadata_missing",
            path=str(path),
            requested_start_sec=start,
            requested_end_sec=end,
            expected_start_pts=expected_start_pts,
            expected_end_pts=expected_end_pts,
        )
    if not path.is_file():
        return _audit_result(FAIL, "media_missing", path=str(path))
    tool = _resolve_tool(ffprobe_bin)
    stream_origin = _finite_float(stream_start_time_sec)
    if stream_start_time_sec is not None and stream_origin is None:
        return _audit_result(
            NOT_CHECKED,
            "invalid_stream_start_time",
            path=str(path),
            requested_start_sec=start,
            requested_end_sec=end,
        )
    stream_origin = stream_origin or 0.0
    read_start = max(0.0, stream_origin + start - tolerance)
    read_end = max(read_start + max(tolerance, 1e-6), stream_origin + end + tolerance)
    read_interval = f"{read_start:.9f}%{read_end:.9f}"
    command = [
        tool or str(ffprobe_bin),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        read_interval,
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp,best_effort_timestamp_time,pts,pts_time",
        "-print_format",
        "json",
        str(path),
    ]
    if tool is None:
        return _tool_missing_result(["ffprobe"], command=command, path=str(path))
    command_result = _run_media_command(command)
    if command_result["status"] != PASS:
        command_result.update({"path": str(path), "read_interval": read_interval})
        return command_result
    payload, parse_error = _parse_probe_payload(command_result, path)
    if parse_error is not None:
        return parse_error
    assert payload is not None
    frames = [item for item in payload.get("frames", []) if isinstance(item, dict)]
    frames_with_pts = [item for item in frames if _frame_pts(item) is not None]
    if not frames_with_pts:
        return _audit_result(
            FAIL,
            "boundary_frames_missing",
            path=str(path),
            read_interval=read_interval,
            frames_scanned=len(frames),
            command=command,
        )

    start_frame = min(frames_with_pts, key=lambda frame: abs(_frame_pts(frame) - expected_start_pts))
    end_frame = min(frames_with_pts, key=lambda frame: abs(_frame_pts(frame) - expected_end_pts))
    decoded_start_pts = _frame_pts(start_frame)
    decoded_end_pts = _frame_pts(end_frame)
    assert decoded_start_pts is not None and decoded_end_pts is not None
    start_error_sec = abs(decoded_start_pts - expected_start_pts) * float(parsed_time_base)
    end_error_sec = abs(decoded_end_pts - expected_end_pts) * float(parsed_time_base)
    boundary = validate_clip_boundaries(
        start,
        end,
        requested_start_pts=expected_start_pts,
        requested_end_pts=expected_end_pts,
        decoded_start_pts=decoded_start_pts,
        decoded_end_pts=decoded_end_pts,
        time_base=parsed_time_base,
        max_frame_boundary_error_sec=tolerance,
    )
    boundary.update({
        "path": str(path),
        "read_interval": read_interval,
        "stream_start_time_sec": stream_origin,
        "frames_scanned": len(frames),
        "command": command,
        "decoded_start_time_sec": _finite_float(start_frame.get("best_effort_timestamp_time") or start_frame.get("pts_time")),
        "decoded_end_time_sec": _finite_float(end_frame.get("best_effort_timestamp_time") or end_frame.get("pts_time")),
        "start_error_sec": start_error_sec,
        "end_error_sec": end_error_sec,
    })
    return boundary


def validate_clip_boundaries(
    requested_start_sec: float,
    requested_end_sec: float,
    *,
    requested_start_pts: int | None,
    requested_end_pts: int | None,
    decoded_start_pts: int | None,
    decoded_end_pts: int | None,
    time_base: str | Fraction | None,
    max_frame_boundary_error_sec: float = 0.05,
) -> dict[str, Any]:
    """Validate requested versus actual decoded frame boundaries."""
    try:
        start, end = _validate_interval(requested_start_sec, requested_end_sec)
        tolerance = _validate_tolerance(max_frame_boundary_error_sec, "max_frame_boundary_error_sec")
    except ValueError as exc:
        return _audit_result(FAIL, "invalid_boundary_request", error=str(exc))
    parsed_time_base = _parse_time_base(time_base)
    if parsed_time_base is None or any(
        value is None
        for value in (requested_start_pts, requested_end_pts, decoded_start_pts, decoded_end_pts)
    ):
        return _audit_result(
            NOT_CHECKED,
            "boundary_pts_metadata_missing",
            requested_start_sec=start,
            requested_end_sec=end,
            requested_start_pts=requested_start_pts,
            requested_end_pts=requested_end_pts,
            decoded_start_pts=decoded_start_pts,
            decoded_end_pts=decoded_end_pts,
            time_base=str(time_base) if time_base is not None else None,
        )
    assert requested_start_pts is not None
    assert requested_end_pts is not None
    assert decoded_start_pts is not None
    assert decoded_end_pts is not None
    checks: dict[str, str] = {}
    details: dict[str, str] = {}
    if requested_end_pts <= requested_start_pts:
        _record_check(checks, details, "requested_pts_order", FAIL, "requested PTS are not increasing")
    else:
        _record_check(checks, details, "requested_pts_order", PASS)
    if decoded_end_pts < decoded_start_pts:
        _record_check(checks, details, "decoded_pts_monotone", FAIL, "decoded PTS are not monotone")
    else:
        _record_check(checks, details, "decoded_pts_monotone", PASS)
    start_error_sec = abs(decoded_start_pts - requested_start_pts) * float(parsed_time_base)
    end_error_sec = abs(decoded_end_pts - requested_end_pts) * float(parsed_time_base)
    within_tolerance = start_error_sec <= tolerance and end_error_sec <= tolerance
    _record_check(
        checks,
        details,
        "frame_boundary_error",
        PASS if within_tolerance else FAIL,
        f"frame boundary error exceeds {tolerance:.9f}s",
    )
    status = _status_for_checks(checks)
    return _audit_result(
        status,
        _reason_for_checks(checks, details),
        boundary_status=BOUNDARY_VERIFIED if status == PASS else "failed",
        requested_start_sec=start,
        requested_end_sec=end,
        requested_start_pts=requested_start_pts,
        requested_end_pts=requested_end_pts,
        decoded_start_pts=decoded_start_pts,
        decoded_end_pts=decoded_end_pts,
        time_base=f"{parsed_time_base.numerator}/{parsed_time_base.denominator}",
        start_error_sec=start_error_sec,
        end_error_sec=end_error_sec,
        checks=checks,
        check_details=details,
    )


def validate_sample_bounds(
    start_sample: int,
    end_sample: int,
    *,
    slot_end_sample: int | None,
    canvas_limit_sample: int | None,
    timeline_end_sample: int | None = None,
    slot_timeline_end_sample: int | None = None,
    timeline_canvas_end_sample: int | None = None,
) -> dict[str, Any]:
    """Validate the W5 round/timeline sample inequalities when supplied."""
    values = (
        start_sample,
        end_sample,
        slot_end_sample,
        canvas_limit_sample,
        timeline_end_sample,
        slot_timeline_end_sample,
        timeline_canvas_end_sample,
    )
    if any(isinstance(value, bool) or (value is not None and not isinstance(value, int)) for value in values):
        return _audit_result(FAIL, "sample_bounds_must_be_integers")
    if end_sample < start_sample:
        return _audit_result(FAIL, "sample_interval_reversed")
    checks: dict[str, str] = {}
    details: dict[str, str] = {}
    if slot_end_sample is None or canvas_limit_sample is None:
        _record_check(checks, details, "round_canvas_bounds", NOT_CHECKED, "round slot/canvas limits are missing")
    else:
        round_ok = start_sample <= end_sample <= slot_end_sample <= canvas_limit_sample
        _record_check(checks, details, "round_canvas_bounds", PASS if round_ok else FAIL, "round sample bounds are invalid")
    timeline_values = (timeline_end_sample, slot_timeline_end_sample, timeline_canvas_end_sample)
    if any(value is None for value in timeline_values):
        _record_check(checks, details, "timeline_bounds", NOT_CHECKED, "timeline sample limits are incomplete")
    else:
        timeline_ok = start_sample <= timeline_end_sample <= slot_timeline_end_sample <= timeline_canvas_end_sample
        _record_check(checks, details, "timeline_bounds", PASS if timeline_ok else FAIL, "timeline sample bounds are invalid")
    status = _status_for_checks(checks)
    return _audit_result(
        status,
        _reason_for_checks(checks, details),
        start_sample=start_sample,
        end_sample=end_sample,
        slot_end_sample=slot_end_sample,
        canvas_limit_sample=canvas_limit_sample,
        timeline_end_sample=timeline_end_sample,
        slot_timeline_end_sample=slot_timeline_end_sample,
        timeline_canvas_end_sample=timeline_canvas_end_sample,
        checks=checks,
        check_details=details,
    )


def build_strict_clip_command(
    source_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    *,
    ffmpeg_bin: str | Path = "ffmpeg",
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    video_preset: str = "veryfast",
    video_crf: int = 20,
    audio_bitrate: str = "192k",
) -> list[str]:
    """Build a decoded/re-encoded clip command; stream-copy codecs are rejected."""
    start, end = _validate_interval(start_sec, end_sec)
    duration = end - start
    if str(video_codec).strip().lower() in {"copy", "-c:v copy", "-codec:v copy"}:
        raise ValueError("strict_decode clip cannot use video stream-copy")
    if str(audio_codec).strip().lower() in {"copy", "-c:a copy", "-codec:a copy"}:
        raise ValueError("strict_decode clip cannot use audio stream-copy")
    if not str(video_codec).strip() or not str(audio_codec).strip():
        raise ValueError("strict_decode clip requires explicit video and audio codecs")
    if isinstance(video_crf, bool) or int(video_crf) < 0:
        raise ValueError("video_crf must be a non-negative integer")
    duration_arg = f"{duration:.9f}"
    return [
        str(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        # 输入侧 seek：帧精确且避免与 trim=[0,duration] 组合成空流
        # （输出侧 -ss 会在 trim 之后丢弃全部帧，导致 video_stream_missing）
        "-ss",
        f"{start:.9f}",
        "-t",
        duration_arg,
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"trim=start=0:end={duration_arg},setpts=PTS-STARTPTS",
        "-af",
        f"atrim=start=0:end={duration_arg},asetpts=PTS-STARTPTS",
        "-c:v",
        str(video_codec),
        "-preset",
        str(video_preset),
        "-crf",
        str(int(video_crf)),
        "-c:a",
        str(audio_codec),
        "-b:a",
        str(audio_bitrate),
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _replace_output_path(command: list[str], output_path: Path) -> list[str]:
    replaced = list(command)
    if replaced:
        replaced[-1] = str(output_path)
    return replaced


def strict_decode_clip(
    source_path: Path,
    output_path: Path | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    *,
    cache_dir: Path | None = None,
    source_sha256: str | None = None,
    sidecar_path: Path | None = None,
    ffmpeg_bin: str | Path = "ffmpeg",
    ffprobe_bin: str | Path = "ffprobe",
    max_frame_boundary_error_sec: float = 0.05,
    max_duration_error_sec: float = 0.05,
    reuse: bool = True,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    video_preset: str = "veryfast",
    video_crf: int = 20,
    audio_bitrate: str = "192k",
) -> dict[str, Any]:
    """Create or strictly reuse a decoded clip and write its verified sidecar."""
    source_path = Path(source_path)
    if start_sec is None or end_sec is None:
        return _audit_result(FAIL, "clip_boundaries_required", source_path=str(source_path))
    try:
        start, end = _validate_interval(start_sec, end_sec)
        frame_tolerance = _validate_tolerance(max_frame_boundary_error_sec, "max_frame_boundary_error_sec")
        duration_tolerance = _validate_tolerance(max_duration_error_sec, "max_duration_error_sec")
    except ValueError as exc:
        return _audit_result(FAIL, "invalid_clip_request", error=str(exc), source_path=str(source_path))
    if not source_path.is_file():
        return _audit_result(FAIL, "source_missing", source_path=str(source_path))
    try:
        actual_source_sha256 = sha256_file(source_path)
    except OSError as exc:
        return _audit_result(FAIL, "source_hash_failed", error=str(exc), source_path=str(source_path))
    if source_sha256 is not None and _digest_text(source_sha256) != actual_source_sha256:
        return _audit_result(
            FAIL,
            "source_hash_mismatch",
            source_path=str(source_path),
            expected_source_sha256=source_sha256,
            actual_source_sha256=actual_source_sha256,
        )
    fingerprint = clip_cache_fingerprint(actual_source_sha256, start, end)
    if output_path is None:
        if cache_dir is None:
            return _audit_result(FAIL, "output_or_cache_dir_required", source_path=str(source_path), fingerprint=fingerprint)
        output_path = clip_cache_path(cache_dir, fingerprint)
    output_path = Path(output_path)
    if output_path.resolve() == source_path.resolve():
        return _audit_result(FAIL, "output_must_differ_from_source", source_path=str(source_path), output_path=str(output_path))
    resolved_sidecar_path = clip_sidecar_path(output_path, sidecar_path)

    resolved_ffmpeg = _resolve_tool(ffmpeg_bin)
    resolved_ffprobe = _resolve_tool(ffprobe_bin)
    missing_tools = [
        name
        for name, resolved in (("ffmpeg", resolved_ffmpeg), ("ffprobe", resolved_ffprobe))
        if resolved is None
    ]
    if missing_tools:
        command = None
        try:
            command = build_strict_clip_command(
                source_path,
                output_path,
                start,
                end,
                ffmpeg_bin=resolved_ffmpeg or ffmpeg_bin,
                video_codec=video_codec,
                audio_codec=audio_codec,
                video_preset=video_preset,
                video_crf=video_crf,
                audio_bitrate=audio_bitrate,
            )
        except ValueError:
            pass
        return _tool_missing_result(
            missing_tools,
            command=command,
            source_path=str(source_path),
            output_path=str(output_path),
            sidecar_path=str(resolved_sidecar_path),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
        )

    assert resolved_ffprobe is not None and resolved_ffmpeg is not None
    source_probe = probe_media(source_path, ffprobe_bin=resolved_ffprobe)
    if source_probe["status"] != PASS:
        return _audit_result(
            source_probe["status"],
            source_probe.get("reason"),
            source_path=str(source_path),
            output_path=str(output_path),
            sidecar_path=str(resolved_sidecar_path),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
            source_probe=source_probe,
        )
    time_base = source_probe.get("time_base")
    stream_start_pts = source_probe.get("stream_start_pts")
    stream_start_time_sec = source_probe.get("stream_start_time_sec")
    if time_base is None or stream_start_pts is None:
        return _audit_result(
            NOT_CHECKED,
            "strict_pts_metadata_missing",
            source_path=str(source_path),
            output_path=str(output_path),
            sidecar_path=str(resolved_sidecar_path),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
            source_probe=source_probe,
        )
    try:
        expected_start_pts, expected_end_pts = expected_boundary_pts(
            start,
            end,
            stream_start_pts=stream_start_pts,
            stream_start_time_sec=stream_start_time_sec,
            time_base=time_base,
        )
    except ValueError as exc:
        return _audit_result(NOT_CHECKED, "strict_pts_metadata_invalid", error=str(exc), source_probe=source_probe)
    boundary_probe = probe_boundary_pts(
        source_path,
        start,
        end,
        expected_start_pts=expected_start_pts,
        expected_end_pts=expected_end_pts,
        time_base=time_base,
        stream_start_time_sec=stream_start_time_sec,
        max_frame_boundary_error_sec=frame_tolerance,
        ffprobe_bin=resolved_ffprobe,
    )
    if boundary_probe["status"] != PASS:
        return _audit_result(
            boundary_probe["status"],
            boundary_probe.get("reason"),
            source_path=str(source_path),
            output_path=str(output_path),
            sidecar_path=str(resolved_sidecar_path),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
            requested_start_sec=start,
            requested_end_sec=end,
            requested_start_pts=expected_start_pts,
            requested_end_pts=expected_end_pts,
            source_probe=source_probe,
            boundary_probe=boundary_probe,
        )
    decoded_start_pts = boundary_probe.get("decoded_start_pts")
    decoded_end_pts = boundary_probe.get("decoded_end_pts")

    cache_validation: dict[str, Any] | None = None
    if reuse and output_path.is_file():
        try:
            cache_validation = validate_clip_cache_entry(
                output_path,
                source_path,
                start,
                end,
                sidecar_path=resolved_sidecar_path,
                expected_start_pts=expected_start_pts,
                expected_end_pts=expected_end_pts,
                observed_start_pts=decoded_start_pts,
                observed_end_pts=decoded_end_pts,
                time_base=time_base,
                max_frame_boundary_error_sec=frame_tolerance,
            )
        except ValueError as exc:
            cache_validation = _audit_result(FAIL, "cache_validation_error", error=str(exc))
        if cache_validation["status"] == PASS:
            return _audit_result(
                PASS,
                None,
                clip_path=str(output_path),
                sidecar_path=str(resolved_sidecar_path),
                fingerprint=fingerprint,
                source_sha256=actual_source_sha256,
                requested_start_sec=start,
                requested_end_sec=end,
                requested_start_pts=expected_start_pts,
                requested_end_pts=expected_end_pts,
                decoded_start_pts=decoded_start_pts,
                decoded_end_pts=decoded_end_pts,
                cache_hit=True,
                generated=False,
                source_probe=source_probe,
                boundary_probe=boundary_probe,
                cache_validation=cache_validation,
                sidecar=cache_validation.get("sidecar"),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix or ".mp4",
            dir=str(output_path.parent),
        )
        os.close(fd)
        temporary_output = Path(temporary_name)
        command = build_strict_clip_command(
            source_path,
            temporary_output,
            start,
            end,
            ffmpeg_bin=resolved_ffmpeg,
            video_codec=video_codec,
            audio_codec=audio_codec,
            video_preset=video_preset,
            video_crf=video_crf,
            audio_bitrate=audio_bitrate,
        )
    except (OSError, ValueError) as exc:
        if temporary_output is not None and temporary_output.exists():
            temporary_output.unlink()
        return _audit_result(FAIL, "strict_clip_command_build_failed", error=str(exc), output_path=str(output_path))

    command_result = _run_media_command(command)
    if command_result["status"] != PASS:
        if temporary_output.exists():
            temporary_output.unlink()
        return _audit_result(
            command_result["status"],
            command_result.get("reason"),
            clip_path=str(output_path),
            sidecar_path=str(resolved_sidecar_path),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
            command=_replace_output_path(command, output_path),
            command_result=command_result,
            cache_validation=cache_validation,
        )
    if not temporary_output.is_file() or temporary_output.stat().st_size <= 0:
        temporary_output.unlink(missing_ok=True)
        return _audit_result(
            FAIL,
            "strict_clip_output_missing_or_empty",
            clip_path=str(output_path),
            sidecar_path=str(resolved_sidecar_path),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
            command=_replace_output_path(command, output_path),
            cache_validation=cache_validation,
        )

    output_probe = probe_media(temporary_output, ffprobe_bin=resolved_ffprobe)
    if output_probe.get("basic_status") != PASS:
        temporary_output.unlink(missing_ok=True)
        output_status = output_probe["status"] if output_probe["status"] != PASS else FAIL
        return _audit_result(
            output_status,
            output_probe.get("reason") or "strict_clip_output_probe_failed",
            clip_path=str(output_path),
            sidecar_path=str(resolved_sidecar_path),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
            command=_replace_output_path(command, output_path),
            output_probe=output_probe,
            cache_validation=cache_validation,
        )
    actual_duration_sec = output_probe.get("duration_sec")
    duration_error_sec = (
        abs(actual_duration_sec - (end - start))
        if isinstance(actual_duration_sec, (int, float))
        else None
    )
    if duration_error_sec is None or duration_error_sec > duration_tolerance:
        temporary_output.unlink(missing_ok=True)
        return _audit_result(
            FAIL,
            "strict_clip_duration_out_of_tolerance",
            clip_path=str(output_path),
            sidecar_path=str(resolved_sidecar_path),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
            command=_replace_output_path(command, output_path),
            output_probe=output_probe,
            actual_duration_sec=actual_duration_sec,
            requested_duration_sec=end - start,
            duration_error_sec=duration_error_sec,
            cache_validation=cache_validation,
        )

    try:
        os.replace(str(temporary_output), str(output_path))
        temporary_output = None
        clip_digest = sha256_file(output_path)
        sidecar = {
            "clip_contract_version": CLIP_CONTRACT_VERSION,
            "source_sha256": actual_source_sha256,
            "requested_start_sec": start,
            "requested_end_sec": end,
            "requested_start_pts": expected_start_pts,
            "requested_end_pts": expected_end_pts,
            "decoded_start_pts": decoded_start_pts,
            "decoded_end_pts": decoded_end_pts,
            "cut_mode": STRICT_DECODE,
            "boundary_status": BOUNDARY_VERIFIED,
            "clip_sha256": clip_digest,
            "cache_fingerprint": fingerprint,
            "time_base": str(time_base),
            "stream_start_pts": stream_start_pts,
            "stream_start_time_sec": stream_start_time_sec,
            "boundary_error_start_sec": boundary_probe.get("start_error_sec"),
            "boundary_error_end_sec": boundary_probe.get("end_error_sec"),
            "requested_duration_sec": end - start,
            "clip_duration_sec": actual_duration_sec,
            "duration_error_sec": duration_error_sec,
            "ffmpeg_command": _replace_output_path(command, output_path),
        }
        written_sidecar = write_clip_sidecar(output_path, sidecar, resolved_sidecar_path)
    except (OSError, ValueError) as exc:
        return _audit_result(
            FAIL,
            "strict_clip_sidecar_write_failed",
            error=str(exc),
            clip_path=str(output_path),
            sidecar_path=str(resolved_sidecar_path),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
            command=_replace_output_path(command, output_path),
            cache_validation=cache_validation,
        )

    final_validation = validate_clip_cache_entry(
        output_path,
        source_path,
        start,
        end,
        sidecar_path=written_sidecar,
        expected_start_pts=expected_start_pts,
        expected_end_pts=expected_end_pts,
        observed_start_pts=decoded_start_pts,
        observed_end_pts=decoded_end_pts,
        time_base=time_base,
        max_frame_boundary_error_sec=frame_tolerance,
    )
    if final_validation["status"] != PASS:
        return _audit_result(
            final_validation["status"],
            final_validation.get("reason") or "strict_clip_final_validation_failed",
            clip_path=str(output_path),
            sidecar_path=str(written_sidecar),
            fingerprint=fingerprint,
            source_sha256=actual_source_sha256,
            command=_replace_output_path(command, output_path),
            source_probe=source_probe,
            boundary_probe=boundary_probe,
            output_probe=output_probe,
            sidecar=sidecar,
            validation=final_validation,
            cache_validation=cache_validation,
        )
    return _audit_result(
        PASS,
        None,
        clip_path=str(output_path),
        sidecar_path=str(written_sidecar),
        fingerprint=fingerprint,
        source_sha256=actual_source_sha256,
        requested_start_sec=start,
        requested_end_sec=end,
        requested_start_pts=expected_start_pts,
        requested_end_pts=expected_end_pts,
        decoded_start_pts=decoded_start_pts,
        decoded_end_pts=decoded_end_pts,
        actual_duration_sec=actual_duration_sec,
        requested_duration_sec=end - start,
        duration_error_sec=duration_error_sec,
        cache_hit=False,
        generated=True,
        command=_replace_output_path(command, output_path),
        source_probe=source_probe,
        boundary_probe=boundary_probe,
        output_probe=output_probe,
        sidecar=sidecar,
        validation=final_validation,
        cache_validation=cache_validation,
    )


def _is_pass_audit(audit: Mapping[str, Any] | None) -> bool:
    return isinstance(audit, Mapping) and audit.get("status") == PASS and audit.get("ok") is not False


def _validate_volume(value: Any, name: str) -> float:
    parsed = _finite_float(value)
    if parsed is None or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return parsed


def build_mux_command(
    clip_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    duration_sec: float,
    has_game_audio: bool,
    game_vol: float = 0.25,
    comm_vol: float = 1.0,
    ffmpeg_bin: str | Path = "ffmpeg",
    video_codec: str = "copy",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    clip_audit: Mapping[str, Any] | None = None,
) -> list[str]:
    """Build a no-``-shortest`` mux command with explicit audio padding/trimming.

    Video stream-copy is allowed only when the caller supplies a passing strict
    clip audit.  A re-encoding video codec can be selected when such an audit
    is unavailable.
    """
    duration = _finite_float(duration_sec)
    if duration is None or duration <= 0.0:
        raise ValueError("duration_sec must be positive")
    if not isinstance(has_game_audio, bool):
        raise ValueError("has_game_audio must be a boolean")
    game_volume = _validate_volume(game_vol, "game_vol")
    commentary_volume = _validate_volume(comm_vol, "comm_vol")
    if str(video_codec).strip().lower() in {"copy", "-c:v copy", "-codec:v copy"} and not _is_pass_audit(clip_audit):
        raise ValueError("video stream-copy requires a passing strict clip audit")
    if not str(video_codec).strip() or not str(audio_codec).strip():
        raise ValueError("mux requires explicit video and audio codecs")
    duration_arg = f"{duration:.6f}"
    if has_game_audio:
        audio_filter = (
            f"[0:a:0]volume={game_volume:g},apad,atrim=duration={duration_arg}[bg];"
            f"[1:a:0]volume={commentary_volume:g},apad,atrim=duration={duration_arg}[sp];"
            "[bg][sp]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
        )
    else:
        audio_filter = f"[1:a:0]volume={commentary_volume:g},apad,atrim=duration={duration_arg}[aout]"
    return [
        str(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clip_path),
        "-i",
        str(audio_path),
        "-filter_complex",
        audio_filter,
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        str(video_codec),
        "-c:a",
        str(audio_codec),
        "-b:a",
        str(audio_bitrate),
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def build_strict_mux_command(
    clip_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    duration_sec: float,
    has_game_audio: bool,
    game_vol: float = 0.25,
    comm_vol: float = 1.0,
    ffmpeg_bin: str | Path = "ffmpeg",
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    clip_audit: Mapping[str, Any] | None = None,
) -> list[str]:
    """Build a strict mux command; re-encoding is the default video policy."""
    return build_mux_command(
        clip_path,
        audio_path,
        output_path,
        duration_sec=duration_sec,
        has_game_audio=has_game_audio,
        game_vol=game_vol,
        comm_vol=comm_vol,
        ffmpeg_bin=ffmpeg_bin,
        video_codec=video_codec,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        clip_audit=clip_audit,
    )


def validate_mux_boundaries(
    clip_duration_sec: float | None,
    output_duration_sec: float | None,
    audio_duration_sec: float | None = None,
    *,
    tolerance_sec: float = 0.01,
    require_audio: bool = True,
) -> dict[str, Any]:
    """Validate that mux output and audio stay on the verified clip interval."""
    try:
        tolerance = _validate_tolerance(tolerance_sec, "tolerance_sec")
    except ValueError as exc:
        return _audit_result(FAIL, "invalid_mux_tolerance", error=str(exc))
    clip_duration = _finite_float(clip_duration_sec)
    output_duration = _finite_float(output_duration_sec)
    if clip_duration is None or clip_duration <= 0.0:
        return _audit_result(FAIL, "clip_duration_missing_or_invalid")
    if output_duration is None or output_duration <= 0.0:
        return _audit_result(FAIL, "output_duration_missing_or_invalid")
    checks: dict[str, str] = {}
    details: dict[str, str] = {}
    duration_error_sec = abs(output_duration - clip_duration)
    _record_check(
        checks,
        details,
        "video_duration",
        PASS if duration_error_sec <= tolerance else FAIL,
        f"mux output duration differs by more than {tolerance:.9f}s",
    )
    audio_duration = _finite_float(audio_duration_sec)
    if audio_duration is None:
        _record_check(
            checks,
            details,
            "audio_within_clip",
            NOT_CHECKED if require_audio else PASS,
            "audio duration is unavailable",
        )
    else:
        audio_ok = audio_duration >= 0.0 and audio_duration <= clip_duration + tolerance
        _record_check(checks, details, "audio_within_clip", PASS if audio_ok else FAIL, "audio exceeds verified clip duration")
    status = _status_for_checks(checks)
    return _audit_result(
        status,
        _reason_for_checks(checks, details),
        clip_duration_sec=clip_duration,
        output_duration_sec=output_duration,
        audio_duration_sec=audio_duration,
        duration_error_sec=duration_error_sec,
        tolerance_sec=tolerance,
        checks=checks,
        check_details=details,
    )


def mux_clip_with_audio(
    clip_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    clip_audit: Mapping[str, Any] | None = None,
    sidecar_path: Path | None = None,
    ffmpeg_bin: str | Path = "ffmpeg",
    ffprobe_bin: str | Path = "ffprobe",
    game_vol: float = 0.25,
    comm_vol: float = 1.0,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    duration_tolerance_sec: float = 0.01,
) -> dict[str, Any]:
    """Mux commentary into a verified clip without ``-shortest``."""
    clip_path = Path(clip_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    if not clip_path.is_file():
        return _audit_result(FAIL, "clip_missing", clip_path=str(clip_path))
    if not audio_path.is_file():
        return _audit_result(FAIL, "audio_missing", audio_path=str(audio_path))
    if clip_audit is None:
        try:
            sidecar = read_clip_sidecar(clip_path, sidecar_path)
        except (FileNotFoundError, ValueError) as exc:
            return _audit_result(
                FAIL,
                "strict_clip_sidecar_required",
                error=str(exc),
                clip_path=str(clip_path),
                sidecar_path=str(clip_sidecar_path(clip_path, sidecar_path)),
            )
        clip_audit = validate_clip_sidecar(sidecar, clip_path=clip_path, require_clip=True)
    if not _is_pass_audit(clip_audit):
        return _audit_result(
            clip_audit.get("status", FAIL),
            "strict_clip_not_verified",
            clip_path=str(clip_path),
            sidecar_path=str(sidecar_path or clip_sidecar_path(clip_path)),
            clip_audit=dict(clip_audit),
        )

    resolved_ffmpeg = _resolve_tool(ffmpeg_bin)
    resolved_ffprobe = _resolve_tool(ffprobe_bin)
    missing_tools = [
        name
        for name, resolved in (("ffmpeg", resolved_ffmpeg), ("ffprobe", resolved_ffprobe))
        if resolved is None
    ]
    if missing_tools:
        return _tool_missing_result(
            missing_tools,
            clip_path=str(clip_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            clip_audit=dict(clip_audit),
        )
    assert resolved_ffmpeg is not None and resolved_ffprobe is not None
    clip_probe = probe_media(clip_path, ffprobe_bin=resolved_ffprobe)
    if clip_probe.get("basic_status") != PASS:
        return _audit_result(
            clip_probe["status"] if clip_probe["status"] != PASS else FAIL,
            clip_probe.get("reason") or "clip_probe_failed",
            clip_path=str(clip_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            clip_probe=clip_probe,
            clip_audit=dict(clip_audit),
        )
    clip_duration_sec = clip_probe.get("duration_sec")
    has_game_audio = clip_probe.get("has_audio")
    if not isinstance(has_game_audio, bool):
        return _audit_result(FAIL, "clip_audio_probe_unknown", clip_probe=clip_probe, clip_audit=dict(clip_audit))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix or ".mp4",
            dir=str(output_path.parent),
        )
        os.close(fd)
        temporary_output = Path(temporary_name)
        command = build_mux_command(
            clip_path,
            audio_path,
            temporary_output,
            duration_sec=clip_duration_sec,
            has_game_audio=has_game_audio,
            game_vol=game_vol,
            comm_vol=comm_vol,
            ffmpeg_bin=resolved_ffmpeg,
            video_codec=video_codec,
            audio_codec=audio_codec,
            audio_bitrate=audio_bitrate,
            clip_audit=clip_audit,
        )
    except (OSError, ValueError) as exc:
        if temporary_output is not None and temporary_output.exists():
            temporary_output.unlink()
        return _audit_result(FAIL, "mux_command_build_failed", error=str(exc), output_path=str(output_path))

    command_result = _run_media_command(command)
    if command_result["status"] != PASS:
        temporary_output.unlink(missing_ok=True)
        return _audit_result(
            command_result["status"],
            command_result.get("reason"),
            clip_path=str(clip_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            command=_replace_output_path(command, output_path),
            command_result=command_result,
            clip_audit=dict(clip_audit),
        )
    if not temporary_output.is_file() or temporary_output.stat().st_size <= 0:
        temporary_output.unlink(missing_ok=True)
        return _audit_result(FAIL, "mux_output_missing_or_empty", output_path=str(output_path), clip_audit=dict(clip_audit))

    output_probe = probe_media(temporary_output, ffprobe_bin=resolved_ffprobe)
    if output_probe.get("basic_status") != PASS:
        temporary_output.unlink(missing_ok=True)
        return _audit_result(
            output_probe["status"] if output_probe["status"] != PASS else FAIL,
            output_probe.get("reason") or "mux_output_probe_failed",
            output_path=str(output_path),
            output_probe=output_probe,
            clip_audit=dict(clip_audit),
        )
    boundary_validation = validate_mux_boundaries(
        clip_duration_sec,
        output_probe.get("duration_sec"),
        output_probe.get("audio_duration_sec"),
        tolerance_sec=duration_tolerance_sec,
        require_audio=True,
    )
    if boundary_validation["status"] != PASS:
        temporary_output.unlink(missing_ok=True)
        return _audit_result(
            boundary_validation["status"],
            boundary_validation.get("reason") or "mux_boundary_validation_failed",
            clip_path=str(clip_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            command=_replace_output_path(command, output_path),
            clip_probe=clip_probe,
            output_probe=output_probe,
            boundary_validation=boundary_validation,
            clip_audit=dict(clip_audit),
        )
    try:
        os.replace(str(temporary_output), str(output_path))
        temporary_output = None
    except OSError as exc:
        return _audit_result(FAIL, "mux_output_commit_failed", error=str(exc), output_path=str(output_path), clip_audit=dict(clip_audit))
    return _audit_result(
        PASS,
        None,
        clip_path=str(clip_path),
        audio_path=str(audio_path),
        output_path=str(output_path),
        command=_replace_output_path(command, output_path),
        clip_probe=clip_probe,
        output_probe=output_probe,
        boundary_validation=boundary_validation,
        clip_audit=dict(clip_audit),
    )


strict_mux = mux_clip_with_audio
validate_clip_cache = validate_clip_cache_entry
write_sidecar = write_clip_sidecar
read_sidecar = read_clip_sidecar


__all__ = [
    "BOUNDARY_VERIFIED",
    "CLIP_CONTRACT_VERSION",
    "FAIL",
    "NOT_CHECKED",
    "PASS",
    "STRICT_DECODE",
    "build_mux_command",
    "build_strict_clip_command",
    "build_strict_mux_command",
    "clip_cache_fingerprint",
    "clip_cache_path",
    "clip_sidecar_path",
    "expected_boundary_pts",
    "mux_clip_with_audio",
    "probe_boundary_pts",
    "probe_media",
    "read_clip_sidecar",
    "read_sidecar",
    "round_half_even",
    "sha256_file",
    "strict_decode_clip",
    "strict_mux",
    "validate_clip_boundaries",
    "validate_clip_cache",
    "validate_clip_cache_entry",
    "validate_clip_sidecar",
    "validate_mux_boundaries",
    "validate_sample_bounds",
    "write_clip_sidecar",
    "write_sidecar",
]
