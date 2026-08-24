"""只读的流水线配置与输入校验（不产生任何写入）。"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from audio_service.emotion import parse_emotional_text
from sbmachine import neutral_contract, speech_measure, voice_task_contract
from sbmachine.common import count_spoken_chars
from sbmachine.phase3b_prompt import LLMB_HARD_CAP_FACTOR


class PublishContractError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def phase_enabled(phases: dict, new_key: str, old_key: str, default: bool = True) -> bool:
    return bool(phases.get(new_key, phases.get(old_key, default)))


def enabled_phases(config: dict) -> list[str]:
    phases = config.get("phases", {}) if isinstance(config.get("phases", {}), dict) else {}
    result: list[str] = []
    flags = (
        ("demo_parse", bool(phases.get("demo_parse", False))),
        ("video_marking", bool(phases.get("video_marking", False))),
        ("phase1", bool(phases.get("preprocess_slice", phases.get("phase1_slice", True)))),
        ("phase2", bool(phases.get("phase2_yolo", True))),
        ("phase3a", phase_enabled(phases, "phase3a_semantic", "phase3_semantic", True)),
        ("phase3b", phase_enabled(phases, "phase3b_semantic", "phase3_semantic", True)),
        ("phase3c", bool(phases.get("phase3c_render", False))),
        ("phase4", bool(phases.get("phase4_assemble", True))),
    )
    for name, active in flags:
        if active:
            result.append(name)
    return result


def _resolve(value: Any, root: Path) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _is_positive_field(name: str) -> bool:
    lowered = name.lower()
    return (
        "interval" in lowered
        or "timeout" in lowered
        or "window" in lowered
        or lowered in {"fps", "worker", "workers"}
        or lowered.endswith("_fps")
        or lowered.endswith("_worker")
        or lowered.endswith("_workers")
    )


def _validate_positive_values(value: Any, errors: list[str], path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_positive_values(child, errors, (*path, str(key)))
        return
    if not path or not _is_positive_field(path[-1]):
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        errors.append(f"{'.'.join(path)} must be a positive number")


def _build_input(name: str, value: Any, root: Path) -> dict:
    path = _resolve(value, root)
    return {"name": name, "path": str(path) if path is not None else "", "exists": bool(path and path.exists())}


def _required_inputs(config: dict, active: set[str], root: Path) -> list[dict]:
    paths = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    demo = config.get("demo", {}) if isinstance(config.get("demo", {}), dict) else {}
    slicer = config.get("slicer", {}) if isinstance(config.get("slicer", {}), dict) else {}
    yolo_root = config.get("yolo", {}) if isinstance(config.get("yolo", {}), dict) else {}
    yolo = yolo_root.get("yolo", {}) if isinstance(yolo_root.get("yolo", {}), dict) else {}
    phases = config.get("phases", {}) if isinstance(config.get("phases", {}), dict) else {}
    phase4 = config.get("phase4", {}) if isinstance(config.get("phase4", {}), dict) else {}
    required: list[dict] = []

    def add(name: str, value: Any) -> None:
        if any(item["name"] == name for item in required):
            return
        required.append(_build_input(name, value, root))

    if "demo_parse" in active:
        add("demo", paths.get("demo"))
    if "video_marking" in active:
        add("video", paths.get("video"))
        add("slicer_model", slicer.get("model", "models/qiepian/frame_type_classifier.pt"))
    if "phase1" in active:
        add("video", paths.get("video"))
        if "video_marking" not in active:
            segments = paths.get("segments_json")
            add("segments" if segments else "hud_detections", segments or paths.get("hud_detections_jsonl"))
    if "phase2" in active:
        if bool(yolo.get("enabled", True)):
            add("vision_yolo_model", yolo.get("model_path"))
        if "phase1" not in active:
            rounds_value = paths.get("rounds_json", "output/sbmachine/rounds.json")
            add("rounds", rounds_value)
            rounds_path = _resolve(rounds_value, root)
            if rounds_path is not None and rounds_path.is_file():
                try:
                    rounds_payload = json.loads(rounds_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    rounds_payload = None
                if isinstance(rounds_payload, dict) and rounds_payload.get("video_path"):
                    add("phase2_video", rounds_payload["video_path"])
        if "demo_parse" not in active:
            add("parsed_demo", demo.get("parsed_dir", "output/demo"))
    if "phase3a" in active or "phase3b" in active:
        if "phase2" not in active:
            add("rounds_with_yolo", paths.get("rounds_with_yolo_json", "output/sbmachine/rounds_with_yolo.json"))
    if "phase3a" in active and "phase2" not in active:
        add(
            "rounds_with_yolo_semantic",
            paths.get("rounds_with_yolo_semantic_json", "output/sbmachine/rounds_with_yolo_semantic.json"),
        )
    if "phase3b" in active:
        if "phase3a" not in active:
            add("rounds_with_neutral", paths.get("rounds_with_neutral_json", "output/sbmachine/rounds_with_neutral.json"))
        add("style_skill", paths.get("style_skill", "Prompt/skill/style_skill.md"))
    if "phase3c" in active:
        add("llmb_draft_package", paths.get("llmb_draft_package_json", "output/sbmachine/llmb_draft_package.json"))
    if "phase4" in active:
        if "phase3b" not in active:
            add("rounds_with_commentary", paths.get("rounds_with_commentary_json", "output/sbmachine/rounds_with_commentary.json"))
            if not bool(phases.get("phase3c_render", False)):
                add("commentary", paths.get("commentary_json", "output/sbmachine/commentary.json"))
        if bool(phases.get("phase3c_render", False)) or str(phase4.get("publish_profile", "legacy")) in {"strict_av", "strict_c", "broadcast"}:
            add("commentary_render_package", paths.get("commentary_render_package_json", "output/sbmachine/commentary_render_package.json"))
        if str(phase4.get("publish_profile", "legacy")) in {"strict_av", "strict_c", "broadcast"} and bool(phase4.get("make_video", False)):
            add("phase4_source_video", paths.get("video"))
        add("tts_config", phase4.get("tts_config", "audio_service/gpt_sovits_runtime.yaml"))
    return required


def preflight_config(config: dict, *, root: Path, only: Iterable[str] | None = None) -> dict:
    errors: list[str] = []
    if not isinstance(config, dict):
        errors.append("config root must be a mapping")
        config = {}
    _validate_positive_values(config, errors)
    semantic = config.get("semantic", {}) if isinstance(config.get("semantic", {}), dict) else {}
    minimum = semantic.get("window_min_sec")
    maximum = semantic.get("window_max_sec")
    if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)) and minimum > maximum:
        errors.append("semantic.window_min_sec must not exceed semantic.window_max_sec")

    phases = config.get("phases", {}) if isinstance(config.get("phases", {}), dict) else {}
    configured = enabled_phases(config)
    active = set(only) if only is not None else set(configured)
    phase4_requested = "phase4" in active

    # ── Phase3c 配置迁移校验（fail-closed，不允许静默兼容）──
    legacy_llmc = semantic.get("llmc")
    if isinstance(legacy_llmc, dict) and "enabled" in legacy_llmc:
        errors.append("semantic.llmc 已废弃，请迁移至 semantic.phase3c.mode")
    phase3c_render = bool(phases.get("phase3c_render", False))
    if phase3c_render:
        paths_cfg = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
        if not str(paths_cfg.get("llmb_draft_package_json") or "").strip():
            errors.append("paths.llmb_draft_package_json is required when phase3c_render is enabled")
        if not str(paths_cfg.get("commentary_render_package_json") or "").strip():
            errors.append("paths.commentary_render_package_json is required when phase3c_render is enabled")
        phase3c_cfg = semantic.get("phase3c")
        phase3c_cfg = phase3c_cfg if isinstance(phase3c_cfg, dict) else {}
        if phase3c_cfg.get("mode") not in {"off", "shadow", "optional", "required"}:
            errors.append("semantic.phase3c.mode must be one of off/shadow/optional/required")
        if phase3c_cfg.get("contract_version", 1) not in {1, 2}:
            errors.append("semantic.phase3c.contract_version must be 1 or 2")

    phase4_cfg = config.get("phase4", {}) if isinstance(config.get("phase4", {}), dict) else {}
    publish_profile = str(phase4_cfg.get("publish_profile", "legacy"))
    if publish_profile not in {"legacy", "strict_av", "strict_c", "broadcast"}:
        errors.append("phase4.publish_profile must be legacy/strict_av/strict_c/broadcast")
    clip_mode = str(phase4_cfg.get("clip_mode", "legacy_copy"))
    if clip_mode not in {"legacy_copy", "strict_decode"}:
        errors.append("phase4.clip_mode must be legacy_copy or strict_decode")
    if phase4_requested and publish_profile in {"strict_av", "strict_c", "broadcast"}:
        if clip_mode != "strict_decode":
            errors.append("strict Phase4 publish profiles require phase4.clip_mode=strict_decode")
        if not phase3c_render:
            errors.append("strict Phase4 publish profiles require phase3c_render")
        phase3c_cfg_for_phase4 = semantic.get("phase3c") if isinstance(semantic.get("phase3c"), dict) else {}
        if phase3c_cfg_for_phase4.get("contract_version", 1) != 2:
            errors.append("strict Phase4 publish profiles require semantic.phase3c.contract_version=2")
        if not bool(phase4_cfg.get("make_video", False)):
            errors.append("strict Phase4 publish profiles require phase4.make_video=true")
        media_probe_cfg = phase4_cfg.get("media_probe") if isinstance(phase4_cfg.get("media_probe"), dict) else {}
        for field in ("ffmpeg_bin", "ffprobe_bin"):
            configured_tool = str(media_probe_cfg.get(field) or "").strip()
            resolved_tool = _resolve(configured_tool, root) if configured_tool else None
            if not configured_tool or not ((resolved_tool is not None and resolved_tool.is_file()) or shutil.which(configured_tool)):
                errors.append(f"strict Phase4 requires executable phase4.media_probe.{field}")

    ordered_names = ("demo_parse", "video_marking", "phase1", "phase2", "phase3a", "phase3b", "phase3c", "phase4")
    inputs = _required_inputs(config, active, root)
    errors.extend(f"required input does not exist: {item['name']}={item['path']}" for item in inputs if not item["exists"])
    if "phase2" in active and "demo_parse" not in active:
        demo = config.get("demo", {}) if isinstance(config.get("demo", {}), dict) else {}
        parsed_dir = _resolve(demo.get("parsed_dir", "output/demo"), root)
        if parsed_dir is not None and parsed_dir.exists():
            try:
                validate_demo_publishable(parsed_dir)
            except PublishContractError as exc:
                errors.append(f"invalid parsed demo: {exc}")
    return {
        "config_valid": not errors,
        "enabled_phases": [name for name in (ordered_names if only is not None else configured) if name in active],
        "required_inputs": inputs,
        "services_started": [],
        "writes_performed": False,
        "errors": errors,
    }


def _read_manifest(path: Path, stage: str) -> dict:
    if not path.is_file():
        raise PublishContractError(stage, f"missing stage output: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishContractError(stage, f"invalid stage output: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublishContractError(stage, f"stage output must be a JSON object: {path}")
    return value


def _read_json(path: Path, stage: str) -> Any:
    if not path.is_file():
        raise PublishContractError(stage, f"missing stage output: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishContractError(stage, f"invalid stage output: {path}: {exc}") from exc


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _rounds(data: Any) -> list[Any]:
    return _as_list(data.get("rounds")) if isinstance(data, dict) else _as_list(data)


def _require_keys(item: Any, keys: Iterable[str], label: str, errors: list[str]) -> None:
    if not isinstance(item, dict):
        errors.append(f"{label} must be an object")
        return
    for key in keys:
        if key not in item:
            errors.append(f"{label} missing {key}")


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate_demo_artifacts(data: Any) -> list[str]:
    errors: list[str] = []
    bundle = data if isinstance(data, dict) else {}
    rounds = _as_list(bundle.get("rounds"))
    kills = _as_list(bundle.get("kills"))
    roster = _as_list(bundle.get("roster"))
    if not rounds:
        errors.append("demo.rounds must be a non-empty list")
    if not roster:
        errors.append("demo.roster must be a non-empty list")
    round_nos: set[int] = set()
    for row in rounds:
        if not isinstance(row, dict):
            continue
        try:
            round_nos.add(int(row.get("round_no", 0)))
        except (TypeError, ValueError):
            pass
    for index, item in enumerate(rounds):
        label = f"demo.rounds[{index}]"
        _require_keys(item, ("round_no", "start_tick", "freeze_end_tick", "end_tick"), label, errors)
        if isinstance(item, dict):
            try:
                invalid_bounds = int(item.get("start_tick", 0)) > int(item.get("end_tick", 0))
            except (TypeError, ValueError):
                errors.append(f"{label} tick fields must be integers")
            else:
                if invalid_bounds:
                    errors.append(f"{label} start_tick must be <= end_tick")
    for index, item in enumerate(kills):
        label = f"demo.kills[{index}]"
        _require_keys(item, ("round_no", "tick", "attacker", "victim"), label, errors)
        if isinstance(item, dict) and round_nos:
            try:
                missing_round = int(item.get("round_no", 0)) not in round_nos
            except (TypeError, ValueError):
                errors.append(f"{label}.round_no must be an integer")
            else:
                if missing_round:
                    errors.append(f"{label} references missing round_no")
    return errors


def validate_phase1_artifacts(data: Any) -> list[str]:
    errors: list[str] = []
    bundle = data if isinstance(data, dict) else {}
    rounds_payload = bundle.get("rounds")
    round_list_payload = bundle.get("round_list")
    segments_payload = bundle.get("segments")
    rounds = _rounds(rounds_payload)
    round_list = _rounds(round_list_payload)
    segments = _as_list(segments_payload.get("segments")) if isinstance(segments_payload, dict) else _as_list(segments_payload)
    if not isinstance(rounds_payload, dict) or not str(rounds_payload.get("video_path", "")).strip():
        errors.append("phase1 rounds must contain video_path")
    if not rounds:
        errors.append("phase1 rounds must be a non-empty list")
    if not round_list:
        errors.append("phase1 round_list must be a non-empty list")
    if not segments:
        errors.append("phase1 segments must be a non-empty list")

    round_nos: list[int] = []
    for index, item in enumerate(rounds):
        label = f"phase1.rounds[{index}]"
        _require_keys(item, ("round_no", "start_sec", "end_sec"), label, errors)
        if not isinstance(item, dict):
            continue
        round_no = item.get("round_no")
        if not isinstance(round_no, int) or isinstance(round_no, bool) or round_no <= 0:
            errors.append(f"{label}.round_no must be a positive integer")
        else:
            round_nos.append(round_no)
        start, end = _float_or_none(item.get("start_sec")), _float_or_none(item.get("end_sec"))
        if start is None or end is None or start >= end:
            errors.append(f"{label} must have finite start_sec < end_sec")
        hint = item.get("demo_round_hint")
        if hint is not None and hint != "unmatched":
            try:
                valid_hint = not isinstance(hint, bool) and int(hint) > 0
            except (TypeError, ValueError):
                valid_hint = False
            if not valid_hint:
                errors.append(f"{label}.demo_round_hint must be a positive round number or unmatched")

    listed_nos = [item.get("round_no") for item in round_list if isinstance(item, dict)]
    if round_nos and listed_nos != round_nos:
        errors.append("phase1 round_list round numbers must match rounds")
    for index, item in enumerate(segments):
        label = f"phase1.segments[{index}]"
        _require_keys(item, ("start_sec", "end_sec"), label, errors)
        if isinstance(item, dict):
            start, end = _float_or_none(item.get("start_sec")), _float_or_none(item.get("end_sec"))
            if start is None or end is None or start >= end:
                errors.append(f"{label} must have finite start_sec < end_sec")
    return errors


def _timeline_rows(data: Any) -> list[Any]:
    rows: list[Any] = []
    for round_data in _rounds(data):
        if not isinstance(round_data, dict):
            continue
        phase2 = round_data.get("_phase2_yolo") or round_data.get("phase2_yolo") or {}
        if isinstance(phase2, dict):
            rows.extend(_as_list(phase2.get("timeline")))
            rows.extend(_as_list(phase2.get("key_frames")))
    return rows


def _row_time(row: Any) -> float | None:
    if not isinstance(row, dict):
        return None
    when = row.get("when") if isinstance(row.get("when"), dict) else {}
    return _float_or_none(row.get("video_time", row.get("time_sec", when.get("video_time"))))


def validate_vision_timeline(data: Any) -> list[str]:
    rows = _timeline_rows(data)
    if not _rounds(data):
        return ["phase2 artifact must contain rounds"]
    if not rows:
        return ["phase2 timeline must contain at least one row"]
    errors: list[str] = []
    last_time = -1.0
    for index, row in enumerate(rows):
        time_value = _row_time(row)
        if time_value is None:
            errors.append(f"phase2.rows[{index}] missing numeric video_time/time_sec")
            continue
        if time_value < last_time:
            errors.append(f"phase2.rows[{index}] time must be monotonic")
        last_time = time_value
    return errors


_PHASE4_V2_CHECK_STATUSES = {"pass", "fail", "not_checked", "not_required", "degraded"}
_PHASE4_V2_PROFILES = {"legacy", "strict_av", "strict_c", "broadcast"}


def _phase4_v2_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_phase4_execution_v2(
    manifest: Any,
    *,
    rounds_final: Any = None,
    render_package: Any = None,
    publish_profile: str | None = None,
) -> list[str]:
    """Validate the strict Phase4 execution contract without weakening legacy checks."""
    errors: list[str] = []
    payload = manifest if isinstance(manifest, dict) else {}
    if payload.get("phase4_execution_contract_version") != 2:
        errors.append("phase4 execution contract version must be 2")
    if payload.get("sync_schema_version") != 2:
        errors.append("phase4 sync_schema_version must be 2")
    profile = publish_profile or payload.get("publish_profile")
    if profile not in _PHASE4_V2_PROFILES:
        errors.append(f"phase4 publish_profile is invalid: {profile!r}")
    if not isinstance(payload.get("render_package_artifact_identity"), str):
        errors.append("phase4 render_package_artifact_identity must be a string")
    for key in ("media_sync_status", "content_gate_status", "delivery_status"):
        if payload.get(key) not in _PHASE4_V2_CHECK_STATUSES:
            errors.append(f"phase4 {key} is invalid")

    media_checks = payload.get("media_checks")
    if not isinstance(media_checks, dict):
        errors.append("phase4 media_checks must be an object")
        media_checks = {}
    for key in (
        "source_identity",
        "decoded_pts_monotone",
        "clip_boundary",
        "audio_within_slot",
        "canvas_bounds",
        "subtitle_within_audio",
    ):
        if key not in media_checks:
            errors.append(f"phase4 media_checks.{key} is required")
    for key, value in media_checks.items():
        if value not in _PHASE4_V2_CHECK_STATUSES:
            errors.append(f"phase4 media_checks.{key} is invalid")

    content_checks = payload.get("content_checks")
    if not isinstance(content_checks, dict):
        errors.append("phase4 content_checks must be an object")
        content_checks = {}
    if content_checks.get("package_status") not in {"ready", "blocked"}:
        errors.append("phase4 content_checks.package_status is invalid")
    if not isinstance(content_checks.get("text_sources"), list) or any(
        not isinstance(item, str) for item in content_checks.get("text_sources", [])
    ):
        errors.append("phase4 content_checks.text_sources must be a string list")
    if content_checks.get("fact_check_scope") not in {"disabled", "strong"}:
        errors.append("phase4 content_checks.fact_check_scope is invalid")
    if not isinstance(content_checks.get("blocked_rounds"), int) or isinstance(content_checks.get("blocked_rounds"), bool):
        errors.append("phase4 content_checks.blocked_rounds must be an integer")

    final_payload = rounds_final if isinstance(rounds_final, dict) else {}
    final_rounds = _rounds(final_payload)
    manifest_rounds = payload.get("rounds")
    if not isinstance(manifest_rounds, list) or not manifest_rounds:
        errors.append("phase4 v2 manifest rounds must be a non-empty list")
        manifest_rounds = []
    final_nos = [item.get("round_no") for item in final_rounds if isinstance(item, dict)]
    manifest_nos = [item.get("round_no") for item in manifest_rounds if isinstance(item, dict)]
    if final_nos and manifest_nos != final_nos:
        errors.append("phase4 v2 manifest round numbers must match rounds_final")

    expected_identity = None
    if profile in {"strict_av", "strict_c", "broadcast"} and not isinstance(render_package, dict):
        errors.append("strict Phase4 requires commentary_render_package_v2")
    if isinstance(render_package, dict):
        expected_identity = render_package.get("artifact_identity")
        if render_package.get("contract") != "commentary_render_package_v2":
            errors.append("strict Phase4 requires commentary_render_package_v2")
        if isinstance(expected_identity, str) and payload.get("render_package_artifact_identity") != expected_identity:
            errors.append("phase4 render package identity does not match manifest")
        package_status = render_package.get("package_status")
        if package_status in {"ready", "blocked"} and content_checks.get("package_status") != package_status:
            errors.append("phase4 content_checks.package_status does not match render package")

    seen_sources: set[str] = set()
    blocked_rounds = 0
    round_media_statuses: list[str] = []
    round_content_statuses: list[str] = []
    round_delivery_statuses: list[str] = []
    round_skipped_flags: list[bool] = []
    for index, item in enumerate(manifest_rounds):
        label = f"phase4.rounds[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label} must be an object")
            continue
        for key in ("round_no", "audio_path", "skipped", "aligned", "segments", "media_sync_status", "content_gate_status", "delivery_status"):
            if key not in item:
                errors.append(f"{label}.{key} is required")
        round_no = item.get("round_no")
        if not isinstance(round_no, int) or isinstance(round_no, bool) or round_no <= 0:
            errors.append(f"{label}.round_no must be a positive integer")
        for key in ("media_sync_status", "content_gate_status", "delivery_status"):
            if item.get(key) not in _PHASE4_V2_CHECK_STATUSES:
                errors.append(f"{label}.{key} is invalid")
        round_media_statuses.append(str(item.get("media_sync_status")))
        round_content_statuses.append(str(item.get("content_gate_status")))
        round_delivery_statuses.append(str(item.get("delivery_status")))
        round_skipped_flags.append(bool(item.get("skipped")))
        if not isinstance(item.get("audio_path"), str):
            errors.append(f"{label}.audio_path must be a string")
        if not isinstance(item.get("skipped"), bool):
            errors.append(f"{label}.skipped must be boolean")
        if not isinstance(item.get("aligned"), bool):
            errors.append(f"{label}.aligned must be boolean")
        elif item.get("aligned") is not (item.get("media_sync_status") == "pass"):
            errors.append(f"{label}.aligned must be derived from media_sync_status")
        segments = item.get("segments")
        if not isinstance(segments, list):
            errors.append(f"{label}.segments must be a list")
            segments = []
        if item.get("content_gate_status") == "fail":
            blocked_rounds += 1
        unit_ids: set[str] = set()
        for segment_index, segment in enumerate(segments):
            segment_label = f"{label}.segments[{segment_index}]"
            if not isinstance(segment, dict):
                errors.append(f"{segment_label} must be an object")
                continue
            for key in (
                "unit_id", "sequence", "final_text", "emotion", "text_source", "render_slot",
                "actual_duration_sec", "asset_frame_count", "sample_rate", "applied_speed_factor",
                "fit_state", "attempted_speed_factors",
            ):
                if key not in segment:
                    errors.append(f"{segment_label}.{key} is required")
            unit_id = segment.get("unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                errors.append(f"{segment_label}.unit_id must be a non-empty string")
            elif unit_id in unit_ids:
                errors.append(f"{segment_label}.unit_id is duplicated")
            else:
                unit_ids.add(unit_id)
            if segment.get("sequence") != segment_index + 1:
                errors.append(f"{segment_label}.sequence must be {segment_index + 1}")
            if not isinstance(segment.get("final_text"), str) or not segment.get("final_text", "").strip():
                errors.append(f"{segment_label}.final_text must be non-empty")
            if not isinstance(segment.get("emotion"), str) or not segment.get("emotion"):
                errors.append(f"{segment_label}.emotion must be a non-empty string")
            text_source = segment.get("text_source")
            if text_source not in {"llmc", "llmb_passthrough"}:
                errors.append(f"{segment_label}.text_source is invalid")
            elif isinstance(text_source, str):
                seen_sources.add(text_source)
            slot = segment.get("render_slot")
            if not isinstance(slot, dict):
                errors.append(f"{segment_label}.render_slot must be an object")
                slot = {}
            for key in ("start_sec", "end_sec"):
                if not _phase4_v2_number(slot.get(key)):
                    errors.append(f"{segment_label}.render_slot.{key} must be finite")
            if _phase4_v2_number(slot.get("start_sec")) and _phase4_v2_number(slot.get("end_sec")) and slot["start_sec"] >= slot["end_sec"]:
                errors.append(f"{segment_label}.render_slot must have start_sec < end_sec")
            for key in ("start_tick", "end_tick"):
                if not isinstance(slot.get(key), int) or isinstance(slot.get(key), bool):
                    errors.append(f"{segment_label}.render_slot.{key} must be an integer")
            if isinstance(slot.get("start_tick"), int) and isinstance(slot.get("end_tick"), int) and slot["start_tick"] >= slot["end_tick"]:
                errors.append(f"{segment_label}.render_slot must have start_tick < end_tick")
            actual_duration = segment.get("actual_duration_sec")
            if not _phase4_v2_number(actual_duration) or float(actual_duration) < 0:
                errors.append(f"{segment_label}.actual_duration_sec must be non-negative")
            frame_count = segment.get("asset_frame_count")
            sample_rate = segment.get("sample_rate")
            if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 0:
                errors.append(f"{segment_label}.asset_frame_count must be a non-negative integer")
            if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
                errors.append(f"{segment_label}.sample_rate must be a positive integer")
            if isinstance(frame_count, int) and isinstance(sample_rate, int) and _phase4_v2_number(actual_duration) and abs(float(actual_duration) - frame_count / sample_rate) > 1e-6:
                errors.append(f"{segment_label}.actual_duration_sec must equal frames/sample_rate")
            if not _phase4_v2_number(segment.get("applied_speed_factor")) or not 1.0 <= float(segment.get("applied_speed_factor")) <= 1.5:
                errors.append(f"{segment_label}.applied_speed_factor must be within [1.0, 1.5]")
            attempted = segment.get("attempted_speed_factors")
            if not isinstance(attempted, list) or any(not _phase4_v2_number(value) for value in attempted):
                errors.append(f"{segment_label}.attempted_speed_factors must be numeric list")
            fit_state = segment.get("fit_state")
            if fit_state not in {"fit", "render_unfit"}:
                errors.append(f"{segment_label}.fit_state is invalid")
            if fit_state == "fit" and (not isinstance(frame_count, int) or frame_count <= 0):
                errors.append(f"{segment_label}.fit asset must contain PCM frames")
            sample_fields = (
                "round_canvas_start_sample", "round_canvas_end_sample", "round_slot_end_sample",
                "round_canvas_limit_sample", "timeline_start_sample", "timeline_end_sample",
                "slot_timeline_end_sample", "timeline_canvas_end_sample",
            )
            for key in sample_fields:
                if key not in segment:
                    errors.append(f"{segment_label}.{key} is required")
            sample_values = {key: segment.get(key) for key in sample_fields if key in segment}
            for key, value in sample_values.items():
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"{segment_label}.{key} must be a non-negative integer")
            if all(key in sample_values for key in ("round_canvas_start_sample", "round_canvas_end_sample", "round_canvas_limit_sample", "round_slot_end_sample")):
                if sample_values["round_canvas_end_sample"] < sample_values["round_canvas_start_sample"]:
                    errors.append(f"{segment_label} round canvas end precedes start")
                if sample_values["round_canvas_end_sample"] > sample_values["round_slot_end_sample"]:
                    errors.append(f"{segment_label} audio exceeds slot sample end")
                if sample_values["round_slot_end_sample"] > sample_values["round_canvas_limit_sample"]:
                    errors.append(f"{segment_label} slot exceeds round canvas")
            if all(key in sample_values for key in ("timeline_start_sample", "timeline_end_sample", "slot_timeline_end_sample", "timeline_canvas_end_sample")):
                if sample_values["timeline_end_sample"] < sample_values["timeline_start_sample"]:
                    errors.append(f"{segment_label} timeline end precedes start")
                if sample_values["timeline_end_sample"] > sample_values["slot_timeline_end_sample"]:
                    errors.append(f"{segment_label} audio exceeds timeline slot")
                if sample_values["slot_timeline_end_sample"] > sample_values["timeline_canvas_end_sample"]:
                    errors.append(f"{segment_label} slot exceeds timeline canvas")

    if isinstance(content_checks.get("text_sources"), list) and sorted(set(content_checks["text_sources"])) != sorted(seen_sources):
        errors.append("phase4 content_checks.text_sources does not match execution segments")
    if isinstance(content_checks.get("blocked_rounds"), int) and content_checks.get("blocked_rounds") != blocked_rounds:
        errors.append("phase4 content_checks.blocked_rounds does not match execution rounds")

    expected_media_status = (
        "fail" if "fail" in round_media_statuses else
        "not_checked" if "not_checked" in round_media_statuses else
        "pass" if round_media_statuses else
        "not_required"
    )
    expected_content_status = (
        "fail" if "fail" in round_content_statuses else
        "degraded" if "degraded" in round_content_statuses else
        "pass" if round_content_statuses else
        "not_required"
    )
    if payload.get("media_sync_status") != expected_media_status:
        errors.append("phase4 media_sync_status does not match round media statuses")
    if payload.get("content_gate_status") != expected_content_status:
        errors.append("phase4 content_gate_status does not match round content statuses")
    if profile == "legacy":
        expected_delivery_status = "fail" if "fail" in round_delivery_statuses else "pass"
    elif profile == "strict_av":
        expected_delivery_status = (
            "pass"
            if expected_media_status == "pass" and expected_content_status in {"pass", "degraded"} and not blocked_rounds
            else "fail"
        )
    elif profile == "strict_c":
        expected_delivery_status = (
            "pass"
            if expected_media_status == "pass" and expected_content_status == "pass" and not blocked_rounds
            else "fail"
        )
    else:
        expected_delivery_status = "fail"
    if payload.get("delivery_status") != expected_delivery_status:
        errors.append("phase4 delivery_status does not match publish profile gate")
    if profile in {"strict_av", "strict_c"}:
        if not any(status == "pass" for status in round_media_statuses):
            errors.append(f"{profile} requires at least one checked media pass")
        if any(
            status not in {"pass", "not_required"}
            or (status == "not_required" and not skipped)
            for status, skipped in zip(round_media_statuses, round_skipped_flags)
        ):
            errors.append(f"{profile} cannot contain unchecked or failed media rounds")
        if any(
            status != "pass"
            and not (status == "not_required" and skipped)
            for status, skipped in zip(round_delivery_statuses, round_skipped_flags)
        ):
            errors.append(f"{profile} requires every delivered round to pass")

    if profile == "strict_av":
        if payload.get("media_sync_status") != "pass":
            errors.append("strict_av requires media_sync_status=pass")
        if payload.get("delivery_status") != "pass":
            errors.append("strict_av requires delivery_status=pass")
        if content_checks.get("package_status") != "ready" or blocked_rounds:
            errors.append("strict_av requires a ready render package with no blocked rounds")
    elif profile == "strict_c":
        if payload.get("media_sync_status") != "pass":
            errors.append("strict_c requires media_sync_status=pass")
        if payload.get("delivery_status") != "pass" or payload.get("content_gate_status") != "pass":
            errors.append("strict_c requires content and delivery status pass")
        if content_checks.get("package_status") != "ready" or content_checks.get("fact_check_scope") != "strong":
            errors.append("strict_c requires ready package and strong fact scope")
        if isinstance(render_package, dict):
            policy = render_package.get("content_policy") or {}
            if policy.get("phase3c_mode") != "required":
                errors.append("strict_c requires phase3c mode required")
        if "llmb_passthrough" in seen_sources:
            errors.append("strict_c does not allow llmb_passthrough text")
    elif profile == "broadcast":
        errors.append("broadcast profile requires the W7 global sync contract")
    return errors


def validate_phase4_execution_v2(
    manifest: Any,
    rounds_final: Any = None,
    render_package: Any = None,
    *,
    publish_profile: str | None = None,
) -> list[str]:
    """Public list-returning validator for execution v2 and publish gates."""
    return _validate_phase4_execution_v2(
        manifest,
        rounds_final=rounds_final,
        render_package=render_package,
        publish_profile=publish_profile,
    )


def validate_final_manifest(data: Any, commentary: Any = None) -> list[str]:
    errors: list[str] = []
    bundle = data if isinstance(data, dict) else {}
    rounds_final = bundle.get("rounds_final") or bundle.get("rounds") or bundle
    rounds = _rounds(rounds_final)
    if not rounds:
        errors.append("phase4 rounds_final must contain rounds")
    if commentary is not None:
        voice_manifest = bundle if isinstance(bundle.get("rounds_final"), dict) else {"rounds_final": rounds_final}
        errors.extend(voice_task_contract.validate_final_voice_task(voice_manifest, commentary=commentary))
    manifest = bundle.get("assemble_manifest") or bundle.get("manifest") or {}
    if not isinstance(manifest, dict):
        errors.append("phase4 assemble_manifest must be an object")
        return errors
    if manifest.get("phase4_execution_contract_version") == 2:
        errors.extend(
            validate_phase4_execution_v2(
                manifest,
                rounds_final=rounds_final,
                render_package=commentary if isinstance(commentary, dict) and commentary.get("contract") == "commentary_render_package_v2" else None,
            )
        )
        return errors
    manifest_rounds = _as_list(manifest.get("rounds"))
    if not manifest_rounds:
        errors.append("phase4 assemble_manifest must contain rounds")
    final_nos = [item.get("round_no") for item in rounds if isinstance(item, dict)]
    manifest_nos = [item.get("round_no") for item in manifest_rounds if isinstance(item, dict)]
    if final_nos and manifest_nos != final_nos:
        errors.append("phase4 assemble_manifest round numbers must match rounds_final")
    for index, item in enumerate(rounds):
        if not isinstance(item, dict):
            errors.append(f"phase4.rounds[{index}] must be an object")
            continue
        audio = item.get("_phase4_audio") or item.get("phase4_audio") or {}
        if not isinstance(audio, dict) or not isinstance(audio.get("audio_path"), str):
            errors.append(f"phase4.rounds[{index}] must contain a string audio_path")
    for index, item in enumerate(manifest_rounds):
        label = f"phase4.assemble_manifest.rounds[{index}]"
        _require_keys(item, ("round_no", "audio_path", "aligned", "segments"), label, errors)
        if isinstance(item, dict):
            if not isinstance(item.get("audio_path"), str):
                errors.append(f"{label}.audio_path must be a string")
            if item.get("aligned") is not True:
                errors.append(f"{label}.aligned must be true")
            if not isinstance(item.get("segments"), list):
                errors.append(f"{label}.segments must be a list")
    return errors


def _raise_contract_errors(stage: str, errors: list[str]) -> None:
    if errors:
        raise PublishContractError(stage, "; ".join(errors))


def validate_demo_publishable(parsed_dir: Path) -> None:
    from tools.demo.demo_manifest import DemoManifestError, validate_demo_manifest

    try:
        validate_demo_manifest(parsed_dir)
    except (DemoManifestError, OSError, ValueError) as exc:
        raise PublishContractError("demo_parse", str(exc)) from exc
    payload = {
        name: _read_json(parsed_dir / f"{name}.json", "demo_parse")
        for name in ("rounds", "kills", "roster")
    }
    _raise_contract_errors("demo_parse", validate_demo_artifacts(payload))


def validate_phase1_publishable(rounds_path: Path, round_list_path: Path, segments_path: Path) -> None:
    payload = {
        "rounds": _read_json(rounds_path, "phase1"),
        "round_list": _read_json(round_list_path, "phase1"),
        "segments": _read_json(segments_path, "phase1"),
    }
    _raise_contract_errors("phase1", validate_phase1_artifacts(payload))


def validate_phase2_publishable(path: Path) -> None:
    _raise_contract_errors("phase2", validate_vision_timeline(_read_json(path, "phase2")))


def validate_phase4_publishable(rounds_path: Path, manifest_path: Path, commentary_path: Path | None = None) -> None:
    commentary = None
    if commentary_path is not None:
        commentary = _read_json(commentary_path, "phase4")
    rounds_final = _read_json(rounds_path, "phase4")
    manifest = _read_json(manifest_path, "phase4")
    if isinstance(manifest, dict) and manifest.get("phase4_execution_contract_version") == 2:
        render_package = commentary if isinstance(commentary, dict) and commentary.get("contract") == "commentary_render_package_v2" else None
        _raise_contract_errors(
            "phase4",
            validate_phase4_execution_v2(
                manifest,
                rounds_final=rounds_final,
                render_package=render_package,
            ),
        )
        return
    payload = {
        "rounds_final": rounds_final,
        "assemble_manifest": manifest,
    }
    _raise_contract_errors("phase4", validate_final_manifest(payload, commentary=commentary))


def validate_neutral_publishable(path: Path) -> None:
    manifest = _read_manifest(path, "phase3a")
    # 方案 R：recovery 配置由 manifest 携带（phase3a 写入）。未声明 → 视为禁用（K=0）。
    recovery_meta = manifest.get("recovery")
    recovery_enabled = bool(recovery_meta.get("enabled", False)) if isinstance(recovery_meta, dict) else False
    rounds = manifest.get("rounds")
    if not isinstance(rounds, list):
        raise PublishContractError("phase3a", "neutral manifest rounds must be a list")
    unrecoverable_count = 0
    infra_error_count = 0
    windows_total = 0
    for round_data in rounds:
        if not isinstance(round_data, dict) or round_data.get("analyst_failed"):
            raise PublishContractError("phase3a", "analyst failure is not publishable")
        round_no = round_data.get("round_no", "?") if isinstance(round_data, dict) else "?"
        scenes = round_data.get("scenes", [])
        if not isinstance(scenes, list):
            raise PublishContractError("phase3a", "neutral scenes must be a list")
        for idx, scene in enumerate(scenes):
            windows_total += 1
            if not isinstance(scene, dict):
                raise PublishContractError("phase3a", f"neutral scene round={round_no}[{idx}] must be an object")
            generation_status = scene.get("generation_status", "")
            neutral_source = scene.get("neutral_source", "")
            # S2: 构建窗口定位信息
            window_id = scene.get("window_id", f"r{round_no:03d}_w{idx+1:02d}" if isinstance(round_no, int) else f"window[{idx}]")
            t_start = scene.get("t_start", "?") if "t_start" in scene else "?"
            t_end = scene.get("t_end", "?") if "t_end" in scene else "?"
            loc = f"phase3a round={round_no} {window_id} t={t_start}-{t_end}"
            # A: 字段不存在 → legacy；字段存在但为空（如 contract_error 故意留空）不是 legacy。
            if "generation_status" not in scene or "neutral_source" not in scene:
                raise PublishContractError(
                    "phase3a", f"{loc}: legacy neutral is not publishable; rerun Phase3a",
                )
            neutral = str(scene.get("neutral") or "")

            # 方案 R：基建错误占比统计（§6.3）——按 first_attempt_status 统计所有
            # 遭遇过基建错误的窗口（无论最终是否恢复），用于 10% 中止裁决。
            first_attempt_status = scene.get("first_attempt_status")
            if first_attempt_status in {"transport_error", "http_error", "response_error"}:
                infra_error_count += 1

            # 新字段优先：generation_status 是权威判断依据
            if generation_status:
                if generation_status == "success":
                    # generation_status == "success": 来源必须是 Phase3a 明确声明的合法值。
                    # "rule" = 规则层 summary 兜底（LLM-A 失败时的确定性中性稿）。
                    if neutral_source == "fallback":
                        raise PublishContractError("phase3a", f"{loc}: rule fallback neutral is not publishable")
                    if neutral_source not in {"llm", "intentional_empty", "llm_retry", "rule"}:
                        raise PublishContractError(
                            "phase3a", f"{loc}: neutral source {neutral_source!r} is not publishable",
                        )
                    continue
                # 方案 R（§6.4）：不可恢复终态——generation_status 保留真实失败类，
                # neutral_source="unrecoverable"，neutral 为空。由 K 配额裁决是否放行。
                if neutral_source == "unrecoverable":
                    unrecoverable_count += 1
                    continue
                # 非 success 且未标 unrecoverable → 不可发布（fail-closed）
                raise PublishContractError(
                    "phase3a",
                    f"{loc}: window generation error ({generation_status}) is not publishable",
                )

            # 向后兼容：旧 JSON 无 generation_status，沿用旧逻辑
            if neutral_source == "fallback" and neutral:
                raise PublishContractError("phase3a", f"{loc}: rule fallback neutral is not publishable")

    # ── 方案 R 配额裁决（§6.5）──
    if windows_total > 0:
        # K = max(2, floor(0.03 × windows_total))；恢复未启用时 K=0（§6.7 零容忍）
        k_quota = max(2, math.floor(0.03 * windows_total)) if recovery_enabled else 0
        if unrecoverable_count > k_quota:
            raise PublishContractError(
                "phase3a",
                f"unrecoverable windows {unrecoverable_count} exceed K quota {k_quota} "
                f"(recovery={'on' if recovery_enabled else 'off'}, windows_total={windows_total})",
            )
        # 基建错误比例（>10%）中止裁决已于 2026-08-16 移除：失败窗口交由
        # K 配额（可发布量兜底）与下游空回合语义处理，不再因占比直接判整场失败。


def _is_v4_silent_scene(scene: Any) -> bool:
    """schema v4 静默 scene：显式 neutral_source=intentional_empty，或
    scene_kind=intentional_empty 且 neutral 为空（golden fixture 的省略写法）。"""
    if not isinstance(scene, dict):
        return False
    if scene.get("neutral_source") == "intentional_empty":
        return True
    return scene.get("scene_kind") == "intentional_empty" and not str(scene.get("neutral") or "").strip()


def _normalize_v4_silent_scenes(manifest: Any) -> None:
    """静默 scene 豁免 rule_capsule/fact_catalog 等 v4 字段（§7.5 静默状态校验）。

    只对内存中的已解析 dict 补齐基础字段，不写回文件、不重算历史结果。
    """
    for round_data in _rounds(manifest):
        if not isinstance(round_data, dict):
            continue
        for scene in _as_list(round_data.get("scenes")):
            if not _is_v4_silent_scene(scene):
                continue
            scene.setdefault("neutral_source", "intentional_empty")
            scene.setdefault("fact_anchors", {})
            scene.setdefault("char_budget", 0)


def validate_neutral_v4_publishable(path: Path) -> None:
    """Phase3a neutral 产物按 schema_version 分流发布校验。

    - 4：调用 neutral_contract.validate_neutral_v4（fail-closed）；
      静默 scene（neutral_source=intentional_empty）不要求 rule_capsule/fact_catalog。
    - 3：现有 validate_neutral_publishable 路径不变。
    - 其他版本：拒绝，不静默修复。
    """
    manifest = _read_manifest(path, "phase3a")
    schema_version = manifest.get("schema_version")
    if schema_version == neutral_contract.SCHEMA_VERSION_V4:
        _normalize_v4_silent_scenes(manifest)
        try:
            neutral_contract.validate_neutral_v4(manifest)
        except ValueError as exc:
            raise PublishContractError("phase3a", str(exc)) from exc
        return
    if schema_version == neutral_contract.SCHEMA_VERSION:
        validate_neutral_publishable(path)
        return
    raise PublishContractError(
        "phase3a", f"unsupported neutral schema_version: {schema_version!r}; expected 3 or 4",
    )


_SPARSE_V1_POLICY: dict[str, tuple[str, ...]] = {
    "green": ("primary", "capsule"),
    "amber": ("primary", "compact", "capsule"),
    "red": ("compact", "capsule"),
}


def validate_commentary_publishable(
    path: Path,
    neutral_path: Path | None = None,
    rounds_commentary_path: Path | None = None,
) -> None:
    manifest = _read_manifest(path, "phase3b")
    if manifest.get("commentary_schema_version") == voice_task_contract.SCHEMA_COMMENTARY_V3:
        _raise_contract_errors("phase3b", _validate_commentary_v3(manifest, path, neutral_path, rounds_commentary_path))
        return
    if manifest.get("commentary_schema_version") != 2:
        raise PublishContractError("phase3b", "commentary_schema_version=2 is required; rerun Phase3b")
    source_run_id = manifest.get("source_neutral_run_id")
    source_sha = manifest.get("source_neutral_sha256")
    source_count = manifest.get("source_window_count")
    if not isinstance(source_run_id, str) or not source_run_id.strip():
        raise PublishContractError("phase3b", "source_neutral_run_id is required")
    if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", source_sha) is None:
        raise PublishContractError("phase3b", "source_neutral_sha256 must be a SHA-256 hex digest")
    if not isinstance(source_count, int) or isinstance(source_count, bool) or source_count < 0:
        raise PublishContractError("phase3b", "source_window_count must be a non-negative integer")
    source_path = neutral_path or path.with_name("rounds_with_neutral.json")
    if neutral_path is not None and not source_path.is_file():
        raise PublishContractError("phase3b", f"missing source neutral artifact: {source_path}")
    if source_path.is_file():
        neutral_bytes = source_path.read_bytes()
        if hashlib.sha256(neutral_bytes).hexdigest() != source_sha:
            raise PublishContractError("phase3b", "source_neutral_sha256 does not match rounds_with_neutral.json")
        try:
            neutral_manifest = json.loads(neutral_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublishContractError("phase3b", f"invalid source neutral artifact: {exc}") from exc
        if not isinstance(neutral_manifest, dict) or neutral_manifest.get("run_id") != source_run_id:
            raise PublishContractError("phase3b", "source_neutral_run_id does not match rounds_with_neutral.json")
        actual_source_count = sum(len(item.get("scenes", [])) for item in neutral_manifest.get("rounds", []) if isinstance(item, dict) and isinstance(item.get("scenes", []), list))
        if actual_source_count != source_count:
            raise PublishContractError("phase3b", "source_window_count does not match rounds_with_neutral.json")
    rounds = manifest.get("rounds")
    if not isinstance(rounds, list):
        raise PublishContractError("phase3b", "commentary manifest rounds must be a list")
    style_config = manifest.get("effective_style_config") or {}
    k_enabled = bool(style_config.get("style_k_enabled", False))
    errors: list[str] = []
    all_window_ids: set[str] = set()
    counted_windows = 0
    success_statuses = {"ok", "retry_success"}
    allowed_statuses = success_statuses | {"skipped_intentional_empty", "skipped_unrecoverable", "style_failed", "upstream_failed"}
    for round_index, round_data in enumerate(rounds):
        label = f"rounds[{round_index}]"
        if not isinstance(round_data, dict):
            errors.append(f"{label} must be an object")
            continue
        window_results = round_data.get("window_results")
        scenes = round_data.get("scenes")
        if not isinstance(window_results, list) or not isinstance(scenes, list):
            errors.append(f"{label} must contain window_results and scenes lists")
            continue
        counted_windows += len(window_results)
        success_indices: set[int] = set()
        required: list[dict] = []
        for window_index, item in enumerate(window_results):
            item_label = f"{label}.window_results[{window_index}]"
            if not isinstance(item, dict):
                errors.append(f"{item_label} must be an object")
                continue
            window_id = item.get("window_id")
            if not isinstance(window_id, str) or not window_id:
                errors.append(f"{item_label}.window_id must be non-empty")
            elif window_id in all_window_ids:
                errors.append(f"duplicate window_id: {window_id}")
            else:
                all_window_ids.add(window_id)
            style_status = item.get("style_status")
            if style_status not in allowed_statuses:
                errors.append(f"{item_label}.style_status is invalid")
            retry_count = item.get("retry_count")
            if not isinstance(retry_count, int) or isinstance(retry_count, bool) or not 0 <= retry_count <= 2:
                errors.append(f"{item_label}.retry_count must be between 0 and 2")
            neutral_nonempty = item.get("neutral_nonempty")
            if not isinstance(neutral_nonempty, bool):
                errors.append(f"{item_label}.neutral_nonempty must be boolean")
            neutral_source = item.get("neutral_source")
            scene_index = item.get("published_scene_index")
            if neutral_nonempty and neutral_source != "unrecoverable":
                required.append(item)
                if style_status == "skipped_intentional_empty":
                    errors.append(f"{item_label} non-empty neutral cannot be intentional empty")
            if style_status in success_statuses:
                if neutral_nonempty is not True:
                    errors.append(f"{item_label} successful style requires non-empty neutral")
                if not isinstance(scene_index, int) or isinstance(scene_index, bool) or not 0 <= scene_index < len(scenes):
                    errors.append(f"{item_label} has invalid published_scene_index")
                    continue
                if scene_index in success_indices:
                    errors.append(f"{label} scene index {scene_index} is referenced more than once")
                success_indices.add(scene_index)
                scene = scenes[scene_index]
                if not isinstance(scene, dict):
                    errors.append(f"{label}.scenes[{scene_index}] must be an object")
                    continue
                for key in ("t_start", "t_end", "text", "emotion"):
                    if key not in scene:
                        errors.append(f"{label}.scenes[{scene_index}] missing {key}")
                # 唯一严格逐窗校验：scenes 与成功 window_results 一对一，
                # 窗口身份/时间/风格/预算审计/口播字数强校验（LLM-C 已独立为
                # Phase3c，不再放宽；整合段由 Phase3c 消费 B 封存包后另行交付）。
                if scene.get("window_id") != window_id:
                    errors.append(f"{item_label} does not match its scene window_id")
                if scene.get("t_start") != item.get("t_start") or scene.get("t_end") != item.get("t_end"):
                    errors.append(f"{item_label} does not match its scene time range")
                if scene.get("style_status") != style_status:
                    errors.append(f"{item_label} does not match its scene style_status")
                if scene.get("char_budget") != item.get("char_budget") or scene.get("output_chars") != item.get("output_chars"):
                    errors.append(f"{item_label} does not match its scene budget audit")
                rendered_scene = f"[{scene.get('emotion', '')}]{scene.get('text', '')}"
                if item.get("output_chars") != count_spoken_chars(re.sub(r"\[[^\]]{1,4}\]", "", rendered_scene).strip()):
                    errors.append(f"{item_label}.output_chars does not match rendered scene text")
            elif scene_index is not None:
                errors.append(f"{item_label} failed/skipped window must not reference a scene")
            if style_status == "skipped_intentional_empty" and (neutral_nonempty or neutral_source != "intentional_empty"):
                errors.append(f"{item_label} has invalid intentional-empty state")
            budget, output_chars = item.get("char_budget"), item.get("output_chars")
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                errors.append(f"{item_label}.char_budget must be positive")
            if output_chars is not None and (not isinstance(output_chars, int) or isinstance(output_chars, bool) or output_chars < 0):
                errors.append(f"{item_label}.output_chars must be null or non-negative")
            elif isinstance(output_chars, int) and isinstance(budget, int) and output_chars > int(budget * LLMB_HARD_CAP_FACTOR):
                errors.append(f"{item_label} exceeds char_budget")
            if style_status in success_statuses and not isinstance(output_chars, int):
                errors.append(f"{item_label} successful style requires output_chars")
        if success_indices != set(range(len(scenes))):
            errors.append(f"{label} scenes must map one-to-one to successful window_results")

        successes = [item for item in required if item.get("style_status") in success_statuses]
        upstream = any(item.get("style_status") in {"upstream_failed", "skipped_unrecoverable"} for item in window_results if isinstance(item, dict))
        analyst_failure = bool(window_results) and all(item.get("style_status") == "upstream_failed" and item.get("failure_reason") == "analyst_failed" for item in window_results if isinstance(item, dict))
        if analyst_failure or (not window_results and round_data.get("status") == "analyst_failed"):
            expected_status = "analyst_failed"
        elif not required and window_results and not upstream and all(item.get("style_status") == "skipped_intentional_empty" for item in window_results if isinstance(item, dict)):
            expected_status = "silent"
        elif (
            round_data.get("status") == "empty"
            and required
            and not successes
        ):
            # 只兼容读取旧版 30% 门禁已经产出的 empty；新产物不再生成该状态。
            expected_status = "empty"
        elif required and len(successes) == len(required) and not upstream:
            expected_status = "ok"
        elif required and not successes and all(item.get("style_status") == "style_failed" for item in required):
            expected_status = "style_failed"
        else:
            expected_status = "partial"
        if round_data.get("status") != expected_status:
            errors.append(f"{label}.status must be {expected_status}")
        # 可发布状态：ok/silent 与历史 empty 放行；partial 仅在「至少有一个成功窗口」
        # （有可播内容，败窗留空）时放行，零窗口/全败的 partial 仍拒。
        if expected_status in {"ok", "silent", "empty"}:
            pass
        elif expected_status == "partial" and len(successes) > 0:
            pass
        else:
            errors.append(f"{label} has non-publishable status {expected_status}")

        rendered = "".join(f"[{scene.get('emotion', '')}]{scene.get('text', '')}" for scene in scenes if isinstance(scene, dict))
        if round_data.get("commentary_text") != rendered:
            errors.append(f"{label}.commentary_text does not match scenes")
        expected_segments = [dict(emotion=seg.emotion, text=seg.text, order=index) for index, seg in enumerate(parse_emotional_text(rendered))]
        if round_data.get("emotion_segments") != expected_segments:
            errors.append(f"{label}.emotion_segments do not match scenes")
    if counted_windows != source_count:
        errors.append(f"source_window_count={source_count} does not match window_results total={counted_windows}")

    # ── K 配额：仅在 style_k_enabled=true 时启用 ──
    if k_enabled:
        style_failed_count = 0
        for round_data in rounds:
            for item in round_data.get("window_results", []):
                if isinstance(item, dict) and item.get("style_status") == "style_failed":
                    style_failed_count += 1
        if style_failed_count > 0:
            k_quota = max(2, math.floor(0.03 * source_count))
            if style_failed_count <= k_quota:
                errors = [e for e in errors if "has non-publishable status" not in e]
            else:
                errors.append(f"style_failed windows {style_failed_count} exceed K quota {k_quota} ({source_count} total)")

    _raise_contract_errors("phase3b", errors)


def _validate_commentary_v3(
    manifest: dict,
    path: Path,
    neutral_path: Path | None,
    rounds_commentary_path: Path | None,
) -> list[str]:
    """commentary v3 的 preflight 校验（§9.6）：

    - 结构契约：voice_task_contract.validate_commentary_v3（候选闭包/事实覆盖/slot）。
    - 来源 hash：source_neutral_sha256/run_id 与 neutral 产物一致（存在时核对）。
    - primary 引用：每个 voice_task_id 在 rounds_with_commentary 中有对应 scene
      （voice_task_id 匹配 window_id、slot 一致、primary_variant_id=primary、文本一致）。
    - 选择策略：候选集合与 selection_order 必须等于 sparse_v1 按 risk_class 的规范组合。
    - 统一语音计量（§11.7）：任务引用的 profile 存在且 validated 时，用
      speech_measure.measure_text 重新核对候选 spoken_units 与安全上界；不用 len() 替代。
    """
    errors: list[str] = list(voice_task_contract.validate_commentary_v3(manifest))
    if manifest.get("speech_metric_version") != speech_measure.METRIC_VERSION:
        errors.append(f"speech_metric_version must be {speech_measure.METRIC_VERSION!r} for commentary v3")

    source_run_id = manifest.get("source_neutral_run_id")
    source_sha = manifest.get("source_neutral_sha256")
    if not isinstance(source_run_id, str) or not source_run_id.strip():
        errors.append("source_neutral_run_id is required")
    if not isinstance(source_sha, str) or not source_sha.strip():
        errors.append("source_neutral_sha256 is required")
    source_path = neutral_path or path.with_name("rounds_with_neutral.json")
    if neutral_path is not None and not source_path.is_file():
        errors.append(f"missing source neutral artifact: {source_path}")
    elif source_path.is_file():
        neutral_bytes = source_path.read_bytes()
        if hashlib.sha256(neutral_bytes).hexdigest() != source_sha:
            errors.append("source_neutral_sha256 does not match rounds_with_neutral.json")
        try:
            neutral_manifest = json.loads(neutral_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid source neutral artifact: {exc}")
        else:
            if not isinstance(neutral_manifest, dict) or neutral_manifest.get("run_id") != source_run_id:
                errors.append("source_neutral_run_id does not match rounds_with_neutral.json")

    commentary_scenes: dict[str, dict[str, Any]] = {}
    rounds_commentary_path = rounds_commentary_path or path.with_name("rounds_with_commentary.json")
    if not rounds_commentary_path.is_file():
        errors.append(f"missing rounds_with_commentary artifact: {rounds_commentary_path}")
    else:
        try:
            rounds_payload = json.loads(rounds_commentary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid rounds_with_commentary artifact: {exc}")
        else:
            if not isinstance(rounds_payload, dict):
                errors.append("rounds_with_commentary artifact must be an object")
            elif rounds_payload.get("source_neutral_sha256") != source_sha:
                errors.append("source_neutral_sha256 does not match rounds_with_commentary")
            for round_data in _rounds(rounds_payload):
                if not isinstance(round_data, dict):
                    continue
                for scene in _as_list(round_data.get("scenes")):
                    if not isinstance(scene, dict) or not isinstance(scene.get("voice_task_id"), str):
                        continue
                    commentary_scenes.setdefault(scene["voice_task_id"], scene)

    voice_tasks = manifest.get("voice_tasks")
    voice_tasks = voice_tasks if isinstance(voice_tasks, list) else []
    for task in voice_tasks:
        task_label = f"voice_task_id {task.get('voice_task_id', '?')}"
        if not isinstance(task, dict):
            errors.append("commentary v3 voice_tasks entries must be objects")
            continue
        task_id = task.get("voice_task_id")
        if not isinstance(task_id, str):
            continue
        scene = commentary_scenes.get(task_id)
        if scene is None:
            errors.append(f"{task_label} is missing from rounds_with_commentary")
            continue
        if scene.get("window_id") != task.get("window_id"):
            errors.append(f"{task_label} primary reference window_id does not match the voice task")
        slot = task.get("render_slot") if isinstance(task.get("render_slot"), dict) else {}
        for slot_key, scene_key in (("start_sec", "t_start"), ("end_sec", "t_end")):
            expected = _float_or_none(slot.get(slot_key))
            actual = _float_or_none(scene.get(scene_key))
            if expected is not None and actual is not None and not math.isclose(actual, expected, abs_tol=1e-6):
                errors.append(f"{task_label} primary reference {scene_key} does not match the voice task render_slot")
                break
        if scene.get("primary_variant_id") != "primary":
            errors.append(f"{task_label} primary reference must declare primary_variant_id=primary")
        primary = next(
            (
                candidate
                for candidate in _as_list(task.get("candidates"))
                if isinstance(candidate, dict) and candidate.get("variant_id") == "primary"
            ),
            None,
        )
        if primary is not None and scene.get("text") != primary.get("text"):
            errors.append(f"{task_label} primary reference text does not match the primary candidate")

    for task in voice_tasks:
        if not isinstance(task, dict):
            continue
        task_label = f"voice_task_id {task.get('voice_task_id', '?')}"
        risk_class = task.get("risk_class")
        policy = _SPARSE_V1_POLICY.get(risk_class)
        if policy is None:
            continue
        expected_order = list(policy)
        if task.get("selection_order") != expected_order:
            errors.append(
                f"{task_label} selection_order must be {expected_order} for risk_class {risk_class}",
            )
        candidate_variants = {
            candidate.get("variant_id")
            for candidate in _as_list(task.get("candidates"))
            if isinstance(candidate, dict)
        }
        if candidate_variants != set(policy):
            errors.append(
                f"{task_label} candidate set {sorted(candidate_variants)} does not match "
                f"risk_class {risk_class} selection policy {expected_order}",
            )

    for task in voice_tasks:
        if not isinstance(task, dict):
            continue
        task_label = f"voice_task_id {task.get('voice_task_id', '?')}"
        profile_id = task.get("speech_profile_id")
        profile = speech_measure.load_profile(profile_id) if isinstance(profile_id, str) else None
        if profile is None:
            continue
        try:
            profile_status = speech_measure.validate_profile_status(profile)
        except speech_measure.ProfileError:
            continue
        if profile_status != "validated" or profile.get("metric_version") != speech_measure.METRIC_VERSION:
            continue
        for candidate in _as_list(task.get("candidates")):
            if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
                continue
            measure = speech_measure.measure_text(candidate["text"], profile_id=profile_id)
            if candidate.get("spoken_units") != measure["spoken_units"]:
                errors.append(f"{task_label} candidate {candidate.get('variant_id', '?')} spoken_units does not match measure_text")
            upper = measure.get("safe_duration_upper_bound_at_base_speed_sec")
            if upper is not None and candidate.get("safe_duration_upper_bound_at_base_speed_sec") != upper:
                errors.append(
                    f"{task_label} candidate {candidate.get('variant_id', '?')} "
                    "safe_duration_upper_bound_at_base_speed_sec does not match measure_text",
                )
    return errors


def validate_llmb_draft_package(path: Path) -> None:
    """B1 出口封存门禁，按合同版本分流 v1/v2。"""
    package = _read_manifest(path, "phase3b")
    if package.get("contract") == "llmb_draft_package_v2":
        _validate_llmb_draft_package_v2(package)
        return
    errors: list[str] = []
    if package.get("contract") != "llmb_draft_package_v1":
        errors.append(f"contract must be llmb_draft_package_v1, got {package.get('contract')!r}")
    if package.get("producer") != "phase3b":
        errors.append(f"producer must be phase3b, got {package.get('producer')!r}")
    if not isinstance(package.get("run_id"), str) or not package.get("run_id"):
        errors.append("run_id is required")
    source = package.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
    else:
        for key in ("neutral_run_id", "neutral_sha256", "timeline_id"):
            if not isinstance(source.get(key), str) or not source.get(key):
                errors.append(f"source.{key} is required")
    if not isinstance(package.get("artifact_identity"), str) or not package.get("artifact_identity"):
        errors.append("artifact_identity is required")
    rounds = package.get("rounds")
    if not isinstance(rounds, list):
        errors.append("rounds must be a list")
    else:
        round_ids: set[str] = set()
        for round_index, round_data in enumerate(rounds):
            label = f"rounds[{round_index}]"
            if not isinstance(round_data, dict):
                errors.append(f"{label} must be an object")
                continue
            round_id = round_data.get("round_id")
            if not isinstance(round_id, str) or not round_id:
                errors.append(f"{label}.round_id is required")
            elif round_id in round_ids:
                errors.append(f"{label} duplicate round_id: {round_id}")
            else:
                round_ids.add(round_id)
            status = round_data.get("status")
            if status not in {"ready", "intentional_silent", "operator_accepted_skip"}:
                errors.append(f"{label}.status is invalid: {status!r}")
            units = round_data.get("units")
            if not isinstance(units, list):
                errors.append(f"{label}.units must be a list")
                continue
            if status == "ready" and not units:
                errors.append(f"{label}.status ready requires non-empty units")
            unit_ids: set[str] = set()
            for unit_index, unit in enumerate(units):
                unit_label = f"{label}.units[{unit_index}]"
                if not isinstance(unit, dict):
                    errors.append(f"{unit_label} must be an object")
                    continue
                unit_id = unit.get("unit_id")
                if not isinstance(unit_id, str) or not unit_id:
                    errors.append(f"{unit_label}.unit_id is required")
                elif unit_id in unit_ids:
                    errors.append(f"{unit_label} duplicate unit_id: {unit_id}")
                else:
                    unit_ids.add(unit_id)
                draft_text = unit.get("draft_text")
                if not isinstance(draft_text, str) or not draft_text.strip():
                    errors.append(f"{unit_label}.draft_text must be non-empty")
                emotion_binding = unit.get("emotion_binding")
                if not isinstance(emotion_binding, dict) or not isinstance(emotion_binding.get("emotion"), str) or not emotion_binding.get("emotion"):
                    errors.append(f"{unit_label}.emotion_binding.emotion is required")
                allowed_fact_ids = unit.get("allowed_fact_ids")
                if not isinstance(allowed_fact_ids, list):
                    errors.append(f"{unit_label}.allowed_fact_ids must be a list")
                elif status == "ready" and not allowed_fact_ids:
                    errors.append(f"{unit_label}.allowed_fact_ids must be non-empty for ready units")
                slot = unit.get("render_slot")
                if not isinstance(slot, dict):
                    errors.append(f"{unit_label}.render_slot must be an object")
                else:
                    start_tick, end_tick = slot.get("start_tick"), slot.get("end_tick")
                    if not isinstance(start_tick, int) or isinstance(start_tick, bool) or not isinstance(end_tick, int) or isinstance(end_tick, bool):
                        errors.append(f"{unit_label}.render_slot start_tick/end_tick must be integers")
                    elif start_tick >= end_tick:
                        errors.append(f"{unit_label}.render_slot start_tick must be < end_tick")
                capacity = unit.get("speech_capacity")
                if not isinstance(capacity, dict):
                    errors.append(f"{unit_label}.speech_capacity must be an object")
                else:
                    for key in ("slot_sec", "safe_upper_sec", "required_speed_factor", "draft_hard_speed_factor"):
                        if not isinstance(capacity.get(key), (int, float)) or isinstance(capacity.get(key), bool) or capacity.get(key) <= 0:
                            errors.append(f"{unit_label}.speech_capacity.{key} must be a positive number")
    _raise_contract_errors("phase3b", errors)


def _validate_llmb_draft_package_v2(package: dict) -> None:
    """Validate the execution-ready B v2 package without changing v1 semantics."""
    errors: list[str] = []
    if package.get("producer") != "phase3b":
        errors.append(f"producer must be phase3b, got {package.get('producer')!r}")
    if not isinstance(package.get("run_id"), str) or not package.get("run_id"):
        errors.append("run_id is required")
    source = package.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
        source = {}
    for key in ("neutral_run_id", "neutral_sha256", "timeline_id", "source_video_sha256"):
        if not isinstance(source.get(key), str):
            errors.append(f"source.{key} must be a string")
    if source.get("neutral_run_id") != package.get("run_id"):
        errors.append("source.neutral_run_id must match run_id")
    tts_policy = package.get("tts_policy")
    if not isinstance(tts_policy, dict):
        errors.append("tts_policy must be an object")
    else:
        if tts_policy.get("profile_status") not in {"validated", "missing", "stale", "mismatch", "not_required"}:
            errors.append("tts_policy.profile_status is invalid")
        if not isinstance(tts_policy.get("speech_profile_id"), str):
            errors.append("tts_policy.speech_profile_id must be a string")
        if tts_policy.get("require_validated_profile") not in {True, False}:
            errors.append("tts_policy.require_validated_profile must be boolean")
        max_speed = tts_policy.get("max_speed_factor")
        if not isinstance(max_speed, (int, float)) or isinstance(max_speed, bool) or not 1.0 <= float(max_speed) <= 1.5:
            errors.append("tts_policy.max_speed_factor must be within [1.0, 1.5]")
    fps = package.get("render_timebase_fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(float(fps)) or float(fps) <= 0:
        errors.append("render_timebase_fps must be a positive number")
    rounds = package.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        errors.append("rounds must be a non-empty list")
        rounds = []
    round_ids: set[str] = set()
    expected_timeline_id = source.get("timeline_id")
    for round_index, round_data in enumerate(rounds):
        label = f"rounds[{round_index}]"
        if not isinstance(round_data, dict):
            errors.append(f"{label} must be an object")
            continue
        round_id = round_data.get("round_id")
        round_no = round_data.get("round_no")
        if not isinstance(round_id, str) or not round_id:
            errors.append(f"{label}.round_id is required")
        elif round_id in round_ids:
            errors.append(f"{label}.round_id is duplicated")
        else:
            round_ids.add(round_id)
        if not isinstance(round_no, int) or isinstance(round_no, bool) or round_no <= 0:
            errors.append(f"{label}.round_no must be a positive integer")
        elif isinstance(round_id, str) and round_id != f"r{round_no:03d}":
            errors.append(f"{label}.round_id must match round_no")
        status = round_data.get("status")
        if status not in {"ready", "intentional_silent", "operator_accepted_skip"}:
            errors.append(f"{label}.status is invalid: {status!r}")
        units = round_data.get("units")
        if not isinstance(units, list):
            errors.append(f"{label}.units must be a list")
            continue
        if status == "ready" and not units:
            errors.append(f"{label}.status ready requires non-empty units")
        if status != "ready" and units:
            errors.append(f"{label}.units must be empty for {status}")
        unit_ids: set[str] = set()
        for unit_index, unit in enumerate(units):
            unit_label = f"{label}.units[{unit_index}]"
            if not isinstance(unit, dict):
                errors.append(f"{unit_label} must be an object")
                continue
            unit_id = unit.get("unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                errors.append(f"{unit_label}.unit_id is required")
            elif unit_id in unit_ids:
                errors.append(f"{unit_label}.unit_id is duplicated")
            else:
                unit_ids.add(unit_id)
            sequence = unit.get("sequence")
            if sequence != unit_index + 1:
                errors.append(f"{unit_label}.sequence must be {unit_index + 1}")
            if not isinstance(unit.get("draft_text"), str) or not unit["draft_text"].strip():
                errors.append(f"{unit_label}.draft_text must be non-empty")
            binding = unit.get("emotion_binding")
            if not isinstance(binding, dict) or not isinstance(binding.get("emotion"), str) or not binding["emotion"]:
                errors.append(f"{unit_label}.emotion_binding.emotion is required")
            for key in ("allowed_fact_ids", "carry_in_fact_ids"):
                if not isinstance(unit.get(key), list) or any(not isinstance(item, str) for item in unit.get(key, [])):
                    errors.append(f"{unit_label}.{key} must be a string list")
            if status == "ready" and not unit.get("allowed_fact_ids"):
                errors.append(f"{unit_label}.allowed_fact_ids must be non-empty for ready units")
            catalog = unit.get("fact_catalog")
            if not isinstance(catalog, (dict, list)):
                errors.append(f"{unit_label}.fact_catalog must be an object or list")
            slot = unit.get("render_slot")
            if not isinstance(slot, dict):
                errors.append(f"{unit_label}.render_slot must be an object")
            else:
                for key in ("slot_id", "timeline_id"):
                    if not isinstance(slot.get(key), str) or not slot[key]:
                        errors.append(f"{unit_label}.render_slot.{key} is required")
                if slot.get("slot_id") != unit_id:
                    errors.append(f"{unit_label}.render_slot.slot_id must match unit_id")
                if slot.get("timeline_id") != expected_timeline_id:
                    errors.append(f"{unit_label}.render_slot.timeline_id must match source.timeline_id")
                seconds = (slot.get("start_sec"), slot.get("end_sec"))
                ticks = (slot.get("start_tick"), slot.get("end_tick"))
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in seconds):
                    errors.append(f"{unit_label}.render_slot.start_sec/end_sec must be finite numbers")
                elif seconds[0] >= seconds[1]:
                    errors.append(f"{unit_label}.render_slot must have start_sec < end_sec")
                if any(isinstance(value, bool) or not isinstance(value, int) for value in ticks):
                    errors.append(f"{unit_label}.render_slot.start_tick/end_tick must be integers")
                elif ticks[0] >= ticks[1]:
                    errors.append(f"{unit_label}.render_slot must have start_tick < end_tick")
                if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in seconds) and all(isinstance(value, int) and not isinstance(value, bool) for value in ticks):
                    if abs(float(ticks[0]) - float(seconds[0]) * float(fps or 30.0)) > 2.0 or abs(float(ticks[1]) - float(seconds[1]) * float(fps or 30.0)) > 2.0:
                        errors.append(f"{unit_label}.render_slot seconds and ticks do not agree")
            capacity = unit.get("speech_capacity")
            if not isinstance(capacity, dict):
                errors.append(f"{unit_label}.speech_capacity must be an object")
            else:
                for key in ("slot_sec", "safe_upper_sec", "required_speed_factor", "draft_hard_speed_factor"):
                    value = capacity.get(key)
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                        errors.append(f"{unit_label}.speech_capacity.{key} must be a positive number")
                if isinstance(slot, dict) and isinstance(slot.get("start_sec"), (int, float)) and isinstance(slot.get("end_sec"), (int, float)):
                    if not math.isclose(float(capacity.get("slot_sec", -1)), float(slot["end_sec"]) - float(slot["start_sec"]), abs_tol=1e-3):
                        errors.append(f"{unit_label}.speech_capacity.slot_sec must match render_slot seconds")
    body = {key: value for key, value in package.items() if key != "artifact_identity"}
    expected_identity = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if package.get("artifact_identity") != expected_identity:
        errors.append("artifact_identity mismatch")
    _raise_contract_errors("phase3b", errors)


def validate_render_package(path: Path, draft_package_path: Path | None = None) -> None:
    """C7 出口封存 + P4 入场门禁，按 render package 版本分流。"""
    payload = _read_manifest(path, "phase3c")
    if payload.get("contract") == "commentary_render_package_v2":
        _validate_render_package_v2(payload, draft_package_path)
        return
    errors: list[str] = []
    if payload.get("contract") != "commentary_render_package_v1":
        errors.append(f"contract must be commentary_render_package_v1, got {payload.get('contract')!r}")
    if payload.get("producer") != "phase3c":
        errors.append(f"producer must be phase3c, got {payload.get('producer')!r}")
    if payload.get("status") != "render_ready":
        errors.append(f"status must be render_ready, got {payload.get('status')!r}")
    if payload.get("llmc_mode") not in {"off", "shadow", "optional", "required"}:
        errors.append(f"llmc_mode is invalid: {payload.get('llmc_mode')!r}")
    source = payload.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
    else:
        for key in ("llmb_artifact_identity", "neutral_run_id", "neutral_sha256", "timeline_id"):
            if not isinstance(source.get(key), str) or not source.get(key):
                errors.append(f"source.{key} is required")
    if not isinstance(payload.get("artifact_identity"), str) or not payload.get("artifact_identity"):
        errors.append("artifact_identity is required")
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        errors.append("rounds must be a list")
    else:
        for round_index, round_data in enumerate(rounds):
            label = f"rounds[{round_index}]"
            if not isinstance(round_data, dict):
                errors.append(f"{label} must be an object")
                continue
            if not isinstance(round_data.get("round_id"), str) or not round_data.get("round_id"):
                errors.append(f"{label}.round_id is required")
            integration_status = round_data.get("integration_status")
            if integration_status not in {"llmc_accepted", "llmb_passthrough", "blocked", "skipped"}:
                errors.append(f"{label}.integration_status is invalid: {integration_status!r}")
            selected_source = round_data.get("selected_source")
            if selected_source not in {"llmc", "llmb_passthrough"}:
                errors.append(f"{label}.selected_source is invalid: {selected_source!r}")
            if integration_status == "llmc_accepted" and selected_source != "llmc":
                errors.append(f"{label} llmc_accepted must select llmc source")
            if integration_status in {"llmb_passthrough", "skipped"} and selected_source != "llmb_passthrough":
                errors.append(f"{label} {integration_status} must select llmb_passthrough source")
            render_units = round_data.get("render_units")
            if not isinstance(render_units, list):
                errors.append(f"{label}.render_units must be a list")
                continue
            if integration_status in {"blocked", "skipped"} and render_units:
                errors.append(f"{label}.render_units must be empty for {integration_status}")
            unit_ids: set[str] = set()
            for unit_index, unit in enumerate(render_units):
                unit_label = f"{label}.render_units[{unit_index}]"
                if not isinstance(unit, dict):
                    errors.append(f"{unit_label} must be an object")
                    continue
                unit_id = unit.get("unit_id")
                if not isinstance(unit_id, str) or not unit_id:
                    errors.append(f"{unit_label}.unit_id is required")
                elif unit_id in unit_ids:
                    errors.append(f"{unit_label} duplicate unit_id: {unit_id}")
                else:
                    unit_ids.add(unit_id)
                text = unit.get("text")
                if not isinstance(text, str) or not text.strip():
                    errors.append(f"{unit_label}.text must be non-empty")
                elif re.search(r"\[(平述|激动|惊叹)\]", text):
                    errors.append(f"{unit_label}.text must not contain emotion tags")
                if not isinstance(unit.get("emotion"), str) or not unit.get("emotion"):
                    errors.append(f"{unit_label}.emotion is required")
                slot = unit.get("render_slot")
                if not isinstance(slot, dict):
                    errors.append(f"{unit_label}.render_slot must be an object")
                else:
                    start_tick, end_tick = slot.get("start_tick"), slot.get("end_tick")
                    if not isinstance(start_tick, int) or isinstance(start_tick, bool) or not isinstance(end_tick, int) or isinstance(end_tick, bool):
                        errors.append(f"{unit_label}.render_slot start_tick/end_tick must be integers")
                    elif start_tick >= end_tick:
                        errors.append(f"{unit_label}.render_slot start_tick must be < end_tick")
                required_fact_ids = unit.get("required_fact_ids")
                if not isinstance(required_fact_ids, list) or not required_fact_ids:
                    errors.append(f"{unit_label}.required_fact_ids must be a non-empty list")
                speed = unit.get("required_speed_factor")
                if not isinstance(speed, (int, float)) or isinstance(speed, bool) or not 1.0 <= float(speed) <= 1.5:
                    errors.append(f"{unit_label}.required_speed_factor must be within [1.0, 1.5]")
                if unit.get("source") not in {"llmc", "llmb"}:
                    errors.append(f"{unit_label}.source must be llmc or llmb")
    if draft_package_path is not None and draft_package_path.is_file():
        try:
            package = _read_manifest(draft_package_path, "phase3c")
        except PublishContractError:
            package = None
        if isinstance(package, dict):
            if source.get("llmb_artifact_identity") != package.get("artifact_identity"):
                errors.append("source.llmb_artifact_identity does not match llmb draft package artifact_identity")
            b_by_round = {r.get("round_id"): r for r in _as_list(package.get("rounds")) if isinstance(r, dict)}
            for round_data in rounds:
                if not isinstance(round_data, dict) or not isinstance(round_data.get("round_id"), str):
                    continue
                b_round = b_by_round.get(round_data["round_id"])
                if b_round is None:
                    errors.append(f"{round_data['round_id']} missing from llmb draft package")
                    continue
                b_unit_ids = [u.get("unit_id") for u in _as_list(b_round.get("units")) if isinstance(u, dict)]
                render_ids = [u.get("unit_id") for u in _as_list(round_data.get("render_units")) if isinstance(u, dict)]
                if round_data.get("integration_status") in {"llmc_accepted", "llmb_passthrough"} and render_ids != b_unit_ids:
                    errors.append(f"{round_data['round_id']} render_units unit_ids must match llmb draft package units")
                if round_data.get("integration_status") == "llmb_passthrough":
                    b_texts = {u.get("unit_id"): u.get("draft_text") for u in _as_list(b_round.get("units")) if isinstance(u, dict)}
                    for unit in _as_list(round_data.get("render_units")):
                        if isinstance(unit, dict) and b_texts.get(unit.get("unit_id")) != unit.get("text"):
                            errors.append(f"{round_data['round_id']}.{unit.get('unit_id')} passthrough text must equal llmb draft_text")
    _raise_contract_errors("phase3c", errors)


def _validate_render_package_v2(payload: dict, draft_package_path: Path | None = None) -> None:
    """Validate the strict Phase4 input contract and its immutable B binding."""
    errors: list[str] = []
    if payload.get("producer") != "phase3c":
        errors.append(f"producer must be phase3c, got {payload.get('producer')!r}")
    package_status = payload.get("package_status")
    if package_status not in {"ready", "blocked"}:
        errors.append(f"package_status is invalid: {package_status!r}")
    source = payload.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
        source = {}
    for key in ("llmb_artifact_identity", "neutral_run_id", "neutral_sha256", "source_video_sha256"):
        if not isinstance(source.get(key), str):
            errors.append(f"source.{key} must be a string")
    content_policy = payload.get("content_policy")
    if not isinstance(content_policy, dict):
        errors.append("content_policy must be an object")
    else:
        if content_policy.get("phase3c_mode") not in {"off", "shadow", "optional", "required"}:
            errors.append("content_policy.phase3c_mode is invalid")
        if content_policy.get("fact_check_scope") not in {"disabled", "strong"}:
            errors.append("content_policy.fact_check_scope is invalid")
        if content_policy.get("allow_llmb_passthrough") not in {True, False}:
            errors.append("content_policy.allow_llmb_passthrough must be boolean")
    tts_policy = payload.get("tts_policy")
    if not isinstance(tts_policy, dict):
        errors.append("tts_policy must be an object")
    else:
        if tts_policy.get("profile_status") not in {"validated", "missing", "stale", "mismatch", "not_required"}:
            errors.append("tts_policy.profile_status is invalid")
        if not isinstance(tts_policy.get("speech_profile_id"), str):
            errors.append("tts_policy.speech_profile_id must be a string")
        if tts_policy.get("require_validated_profile") not in {True, False}:
            errors.append("tts_policy.require_validated_profile must be boolean")
        max_speed = tts_policy.get("max_speed_factor")
        if not isinstance(max_speed, (int, float)) or isinstance(max_speed, bool) or not 1.0 <= float(max_speed) <= 1.5:
            errors.append("tts_policy.max_speed_factor must be within [1.0, 1.5]")
    timeline = payload.get("timeline")
    if not isinstance(timeline, dict):
        errors.append("timeline must be an object")
        timeline = {}
    for key in ("timeline_id", "source_video_sha256"):
        if not isinstance(timeline.get(key), str):
            errors.append(f"timeline.{key} must be a string")
    origin = timeline.get("timeline_origin_sec")
    fps = timeline.get("render_tick_rate")
    if isinstance(origin, bool) or not isinstance(origin, (int, float)) or not math.isfinite(float(origin)):
        errors.append("timeline.timeline_origin_sec must be finite")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
        errors.append("timeline.render_tick_rate must be positive")
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        errors.append("rounds must be a non-empty list")
        rounds = []
    blocked_rounds = 0
    for round_index, round_data in enumerate(rounds):
        label = f"rounds[{round_index}]"
        if not isinstance(round_data, dict):
            errors.append(f"{label} must be an object")
            continue
        round_id = round_data.get("round_id")
        round_no = round_data.get("round_no")
        if not isinstance(round_id, str) or not round_id:
            errors.append(f"{label}.round_id is required")
        if not isinstance(round_no, int) or isinstance(round_no, bool) or round_no <= 0:
            errors.append(f"{label}.round_no must be a positive integer")
        elif isinstance(round_id, str) and round_id != f"r{round_no:03d}":
            errors.append(f"{label}.round_id must match round_no")
        integration_status = round_data.get("integration_status")
        if integration_status not in {"llmc_accepted", "llmb_passthrough", "blocked", "skipped"}:
            errors.append(f"{label}.integration_status is invalid")
        if integration_status == "blocked":
            blocked_rounds += 1
        units = round_data.get("render_units")
        if not isinstance(units, list):
            errors.append(f"{label}.render_units must be a list")
            continue
        if integration_status in {"blocked", "skipped"} and units:
            errors.append(f"{label}.render_units must be empty for {integration_status}")
        unit_ids: set[str] = set()
        for unit_index, unit in enumerate(units):
            unit_label = f"{label}.render_units[{unit_index}]"
            if not isinstance(unit, dict):
                errors.append(f"{unit_label} must be an object")
                continue
            unit_id = unit.get("unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                errors.append(f"{unit_label}.unit_id is required")
            elif unit_id in unit_ids:
                errors.append(f"{unit_label}.unit_id is duplicated")
            else:
                unit_ids.add(unit_id)
            if unit.get("sequence") != unit_index + 1:
                errors.append(f"{unit_label}.sequence must be {unit_index + 1}")
            if not isinstance(unit.get("final_text"), str) or not unit["final_text"].strip():
                errors.append(f"{unit_label}.final_text must be non-empty")
            if not isinstance(unit.get("emotion"), str) or not unit["emotion"]:
                errors.append(f"{unit_label}.emotion is required")
            expected_source = "llmc" if integration_status == "llmc_accepted" else "llmb_passthrough"
            if unit.get("text_source") != expected_source:
                errors.append(f"{unit_label}.text_source must be {expected_source}")
            slot = unit.get("render_slot")
            if not isinstance(slot, dict):
                errors.append(f"{unit_label}.render_slot must be an object")
            else:
                if not isinstance(slot.get("slot_id"), str) or slot.get("slot_id") != unit_id:
                    errors.append(f"{unit_label}.render_slot.slot_id must match unit_id")
                if not isinstance(slot.get("timeline_id"), str) or slot.get("timeline_id") != timeline.get("timeline_id"):
                    errors.append(f"{unit_label}.render_slot.timeline_id must match timeline.timeline_id")
                seconds = (slot.get("start_sec"), slot.get("end_sec"))
                ticks = (slot.get("start_tick"), slot.get("end_tick"))
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in seconds):
                    errors.append(f"{unit_label}.render_slot seconds must be finite numbers")
                elif seconds[0] >= seconds[1]:
                    errors.append(f"{unit_label}.render_slot must have start_sec < end_sec")
                if any(isinstance(value, bool) or not isinstance(value, int) for value in ticks):
                    errors.append(f"{unit_label}.render_slot ticks must be integers")
                elif ticks[0] >= ticks[1]:
                    errors.append(f"{unit_label}.render_slot must have start_tick < end_tick")
                if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in seconds) and all(isinstance(value, int) and not isinstance(value, bool) for value in ticks) and isinstance(fps, (int, float)):
                    if abs(float(ticks[0]) - float(seconds[0]) * float(fps)) > 2.0 or abs(float(ticks[1]) - float(seconds[1]) * float(fps)) > 2.0:
                        errors.append(f"{unit_label}.render_slot seconds and ticks do not agree")
            speed = unit.get("required_speed_factor")
            max_unit_speed = unit.get("max_speed_factor")
            if not isinstance(speed, (int, float)) or isinstance(speed, bool) or not 1.0 <= float(speed) <= 1.5:
                errors.append(f"{unit_label}.required_speed_factor must be within [1.0, 1.5]")
            if not isinstance(max_unit_speed, (int, float)) or isinstance(max_unit_speed, bool) or not 1.0 <= float(max_unit_speed) <= 1.5:
                errors.append(f"{unit_label}.max_speed_factor must be within [1.0, 1.5]")
            if isinstance(speed, (int, float)) and isinstance(max_unit_speed, (int, float)) and float(speed) > float(max_unit_speed) + 1e-9:
                errors.append(f"{unit_label}.required_speed_factor must not exceed max_speed_factor")
            if not isinstance(unit.get("required_fact_ids"), list) or any(not isinstance(item, str) for item in unit.get("required_fact_ids", [])):
                errors.append(f"{unit_label}.required_fact_ids must be a string list")
    if (package_status == "blocked") != (blocked_rounds > 0):
        errors.append("package_status must be blocked iff a round is blocked")
    body = {key: value for key, value in payload.items() if key != "artifact_identity"}
    expected_identity = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if payload.get("artifact_identity") != expected_identity:
        errors.append("artifact_identity mismatch")

    if draft_package_path is not None:
        package = _read_manifest(draft_package_path, "phase3c")
        if package.get("contract") != "llmb_draft_package_v2":
            errors.append("commentary_render_package_v2 requires llmb_draft_package_v2")
        else:
            if source.get("llmb_artifact_identity") != package.get("artifact_identity"):
                errors.append("source.llmb_artifact_identity does not match B v2 artifact_identity")
            if timeline.get("timeline_id") != (package.get("source") or {}).get("timeline_id"):
                errors.append("timeline.timeline_id does not match B v2 source.timeline_id")
            b_rounds = package.get("rounds") if isinstance(package.get("rounds"), list) else []
            if [item.get("round_id") for item in rounds if isinstance(item, dict)] != [item.get("round_id") for item in b_rounds if isinstance(item, dict)]:
                errors.append("C v2 rounds must match B v2 round order")
            b_by_round = {item.get("round_id"): item for item in b_rounds if isinstance(item, dict)}
            for round_data in rounds:
                if not isinstance(round_data, dict):
                    continue
                b_round = b_by_round.get(round_data.get("round_id"))
                if not isinstance(b_round, dict):
                    continue
                b_units = [item for item in b_round.get("units", []) if isinstance(item, dict)]
                c_units = [item for item in round_data.get("render_units", []) if isinstance(item, dict)]
                if round_data.get("integration_status") in {"llmc_accepted", "llmb_passthrough"}:
                    if [item.get("unit_id") for item in c_units] != [item.get("unit_id") for item in b_units]:
                        errors.append(f"{round_data.get('round_id')} render_units must match B v2 unit order")
                    b_by_id = {item.get("unit_id"): item for item in b_units}
                    for unit in c_units:
                        b_unit = b_by_id.get(unit.get("unit_id"))
                        if not isinstance(b_unit, dict):
                            continue
                        if unit.get("render_slot") != b_unit.get("render_slot"):
                            errors.append(f"{round_data.get('round_id')}.{unit.get('unit_id')} render_slot must match B v2")
                        if unit.get("required_fact_ids") != b_unit.get("allowed_fact_ids"):
                            errors.append(f"{round_data.get('round_id')}.{unit.get('unit_id')} required_fact_ids must match B v2")
                        if round_data.get("integration_status") == "llmb_passthrough" and unit.get("final_text") != b_unit.get("draft_text"):
                            errors.append(f"{round_data.get('round_id')}.{unit.get('unit_id')} passthrough final_text must equal B draft_text")
    _raise_contract_errors("phase3c", errors)


def require_outputs(stage: str, paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise PublishContractError(stage, f"missing stage outputs: {missing}")
