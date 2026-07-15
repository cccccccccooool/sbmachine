"""只读的流水线配置与输入校验（不产生任何写入）。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


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
    vision = config.get("vision", {}) if isinstance(config.get("vision", {}), dict) else {}
    yolo = vision.get("yolo", {}) if isinstance(vision.get("yolo", {}), dict) else {}
    tts = config.get("tts", {}) if isinstance(config.get("tts", {}), dict) else {}
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
    if "phase4" in active:
        if "phase3b" not in active:
            add("rounds_with_commentary", paths.get("rounds_with_commentary_json", "output/sbmachine/rounds_with_commentary.json"))
            add("commentary", paths.get("commentary_json", "output/sbmachine/commentary.json"))
        add("tts_config", tts.get("config", "audio_service/gpt_sovits_runtime.yaml"))
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

    configured = enabled_phases(config)
    active = set(only) if only is not None else set(configured)
    ordered_names = ("demo_parse", "video_marking", "phase1", "phase2", "phase3a", "phase3b", "phase4")
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


def validate_final_manifest(data: Any) -> list[str]:
    errors: list[str] = []
    bundle = data if isinstance(data, dict) else {}
    rounds_final = bundle.get("rounds_final") or bundle.get("rounds") or bundle
    rounds = _rounds(rounds_final)
    if not rounds:
        errors.append("phase4 rounds_final must contain rounds")
    manifest = bundle.get("assemble_manifest") or bundle.get("manifest") or {}
    if not isinstance(manifest, dict):
        errors.append("phase4 assemble_manifest must be an object")
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


def validate_phase4_publishable(rounds_path: Path, manifest_path: Path) -> None:
    payload = {
        "rounds_final": _read_json(rounds_path, "phase4"),
        "assemble_manifest": _read_json(manifest_path, "phase4"),
    }
    _raise_contract_errors("phase4", validate_final_manifest(payload))


def validate_neutral_publishable(path: Path) -> None:
    from sbmachine.commentary_planner import fallback_neutral

    manifest = _read_manifest(path, "phase3a")
    rounds = manifest.get("rounds")
    if not isinstance(rounds, list):
        raise PublishContractError("phase3a", "neutral manifest rounds must be a list")
    for round_data in rounds:
        if not isinstance(round_data, dict) or round_data.get("analyst_failed"):
            raise PublishContractError("phase3a", "analyst failure is not publishable")
        scenes = round_data.get("scenes", [])
        if not isinstance(scenes, list):
            raise PublishContractError("phase3a", "neutral scenes must be a list")
        for scene in scenes:
            if not isinstance(scene, dict):
                raise PublishContractError("phase3a", "neutral scene must be an object")
            plan = scene.get("commentary_plan")
            neutral = str(scene.get("neutral") or "")
            if isinstance(plan, dict) and neutral and neutral == fallback_neutral(plan):
                raise PublishContractError("phase3a", "rule fallback neutral is not publishable")


def validate_commentary_publishable(path: Path) -> None:
    manifest = _read_manifest(path, "phase3b")
    rounds = manifest.get("rounds")
    if not isinstance(rounds, list):
        raise PublishContractError("phase3b", "commentary manifest rounds must be a list")
    rejected = []
    for round_data in rounds:
        status = round_data.get("status") if isinstance(round_data, dict) else None
        if status not in {"ok", "silent"}:
            rejected.append(status)
    if rejected:
        raise PublishContractError("phase3b", f"non-publishable commentary status: {rejected}")


def require_outputs(stage: str, paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise PublishContractError(stage, f"missing stage outputs: {missing}")
