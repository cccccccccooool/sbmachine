"""Stage contract validators for JSON-in/JSON-out pipeline artifacts."""
from __future__ import annotations

from sbmachine.neutral_contract import SCHEMA_VERSION, SUPPORTED_PHASE3A_MODES
from sbmachine.preflight import validate_demo_artifacts, validate_final_manifest, validate_vision_timeline

import argparse
import json
from pathlib import Path
from typing import Any, Callable

Validator = Callable[[Any], list[str]]

EMOTIONS = {"平述", "激动", "惊叹"}


def load_json(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_neutral(data: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["phase3a artifact must be an object"]
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append("phase3a artifact has unsupported schema_version")
    if data.get("phase3a_mode") not in SUPPORTED_PHASE3A_MODES:
        errors.append("phase3a artifact has unsupported mode")
    if not isinstance(data.get("run_id"), str) or not data["run_id"].strip():
        errors.append("phase3a artifact is missing run_id")
    if not isinstance(data.get("source_rounds_sha256"), str) or not data["source_rounds_sha256"].strip():
        errors.append("phase3a artifact is missing source_rounds_sha256")
    rounds = _rounds(data)
    if not rounds:
        errors.append("phase3a artifact must contain rounds")
        return errors
    for ridx, round_data in enumerate(rounds):
        scenes = _as_list(round_data.get("scenes") if isinstance(round_data, dict) else None)
        # Empty scenes represent a deliberate silent round and are consumed by Phase3b.
        if not scenes:
            continue
        last_end = None
        for sidx, scene in enumerate(scenes):
            label = f"phase3a.rounds[{ridx}].scenes[{sidx}]"
            _require(scene, ["t_start", "t_end", "neutral", "commentary_plan"], label, errors)
            if not isinstance(scene, dict):
                continue
            start = _float_or_none(scene.get("t_start"))
            end = _float_or_none(scene.get("t_end"))
            if start is None or end is None or start >= end:
                errors.append(f"{label} must have t_start < t_end")
            if last_end is not None and start is not None and start < last_end:
                errors.append(f"{label} overlaps previous scene")
            if end is not None:
                last_end = end
    return errors


def validate_commentary(data: Any) -> list[str]:
    errors: list[str] = []
    items = _commentary_items(data)
    if not items:
        return ["phase3b artifact must contain commentary rounds"]
    for idx, item in enumerate(items):
        label = f"phase3b.rounds[{idx}]"
        _require(item, ["commentary_text", "emotion_segments"], label, errors)
        if not isinstance(item, dict):
            continue
        text = str(item.get("commentary_text", "")).strip()
        segments = _as_list(item.get("emotion_segments"))
        if text and not segments:
            errors.append(f"{label}.emotion_segments must be non-empty when commentary_text is present")
        if not text and segments:
            errors.append(f"{label}.emotion_segments must be empty when commentary_text is empty")
        for sidx, segment in enumerate(segments):
            slabel = f"{label}.emotion_segments[{sidx}]"
            _require(segment, ["emotion", "text"], slabel, errors)
            if isinstance(segment, dict) and segment.get("emotion") not in EMOTIONS:
                errors.append(f"{slabel}.emotion has unsupported value")
    return errors


STAGE_VALIDATORS: dict[str, Validator] = {
    "demo": validate_demo_artifacts,
    "phase2": validate_vision_timeline,
    "phase3a": validate_neutral,
    "phase3b": validate_commentary,
    "phase4": validate_final_manifest,
}

DEFAULT_STAGE_FILES = {
    "phase2": "rounds_with_yolo.json",
    "phase3a": "rounds_with_neutral.json",
    "phase3b": "commentary.json",
    "phase4": "rounds_final.json",
}


def validate_stage(stage: str, payload: Any) -> list[str]:
    return STAGE_VALIDATORS[stage](payload)


def validate_output_dir(output_dir: Path) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {}
    demo_dir = output_dir / "demo"
    if demo_dir.exists():
        report["demo"] = validate_demo_artifacts(_load_demo_dir(demo_dir))
    for stage, filename in DEFAULT_STAGE_FILES.items():
        path = output_dir / filename
        if path.exists():
            report[stage] = validate_stage(stage, load_json(path))
        else:
            report[stage] = [f"missing {filename}"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate sbmachine stage JSON contracts.")
    parser.add_argument("stage", choices=[*STAGE_VALIDATORS.keys(), "all"])
    parser.add_argument("--file", type=Path, help="single artifact JSON/JSONL file")
    parser.add_argument("--output-dir", type=Path, help="pipeline output directory")
    args = parser.parse_args(argv)

    if args.stage == "all":
        if not args.output_dir:
            parser.error("all requires --output-dir")
        report = validate_output_dir(args.output_dir)
    else:
        if args.file:
            report = {args.stage: validate_stage(args.stage, load_json(args.file))}
        elif args.output_dir and args.stage == "demo":
            report = {"demo": validate_demo_artifacts(_load_demo_dir(args.output_dir))}
        else:
            parser.error("stage validation requires --file")

    has_errors = False
    for stage, errors in report.items():
        if errors:
            has_errors = True
            print(f"[FAIL] {stage}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK] {stage}")
    return 1 if has_errors else 0


def _load_demo_dir(path: Path) -> dict[str, Any]:
    return {
        "rounds": load_json(path / "rounds.json") if (path / "rounds.json").exists() else [],
        "kills": load_json(path / "kills.json") if (path / "kills.json").exists() else [],
        "grenades": load_json(path / "grenades.json") if (path / "grenades.json").exists() else [],
        "roster": load_json(path / "roster.json") if (path / "roster.json").exists() else [],
    }


def _as_bundle(data: Any) -> dict[str, Any]:
    return data if isinstance(data, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _rounds(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        return _as_list(data.get("rounds"))
    return _as_list(data)


def _timeline_rows(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    rows: list[Any] = []
    for round_data in _rounds(data):
        phase2 = (
            round_data.get("_phase2_yolo")
            or round_data.get("phase2_yolo")
            or {}
        )
        rows.extend(_as_list(phase2.get("timeline")))
        rows.extend(_as_list(phase2.get("key_frames")))
    return rows


def _commentary_items(data: Any) -> list[Any]:
    if isinstance(data, dict) and isinstance(data.get("rounds"), list):
        return data["rounds"]
    return []


def _require(item: Any, keys: list[str], label: str, errors: list[str]) -> None:
    if not isinstance(item, dict):
        errors.append(f"{label} must be an object")
        return
    for key in keys:
        if key not in item:
            errors.append(f"{label} missing {key}")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_time(row: Any) -> float | None:
    if not isinstance(row, dict):
        return None
    when = row.get("when") if isinstance(row.get("when"), dict) else {}
    return _float_or_none(row.get("video_time", row.get("time_sec", when.get("video_time"))))


if __name__ == "__main__":
    raise SystemExit(main())

