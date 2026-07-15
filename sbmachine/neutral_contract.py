"""Phase 3a 中性稿产物的严格契约校验（fail-closed，任何不符即抛错）。"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
PHASE3A_MODE = "llma_slicer_then_llma_analyze"
CLOUD_PHASE3A_MODE = "cloud_round_timeline"
SUPPORTED_PHASE3A_MODES = {PHASE3A_MODE, CLOUD_PHASE3A_MODE}


def rounds_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_manifest_metadata(rounds_path: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase3a_mode": PHASE3A_MODE,
        "run_id": uuid.uuid4().hex,
        "source_rounds_sha256": rounds_sha256(rounds_path),
    }


def validate_neutral_manifest(payload: Any, rounds_path: Path) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("neutral artifact must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("neutral artifact schema_version is unsupported; rerun phase3a")
    if payload.get("phase3a_mode") not in SUPPORTED_PHASE3A_MODES:
        raise ValueError("neutral artifact was not produced by the current phase3a mode")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
        raise ValueError("neutral artifact is missing run_id")
    if payload.get("source_rounds_sha256") != rounds_sha256(rounds_path):
        raise ValueError("neutral artifact does not match the current rounds_with_yolo input")
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("neutral artifact is missing rounds")

    try:
        source_payload = json.loads(rounds_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("source rounds input is not valid JSON") from exc
    source_rounds = source_payload.get("rounds") if isinstance(source_payload, dict) else None
    if not isinstance(source_rounds, list):
        raise ValueError("source rounds input is missing rounds")
    source_by_round: dict[int, tuple[float, float]] = {}
    for source_index, source_round in enumerate(source_rounds):
        label = f"source rounds[{source_index}]"
        if not isinstance(source_round, dict):
            raise ValueError(f"{label} must be an object")
        round_no = source_round.get("round_no")
        if not isinstance(round_no, int) or isinstance(round_no, bool):
            raise ValueError(f"{label}.round_no must be an integer")
        if round_no in source_by_round:
            raise ValueError(f"source round_no {round_no} is duplicated")
        start, end = source_round.get("start_sec"), source_round.get("end_sec")
        if (
            isinstance(start, bool) or not isinstance(start, (int, float))
            or isinstance(end, bool) or not isinstance(end, (int, float))
            or not math.isfinite(float(start)) or not math.isfinite(float(end))
            or float(start) >= float(end)
        ):
            raise ValueError(f"{label} must have finite start_sec < end_sec")
        source_by_round[round_no] = (float(start), float(end))

    output_round_nos: set[int] = set()
    for round_index, round_data in enumerate(rounds):
        round_label = f"neutral rounds[{round_index}]"
        if not isinstance(round_data, dict):
            raise ValueError(f"{round_label} must be an object")
        round_no = round_data.get("round_no")
        if not isinstance(round_no, int) or isinstance(round_no, bool):
            raise ValueError(f"{round_label}.round_no must be an integer")
        if round_no in output_round_nos:
            raise ValueError(f"neutral round_no {round_no} is duplicated")
        output_round_nos.add(round_no)
    if output_round_nos != set(source_by_round):
        raise ValueError("neutral rounds must correspond one-to-one with source rounds")

    for round_index, round_data in enumerate(rounds):
        round_label = f"neutral rounds[{round_index}]"
        round_start, round_end = source_by_round[round_data["round_no"]]
        scenes = round_data.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError(f"{round_label}.scenes must be a list")
        previous_end: float | None = None
        for scene_index, scene in enumerate(scenes):
            scene_label = f"{round_label}.scenes[{scene_index}]"
            if not isinstance(scene, dict):
                raise ValueError(f"{scene_label} must be an object")
            missing = [key for key in ("t_start", "t_end", "scene", "commentary_plan", "neutral") if key not in scene]
            if missing:
                raise ValueError(f"{scene_label} is missing fields: {', '.join(missing)}")
            start, end = scene["t_start"], scene["t_end"]
            if (
                isinstance(start, bool) or not isinstance(start, (int, float))
                or isinstance(end, bool) or not isinstance(end, (int, float))
                or not math.isfinite(float(start)) or not math.isfinite(float(end))
                or float(start) >= float(end)
            ):
                raise ValueError(f"{scene_label} must have finite t_start < t_end")
            if previous_end is not None and float(start) < previous_end:
                raise ValueError(f"{scene_label} overlaps the previous scene")
            if float(start) < round_start or float(end) > round_end:
                raise ValueError(f"{scene_label} must stay within its source round range")
            if not isinstance(scene["scene"], str) or not scene["scene"].strip():
                raise ValueError(f"{scene_label}.scene must be a non-empty string")
            if not isinstance(scene["commentary_plan"], dict):
                raise ValueError(f"{scene_label}.commentary_plan must be an object")
            if not isinstance(scene["neutral"], str):
                raise ValueError(f"{scene_label}.neutral must be a string")
            hype = scene.get("hype")
            if "hype" in scene and (
                isinstance(hype, bool) or not isinstance(hype, (int, float)) or not math.isfinite(float(hype))
            ):
                raise ValueError(f"{scene_label}.hype must be a finite number")
            char_budget = scene.get("char_budget")
            if "char_budget" in scene and (
                not isinstance(char_budget, int) or isinstance(char_budget, bool)
            ):
                raise ValueError(f"{scene_label}.char_budget must be an integer")
            scream = scene.get("scream_eligible")
            if "scream_eligible" in scene and not isinstance(scream, bool):
                raise ValueError(f"{scene_label}.scream_eligible must be a boolean")
            previous_end = float(end)
    return payload
