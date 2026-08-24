"""Phase 3a 中性稿产物的严格契约校验（fail-closed，任何不符即抛错）。"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
PHASE3A_MODE = "llma_slicer_then_llma_analyze"
CLOUD_PHASE3A_MODE = "cloud_round_timeline"
SUPPORTED_PHASE3A_MODES = {PHASE3A_MODE, CLOUD_PHASE3A_MODE}

# v4 契约（§7.3/§7.4）
SCHEMA_VERSION_V4 = 4
PHASE3A_MODE_V4 = "rule_neutral_renderer"
FACT_ID_RE = re.compile(r"^fact:v1:[^:]+:[^:]+:\d{5}:[0-9a-f]{8}$")
_TICK_PER_SEC = 30


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
            missing = [key for key in ("t_start", "t_end", "scene", "commentary_plan", "neutral", "fact_anchors") if key not in scene]
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
            fact_anchors = scene["fact_anchors"]
            if not isinstance(fact_anchors, dict):
                raise ValueError(f"{scene_label}.fact_anchors must be an object")
            for key in ("players", "teams", "numbers", "events", "results", "locations", "weapons"):
                values = fact_anchors.get(key)
                if not isinstance(values, list):
                    raise ValueError(f"{scene_label}.fact_anchors.{key} must be a list")
                if key == "numbers":
                    if any(
                        not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        for value in values
                    ):
                        raise ValueError(f"{scene_label}.fact_anchors.numbers must contain numbers")
                elif any(not isinstance(value, str) or not value for value in values):
                    raise ValueError(f"{scene_label}.fact_anchors.{key} must contain non-empty strings")
            if any(side not in {"T", "CT"} for side in fact_anchors["teams"]):
                raise ValueError(f"{scene_label}.fact_anchors.teams contains an unsupported side")
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


def validate_neutral_v4(payload: Any) -> dict:
    """schema v4 校验：mode、render_slot、fact catalog、原子 ID 与 capsule 完整性。

    非静默 scene 必须包含唯一 neutral、rule_capsule、fact_catalog、required_fact_ids、
    render_slot、speech_budget 和 neutral_renderer；任一不符抛 ValueError（fail-closed）。
    """
    if not isinstance(payload, dict):
        raise ValueError("neutral v4 artifact must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION_V4:
        raise ValueError(f"neutral v4 artifact schema_version must be {SCHEMA_VERSION_V4}")
    if payload.get("phase3a_mode") != PHASE3A_MODE_V4:
        raise ValueError("neutral v4 artifact phase3a_mode must be rule_neutral_renderer")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
        raise ValueError("neutral v4 artifact is missing run_id")
    if not isinstance(payload.get("source_rounds_sha256"), str) or not payload["source_rounds_sha256"].strip():
        raise ValueError("neutral v4 artifact is missing source_rounds_sha256")
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("neutral v4 artifact is missing rounds")

    for round_index, round_data in enumerate(rounds):
        round_label = f"neutral v4 rounds[{round_index}]"
        if not isinstance(round_data, dict):
            raise ValueError(f"{round_label} must be an object")
        scenes = round_data.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError(f"{round_label}.scenes must be a list")
        for scene_index, scene in enumerate(scenes):
            scene_label = f"{round_label}.scenes[{scene_index}]"
            if not isinstance(scene, dict):
                raise ValueError(f"{scene_label} must be an object")
            missing = [
                key for key in
                ("t_start", "t_end", "window_id", "neutral", "neutral_source", "fact_anchors", "char_budget")
                if key not in scene
            ]
            if missing:
                raise ValueError(f"{scene_label} is missing fields: {', '.join(missing)}")
            neutral = scene["neutral"]
            if not isinstance(neutral, str):
                raise ValueError(f"{scene_label}.neutral must be a string")
            is_silence = str(scene.get("neutral_source") or "") == "intentional_empty"
            if is_silence:
                continue
            required_v4 = [
                "neutral_renderer", "rule_capsule", "fact_catalog",
                "required_fact_ids", "render_slot", "speech_budget",
            ]
            missing_v4 = [key for key in required_v4 if key not in scene]
            if missing_v4:
                raise ValueError(f"{scene_label} is missing v4 fields: {', '.join(missing_v4)}")
            if not neutral.strip():
                raise ValueError(f"{scene_label} non-silence neutral must be non-empty")

            renderer = scene["neutral_renderer"]
            if not isinstance(renderer, dict) or renderer.get("selected") not in {"rule_template", "tiny_assembler"}:
                raise ValueError(f"{scene_label}.neutral_renderer must select rule_template or tiny_assembler")
            capsule = scene["rule_capsule"]
            if not isinstance(capsule, str) or not capsule.strip():
                raise ValueError(f"{scene_label}.rule_capsule must be a non-empty string")

            catalog = scene["fact_catalog"]
            if not isinstance(catalog, list) or not catalog:
                raise ValueError(f"{scene_label}.fact_catalog must be a non-empty list")
            catalog_ids: set[str] = set()
            for fact_index, fact in enumerate(catalog):
                fact_label = f"{scene_label}.fact_catalog[{fact_index}]"
                if not isinstance(fact, dict):
                    raise ValueError(f"{fact_label} must be an object")
                fact_id = fact.get("fact_id")
                if not isinstance(fact_id, str) or not FACT_ID_RE.match(fact_id):
                    raise ValueError(f"{fact_label}.fact_id must match {FACT_ID_RE.pattern!r}")
                if fact_id in catalog_ids:
                    raise ValueError(f"{fact_label} fact_id is duplicated: {fact_id}")
                catalog_ids.add(fact_id)
                for key in ("kind", "origin", "canonical_clause"):
                    if not isinstance(fact.get(key), str) or not fact[key]:
                        raise ValueError(f"{fact_label}.{key} must be a non-empty string")
                if fact.get("origin") not in {"event", "derived"}:
                    raise ValueError(f"{fact_label}.origin must be event or derived")
                anchor_tick = fact.get("anchor_tick")
                if not isinstance(anchor_tick, int) or isinstance(anchor_tick, bool) or anchor_tick < 0:
                    raise ValueError(f"{fact_label}.anchor_tick must be a non-negative integer")
                source_range = fact.get("source_tick_range")
                if (
                    not isinstance(source_range, list) or len(source_range) != 2
                    or not all(isinstance(tick, int) and not isinstance(tick, bool) for tick in source_range)
                    or source_range[0] > source_range[1]
                ):
                    raise ValueError(f"{fact_label}.source_tick_range must be a [start, end] tick range")

            required_ids = scene["required_fact_ids"]
            if not isinstance(required_ids, list) or not required_ids:
                raise ValueError(f"{scene_label}.required_fact_ids must be a non-empty list")
            if not all(isinstance(fid, str) and FACT_ID_RE.match(fid) for fid in required_ids):
                raise ValueError(f"{scene_label}.required_fact_ids entries must match {FACT_ID_RE.pattern!r}")
            if not set(required_ids).issubset(catalog_ids):
                raise ValueError(f"{scene_label}.required_fact_ids must be a subset of fact_catalog ids")

            slot = scene["render_slot"]
            if not isinstance(slot, dict):
                raise ValueError(f"{scene_label}.render_slot must be an object")
            start_tick, end_tick = slot.get("start_tick"), slot.get("end_tick")
            if (
                not isinstance(start_tick, int) or isinstance(start_tick, bool)
                or not isinstance(end_tick, int) or isinstance(end_tick, bool)
                or start_tick >= end_tick
            ):
                raise ValueError(f"{scene_label}.render_slot must have start_tick < end_tick")
            expected_start = int(round(float(scene["t_start"]) * _TICK_PER_SEC))
            expected_end = int(round(float(scene["t_end"]) * _TICK_PER_SEC))
            if abs(start_tick - expected_start) > 2 or abs(end_tick - expected_end) > 2:
                raise ValueError(f"{scene_label}.render_slot ticks are inconsistent with t_start/t_end")
            if slot.get("gap_policy") != "independent_window":
                raise ValueError(f"{scene_label}.render_slot.gap_policy must be independent_window")

            budget = scene["speech_budget"]
            if not isinstance(budget, dict):
                raise ValueError(f"{scene_label}.speech_budget must be an object")
            target_units, hard_units = budget.get("target_units"), budget.get("hard_units")
            if (
                not isinstance(target_units, int) or isinstance(target_units, bool)
                or not isinstance(hard_units, int) or isinstance(hard_units, bool)
                or target_units < 0 or hard_units < target_units
            ):
                raise ValueError(f"{scene_label}.speech_budget must have 0 <= target_units <= hard_units")
            if not isinstance(budget.get("profile_id"), str) or not budget["profile_id"].strip():
                raise ValueError(f"{scene_label}.speech_budget.profile_id must be non-empty")
    return payload
