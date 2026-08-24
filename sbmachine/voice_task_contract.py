"""单向配音任务单（voice task）产物的纯结构契约校验。

本模块只做结构/取值校验，返回 errors 列表（空列表=通过）；
只依赖标准库，禁止 import 任何 sbmachine 模块，避免循环依赖。
"""
from __future__ import annotations

import math
import re
from typing import Any

SCHEMA_NEUTRAL_V4 = 4
SCHEMA_COMMENTARY_V3 = 3
VOICE_TASK_CONTRACT_VERSION = 1
CANDIDATE_POLICY_SPARSE_V1 = "sparse_v1"
SPEECH_METRIC_UNITS_V1 = "speech_units_v1"
MAX_CANDIDATES_PER_SCENE = 3

PHASE3A_MODE_RULE_NEUTRAL_RENDERER = "rule_neutral_renderer"

FACT_ID_RE = re.compile(r"^fact:v1:[\w:.-]+:\d{5}:[0-9a-f]{8}$")
RISK_CLASSES = {"green", "amber", "red"}
VARIANT_IDS = {"primary", "compact", "capsule"}
CANDIDATE_SOURCES = {"llmb", "llmb_compact", "rule_capsule"}
FIT_STATES = {"fit", "render_unfit"}
ANCHOR_KEYS = ("players", "teams", "numbers", "events", "results", "locations", "weapons")

_TICK_FPS = 30.0
_TICK_TOLERANCE = 2


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _slot_tick_mismatch(sec_value: Any, tick_value: Any) -> bool:
    if not _is_number(sec_value) or not _is_int(tick_value):
        return True
    return abs(float(tick_value) - float(sec_value) * _TICK_FPS) > _TICK_TOLERANCE


def _validate_render_slot(slot: Any, label: str, errors: list[str]) -> None:
    if not isinstance(slot, dict):
        errors.append(f"{label}.render_slot must be an object")
        return
    for key in ("start_tick", "end_tick"):
        if not _is_int(slot.get(key)):
            errors.append(f"{label}.render_slot.{key} must be an integer")
    start_tick, end_tick = slot.get("start_tick"), slot.get("end_tick")
    if _is_int(start_tick) and _is_int(end_tick) and start_tick >= end_tick:
        errors.append(f"{label}.render_slot must have start_tick < end_tick")
    if _slot_tick_mismatch(slot.get("start_sec"), slot.get("start_tick")):
        errors.append(f"{label}.render_slot.start_tick must equal start_sec * 30 within ±{_TICK_TOLERANCE}")
    if _slot_tick_mismatch(slot.get("end_sec"), slot.get("end_tick")):
        errors.append(f"{label}.render_slot.end_tick must equal end_sec * 30 within ±{_TICK_TOLERANCE}")


def _validate_fact_catalog(catalog: Any, required_ids: Any, label: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(catalog, list) or not catalog:
        errors.append(f"{label}.fact_catalog must be a non-empty list")
        return by_id
    for index, fact in enumerate(catalog):
        fact_label = f"{label}.fact_catalog[{index}]"
        if not isinstance(fact, dict):
            errors.append(f"{fact_label} must be an object")
            continue
        fact_id = fact.get("fact_id")
        if not isinstance(fact_id, str) or not FACT_ID_RE.match(fact_id):
            errors.append(f"{fact_label}.fact_id has an invalid format")
        elif fact_id in by_id:
            errors.append(f"{fact_label}.fact_id is duplicated")
        if isinstance(fact_id, str):
            by_id[fact_id] = fact
        if not _is_nonempty_str(fact.get("kind")):
            errors.append(f"{fact_label}.kind must be a non-empty string")
        if not _is_nonempty_str(fact.get("canonical_clause")):
            errors.append(f"{fact_label}.canonical_clause must be a non-empty string")
        origin = fact.get("origin")
        if origin not in {"event", "derived"}:
            errors.append(f"{fact_label}.origin must be 'event' or 'derived'")
        if not _is_int(fact.get("anchor_tick")):
            errors.append(f"{fact_label}.anchor_tick must be an integer")
        if fact.get("required") is not True and fact.get("required") is not False:
            errors.append(f"{fact_label}.required must be a boolean")
        if not _is_number(fact.get("priority")):
            errors.append(f"{fact_label}.priority must be a number")
        tick_range = fact.get("source_tick_range")
        if (
            not isinstance(tick_range, list)
            or len(tick_range) != 2
            or not _is_int(tick_range[0])
            or not _is_int(tick_range[1])
            or tick_range[0] > tick_range[1]
        ):
            errors.append(f"{fact_label}.source_tick_range must be [start, end] with start <= end")
    if required_ids is None:
        errors.append(f"{label}.required_fact_ids must be a list")
    elif isinstance(required_ids, list):
        for index, fact_id in enumerate(required_ids):
            fact_label = f"{label}.required_fact_ids[{index}]"
            if not isinstance(fact_id, str):
                errors.append(f"{fact_label} must be a string")
                continue
            fact = by_id.get(fact_id)
            if fact is None:
                errors.append(f"{fact_label} references a fact_id missing from fact_catalog")
            elif fact.get("required") is not True:
                errors.append(f"{fact_label} references a fact whose required flag is not true")
    return by_id


def validate_neutral_v4(manifest: Any) -> list[str]:
    """校验 Phase3a schema v4（rounds_with_neutral）结构契约。"""
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["neutral v4 artifact must be an object"]
    if manifest.get("schema_version") != SCHEMA_NEUTRAL_V4:
        errors.append("neutral v4 artifact schema_version must be 4")
    if manifest.get("phase3a_mode") != PHASE3A_MODE_RULE_NEUTRAL_RENDERER:
        errors.append("neutral v4 artifact phase3a_mode must be rule_neutral_renderer")
    if not _is_nonempty_str(manifest.get("run_id")):
        errors.append("neutral v4 artifact is missing run_id")
    if not _is_nonempty_str(manifest.get("source_rounds_sha256")):
        errors.append("neutral v4 artifact is missing source_rounds_sha256")
    speech_metric = manifest.get("speech_metric_version")
    if speech_metric is not None and not _is_nonempty_str(speech_metric):
        errors.append("neutral v4 artifact speech_metric_version must be a non-empty string")
    rounds = manifest.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        errors.append("neutral v4 artifact must contain rounds")
        return errors
    for ridx, round_data in enumerate(rounds):
        if not isinstance(round_data, dict):
            errors.append(f"neutral v4 rounds[{ridx}] must be an object")
            continue
        scenes = round_data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            errors.append(f"neutral v4 rounds[{ridx}].scenes must be a non-empty list")
            continue
        for sidx, scene in enumerate(scenes):
            label = f"neutral v4 rounds[{ridx}].scenes[{sidx}]"
            if not isinstance(scene, dict):
                errors.append(f"{label} must be an object")
                continue
            if not _is_nonempty_str(scene.get("window_id")):
                errors.append(f"{label}.window_id must be a non-empty string")
            start, end = scene.get("t_start"), scene.get("t_end")
            if not _is_number(start) or not _is_number(end) or float(start) >= float(end):
                errors.append(f"{label} must have finite t_start < t_end")
            neutral = scene.get("neutral")
            if not isinstance(neutral, str) or not neutral.strip():
                continue
            if not _is_nonempty_str(scene.get("neutral_source")):
                errors.append(f"{label}.neutral_source must be a non-empty string")
            renderer = scene.get("neutral_renderer")
            if not isinstance(renderer, dict):
                errors.append(f"{label}.neutral_renderer must be an object")
            else:
                if not _is_nonempty_str(renderer.get("selected")):
                    errors.append(f"{label}.neutral_renderer.selected must be a non-empty string")
                if renderer.get("policy") is not None and not _is_nonempty_str(renderer.get("policy")):
                    errors.append(f"{label}.neutral_renderer.policy must be a non-empty string")
            if not _is_nonempty_str(scene.get("rule_capsule")):
                errors.append(f"{label}.rule_capsule must be a non-empty string")
            _validate_fact_catalog(
                scene.get("fact_catalog"), scene.get("required_fact_ids"), label, errors
            )
            anchors = scene.get("fact_anchors")
            if not isinstance(anchors, dict):
                errors.append(f"{label}.fact_anchors must be an object")
            else:
                for key in ANCHOR_KEYS:
                    if not isinstance(anchors.get(key), list):
                        errors.append(f"{label}.fact_anchors.{key} must be a list")
            _validate_render_slot(scene.get("render_slot"), label, errors)
            budget = scene.get("speech_budget")
            if not isinstance(budget, dict):
                errors.append(f"{label}.speech_budget must be an object")
            else:
                for key in ("target_units", "hard_units"):
                    if not _is_int(budget.get(key)):
                        errors.append(f"{label}.speech_budget.{key} must be an integer")
                if not _is_nonempty_str(budget.get("profile_id")):
                    errors.append(f"{label}.speech_budget.profile_id must be a non-empty string")
            char_budget = scene.get("char_budget")
            if "char_budget" in scene and not _is_int(char_budget):
                errors.append(f"{label}.char_budget must be an integer")
    return errors


def validate_commentary_v3(manifest: Any) -> list[str]:
    """校验 Phase3b commentary schema v3（voice_tasks 稀疏候选目录）结构契约。"""
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["commentary v3 artifact must be an object"]
    if manifest.get("commentary_schema_version") != SCHEMA_COMMENTARY_V3:
        errors.append("commentary v3 artifact commentary_schema_version must be 3")
    if manifest.get("voice_task_contract_version") != VOICE_TASK_CONTRACT_VERSION:
        errors.append("commentary v3 artifact voice_task_contract_version must be 1")
    if manifest.get("candidate_policy") != CANDIDATE_POLICY_SPARSE_V1:
        errors.append("commentary v3 artifact candidate_policy must be sparse_v1")
    if not _is_nonempty_str(manifest.get("speech_metric_version")):
        errors.append("commentary v3 artifact is missing speech_metric_version")
    if not _is_nonempty_str(manifest.get("source_neutral_run_id")):
        errors.append("commentary v3 artifact is missing source_neutral_run_id")
    if not _is_nonempty_str(manifest.get("source_neutral_sha256")):
        errors.append("commentary v3 artifact is missing source_neutral_sha256")
    voice_tasks = manifest.get("voice_tasks")
    if not isinstance(voice_tasks, list) or not voice_tasks:
        errors.append("commentary v3 artifact must contain voice_tasks")
        return errors
    for tidx, task in enumerate(voice_tasks):
        label = f"commentary v3 voice_tasks[{tidx}]"
        if not isinstance(task, dict):
            errors.append(f"{label} must be an object")
            continue
        if not _is_nonempty_str(task.get("voice_task_id")):
            errors.append(f"{label}.voice_task_id must be a non-empty string")
        if not _is_nonempty_str(task.get("window_id")):
            errors.append(f"{label}.window_id must be a non-empty string")
        _validate_render_slot(task.get("render_slot"), label, errors)
        required_ids = task.get("required_fact_ids")
        if not isinstance(required_ids, list) or not required_ids:
            errors.append(f"{label}.required_fact_ids must be a non-empty list")
        elif any(not isinstance(fact_id, str) or not FACT_ID_RE.match(fact_id) for fact_id in required_ids):
            errors.append(f"{label}.required_fact_ids contains an invalid fact_id")
        if not _is_nonempty_str(task.get("speech_profile_id")):
            errors.append(f"{label}.speech_profile_id must be a non-empty string")
        risk_class = task.get("risk_class")
        if risk_class not in RISK_CLASSES:
            errors.append(f"{label}.risk_class must be one of green/amber/red")
        selection_order = task.get("selection_order")
        if not isinstance(selection_order, list) or not selection_order:
            errors.append(f"{label}.selection_order must be a non-empty list")
        elif any(variant not in VARIANT_IDS for variant in selection_order):
            errors.append(f"{label}.selection_order contains an unsupported variant_id")
        max_speed_factor = task.get("max_speed_factor")
        if not _is_number(max_speed_factor) or float(max_speed_factor) < 1.0:
            errors.append(f"{label}.max_speed_factor must be a finite number >= 1.0")
        candidates = task.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"{label}.candidates must be a non-empty list")
        elif len(candidates) > MAX_CANDIDATES_PER_SCENE:
            errors.append(f"{label}.candidates must not exceed 3 per scene")
        seen_variants: set[str] = set()
        if isinstance(candidates, list):
            for cidx, candidate in enumerate(candidates):
                clabel = f"{label}.candidates[{cidx}]"
                if not isinstance(candidate, dict):
                    errors.append(f"{clabel} must be an object")
                    continue
                variant_id = candidate.get("variant_id")
                if variant_id not in VARIANT_IDS:
                    errors.append(f"{clabel}.variant_id must be one of primary/compact/capsule")
                elif variant_id in seen_variants:
                    errors.append(f"{clabel}.variant_id is duplicated")
                else:
                    seen_variants.add(variant_id)
                source = candidate.get("source")
                if source not in CANDIDATE_SOURCES:
                    errors.append(f"{clabel}.source must be one of llmb/llmb_compact/rule_capsule")
                if variant_id == "capsule" and source != "rule_capsule":
                    errors.append(f"{clabel}.capsule candidate must have source rule_capsule")
                if not _is_nonempty_str(candidate.get("text")):
                    errors.append(f"{clabel}.text must be a non-empty string")
                preserved = candidate.get("preserved_fact_ids")
                if not isinstance(preserved, list):
                    errors.append(f"{clabel}.preserved_fact_ids must be a list")
                elif isinstance(required_ids, list):
                    missing = [fact_id for fact_id in required_ids if fact_id not in preserved]
                    if missing:
                        errors.append(f"{clabel}.preserved_fact_ids must cover all required_fact_ids")
            if isinstance(selection_order, list) and seen_variants:
                uncovered = sorted(seen_variants - set(selection_order))
                if uncovered:
                    errors.append(f"{label}.candidates must be covered by selection_order")
            if risk_class == "green" and "compact" in seen_variants:
                errors.append(f"{label}.green risk_class must not contain compact candidates")
            if risk_class == "red" and "primary" in seen_variants:
                errors.append(f"{label}.red risk_class must not contain primary candidates")
        if task.get("semantic_state") != "ok":
            errors.append(f"{label}.semantic_state must be 'ok'")
    return errors


def _final_scenes(manifest: Any) -> list[tuple[str, dict[str, Any]]]:
    scenes: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(manifest, dict):
        return scenes
    rounds_final = manifest.get("rounds_final") or manifest.get("rounds") or manifest
    rounds = rounds_final.get("rounds") if isinstance(rounds_final, dict) else None
    if not isinstance(rounds, list):
        return scenes
    for ridx, round_data in enumerate(rounds):
        if not isinstance(round_data, dict):
            continue
        for sidx, scene in enumerate(round_data.get("scenes") or []):
            if isinstance(scene, dict) and _is_nonempty_str(scene.get("voice_task_id")):
                scenes.append((f"final rounds[{ridx}].scenes[{sidx}]", scene))
    return scenes


def _commentary_task_by_id(commentary: Any) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if isinstance(commentary, dict):
        for task in commentary.get("voice_tasks") or []:
            if isinstance(task, dict) and isinstance(task.get("voice_task_id"), str):
                by_id[task["voice_task_id"]] = task
    return by_id


def validate_final_voice_task(manifest: Any, commentary: Any = None) -> list[str]:
    """校验 Phase4 rounds_final 每 scene 的配音任务单执行结果。

    commentary 可选：提供 commentary v3 payload 时进一步校验
    selected_variant_id 属于任务单候选、audio tick 不越出任务单 render_slot。
    """
    errors: list[str] = []
    for label, scene in _final_scenes(manifest):
        missing = [
            key
            for key in (
                "selected_variant_id",
                "selected_text",
                "actual_duration_sec",
                "applied_speed_factor",
                "audio_start_tick",
                "audio_end_tick",
                "fit_state",
            )
            if key not in scene
        ]
        if missing:
            errors.append(f"{label} is missing fields: {', '.join(missing)}")
            continue
        fit_state = scene.get("fit_state")
        if fit_state not in FIT_STATES:
            errors.append(f"{label}.fit_state must be one of fit/render_unfit")
            continue
        audio_start, audio_end = scene.get("audio_start_tick"), scene.get("audio_end_tick")
        if fit_state == "fit":
            if not _is_nonempty_str(scene.get("selected_variant_id")):
                errors.append(f"{label}.selected_variant_id must be non-empty when fit_state is fit")
            if not _is_nonempty_str(scene.get("selected_text")):
                errors.append(f"{label}.selected_text must be non-empty when fit_state is fit")
            if not _is_number(scene.get("actual_duration_sec")) or float(scene["actual_duration_sec"]) < 0:
                errors.append(f"{label}.actual_duration_sec must be a finite number >= 0 when fit_state is fit")
            if not _is_number(scene.get("applied_speed_factor")) or float(scene["applied_speed_factor"]) < 1.0:
                errors.append(f"{label}.applied_speed_factor must be a finite number >= 1.0 when fit_state is fit")
            if not _is_int(audio_start) or not _is_int(audio_end):
                errors.append(f"{label}.audio_start_tick/audio_end_tick must be integers when fit_state is fit")
        if _is_int(audio_start) and _is_int(audio_end) and audio_end < audio_start:
            errors.append(f"{label} must have audio_end_tick >= audio_start_tick")
        slot = scene.get("render_slot") if isinstance(scene.get("render_slot"), dict) else None
        voice_task_id = scene.get("voice_task_id")
        task = None
        if commentary is not None:
            tasks_by_id = _commentary_task_by_id(commentary)
            task = tasks_by_id.get(voice_task_id)
            if voice_task_id not in tasks_by_id:
                errors.append(f"{label}.voice_task_id is not found in the commentary voice_tasks")
        if task is not None and isinstance(task.get("render_slot"), dict):
            slot = task["render_slot"]
        if slot is not None:
            slot_start, slot_end = slot.get("start_tick"), slot.get("end_tick")
            if _is_int(audio_start) and _is_int(slot_start) and audio_start < slot_start:
                errors.append(f"{label}.audio_start_tick must not precede render_slot.start_tick")
            if _is_int(audio_end) and _is_int(slot_end) and audio_end > slot_end:
                errors.append(f"{label}.audio_end_tick must not exceed render_slot.end_tick")
        if task is not None and fit_state == "fit":
            variant_ids = [
                candidate.get("variant_id")
                for candidate in (task.get("candidates") or [])
                if isinstance(candidate, dict)
            ]
            if scene.get("selected_variant_id") not in variant_ids:
                errors.append(f"{label}.selected_variant_id is not among the voice task candidates")
    return errors
