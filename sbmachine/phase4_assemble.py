"""Phase 4：逐场景合成解说 TTS，并与每局音频/视频对齐混流。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from sbmachine import speech_measure
from sbmachine.common import load_config, read_json, require_path, resolve_path, write_json
from sbmachine.phase4_av import (
    _assemble_scene_wav,
    _mux_round_video,
    _run_ffmpeg,
    _wav_info,
    audio_end_tick,
    check_scene_slot_fit,
)
from sbmachine.schemas import AudioData, load_match, save_match
from sbmachine.voice_task_contract import validate_commentary_v3


_EMOTIONS = {"平述", "激动", "惊叹"}
_RENDER_UNFIT_REASON = "all candidates exceeded the fixed slot"

_SELECTION_RESULT_KEYS = (
    "voice_task_id",
    "window_id",
    "render_slot",
    "selected_variant_id",
    "selected_text",
    "actual_duration_sec",
    "applied_speed_factor",
    "audio_start_tick",
    "audio_end_tick",
    "fit_state",
    "attempted_variants",
    "render_unfit_reason",
)


@dataclass(frozen=True)
class _VoiceTaskPlan:
    """§10.2 只读执行对象：任务单 + 对应 scene 的绑定信息，构造后不可修改。"""

    round_no: int
    voice_task_id: str
    window_id: str
    scene: dict
    slot_start_tick: int
    slot_end_tick: int
    slot_start_sec: float
    slot_end_sec: float
    selection_order: tuple[str, ...]
    speech_profile_id: str
    candidates: tuple[dict, ...]
    max_speed_factor: float

    @property
    def slot_duration_sec(self) -> float:
        return self.slot_end_sec - self.slot_start_sec


@dataclass(frozen=True)
class _RenderUnitPlan:
    """Read-only execution unit sourced exclusively from C v2 final_text."""

    round_no: int
    unit_id: str
    sequence: int
    final_text: str
    emotion: str
    text_source: str
    slot_start_sec: float
    slot_end_sec: float
    slot_start_tick: int
    slot_end_tick: int
    required_speed_factor: float
    max_speed_factor: float
    speech_profile_id: str
    timeline_id: str
    render_package_artifact_identity: str

    @property
    def slot_duration_sec(self) -> float:
        return self.slot_end_sec - self.slot_start_sec

    @property
    def tts_text(self) -> str:
        # The emotion tag is a TTS routing instruction; final_text remains unmodified in the contract.
        return f"[{self.emotion}]{self.final_text}"


def _tts_cache_key(text: str, fingerprint: str) -> str:
    """由运行时指纹加文本算出 TTS 缓存文件名；指纹变则缓存自动失效。"""
    return hashlib.sha256(f"{fingerprint}\0{text}".encode("utf-8")).hexdigest()


def _strict_round_samples(value: float) -> int:
    """Use the centralized media-clock rounding policy for PCM endpoints."""
    try:
        from sbmachine.media_clock import round_half_even

        return int(round_half_even(value))
    except (ImportError, AttributeError):  # pragma: no cover - compatibility during partial installs
        return int(round(float(value)))


def _load_render_package_v2(render_package_path: Path, match, publish_profile: str) -> tuple[dict, dict[int, list[_RenderUnitPlan]]]:
    """Load and bind a C v2 package without consulting commentary candidates."""
    from sbmachine.preflight import validate_render_package

    payload = read_json(render_package_path)
    if not isinstance(payload, dict) or payload.get("contract") != "commentary_render_package_v2":
        raise ValueError("strict Phase4 requires commentary_render_package_v2")
    validate_render_package(render_package_path)
    if payload.get("package_status") == "blocked":
        raise ValueError("strict Phase4 cannot execute a blocked render package")
    timeline = payload.get("timeline")
    tts_policy = payload.get("tts_policy")
    content_policy = payload.get("content_policy")
    if not isinstance(timeline, dict) or not isinstance(tts_policy, dict) or not isinstance(content_policy, dict):
        raise ValueError("commentary_render_package_v2 is missing execution policy blocks")
    if publish_profile in {"strict_av", "strict_c"} and tts_policy.get("require_validated_profile") is True:
        if tts_policy.get("profile_status") != "validated":
            raise ValueError(
                f"strict Phase4 requires a validated speech profile, got {tts_policy.get('profile_status')!r}"
            )
    if publish_profile == "strict_c":
        if content_policy.get("phase3c_mode") != "required":
            raise ValueError("strict_c requires semantic.phase3c.mode=required")
        if content_policy.get("fact_check_scope") != "strong":
            raise ValueError("strict_c requires strong fact scope")
        if any(
            isinstance(round_data, dict)
            and any(unit.get("text_source") == "llmb_passthrough" for unit in round_data.get("render_units", []) if isinstance(unit, dict))
            for round_data in payload.get("rounds", [])
        ):
            raise ValueError("strict_c does not allow llmb_passthrough units")
    package_rounds = payload.get("rounds")
    if not isinstance(package_rounds, list) or not package_rounds:
        raise ValueError("commentary_render_package_v2 rounds must be non-empty")
    match_rounds = {round_record.round_no: round_record for round_record in match.rounds}
    package_nos = [item.get("round_no") for item in package_rounds if isinstance(item, dict)]
    if package_nos != [round_record.round_no for round_record in match.rounds]:
        raise ValueError("render package round order does not match rounds_with_commentary")
    profile_id = str(tts_policy.get("speech_profile_id") or "")
    package_identity = str(payload.get("artifact_identity") or "")
    timeline_id = str(timeline.get("timeline_id") or "")
    plans_by_round: dict[int, list[_RenderUnitPlan]] = {}
    for round_data in package_rounds:
        if not isinstance(round_data, dict):
            raise ValueError("render package round must be an object")
        round_no = int(round_data["round_no"])
        round_record = match_rounds[round_no]
        integration_status = round_data.get("integration_status")
        if integration_status in {"blocked", "skipped"}:
            plans_by_round[round_no] = []
            continue
        if integration_status not in {"llmc_accepted", "llmb_passthrough"}:
            raise ValueError(f"render package round {round_no} has invalid integration status")
        units = round_data.get("render_units")
        if not isinstance(units, list):
            raise ValueError(f"render package round {round_no} render_units must be a list")
        plans: list[_RenderUnitPlan] = []
        for index, unit in enumerate(units, start=1):
            if not isinstance(unit, dict):
                raise ValueError(f"render package round {round_no} unit must be an object")
            slot = unit.get("render_slot")
            if not isinstance(slot, dict):
                raise ValueError(f"render package unit {unit.get('unit_id')!r} is missing render_slot")
            start_sec = float(slot["start_sec"])
            end_sec = float(slot["end_sec"])
            if start_sec < round_record.start_sec - 1e-6 or end_sec > round_record.end_sec + 1e-6:
                raise ValueError(f"render package unit {unit.get('unit_id')!r} lies outside round {round_no}")
            plans.append(
                _RenderUnitPlan(
                    round_no=round_no,
                    unit_id=str(unit["unit_id"]),
                    sequence=int(unit.get("sequence") or index),
                    final_text=str(unit["final_text"]),
                    emotion=str(unit["emotion"]),
                    text_source=str(unit["text_source"]),
                    slot_start_sec=start_sec,
                    slot_end_sec=end_sec,
                    slot_start_tick=int(slot["start_tick"]),
                    slot_end_tick=int(slot["end_tick"]),
                    required_speed_factor=float(unit["required_speed_factor"]),
                    max_speed_factor=float(unit["max_speed_factor"]),
                    speech_profile_id=profile_id,
                    timeline_id=timeline_id,
                    render_package_artifact_identity=package_identity,
                )
            )
        plans_by_round[round_no] = plans
    return payload, plans_by_round


def _strict_tts_fingerprint(
    fingerprint_fn,
    tts_runtime: dict,
    plan: _RenderUnitPlan,
    speed_factor: float,
) -> str:
    """Add immutable C identity and unit identity to the runtime TTS fingerprint."""
    runtime_fingerprint = fingerprint_fn(
        tts_runtime,
        plan.tts_text,
        speed_factor=speed_factor,
        profile_id=plan.speech_profile_id or None,
    )
    identity = {
        "render_package_artifact_identity": plan.render_package_artifact_identity,
        "unit_id": plan.unit_id,
        "final_text": plan.final_text,
        "text_source": plan.text_source,
        "emotion": plan.emotion,
        "speech_profile_id": plan.speech_profile_id,
        "speed_factor": float(speed_factor),
        "tts_runtime_fingerprint": runtime_fingerprint,
    }
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _strict_select_render_unit(
    plan: _RenderUnitPlan,
    *,
    tts_runtime: dict,
    cache_dir: Path,
    audio_path: Path,
    synthesize_emotional,
    fingerprint_fn,
    sample_rate: int,
    dry_run: bool = False,
) -> dict:
    """Synthesize only C final_text, with one bounded speed retry."""
    result: dict = {
        "unit_id": plan.unit_id,
        "sequence": plan.sequence,
        "final_text": plan.final_text,
        "emotion": plan.emotion,
        "text_source": plan.text_source,
        "render_slot": {
            "slot_id": plan.unit_id,
            "timeline_id": plan.timeline_id,
            "start_sec": plan.slot_start_sec,
            "end_sec": plan.slot_end_sec,
            "start_tick": plan.slot_start_tick,
            "end_tick": plan.slot_end_tick,
        },
        "required_speed_factor": round(plan.required_speed_factor, 3),
        "max_speed_factor": round(plan.max_speed_factor, 3),
        "speech_profile_id": plan.speech_profile_id,
        "audio_asset": None,
        "actual_duration_sec": None,
        "asset_frame_count": 0,
        "sample_rate": sample_rate,
        "applied_speed_factor": round(plan.required_speed_factor, 3),
        "fit_state": "render_unfit",
        "attempted_speed_factors": [],
        "cache_fingerprint": None,
        "render_unfit_reason": _RENDER_UNFIT_REASON,
    }
    initial_speed = max(1.0, min(plan.required_speed_factor, plan.max_speed_factor))
    if dry_run:
        result["attempted_speed_factors"] = [round(initial_speed, 3)]
        result["fit_state"] = "fit"
        result["render_unfit_reason"] = None
        return result

    attempted: list[float] = []
    speed = initial_speed
    for attempt in range(2):
        if any(abs(speed - previous) <= 1e-9 for previous in attempted):
            break
        attempted.append(speed)
        fingerprint = _strict_tts_fingerprint(fingerprint_fn, tts_runtime, plan, speed)
        try:
            _synthesize_with_cache(
                tts_runtime,
                plan.tts_text,
                fingerprint,
                audio_path,
                cache_dir,
                synthesize_emotional,
                speed_factor=speed,
            )
            params, frame_count, actual_duration = _wav_info(audio_path)
            if params.framerate != sample_rate:
                raise ValueError(f"strict TTS sample rate {params.framerate} does not match configured {sample_rate}")
        except Exception as exc:
            result["render_unfit_reason"] = f"tts_error:{type(exc).__name__}"
            continue
        slot_frames = _strict_round_samples(plan.slot_duration_sec * sample_rate)
        if frame_count <= slot_frames:
            result.update({
                "audio_asset": str(audio_path),
                "actual_duration_sec": frame_count / params.framerate,
                "asset_frame_count": frame_count,
                "sample_rate": params.framerate,
                "applied_speed_factor": round(speed, 3),
                "fit_state": "fit",
                "attempted_speed_factors": [round(value, 3) for value in attempted],
                "cache_fingerprint": fingerprint,
                "render_unfit_reason": None,
            })
            return result
        required_speed = speed * (frame_count / max(1, slot_frames))
        speed = min(plan.max_speed_factor, max(speed, required_speed))
        result["render_unfit_reason"] = _RENDER_UNFIT_REASON
    result["attempted_speed_factors"] = [round(value, 3) for value in attempted]
    return result


def _synthesize_with_cache(
    tts_runtime: dict,
    text: str,
    fingerprint: str,
    audio_path: Path,
    cache_dir: Path,
    synthesize_emotional,
    *,
    budget_overage: float = 1.0,
    speed_factor: float = 1.0,
) -> Path:
    """命中缓存则复用，否则合成到临时文件校验通过后再落缓存，最后拷到目标路径。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_tts_cache_key(text, fingerprint)}.wav"
    if cache_path.exists():
        try:
            _wav_info(cache_path)
        except ValueError:
            cache_path.unlink()
        else:
            print(f"[phase4] 复用 TTS 缓存: {cache_path}")
            shutil.copy2(cache_path, audio_path)
            return audio_path

    print(f"[phase4] 生成 TTS 缓存: {cache_path}")
    fd, tmp_name = tempfile.mkstemp(prefix="tts_", suffix=".wav", dir=str(cache_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        synthesize_emotional(
            tts_runtime,
            text,
            tmp_path,
            budget_overage=budget_overage,
            speed_factor=speed_factor,
        )
        _wav_info(tmp_path)
        shutil.move(str(tmp_path), str(cache_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    shutil.copy2(cache_path, audio_path)
    return audio_path


def _load_commentary_scenes(commentary_path: Path, match) -> tuple[dict[int, list[dict]], dict[int, str]]:
    payload = read_json(commentary_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("rounds"), list):
        raise ValueError("commentary.json must be an object containing a rounds array")
    if payload.get("video_path") != match.video_path:
        raise ValueError("commentary.json and rounds_with_commentary must have the same video_path")
    if payload.get("map_name") != match.map_name:
        raise ValueError("commentary.json and rounds_with_commentary must have the same map_name")

    rounds_by_no = {round_record.round_no: round_record for round_record in match.rounds}
    scenes_by_round: dict[int, list[dict]] = {}
    silent_reason_by_round: dict[int, str] = {}
    for round_index, item in enumerate(payload["rounds"]):
        label = f"commentary.rounds[{round_index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        if "round_no" not in item:
            raise ValueError(f"{label}.round_no is required")
        if not isinstance(item["round_no"], int) or isinstance(item["round_no"], bool):
            raise ValueError(f"{label}.round_no must be an integer")
        round_no = item["round_no"]
        if round_no in scenes_by_round:
            raise ValueError(f"duplicate commentary round_no: {round_no}")
        round_record = rounds_by_no.get(round_no)
        if round_record is None:
            raise ValueError(f"commentary contains unknown round_no: {round_no}")
        status = item.get("status")
        if status not in {"ok", "silent"}:
            raise ValueError(f"{label}.status must be ok or silent, got {status!r}")
        for field, expected in (("start_sec", round_record.start_sec), ("end_sec", round_record.end_sec)):
            value = item.get(field)
            if isinstance(value, bool):
                raise ValueError(f"{label}.{field} must match rounds_with_commentary")
            try:
                source_time = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}.{field} must match rounds_with_commentary") from exc
            if not math.isfinite(source_time) or not math.isclose(source_time, expected, abs_tol=1e-6):
                raise ValueError(f"{label}.{field} does not match rounds_with_commentary")
        if not isinstance(item.get("commentary_text"), str):
            raise ValueError(f"{label}.commentary_text must be a string")
        semantic = round_record.phase3_semantic
        if semantic is None:
            raise ValueError(f"rounds_with_commentary round {round_no} is missing phase3 commentary")
        if item["commentary_text"] != semantic.commentary_text:
            raise ValueError(f"{label}.commentary_text does not match rounds_with_commentary")
        if not isinstance(item.get("emotion_segments"), list):
            raise ValueError(f"{label}.emotion_segments must be an array")
        manifest_segments = []
        for segment_index, segment in enumerate(item["emotion_segments"]):
            if not isinstance(segment, dict):
                raise ValueError(f"{label}.emotion_segments[{segment_index}] must be an object")
            manifest_segments.append((segment.get("emotion"), segment.get("text"), segment.get("order")))
        semantic_segments = [
            (segment.emotion, segment.text, segment.order)
            for segment in semantic.emotion_segments
        ]
        if manifest_segments != semantic_segments:
            raise ValueError(f"{label}.emotion_segments do not match rounds_with_commentary")
        if "scenes" not in item:
            raise ValueError(f"{label}.scenes is required; legacy round-level commentary is unsupported")
        if not isinstance(item["scenes"], list):
            raise ValueError(f"{label}.scenes must be an array")
        if bool(item["scenes"]) != bool(item["commentary_text"].strip()):
            raise ValueError(f"{label}.commentary_text and scenes must both be empty or both be populated")
        if bool(item["emotion_segments"]) != bool(item["commentary_text"].strip()):
            raise ValueError(f"{label}.commentary_text and emotion_segments must both be empty or both be populated")
        if status == "ok" and not item["commentary_text"].strip():
            raise ValueError(f"{label}.status ok requires commentary text")
        if status == "silent" and item["commentary_text"].strip():
            raise ValueError(f"{label}.status silent requires empty commentary text")

        normalized: list[dict] = []
        last_end: float | None = None
        for scene_index, scene in enumerate(item["scenes"]):
            scene_label = f"{label}.scenes[{scene_index}]"
            if not isinstance(scene, dict):
                raise ValueError(f"{scene_label} must be an object")
            missing = {"t_start", "t_end", "text", "emotion"} - scene.keys()
            if missing:
                raise ValueError(f"{scene_label} is missing fields: {', '.join(sorted(missing))}")
            if isinstance(scene["t_start"], bool) or isinstance(scene["t_end"], bool):
                raise ValueError(f"{scene_label} times must be numeric")
            try:
                t_start = float(scene["t_start"])
                t_end = float(scene["t_end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{scene_label} times must be numeric") from exc
            if not math.isfinite(t_start) or not math.isfinite(t_end) or t_start >= t_end:
                raise ValueError(f"{scene_label} must have finite t_start < t_end")
            if t_start < round_record.start_sec - 1e-6 or t_end > round_record.end_sec + 1e-6:
                raise ValueError(f"{scene_label} lies outside round {round_no}")
            if last_end is not None and t_start < last_end - 1e-6:
                raise ValueError(f"{scene_label} overlaps the previous scene")
            text = scene["text"]
            emotion = scene["emotion"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{scene_label}.text must be a non-empty string")
            if not isinstance(emotion, str) or emotion not in _EMOTIONS:
                raise ValueError(f"{scene_label}.emotion must be one of {sorted(_EMOTIONS)}")
            normalized.append({
                "t_start": t_start,
                "t_end": t_end,
                "text": text.strip(),
                "emotion": emotion,
            })
            last_end = t_end
        rendered_text = "".join(f"[{scene['emotion']}]{scene['text']}" for scene in normalized)
        if rendered_text != item["commentary_text"]:
            raise ValueError(f"{label}.scenes text does not match commentary_text")
        scenes_by_round[round_no] = normalized
        # 方案 R（§6.4）：透传 Phase3b 的 silent_reason，供 manifest 标 skipped(reason=...)。
        silent_reason = item.get("silent_reason")
        if isinstance(silent_reason, str) and silent_reason:
            silent_reason_by_round[round_no] = silent_reason

    missing_rounds = sorted(set(rounds_by_no) - set(scenes_by_round))
    if missing_rounds:
        raise ValueError(f"commentary.json is missing rounds: {missing_rounds}")
    return scenes_by_round, silent_reason_by_round


def _build_voice_task_plans(commentary_payload, rounds_payload, match) -> dict[int, list[_VoiceTaskPlan]]:
    """解析 commentary v3 任务单与 rounds_with_commentary 的 scene，形成只读执行对象。

    §10.3 步骤 1：校验 task 与 scene 的 window/round/start/end/source hash 一致；
    缺字段或任一项不一致直接拒绝，不猜测、不隐式退化（§9.7）。
    """
    contract_errors = validate_commentary_v3(commentary_payload)
    if contract_errors:
        raise ValueError("commentary v3 contract validation failed: " + "; ".join(contract_errors))
    if not isinstance(rounds_payload, dict):
        raise ValueError("rounds_with_commentary.json must be an object")

    tasks = list(commentary_payload.get("voice_tasks") or [])
    tasks_by_id = {task["voice_task_id"]: task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError("commentary v3 contains duplicate voice_task_id")
    rounds_hash = rounds_payload.get("source_neutral_sha256")
    if not isinstance(rounds_hash, str) or not rounds_hash.strip():
        raise ValueError("rounds_with_commentary v3 is missing source_neutral_sha256")
    if commentary_payload.get("source_neutral_sha256") != rounds_hash:
        raise ValueError("commentary v3 source_neutral_sha256 does not match rounds_with_commentary")

    rounds_by_no = {round_record.round_no: round_record for round_record in match.rounds}
    raw_rounds = rounds_payload.get("rounds")
    if not isinstance(raw_rounds, list):
        raise ValueError("rounds_with_commentary v3 must contain rounds")
    plans_by_round: dict[int, list[_VoiceTaskPlan]] = {}
    used_task_ids: set[str] = set()
    for ridx, round_data in enumerate(raw_rounds):
        label = f"rounds_with_commentary v3 rounds[{ridx}]"
        if not isinstance(round_data, dict):
            raise ValueError(f"{label} must be an object")
        round_no = round_data.get("round_no")
        if not isinstance(round_no, int) or isinstance(round_no, bool):
            raise ValueError(f"{label}.round_no must be an integer")
        round_record = rounds_by_no.get(round_no)
        if round_record is None:
            raise ValueError(f"{label} contains unknown round_no: {round_no}")
        scenes = round_data.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError(f"{label} must contain a scenes array")
        plans: list[_VoiceTaskPlan] = []
        last_end: float | None = None
        for sidx, scene in enumerate(scenes):
            scene_label = f"{label}.scenes[{sidx}]"
            if not isinstance(scene, dict):
                raise ValueError(f"{scene_label} must be an object")
            missing = {"window_id", "voice_task_id", "t_start", "t_end", "text", "emotion"} - scene.keys()
            if missing:
                raise ValueError(f"{scene_label} is missing fields: {', '.join(sorted(missing))}")
            voice_task_id = scene["voice_task_id"]
            if not isinstance(voice_task_id, str) or not voice_task_id.strip():
                raise ValueError(f"{scene_label}.voice_task_id must be a non-empty string")
            if voice_task_id in used_task_ids:
                raise ValueError(f"{scene_label}.voice_task_id is duplicated across scenes")
            used_task_ids.add(voice_task_id)
            task = tasks_by_id.get(voice_task_id)
            if task is None:
                raise ValueError(f"{scene_label}.voice_task_id is not found in the commentary voice_tasks")
            window_id = scene["window_id"]
            if not isinstance(window_id, str) or window_id != task.get("window_id"):
                raise ValueError(f"{scene_label}.window_id does not match the voice task")
            if isinstance(scene["t_start"], bool) or isinstance(scene["t_end"], bool):
                raise ValueError(f"{scene_label} times must be numeric")
            try:
                t_start = float(scene["t_start"])
                t_end = float(scene["t_end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{scene_label} times must be numeric") from exc
            if not math.isfinite(t_start) or not math.isfinite(t_end) or t_start >= t_end:
                raise ValueError(f"{scene_label} must have finite t_start < t_end")
            slot = task.get("render_slot") or {}
            if not math.isclose(t_start, float(slot.get("start_sec")), abs_tol=1e-6) or not math.isclose(
                t_end, float(slot.get("end_sec")), abs_tol=1e-6
            ):
                raise ValueError(f"{scene_label} window does not match the voice task render_slot")
            if t_start < round_record.start_sec - 1e-6 or t_end > round_record.end_sec + 1e-6:
                raise ValueError(f"{scene_label} lies outside round {round_no}")
            if last_end is not None and t_start < last_end - 1e-6:
                raise ValueError(f"{scene_label} overlaps the previous scene")
            last_end = t_end
            text = scene["text"]
            emotion = scene["emotion"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{scene_label}.text must be a non-empty string")
            if not isinstance(emotion, str) or emotion not in _EMOTIONS:
                raise ValueError(f"{scene_label}.emotion must be one of {sorted(_EMOTIONS)}")
            primary = next(
                (candidate for candidate in task.get("candidates") or [] if candidate.get("variant_id") == "primary"),
                None,
            )
            if primary is not None and text.strip() != str(primary.get("text", "")):
                raise ValueError(f"{scene_label}.text must match the primary candidate text")
            plans.append(
                _VoiceTaskPlan(
                    round_no=round_no,
                    voice_task_id=voice_task_id,
                    window_id=window_id,
                    scene=dict(scene),
                    slot_start_tick=int(slot["start_tick"]),
                    slot_end_tick=int(slot["end_tick"]),
                    slot_start_sec=t_start,
                    slot_end_sec=t_end,
                    selection_order=tuple(task.get("selection_order") or ()),
                    speech_profile_id=task.get("speech_profile_id") or "",
                    candidates=tuple(dict(candidate) for candidate in (task.get("candidates") or [])),
                    max_speed_factor=float(task.get("max_speed_factor") or 1.0),
                )
            )
        plans_by_round[round_no] = plans
    missing_rounds = sorted(set(rounds_by_no) - set(plans_by_round))
    if missing_rounds:
        raise ValueError(f"rounds_with_commentary v3 is missing rounds: {missing_rounds}")
    return plans_by_round


def _check_task_profiles(plans_by_round: dict[int, list[_VoiceTaskPlan]], tts_runtime: dict, sample_rate: int) -> None:
    """§10.3 步骤 2：任务单 profile_id 必须与当前 TTS 运行指纹完全匹配。

    profile 缺失、状态非 validated 或指纹不匹配一律 profile_mismatch 停止，不猜测。
    """
    from audio_service.gpt_sovits_client import tts_runtime_fingerprint

    seen: set[str] = set()
    runtime_fingerprints: dict | None = None
    for plans in plans_by_round.values():
        for plan in plans:
            if plan.speech_profile_id in seen:
                continue
            seen.add(plan.speech_profile_id)
            profile = speech_measure.load_profile(plan.speech_profile_id)
            if profile is None:
                raise ValueError(f"profile_mismatch: speech profile {plan.speech_profile_id!r} not found")
            status = speech_measure.validate_profile_status(profile)
            if status != "validated":
                raise ValueError(
                    f"profile_mismatch: speech profile {plan.speech_profile_id!r} status is {status!r}, not validated"
                )
            if runtime_fingerprints is None:
                runtime_fingerprints = tts_runtime_fingerprint(tts_runtime, sample_rate_hz=sample_rate)
            if not speech_measure.check_profile_match(
                profile,
                engine_fingerprint=runtime_fingerprints["engine_fingerprint"],
                voice_fingerprint=runtime_fingerprints["voice_fingerprint"],
                preprocess_fingerprint=runtime_fingerprints["preprocess_fingerprint"],
            ):
                raise ValueError(
                    f"profile_mismatch: speech profile {plan.speech_profile_id!r} "
                    "fingerprints do not match the current TTS runtime"
                )


def _synthesize_variant_attempt(
    plan: _VoiceTaskPlan,
    candidate: dict,
    speed_factor: float,
    *,
    tts_runtime: dict,
    cache_dir: Path,
    scene_audio_path: Path,
    synthesize_emotional,
    fingerprint_fn,
) -> tuple[float | None, Path | None, str | None]:
    """惰性合成单个候选（§5.3）：只合成当前候选，不预合成其他候选。"""
    text = f"[{plan.scene['emotion']}]{candidate['text']}"
    try:
        fingerprint = fingerprint_fn(
            tts_runtime,
            text,
            speed_factor=speed_factor,
            variant_id=candidate["variant_id"],
            profile_id=plan.speech_profile_id,
        )
        audio_path = _synthesize_with_cache(
            tts_runtime,
            text,
            fingerprint,
            scene_audio_path,
            cache_dir,
            synthesize_emotional,
            budget_overage=1.0,
            speed_factor=speed_factor,
        )
        _, _, duration = _wav_info(audio_path)
        return duration, audio_path, fingerprint
    except Exception as exc:
        print(f"[phase4] TTS 失败 {plan.voice_task_id}/{candidate['variant_id']} @{speed_factor:.3f}: {exc}")
        return None, None, None


def _select_voice_variant(
    plan: _VoiceTaskPlan,
    *,
    tts_runtime: dict,
    cache_dir: Path,
    scene_audio_path: Path,
    synthesize_emotional,
    fingerprint_fn,
    dry_run: bool = False,
) -> dict:
    """§10.3 选择算法：按 selection_order 逐候选合成，以实际 PCM 时长裁决。

    - 使用候选建议的最低速度合成；适配即停（惰性）。
    - 超长只能在 task 的 max_speed_factor 内再试一次；仍失败进入下一候选。
    - 全部失败写 render_unfit。
    """
    result: dict = {
        "render_slot": {
            "start_sec": plan.slot_start_sec,
            "end_sec": plan.slot_end_sec,
            "start_tick": plan.slot_start_tick,
            "end_tick": plan.slot_end_tick,
        },
        "voice_task_id": plan.voice_task_id,
        "window_id": plan.window_id,
        "selected_variant_id": None,
        "selected_text": None,
        "actual_duration_sec": None,
        "applied_speed_factor": None,
        "audio_start_tick": None,
        "audio_end_tick": None,
        "fit_state": "render_unfit",
        "attempted_variants": [],
        "render_unfit_reason": _RENDER_UNFIT_REASON,
        "audio_path": None,
        "cache_fingerprint": None,
    }
    if dry_run:
        first = next((candidate for candidate in plan.candidates if candidate["variant_id"] in plan.selection_order), None)
        if first is not None:
            result.update({
                "selected_variant_id": first["variant_id"],
                "selected_text": first["text"],
                "applied_speed_factor": max(
                    1.0, min(float(first.get("minimum_required_speed_factor") or 1.0), plan.max_speed_factor)
                ),
                "fit_state": "fit",
                "render_unfit_reason": None,
            })
        return result

    by_id = {candidate["variant_id"]: candidate for candidate in plan.candidates}
    for variant_id in plan.selection_order:
        candidate = by_id.get(variant_id)
        if candidate is None:
            continue
        suggested = max(1.0, float(candidate.get("minimum_required_speed_factor") or 1.0))
        speed = min(suggested, plan.max_speed_factor)
        duration, audio_path, fingerprint = _synthesize_variant_attempt(
            plan,
            candidate,
            speed,
            tts_runtime=tts_runtime,
            cache_dir=cache_dir,
            scene_audio_path=scene_audio_path,
            synthesize_emotional=synthesize_emotional,
            fingerprint_fn=fingerprint_fn,
        )
        result["attempted_variants"].append(variant_id)
        if duration is None:
            continue
        if check_scene_slot_fit(duration, plan.slot_start_tick, plan.slot_end_tick):
            result.update({
                "selected_variant_id": variant_id,
                "selected_text": candidate["text"],
                "actual_duration_sec": round(duration, 3),
                "applied_speed_factor": round(speed, 3),
                "audio_start_tick": plan.slot_start_tick,
                "audio_end_tick": audio_end_tick(duration, plan.slot_start_tick),
                "fit_state": "fit",
                "render_unfit_reason": None,
                "audio_path": str(audio_path),
                "cache_fingerprint": fingerprint,
            })
            break
        required = duration / plan.slot_duration_sec
        if required > plan.max_speed_factor + 1e-9:
            continue
        retry_speed = max(speed, min(required, plan.max_speed_factor))
        if retry_speed - speed <= 1e-6:
            continue
        duration2, audio_path2, fingerprint2 = _synthesize_variant_attempt(
            plan,
            candidate,
            retry_speed,
            tts_runtime=tts_runtime,
            cache_dir=cache_dir,
            scene_audio_path=scene_audio_path,
            synthesize_emotional=synthesize_emotional,
            fingerprint_fn=fingerprint_fn,
        )
        result["attempted_variants"].append(variant_id)
        if duration2 is None:
            continue
        if check_scene_slot_fit(duration2, plan.slot_start_tick, plan.slot_end_tick):
            result.update({
                "selected_variant_id": variant_id,
                "selected_text": candidate["text"],
                "actual_duration_sec": round(duration2, 3),
                "applied_speed_factor": round(retry_speed, 3),
                "audio_start_tick": plan.slot_start_tick,
                "audio_end_tick": audio_end_tick(duration2, plan.slot_start_tick),
                "fit_state": "fit",
                "render_unfit_reason": None,
                "audio_path": str(audio_path2),
                "cache_fingerprint": fingerprint2,
            })
            break
    return result


def _rounds_final_scene(scene_record: dict) -> dict:
    """§10.4：rounds_final.json 每 scene 的配音任务单执行结果（契约 validate_final_voice_task）。"""
    return {
        key: scene_record[key]
        for key in (
            "window_id",
            "voice_task_id",
            "render_slot",
            "selected_variant_id",
            "selected_text",
            "actual_duration_sec",
            "applied_speed_factor",
            "audio_start_tick",
            "audio_end_tick",
            "fit_state",
            "attempted_variants",
            "render_unfit_reason",
        )
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip_cache_fingerprint(source_sha256: str, start_sec: float, end_sec: float) -> str:
    """clip 缓存指纹：源视频内容哈希 + 起止时间戳，任一变化即视为新 clip。"""
    identity = f"{source_sha256}\0{float(start_sec):.17g}\0{float(end_sec):.17g}"
    return hashlib.sha256(identity.encode("ascii")).hexdigest()


def _resolve_existing_clip(round_record, clip_cache_dir: Path, fingerprint: str | None = None) -> Path | None:
    """优先复用 phase1 切好的 segment_video，其次命中指纹缓存，都没有返回 None。"""
    segment_video = resolve_path(getattr(round_record, "segment_video", ""))
    if segment_video is not None and segment_video.exists():
        return segment_video
    if fingerprint is not None:
        candidate = clip_cache_dir / f"clip_{fingerprint}.mp4"
        if candidate.exists():
            return candidate
    return None


def _clip_for_round(
    round_record,
    source_video: Path,
    clip_cache_dir: Path,
    source_sha256: str | None = None,
) -> Path:
    existing = _resolve_existing_clip(round_record, clip_cache_dir)
    if existing is not None:
        print(f"[phase4] 复用 clip: {existing}")
        return existing

    source_sha256 = source_sha256 or _sha256_file(source_video)
    fingerprint = _clip_cache_fingerprint(source_sha256, round_record.start_sec, round_record.end_sec)
    existing = _resolve_existing_clip(round_record, clip_cache_dir, fingerprint)
    if existing is not None:
        print(f"[phase4] 复用 clip: {existing}")
        return existing

    clip_cache_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_cache_dir / f"clip_{fingerprint}.mp4"
    print(f"[phase4] 切 clip: {clip_path}")
    _run_ffmpeg([
        "-ss", str(round_record.start_sec),
        "-to", str(round_record.end_sec),
        "-i", str(source_video),
        "-c", "copy",
        str(clip_path),
    ])
    return clip_path


def _strict_probe_media(source_video: Path, p4: dict) -> tuple[dict | None, str | None]:
    """Probe a source once; missing tools become auditable not_checked state."""
    try:
        from sbmachine.media_probe import probe_media

        probe_cfg = p4.get("media_probe") if isinstance(p4.get("media_probe"), dict) else {}
        ffprobe_bin = str(probe_cfg.get("ffprobe_bin") or "ffprobe")
        result = probe_media(source_video, ffprobe_bin=ffprobe_bin)
        if result.get("probe_status") != "checked":
            details = result.get("not_checked") if isinstance(result.get("not_checked"), dict) else {}
            reason = details.get("reason") or result.get("media_sync_reason") or "media_probe_not_checked"
            return None, str(reason)
        return result, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _strict_apply_clock_evidence(
    scene_record: dict,
    plan: _RenderUnitPlan,
    *,
    probe: dict | None,
    source_video: Path | None,
    p4: dict,
    timeline_origin_sec: float = 0.0,
) -> None:
    """Attach independent sample/PTS evidence; expected PTS never becomes decoded PTS."""
    from sbmachine.media_clock import MediaClockAdapter

    if not probe:
        return
    time_base = probe.get("time_base")
    start_pts = probe.get("stream_start_pts")
    if time_base is None or start_pts is None:
        return
    try:
        adapter = MediaClockAdapter.from_probe(
            probe,
            sample_rate=int(p4.get("sample_rate", 32000)),
            timeline_origin_sec=timeline_origin_sec,
            timeline_id=plan.timeline_id,
        )
        clock_mapping = adapter.map_interval(
            plan.slot_start_sec,
            plan.slot_end_sec,
            slot_id=plan.unit_id,
        )
    except Exception:
        return
    scene_record["clock_map_version"] = clock_mapping.get("clock_map_version", 1)
    scene_record["expected_start_pts"] = clock_mapping["expected_start_pts"]
    scene_record["expected_end_pts"] = clock_mapping["expected_end_pts"]
    if source_video is None:
        return
    probe_cfg = p4.get("media_probe") if isinstance(p4.get("media_probe"), dict) else {}
    if probe_cfg.get("boundary_probe", True) is False:
        return
    try:
        from sbmachine.media_probe import probe_frame_boundaries

        frames = probe_frame_boundaries(
            source_video,
            plan.slot_start_sec,
            plan.slot_end_sec,
            probe=probe,
            timeline_origin_sec=timeline_origin_sec,
            ffprobe_bin=str(probe_cfg.get("ffprobe_bin") or "ffprobe"),
        )
    except Exception:
        return
    if isinstance(frames, dict):
        start_value = frames.get("decoded_start_pts", frames.get("start_pts"))
        end_value = frames.get("decoded_end_pts", frames.get("end_pts"))
        if start_value is not None:
            scene_record["decoded_start_pts"] = int(start_value)
        if end_value is not None:
            scene_record["decoded_end_pts"] = int(end_value)
    elif isinstance(frames, list) and frames:
        pts_values = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            value = frame.get("pts", frame.get("best_effort_timestamp"))
            if isinstance(value, int) and not isinstance(value, bool):
                pts_values.append(value)
            elif isinstance(value, float) and math.isfinite(value):
                pts_values.append(int(round(value)))
        if pts_values:
            scene_record["decoded_start_pts"] = min(pts_values)
            scene_record["decoded_end_pts"] = max(pts_values)


def _strict_clip_for_round(
    round_record,
    source_video: Path,
    clip_cache_dir: Path,
    *,
    source_sha256: str,
    probe: dict,
    p4: dict,
) -> dict:
    """Create or reuse a decoded clip and preserve its audit result."""
    from sbmachine import phase4_media

    probe_cfg = p4.get("media_probe") if isinstance(p4.get("media_probe"), dict) else {}
    tolerance_cfg = p4.get("media_tolerances") if isinstance(p4.get("media_tolerances"), dict) else {}
    result = phase4_media.strict_decode_clip(
        source_video,
        start_sec=float(round_record.start_sec),
        end_sec=float(round_record.end_sec),
        cache_dir=clip_cache_dir,
        source_sha256=source_sha256,
        ffmpeg_bin=str(probe_cfg.get("ffmpeg_bin") or p4.get("ffmpeg_bin") or "ffmpeg"),
        ffprobe_bin=str(probe_cfg.get("ffprobe_bin") or "ffprobe"),
        max_frame_boundary_error_sec=float(tolerance_cfg.get("max_frame_boundary_error_sec", 0.05)),
        # 切片时长受视频帧量化约束（60fps 粒度约 0.0167s），应使用帧界容差而非音频边界容差
        max_duration_error_sec=float(tolerance_cfg.get("max_frame_boundary_error_sec", 0.05)),
        reuse=True,
    )
    if not isinstance(result, dict):
        raise ValueError("strict clip helper returned a non-object audit")
    normalized = dict(result)
    clip_value = normalized.get("clip_path") or normalized.get("output_path")
    if isinstance(clip_value, (str, Path)):
        normalized["path"] = Path(clip_value)
    sidecar = normalized.get("sidecar")
    normalized["sidecar"] = dict(sidecar) if isinstance(sidecar, dict) else {}
    if normalized.get("status") == "pass":
        if not isinstance(normalized.get("path"), Path) or not normalized["path"].is_file():
            normalized.update({"status": "fail", "ok": False, "reason": "strict_clip_path_missing"})
        elif normalized["sidecar"].get("boundary_status") != "verified":
            normalized.update({"status": "fail", "ok": False, "reason": "strict_clip_sidecar_unverified"})
        elif normalized["sidecar"].get("source_sha256") != source_sha256:
            normalized.update({"status": "fail", "ok": False, "reason": "strict_clip_source_identity_mismatch"})
    return normalized


def _run_phase4_execution_v2(
    *,
    match,
    render_package_path: Path,
    commentary_path: Path | None,
    output_rounds_path: Path,
    manifest_path: Path,
    config: dict,
    dry_run: bool,
    progress_sink=None,
) -> dict:
    """Strict Phase4 execution path; C v2 owns all final text."""
    from audio_service.gpt_sovits_client import read_config as read_tts_config
    from audio_service.gpt_sovits_client import synthesize_emotional
    from audio_service.gpt_sovits_client import tts_cache_fingerprint
    from sbmachine import phase4_media
    from sbmachine.phase4_av import assemble_scene_canvas_v2

    p4 = config.get("phase4", {}) if isinstance(config.get("phase4", {}), dict) else {}
    publish_profile = str(p4.get("publish_profile", "legacy"))
    render_package, plans_by_round = _load_render_package_v2(render_package_path, match, publish_profile)
    tts_runtime_path = require_path(
        p4.get("tts_config", "audio_service/gpt_sovits_runtime.yaml"),
        "phase4.tts_config",
    )
    tts_runtime = read_tts_config(tts_runtime_path) if not dry_run else {}
    output_dir = resolve_path(p4.get("output_dir", "output/sbmachine/rounds")) or Path("output/sbmachine/rounds")
    tts_cache_dir = resolve_path(p4.get("tts_cache_dir", "output/tts_cache")) or Path("output/tts_cache")
    clip_cache_dir = resolve_path(p4.get("clip_cache_dir", "output/clips")) or Path("output/clips")
    sample_rate = int(p4.get("sample_rate", 32000))
    if sample_rate <= 0:
        raise ValueError("phase4.sample_rate must be positive")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    source_video = resolve_path(getattr(match, "video_path", None))
    make_video = bool(p4.get("make_video", False))
    probe: dict | None = None
    probe_error: str | None = None
    if make_video and source_video is not None and source_video.is_file():
        probe, probe_error = _strict_probe_media(source_video, p4)
    elif make_video:
        probe_error = "source video is missing"
    source_probe_summary = {
        "probe_status": "not_checked",
        "media_sync_status": "not_checked",
        "reason": probe_error or "video_probe_not_requested",
    }
    if isinstance(probe, dict):
        source_probe_summary.update({
            key: probe.get(key)
            for key in (
                "probe_schema_version",
                "source_sha256",
                "has_audio",
                "duration_sec",
                "video_stream_index",
                "time_base",
                "stream_start_pts",
                "stream_start_time_sec",
                "avg_frame_rate",
                "r_frame_rate",
                "variable_frame_rate",
                "probe_status",
                "media_sync_status",
                "media_sync_reason",
            )
            if key in probe
        })

    records: list[dict] = []
    skipped_count = 0
    all_sources: set[str] = set()
    round_media_statuses: list[str] = []
    round_content_statuses: list[str] = []
    round_delivery_statuses: list[str] = []
    source_sha256: str | None = None
    timeline_origin_sec = float((render_package.get("timeline") or {}).get("timeline_origin_sec", 0.0))
    for completed, round_record in enumerate(tqdm(match.rounds, desc="Phase4 strict TTS+Mux", unit="round"), start=1):
        rno = round_record.round_no
        plans = plans_by_round.get(rno, [])
        skipped = not plans
        if skipped:
            skipped_count += 1
        scene_records: list[dict] = []
        scene_inputs: list[dict] = []
        round_unfit = False
        for plan_index, plan in enumerate(plans, start=1):
            scene_audio_path = output_dir / f"round_{rno:03d}_scene_{plan_index:03d}.wav"
            selection = _strict_select_render_unit(
                plan,
                tts_runtime=tts_runtime,
                cache_dir=tts_cache_dir,
                audio_path=scene_audio_path,
                synthesize_emotional=synthesize_emotional,
                fingerprint_fn=tts_cache_fingerprint,
                sample_rate=sample_rate,
                dry_run=dry_run,
            )
            scene_records.append(selection)
            all_sources.add(plan.text_source)
            if selection["fit_state"] == "fit" and not dry_run:
                scene_inputs.append({
                    "unit_id": plan.unit_id,
                    "audio_asset": selection["audio_asset"],
                    "slot_start_sec": plan.slot_start_sec,
                    "slot_end_sec": plan.slot_end_sec,
                })
            elif selection["fit_state"] != "fit":
                round_unfit = True

        audio_path = output_dir / f"round_{rno:03d}.wav"
        round_duration = float(round_record.end_sec) - float(round_record.start_sec)
        if round_duration <= 0:
            raise ValueError(f"round {rno} must have start_sec < end_sec")
        canvas_result: dict = {"units": [], "round_canvas_limit_sample": _strict_round_samples(round_duration * sample_rate), "sample_rate": sample_rate}
        if not dry_run and not round_unfit:
            canvas_result = assemble_scene_canvas_v2(
                scene_inputs,
                audio_path,
                round_record.start_sec,
                round_record.end_sec,
                default_sample_rate=sample_rate,
                timeline_origin_sec=timeline_origin_sec,
            )
            canvas_by_id = {item.get("unit_id"): item for item in canvas_result.get("units", []) if isinstance(item, dict)}
            for record in scene_records:
                canvas_item = canvas_by_id.get(record.get("unit_id"))
                if canvas_item:
                    record.update(canvas_item)
        elif not dry_run and skipped:
            canvas_result = assemble_scene_canvas_v2(
                [],
                audio_path,
                round_record.start_sec,
                round_record.end_sec,
                default_sample_rate=sample_rate,
                timeline_origin_sec=timeline_origin_sec,
            )
        round_record.phase4_audio = AudioData(
            audio_path=str(audio_path) if not round_unfit else "",
            duration_sec=(round_duration if dry_run else (_wav_info(audio_path)[2] if audio_path.exists() else None)),
        )
        for plan, record in zip(plans, scene_records):
            _strict_apply_clock_evidence(
                record,
                plan,
                probe=probe,
                source_video=source_video,
                p4=p4,
                timeline_origin_sec=timeline_origin_sec,
            )

        video_path: str | None = None
        sidecar: dict = {}
        media_audit: dict = {}
        media_reason: str | None = None
        media_checks = {
            "source_identity": "not_required" if skipped else "not_checked",
            "decoded_pts_monotone": "not_required" if skipped else "not_checked",
            "clip_boundary": "not_required" if skipped else "not_checked",
            "audio_within_slot": "not_required" if skipped else ("fail" if round_unfit else "pass"),
            "canvas_bounds": "not_required" if skipped else ("fail" if round_unfit else "pass"),
            "subtitle_within_audio": "not_required",
        }
        if round_unfit:
            round_media_status = "fail"
        elif skipped:
            round_media_status = "not_required"
        elif make_video and source_video is not None and probe is not None and not dry_run:
            clip_mode = str(p4.get("clip_mode", "legacy_copy"))
            if clip_mode != "strict_decode":
                raise ValueError("strict Phase4 requires phase4.clip_mode=strict_decode")
            source_sha256 = source_sha256 or str(probe.get("source_sha256") or _sha256_file(source_video))
            clip_result = _strict_clip_for_round(
                round_record,
                source_video,
                clip_cache_dir,
                source_sha256=source_sha256,
                probe=probe,
                p4=p4,
            )
            sidecar = clip_result.get("sidecar") if isinstance(clip_result.get("sidecar"), dict) else {}
            raw_clip_status = str(clip_result.get("status") or "fail")
            clip_status = raw_clip_status if raw_clip_status in {"pass", "fail", "not_checked"} else "fail"
            sidecar_source_sha = sidecar.get("source_sha256")
            media_checks.update({
                "source_identity": (
                    "pass" if sidecar_source_sha == source_sha256 else
                    clip_status if clip_status == "fail" else
                    "not_checked"
                ),
                "decoded_pts_monotone": "pass" if sidecar.get("boundary_status") == "verified" else clip_status,
                "clip_boundary": "pass" if sidecar.get("boundary_status") == "verified" else clip_status,
            })
            media_audit["clip"] = {
                "status": clip_status,
                "reason": clip_result.get("reason") or clip_result.get("media_sync_reason"),
                "fingerprint": clip_result.get("fingerprint"),
            }
            if clip_status == "pass" and isinstance(clip_result.get("path"), (str, Path)):
                clip_path = Path(clip_result["path"])
                out_mp4 = output_dir / f"round_{rno:03d}.mp4"
                probe_cfg = p4.get("media_probe") if isinstance(p4.get("media_probe"), dict) else {}
                tolerance_cfg = p4.get("media_tolerances") if isinstance(p4.get("media_tolerances"), dict) else {}
                mux_result = phase4_media.mux_clip_with_audio(
                    clip_path,
                    audio_path,
                    out_mp4,
                    clip_audit=clip_result,
                    ffmpeg_bin=str(probe_cfg.get("ffmpeg_bin") or p4.get("ffmpeg_bin") or "ffmpeg"),
                    ffprobe_bin=str(probe_cfg.get("ffprobe_bin") or "ffprobe"),
                    game_vol=float(p4.get("game_audio_volume", 0.25)),
                    comm_vol=float(p4.get("commentary_volume", 1.0)),
                    video_codec=str(p4.get("strict_video_codec") or "libx264"),
                    # 混流输出时长含视频帧量化（60fps 粒度约 0.0167s），应使用帧界容差
                    duration_tolerance_sec=float(tolerance_cfg.get("max_frame_boundary_error_sec", 0.05)),
                )
                raw_mux_status = str(mux_result.get("status") or "fail")
                mux_status = raw_mux_status if raw_mux_status in {"pass", "fail", "not_checked"} else "fail"
                media_audit["mux"] = {
                    "status": mux_status,
                    "reason": mux_result.get("reason") or mux_result.get("media_sync_reason"),
                }
                media_checks["mux_boundary"] = mux_status
                if mux_status == "pass":
                    video_path = str(out_mp4)
                else:
                    media_reason = str(mux_result.get("reason") or "strict_mux_not_pass")
            else:
                media_reason = str(clip_result.get("reason") or "strict_clip_not_pass")
            statuses = list(media_checks.values())
            round_media_status = (
                "fail" if "fail" in statuses else
                "not_checked" if "not_checked" in statuses else
                "pass"
            )
        else:
            round_media_status = "not_checked"
        round_content_status = "degraded" if "llmb_passthrough" in {plan.text_source for plan in plans} else "pass"
        if round_unfit:
            round_content_status = "fail"
        round_delivery_status = "not_required" if skipped else "fail" if round_unfit else (
            "pass" if publish_profile == "legacy" and round_media_status in {"pass", "not_checked", "not_required"}
            else "pass" if publish_profile == "strict_av" and round_media_status == "pass" and round_content_status in {"pass", "degraded"}
            else "pass" if publish_profile == "strict_c" and round_media_status == "pass" and round_content_status == "pass"
            else "not_checked"
        )
        round_media_statuses.append(round_media_status)
        round_content_statuses.append(round_content_status)
        round_delivery_statuses.append(round_delivery_status)
        round_record.scenes = scene_records
        record = {
            "round_no": rno,
            "audio_path": str(audio_path) if not round_unfit else "",
            "video_path": video_path,
            "skipped": skipped,
            "aligned": round_media_status == "pass",
            "media_sync_status": round_media_status,
            "content_gate_status": round_content_status,
            "delivery_status": round_delivery_status,
            "media_checks": media_checks,
            "content_checks": {
                "text_sources": sorted({plan.text_source for plan in plans}),
                "blocked_round": False,
            },
            "segments": scene_records,
        }
        if sidecar:
            record["clip_sidecar"] = sidecar
        if media_audit:
            record["media_audit"] = media_audit
        if probe_error:
            record["media_sync_reason"] = probe_error
        if media_reason:
            record["media_sync_reason"] = media_reason
        if round_unfit:
            record["render_unfit"] = True
        records.append(record)
        if progress_sink is not None:
            try:
                progress_sink(completed, len(match.rounds), "round", None)
            except Exception:
                pass

    top_media = "fail" if "fail" in round_media_statuses else "not_checked" if "not_checked" in round_media_statuses else "pass"
    if round_media_statuses and all(status in {"pass", "not_required"} for status in round_media_statuses):
        top_media = "pass"
    top_content = "fail" if "fail" in round_content_statuses else "degraded" if "degraded" in round_content_statuses else "pass"
    blocked_rounds = sum(1 for item in render_package.get("rounds", []) if isinstance(item, dict) and item.get("integration_status") == "blocked")
    if blocked_rounds:
        top_content = "fail"
    if publish_profile == "legacy":
        top_delivery = "fail" if "fail" in round_delivery_statuses else "pass"
    elif publish_profile == "strict_av":
        top_delivery = "pass" if top_media == "pass" and top_content in {"pass", "degraded"} and not blocked_rounds else "fail"
    elif publish_profile == "strict_c":
        top_delivery = "pass" if top_media == "pass" and top_content == "pass" and not blocked_rounds else "fail"
    else:
        top_delivery = "fail"
    manifest = {
        "phase4_execution_contract_version": 2,
        "sync_schema_version": 2,
        "publish_profile": publish_profile,
        "render_package_artifact_identity": str(render_package.get("artifact_identity") or ""),
        "media_sync_status": top_media,
        "content_gate_status": top_content,
        "delivery_status": top_delivery,
        "media_checks": {
            "source_identity": "pass" if source_sha256 else "not_checked",
            "decoded_pts_monotone": (
                "pass" if top_media == "pass" else
                "fail" if top_media == "fail" else
                "not_checked"
            ),
            "clip_boundary": (
                "pass" if top_media == "pass" else
                "fail" if top_media == "fail" else
                "not_checked"
            ),
            "audio_within_slot": (
                "fail" if "fail" in round_media_statuses else
                "pass" if top_media == "pass" else
                "not_checked"
            ),
            "canvas_bounds": (
                "fail" if "fail" in round_media_statuses else
                "pass" if top_media == "pass" else
                "not_checked"
            ),
            "subtitle_within_audio": "not_required",
        },
        "content_checks": {
            "package_status": str(render_package.get("package_status") or ""),
            "text_sources": sorted(all_sources),
            "fact_check_scope": str((render_package.get("content_policy") or {}).get("fact_check_scope") or "disabled"),
            "blocked_rounds": blocked_rounds,
        },
        "rounds": records,
        "total_rounds": len(match.rounds),
        "skipped_rounds": skipped_count,
        "output_dir": str(output_dir),
        "render_package_path": str(render_package_path),
        "source_probe": source_probe_summary,
    }
    if not dry_run:
        save_match(output_rounds_path, match)
        write_json(manifest_path, manifest)
    return manifest


def run_phase4(
    *,
    rounds_path: Path,
    commentary_path: Path | None,
    render_package_path: Path | None = None,
    output_rounds_path: Path,
    manifest_path: Path,
    config_path: Path,
    dry_run: bool = False,
    progress_sink=None,
) -> dict:
    from audio_service.gpt_sovits_client import read_config as read_tts_config
    from audio_service.gpt_sovits_client import synthesize_emotional
    from audio_service.gpt_sovits_client import tts_cache_fingerprint

    config = load_config(config_path)
    match = load_match(rounds_path)

    phase4_cfg = config.get("phase4", {}) if isinstance(config.get("phase4", {}), dict) else {}
    publish_profile = str(phase4_cfg.get("publish_profile", "legacy"))
    if render_package_path is not None:
        return _run_phase4_execution_v2(
            match=match,
            render_package_path=render_package_path,
            commentary_path=commentary_path,
            output_rounds_path=output_rounds_path,
            manifest_path=manifest_path,
            config=config,
            dry_run=dry_run,
            progress_sink=progress_sink,
        )
    if publish_profile in {"strict_av", "strict_c", "broadcast"}:
        raise ValueError(f"{publish_profile} Phase4 requires commentary_render_package_v2")

    # ── v2/v3 双契约分流：commentary_schema_version=3 走配音任务单；否则走 v2 单稿路径 ──
    commentary_payload = read_json(commentary_path)
    voice_task_contract_v3 = (
        isinstance(commentary_payload, dict) and commentary_payload.get("commentary_schema_version") == 3
    )
    if voice_task_contract_v3:
        plans_by_round = _build_voice_task_plans(commentary_payload, read_json(rounds_path), match)
        silent_reason_by_round = {}
    else:
        scenes_by_round, silent_reason_by_round = _load_commentary_scenes(commentary_path, match)

    # ── TTS 配置 ──
    tts_runtime_path = require_path(
        config.get("phase4", {}).get("tts_config", "audio_service/gpt_sovits_runtime.yaml"),
        "phase4.tts_config",
    )
    tts_runtime = read_tts_config(tts_runtime_path) if not dry_run else {}

    # ── Phase4 配置（从 pipeline.yaml phase4 节读取，缺则用默认值）──
    p4 = config.get("phase4", {})
    output_dir = resolve_path(p4.get("output_dir", "output/sbmachine/rounds")) or Path("output/sbmachine/rounds")
    tts_cache_dir = resolve_path(p4.get("tts_cache_dir", "output/tts_cache")) or Path("output/tts_cache")
    clip_cache_dir = resolve_path(p4.get("clip_cache_dir", "output/clips")) or Path("output/clips")
    comm_vol = float(p4.get("commentary_volume", 1.0))
    game_vol = float(p4.get("game_audio_volume", 0.25))
    sample_rate = int(p4.get("sample_rate", 32000))
    if sample_rate <= 0:
        raise ValueError("phase4.sample_rate must be positive")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if voice_task_contract_v3 and not dry_run:
        _check_task_profiles(plans_by_round, tts_runtime, sample_rate)

    # ── 源视频（可选，用于逐局视频混音）──
    source_video = resolve_path(getattr(match, "video_path", None))

    records = []
    skipped_count = 0
    source_video_sha256: str | None = None

    for completed, round_record in enumerate(tqdm(match.rounds, desc="Phase4 TTS+Mux", unit="round"), start=1):
        rno = round_record.round_no
        if voice_task_contract_v3:
            plans = plans_by_round[rno]
            scenes = [plan.scene for plan in plans]
        else:
            scenes = scenes_by_round[rno]
        audio_path = output_dir / f"round_{rno:03d}.wav"
        round_duration = round_record.end_sec - round_record.start_sec
        if round_duration <= 0:
            raise ValueError(f"round {rno} must have start_sec < end_sec")
        skipped = not scenes
        if skipped:
            skipped_count += 1

        scene_inputs: list[tuple[Path, float, float]] = []
        scene_records: list[dict] = []
        if voice_task_contract_v3 and not skipped:
            # ── v3：逐 task 惰性选择（§10.3），只合成需要的候选 ──
            for scene_index, plan in enumerate(plans, start=1):
                scene_audio_path = output_dir / f"round_{rno:03d}_scene_{scene_index:03d}.wav"
                selection = _select_voice_variant(
                    plan,
                    tts_runtime=tts_runtime,
                    cache_dir=tts_cache_dir,
                    scene_audio_path=scene_audio_path,
                    synthesize_emotional=synthesize_emotional,
                    fingerprint_fn=tts_cache_fingerprint,
                    dry_run=dry_run,
                )
                scene_record = {
                    "source_start_sec": plan.slot_start_sec,
                    "source_end_sec": plan.slot_end_sec,
                    "relative_start_sec": plan.slot_start_sec - round_record.start_sec,
                    "audio_duration_sec": selection["actual_duration_sec"],
                    "audio_path": selection["audio_path"],
                    "cache_fingerprint": selection["cache_fingerprint"],
                    **{key: selection[key] for key in _SELECTION_RESULT_KEYS},
                }
                scene_records.append(scene_record)
                if selection["fit_state"] == "fit" and not dry_run:
                    scene_inputs.append(
                        (Path(selection["audio_path"]), scene_record["relative_start_sec"], plan.slot_duration_sec)
                    )
        else:
            # ── v2：现有单文本路径，行为不变 ──
            for scene_index, scene in enumerate(scenes, start=1):
                relative_start = scene["t_start"] - round_record.start_sec
                window_duration = scene["t_end"] - scene["t_start"]
                scene_audio_path = output_dir / f"round_{rno:03d}_scene_{scene_index:03d}.wav"
                fingerprint = tts_cache_fingerprint(tts_runtime, f"[{scene['emotion']}]{scene['text']}") if not dry_run else None
                overage = float(scene.get("budget_overage", 1.0) or 1.0)
                if not dry_run:
                    _synthesize_with_cache(
                        tts_runtime,
                        f"[{scene['emotion']}]{scene['text']}",
                        fingerprint,
                        scene_audio_path,
                        tts_cache_dir,
                        synthesize_emotional,
                        budget_overage=overage,
                    )
                    scene_inputs.append((scene_audio_path, relative_start, window_duration))
                scene_records.append({
                    "source_start_sec": scene["t_start"],
                    "source_end_sec": scene["t_end"],
                    "relative_start_sec": relative_start,
                    "audio_duration_sec": None,
                    "audio_path": str(scene_audio_path),
                    "cache_fingerprint": fingerprint,
                })

        # ── 固定 slot：任一 task 失败则本回合不写最终 WAV/MP4（§10.3 步骤 7/8）──
        round_unfit = voice_task_contract_v3 and not skipped and any(
            scene_record.get("fit_state") == "render_unfit" for scene_record in scene_records
        )
        if not dry_run and not round_unfit:
            durations = _assemble_scene_wav(
                scene_inputs,
                audio_path,
                round_duration,
                default_sample_rate=sample_rate,
            )
            for scene_record, duration in zip(scene_records, durations):
                scene_record["audio_duration_sec"] = duration
            _, _, round_audio_duration = _wav_info(audio_path)
        elif not dry_run:
            round_audio_duration = None
        else:
            round_audio_duration = round_duration
        round_record.phase4_audio = AudioData(
            audio_path=str(audio_path) if not round_unfit else "",
            duration_sec=round_audio_duration,
        )

        if voice_task_contract_v3 and not skipped:
            round_record.scenes = [_rounds_final_scene(scene_record) for scene_record in scene_records]

        # 逐局视频混音（需要 make_video=true、源视频和该局时间戳）
        video_path: str | None = None
        if (
            p4.get("make_video", False)
            and source_video is not None
            and scenes
            and not dry_run
            and not round_unfit
        ):
            existing_clip = _resolve_existing_clip(round_record, clip_cache_dir)
            if existing_clip is not None:
                print(f"[phase4] 复用 clip: {existing_clip}")
                clip_path = existing_clip
            else:
                if source_video_sha256 is None:
                    source_video_sha256 = _sha256_file(source_video)
                clip_path = _clip_for_round(
                    round_record,
                    source_video,
                    clip_cache_dir,
                    source_sha256=source_video_sha256,
                )
            out_mp4 = output_dir / f"round_{rno:03d}.mp4"
            _mux_round_video(clip_path, audio_path, out_mp4, game_vol=game_vol, comm_vol=comm_vol)
            video_path = str(out_mp4)

        record = {
            "round_no": rno,
            "audio_path": str(audio_path) if not round_unfit else "",
            "video_path": video_path,
            "skipped": skipped,
            "aligned": not round_unfit,
            "segments": scene_records,
        }
        if round_unfit:
            record["render_unfit"] = True
        # 方案 R（§6.4）：该段无解说时 manifest 标 skipped(reason=...)；
        # reason 来自 Phase3b 的 silent_reason，区分不可恢复失败与规则层静默。
        if skipped:
            record["skipped_reason"] = silent_reason_by_round.get(rno, "")
        records.append(record)
        if progress_sink is not None:
            try:
                progress_sink(completed, len(match.rounds), "round", None)
            except Exception:
                pass

    manifest = {
        "rounds": records,
        "total_rounds": len(match.rounds),
        "skipped_rounds": skipped_count,
        "output_dir": str(output_dir),
        "commentary_path": str(commentary_path),
    }
    if not dry_run:
        save_match(output_rounds_path, match)
        write_json(manifest_path, manifest)
    return manifest
