"""Cloud Phase3a runner: one request per round over rule-layer window projections."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from sbmachine.commentary_planner import PlannerState
from sbmachine.common import load_config, resolve_path
from sbmachine.hype_score import _compute_char_budget, _scene_hype, _scene_scream_eligible, _speech_rate_config, compute_hype, dominant_round_emotion
from sbmachine.neutral_contract import CLOUD_PHASE3A_MODE, new_manifest_metadata, rounds_sha256, validate_neutral_manifest
from sbmachine.phase3a_cloud_api import generate_cloud_round
from sbmachine.phase3a_cloud_payload import build_cloud_round_payload
from sbmachine.phase3a_cloud_prompt import build_cloud_round_prompt, cloud_system_prompt, parse_cloud_response
from sbmachine.phase3a_prompt import validate_neutral_semantics
from sbmachine.phase3a_payload import _semantic_payload, load_semantic_frames
from sbmachine.llm_projection import build_rule_state_delta, merge_required_fact_anchors
from sbmachine.scene_context import build_scene_contexts
from sbmachine.schemas import load_match
from sbmachine.tactic_book import load_tactic_book
from sbmachine.tactic_projection import build_window_rule_projection


def _video_time(frame: dict) -> float:
    return float((frame.get("when") or {}).get("video_time", 0.0))


def _validate_response(response: dict, windows: dict[str, dict]) -> dict:
    neutral = response["neutral"].strip()
    window_id = response.get("window_id")
    if window_id is not None and not isinstance(window_id, str):
        raise ValueError("cloud analyst window_id is invalid")
    if not neutral:
        if window_id:
            raise ValueError("cloud analyst silence must not select a window")
        return {"neutral": "", "window_id": None}
    if not window_id or window_id not in windows:
        raise ValueError("cloud analyst non-empty neutral must select a known window")
    selected = windows[window_id]
    char_limit = selected.get("character_limit", 100)
    if len(neutral) > char_limit:
        raise ValueError("cloud analyst neutral exceeds window character_limit")
    semantic_error, detail = validate_neutral_semantics(neutral, selected)
    if semantic_error is not None:
        raise ValueError(f"{semantic_error}: {detail}")
    return {"neutral": neutral, "window_id": window_id}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def run_cloud_phase3a(*, rounds_path: Path, output_path: Path, config_path: Path, dry_run: bool = False) -> dict:
    """One cloud call per round; cloud only sees projected windows and selects one id."""
    source_hash = rounds_sha256(rounds_path)
    config = load_config(config_path)
    semantic = config.get("semantic", {}) if isinstance(config.get("semantic"), dict) else {}
    llm_cfg = dict(config.get("llm", {}))
    model = semantic.get("analyst_model") or semantic.get("model")
    if model:
        llm_cfg["model"] = model
    if semantic.get("analyst_request_interval_sec"):
        llm_cfg["request_interval_sec"] = float(semantic["analyst_request_interval_sec"])
    llm_cfg["temperature"] = float(semantic.get("cloud_analyst_temperature", 0.2))
    llm_cfg["top_p"] = float(semantic.get("cloud_analyst_top_p", 0.9))
    max_tokens = int(semantic.get("cloud_analyst_max_tokens", 2048))
    match = load_match(rounds_path)
    configured_semantic_path = resolve_path(config.get("paths", {}).get("rounds_with_yolo_semantic_json"))
    semantic_path = configured_semantic_path or rounds_path.with_name("rounds_with_yolo_semantic.json")
    if configured_semantic_path is not None and not semantic_path.is_file():
        raise ValueError(f"Phase3a semantic input is missing: {semantic_path}")
    semantic_frames = load_semantic_frames(semantic_path)
    window_max_sec = float(semantic.get("window_max_sec", 10.0))
    window_min_sec = float(semantic.get("window_min_sec", 3.0))
    speech_rate = _speech_rate_config(config)
    system_prompt = cloud_system_prompt()
    tactic_book = load_tactic_book(match.map_name)
    output_rounds = []
    window_count = 0

    for round_record in match.rounds:
        planning_frames = _semantic_payload(round_record, external_frames=semantic_frames.get(round_record.round_no)).get("keyframes", [])
        hypes = compute_hype(planning_frames)
        peak_hype = max(hypes) if hypes else 0.0
        avg_hype = round(sum(hypes) / len(hypes), 3) if hypes else 0.0
        planner_state = PlannerState()
        reported_rule_state: dict[str, dict] = {}
        windows = build_scene_contexts(planning_frames, round_record.start_sec, round_record.end_sec, window_max_sec=window_max_sec, window_min_sec=window_min_sec, runtime_config=config)
        public_windows: list[dict] = []
        plans_by_id: dict[str, dict] = {}
        for index, window in enumerate(windows, start=1):
            is_last = index == len(windows)
            own = [frame for frame in planning_frames if window.t_start <= _video_time(frame) and (_video_time(frame) <= window.t_end if is_last else _video_time(frame) < window.t_end)]
            context = [frame for frame in planning_frames if window.context_start <= _video_time(frame) and (_video_time(frame) <= window.context_end if is_last else _video_time(frame) < window.context_end)]
            emotion = dominant_round_emotion(_scene_hype(planning_frames, hypes, window.t_start, window.t_end))
            character_limit = min(
                100,
                max(1, _compute_char_budget(max(1.0, window.t_end - window.t_start), emotion, speech_rate)),
            )
            projection = build_window_rule_projection(
                match.map_name, window, own, context, planning_frames, planner_state, tactic_book,
                is_last_window=is_last,
                char_budget=character_limit,
            )
            plan = projection.plan
            item = {
                "t_start": window.t_start, "t_end": window.t_end, "scene": window.scene,
                "commentary_plan": plan,
                "character_limit": character_limit,
            }
            rule_state = build_rule_state_delta(own, reported_rule_state)
            if rule_state is not None:
                item["rule_state"] = rule_state
            public_windows.append(item)
            plans_by_id[f"window-{index}"] = plan
        payload, safe_windows = build_cloud_round_payload(round_no=round_record.round_no, map_name=match.map_name, windows=public_windows)
        window_count += len(public_windows)
        if dry_run or not payload["windows"]:
            selected = {"neutral": "", "window_id": None}
        else:
            raw = generate_cloud_round(
                build_cloud_round_prompt(payload), llm_cfg, system_prompt=system_prompt, max_tokens=max_tokens,
                log_ctx={"round": f"round{round_record.round_no}", "scene": "round_windows"},
            )
            selected = _validate_response(parse_cloud_response(raw), safe_windows)
        scenes = []
        if selected["neutral"]:
            selected_window = safe_windows[selected["window_id"]]
            start, end = selected_window["t_start"], selected_window["t_end"]
            emotion = dominant_round_emotion(_scene_hype(planning_frames, hypes, start, end))
            scenes.append({
                "t_start": start, "t_end": end, "context_start": start, "context_end": end,
                "scene": selected_window["scene"],
                "window_id": f"r{round_record.round_no:03d}_w{int(selected_window['order']):02d}",
                "actions": [action.get("type") for action in selected_window["selected_actions"]],
                "commentary_plan": plans_by_id[selected["window_id"]], "neutral": selected["neutral"],
                "neutral_source": "llm", "generation_status": "success",
                "fact_anchors": merge_required_fact_anchors(selected_window.get("required_facts")),
                "hype": _scene_hype(planning_frames, hypes, start, end),
                "scream_eligible": _scene_scream_eligible(planning_frames, start, end),
                "char_budget": _compute_char_budget(max(1.0, end - start), emotion, speech_rate),
            })
        output_rounds.append({
            "round_no": round_record.round_no, "start_sec": round_record.start_sec, "end_sec": round_record.end_sec,
            "demo_round_hint": round_record.demo_round_hint, "round_emotion": dominant_round_emotion(peak_hype),
            "peak_hype": peak_hype, "avg_hype": avg_hype, "analyst_failed": False, "scenes": scenes,
        })

    if rounds_sha256(rounds_path) != source_hash:
        raise RuntimeError("rounds_with_yolo changed while cloud analysis was running")
    manifest = {
        **new_manifest_metadata(rounds_path), "phase3a_mode": CLOUD_PHASE3A_MODE,
        "video_path": match.video_path, "map_name": match.map_name,
        "model": llm_cfg.get("model", ""), "rounds": output_rounds,
    }
    if dry_run:
        return {
            "mode": "phase3a_dry_run",
            "writes_performed": False,
            "publish_path": None,
            "rounds": len(output_rounds),
            "windows": window_count,
            "fallback_windows": 0,
        }
    validate_neutral_manifest(manifest, rounds_path)
    _atomic_write_json(output_path, manifest)
    return manifest
