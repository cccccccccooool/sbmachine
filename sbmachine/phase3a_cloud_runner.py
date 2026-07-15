"""云端 Phase3a 执行器：每回合一次结构化模型请求。"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from sbmachine.common import load_config, resolve_path
from sbmachine.hype_score import _compute_char_budget, _scene_hype, _scene_scream_eligible, _speech_rate_config, compute_hype, dominant_round_emotion
from sbmachine.neutral_contract import CLOUD_PHASE3A_MODE, new_manifest_metadata, rounds_sha256, validate_neutral_manifest
from sbmachine.phase3a_cloud_api import generate_cloud_round
from sbmachine.phase3a_cloud_payload import build_cloud_round_payload
from sbmachine.phase3a_cloud_prompt import build_cloud_round_prompt, cloud_system_prompt, parse_cloud_response
from sbmachine.phase3a_payload import _semantic_payload, load_semantic_frames
from sbmachine.schemas import load_match


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"cloud analyst {label} must be a finite number")
    return float(value)


def _validate_response(response: dict, evidence: dict[str, dict], start: float, end: float) -> dict:
    neutral = response["neutral"].strip()
    ids = response["evidence_ids"]
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("cloud analyst evidence_ids are invalid")
    if len(neutral) > 100:
        raise ValueError("cloud analyst neutral exceeds 100 characters")
    if not neutral:
        if ids:
            raise ValueError("cloud analyst silence must not cite evidence")
        return {"neutral": "", "evidence_ids": [], "t_start": start, "t_end": end}
    if not ids:
        raise ValueError("cloud analyst non-empty neutral must cite evidence")
    unknown = [item for item in ids if item not in evidence]
    if unknown:
        raise ValueError(f"cloud analyst cited unknown evidence: {unknown[0]}")
    # 时间范围由本地根据引用的证据推导，不信任模型给出的时间。
    evidence_times = [float(evidence[item]["t"]) for item in ids]
    lo = max(start, min(evidence_times) - 0.25)
    hi = min(end, max(evidence_times) + 0.25)
    if not lo < hi:
        lo, hi = start, end
    return {"neutral": neutral, "evidence_ids": ids, "t_start": lo, "t_end": hi}


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
    """每回合最多生成一条中性稿场景，不做本地话题规划。"""
    source_hash = rounds_sha256(rounds_path)
    config = load_config(config_path)
    semantic = config.get("semantic", {}) if isinstance(config.get("semantic"), dict) else {}
    llm_cfg = dict(config.get("llm", {}))
    model = semantic.get("analyst_model") or semantic.get("model")
    if model:
        llm_cfg["model"] = model
    llm_cfg["temperature"] = float(semantic.get("cloud_analyst_temperature", 0.2))
    llm_cfg["top_p"] = float(semantic.get("cloud_analyst_top_p", 0.9))
    max_tokens = int(semantic.get("cloud_analyst_max_tokens", 2048))
    match = load_match(rounds_path)
    # 物理帧来源是 phase2 产出的精简 DEM-fact sidecar；
    # rounds_with_yolo.json 仍通过 load_match 作为契约/身份锚点。
    configured_semantic_path = resolve_path(config.get("paths", {}).get("rounds_with_yolo_semantic_json"))
    semantic_path = configured_semantic_path or rounds_path.with_name("rounds_with_yolo_semantic.json")
    if configured_semantic_path is not None and not semantic_path.is_file():
        raise ValueError(f"Phase3a semantic input is missing: {semantic_path}")
    semantic_frames = load_semantic_frames(semantic_path)
    speech_rate = _speech_rate_config()
    system_prompt = cloud_system_prompt()
    output_rounds = []

    for round_record in match.rounds:
        planning_frames = _semantic_payload(
            round_record, external_frames=semantic_frames.get(round_record.round_no)
        ).get("keyframes", [])
        hypes = compute_hype(planning_frames)
        peak_hype = max(hypes) if hypes else 0.0
        avg_hype = round(sum(hypes) / len(hypes), 3) if hypes else 0.0
        payload, evidence = build_cloud_round_payload(round_record, match.map_name)
        if dry_run:
            selected = {"neutral": "", "evidence_ids": [], "t_start": round_record.start_sec, "t_end": round_record.end_sec}
        else:
            raw = generate_cloud_round(
                build_cloud_round_prompt(payload), llm_cfg, system_prompt=system_prompt, max_tokens=max_tokens,
                log_ctx={"round": f"round{round_record.round_no}", "scene": "round_timeline"},
            )
            selected = _validate_response(parse_cloud_response(raw), evidence, round_record.start_sec, round_record.end_sec)
        scenes = []
        if selected["neutral"]:
            emotion = dominant_round_emotion(_scene_hype(planning_frames, hypes, selected["t_start"], selected["t_end"]))
            scenes.append({
                "t_start": selected["t_start"], "t_end": selected["t_end"],
                "context_start": selected["t_start"], "context_end": selected["t_end"],
                "scene": "cloud_round_focus",
                "actions": [evidence[item]["type"] for item in selected["evidence_ids"]],
                "commentary_plan": {"version": 2, "source": CLOUD_PHASE3A_MODE, "evidence_ids": selected["evidence_ids"]},
                "neutral": selected["neutral"],
                "hype": _scene_hype(planning_frames, hypes, selected["t_start"], selected["t_end"]),
                "scream_eligible": _scene_scream_eligible(planning_frames, selected["t_start"], selected["t_end"]),
                "char_budget": _compute_char_budget(max(1.0, selected["t_end"] - selected["t_start"]), emotion, speech_rate),
            })
        output_rounds.append({
            "round_no": round_record.round_no, "start_sec": round_record.start_sec, "end_sec": round_record.end_sec,
            "demo_round_hint": round_record.demo_round_hint, "round_emotion": dominant_round_emotion(peak_hype),
            "peak_hype": peak_hype, "avg_hype": avg_hype, "analyst_failed": False, "scenes": scenes,
        })

    if rounds_sha256(rounds_path) != source_hash:
        raise RuntimeError("rounds_with_yolo changed while cloud analysis was running")
    manifest = {
        **new_manifest_metadata(rounds_path),
        "phase3a_mode": CLOUD_PHASE3A_MODE,
        "video_path": match.video_path,
        "map_name": match.map_name,
        "model": llm_cfg.get("model", ""),
        "rounds": output_rounds,
    }
    validate_neutral_manifest(manifest, rounds_path)
    _atomic_write_json(output_path, manifest)
    return manifest
