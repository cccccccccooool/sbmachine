"""Phase 3a 分析模型：确定性时间窗 -> 每窗一条中性稿。"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from sbmachine.common import load_config, resolve_backend, resolve_path, write_json
from sbmachine.hype_score import _compute_char_budget, _scene_hype, _scene_scream_eligible, _speech_rate_config, compute_hype, dominant_round_emotion
from sbmachine.llm_shim import accept_api_response
from sbmachine.phase3a_payload import _semantic_payload, _slim_frame_for_prompt, build_state_block, load_semantic_frames
from sbmachine.phase3a_prompt import _build_analyst_system, _build_window_prompt, _parse_window_neutral_response
from sbmachine.neutral_contract import new_manifest_metadata
from sbmachine.commentary_planner import PlannerState, fallback_neutral, plan_window
from sbmachine.scene_context import build_scene_contexts
from sbmachine.schemas import load_match

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ANALYST_FAILED = "__ANALYST_FAILED__"
_MAX_WINDOW_FRAMES = 5
_MIN_WINDOW_FRAMES = 2


def _call_analyst(
    prompt: str,
    llm_cfg: dict,
    gen_fn,
    system_prompt: str | None = None,
    round_no: int = 0,
    debug: bool = False,
    seg: int = 0,
    max_tokens: int | None = None,
) -> str:
    try:
        log_ctx = {"round": f"round{round_no}"}
        if seg > 0:
            log_ctx["scene"] = f"win{seg}"
        result = gen_fn(prompt, llm_cfg, system_prompt=system_prompt, max_tokens=max_tokens, log_ctx=log_ctx)
    except Exception as exc:
        print(f"[phase3a] round {round_no} analyst error: {exc}", file=sys.stderr)
        return _ANALYST_FAILED

    if debug:
        debug_dir = _PROJECT_ROOT / "output" / "debug_phase3"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dump = {
            "round_no": round_no,
            "seg": seg,
            "model": llm_cfg.get("model", ""),
            "phase": "3a_analyst_window",
            "prompt": prompt,
            "response": result,
        }
        name = f"r{round_no:03d}_w{seg:02d}_3a_analyst.json" if seg else f"r{round_no:03d}_3a_analyst.json"
        (debug_dir / name).write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _video_time(frame: dict) -> float:
    return float((frame.get("when") or {}).get("video_time", 0.0))


def _evenly_sample(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return list(items)
    if limit <= 0:
        return []
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def _is_priority_frame(frame: dict) -> bool:
    ev = frame.get("events") or {}
    c4 = ev.get("c4") or {}
    return bool(ev.get("kills") or c4.get("planted") or c4.get("begin_defuse_tick"))


def _is_context_frame(frame: dict) -> bool:
    ev = frame.get("events") or {}
    return bool(
        ev.get("damages")
        or ev.get("flashes")
        or ev.get("smokes_active")
        or ev.get("infernos_active")
    )


def _select_window_frames(frames: list[dict], lo: float, hi: float, *, is_last: bool) -> list[dict]:
    selected: list[dict] = []
    for frame in frames:
        t = _video_time(frame)
        upper_ok = t <= hi if is_last else t < hi
        if lo <= t and upper_ok:
            selected.append(frame)

    if len(selected) < _MIN_WINDOW_FRAMES and frames:
        mid = (lo + hi) / 2.0
        existing = {_video_time(frame) for frame in selected}
        for frame in sorted(frames, key=lambda item: abs(_video_time(item) - mid)):
            if _video_time(frame) in existing:
                continue
            selected.append(frame)
            existing.add(_video_time(frame))
            if len(selected) >= _MIN_WINDOW_FRAMES:
                break

    selected.sort(key=_video_time)
    if len(selected) <= _MAX_WINDOW_FRAMES:
        return selected

    priority = [frame for frame in selected if _is_priority_frame(frame)]
    priority_times = {_video_time(frame) for frame in priority}
    if len(priority) >= _MAX_WINDOW_FRAMES:
        return _evenly_sample(priority, _MAX_WINDOW_FRAMES)

    context = [
        frame
        for frame in selected
        if _video_time(frame) not in priority_times and _is_context_frame(frame)
    ]
    keep = priority + _evenly_sample(context, _MAX_WINDOW_FRAMES - len(priority))
    keep_times = {_video_time(frame) for frame in keep}
    if len(keep) < _MAX_WINDOW_FRAMES:
        rest = [frame for frame in selected if _video_time(frame) not in keep_times]
        keep.extend(_evenly_sample(rest, _MAX_WINDOW_FRAMES - len(keep)))
    return sorted(keep, key=_video_time)


def _build_window_payload(payload: dict, frames: list[dict], plan: dict) -> dict:
    out = {k: payload[k] for k in ("round_no", "demo_round_hint") if k in payload}
    ownership = plan["ownership"]
    out["t_start"] = ownership["t_start"]
    out["t_end"] = ownership["t_end"]
    out["scene"] = plan["scene"]
    out["commentary_plan"] = plan
    out["context_frames"] = [_slim_frame_for_prompt(frame) for frame in frames]
    return out


def run_phase3a(
    *,
    rounds_path: Path,
    output_path: Path,
    config_path: Path,
    dry_run: bool = False,
) -> dict:
    import os

    config = load_config(config_path)
    debug_enabled = bool(config.get("debug", {}).get("phase3", False) or os.getenv("AI6657_DEBUG_PHASE3"))
    llm_cfg = dict(config.get("llm", {}))
    semantic_cfg = config.get("semantic", {}) if isinstance(config.get("semantic", {}), dict) else {}
    analyst_model = semantic_cfg.get("analyst_model") or semantic_cfg.get("model", "")
    if analyst_model:
        llm_cfg["model"] = analyst_model
    llm_cfg["max_tokens"] = int(semantic_cfg.get("analyst_max_tokens", 256))
    llm_cfg["temperature"] = float(semantic_cfg.get("analyst_temperature", 0.3))
    llm_cfg["repeat_penalty"] = float(semantic_cfg.get("analyst_repeat_penalty", 1.3))
    llm_cfg["top_p"] = float(semantic_cfg.get("analyst_top_p", 0.9))

    backend = resolve_backend(config, "analyst")
    if backend not in {"api", "vllm"}:
        raise ValueError(f"unsupported analyst backend: {backend}; use vllm or api")
    from sbmachine import llma_api as _llma_backend
    gen_fn = _llma_backend.generate

    match = load_match(rounds_path)
    # 物理帧来源是 phase2 产出的精简 DEM-fact sidecar；
    # rounds_with_yolo.json 仍通过 load_match 作为契约/身份锚点。
    configured_semantic_path = resolve_path(
        config.get("paths", {}).get("rounds_with_yolo_semantic_json")
    )
    semantic_path = configured_semantic_path or rounds_path.with_name("rounds_with_yolo_semantic.json")
    if configured_semantic_path is not None and not semantic_path.is_file():
        raise ValueError(f"Phase3a semantic input is missing: {semantic_path}")
    semantic_frames = load_semantic_frames(semantic_path)

    concurrent_rounds = max(1, int(semantic_cfg.get("analyst_concurrent_rounds", 1)))
    analyst_system = _build_analyst_system()
    window_max_sec = float(semantic_cfg.get("window_max_sec", 10.0))
    window_min_sec = float(semantic_cfg.get("window_min_sec", 3.0))
    speech_rate = _speech_rate_config()

    def _process_round(rnd) -> tuple[dict, list[tuple[object, str]]]:
        payload = _semantic_payload(rnd, external_frames=semantic_frames.get(rnd.round_no))
        beats = payload.get("keyframes", [])
        hypes = compute_hype(beats)
        peak_hype = max(hypes) if hypes else 0.0
        avg_hype = round(sum(hypes) / len(hypes), 3) if hypes else 0.0
        round_emotion = dominant_round_emotion(peak_hype)

        windows = build_scene_contexts(
            beats, rnd.start_sec, rnd.end_sec,
            window_max_sec=window_max_sec,
            window_min_sec=window_min_sec,
        )
        scenes_out: list[dict] = []
        failed_windows = 0
        planner_state = PlannerState()
        reported_state: dict[str, dict] = {}
        accepted_samples: list[tuple[object, str]] = []
        llma_inputs: list[dict] = []

        for idx, window in enumerate(windows, start=1):
            t_start, t_end, scene_label = window.t_start, window.t_end, window.scene
            is_last = idx == len(windows)
            ownership_frames = [frame for frame in beats if t_start <= _video_time(frame) and (_video_time(frame) <= t_end if is_last else _video_time(frame) < t_end)]
            context_frames = [frame for frame in beats if window.context_start <= _video_time(frame) and (_video_time(frame) <= window.context_end if is_last else _video_time(frame) < window.context_end)]
            win_frames = _select_window_frames(ownership_frames or beats, t_start, t_end, is_last=is_last)
            plan = plan_window(match.map_name, window, ownership_frames, context_frames, beats, planner_state)
            state_block = build_state_block(ownership_frames, reported_state)
            sc_hype = _scene_hype(beats, hypes, t_start, t_end)
            sc_emotion = dominant_round_emotion(sc_hype)
            char_budget = _compute_char_budget(max(1.0, t_end - t_start), sc_emotion, speech_rate)

            window_payload = _build_window_payload(payload, win_frames, plan)
            # LLM-A 输入定版：plan + 增量状态报告 + 氛围帧，作为中间产物完整留档。
            llma_inputs.append({
                "seg": idx, "scene": scene_label,
                "state_block": state_block,
                **window_payload,
            })

            if dry_run:
                neutral = fallback_neutral(plan)
            else:
                accepted_response = None
                accepted_output = None
                prompt = _build_window_prompt(
                    window_payload,
                    t_start=t_start, t_end=t_end, scene=scene_label,
                    state_block=state_block,
                )
                raw = _call_analyst(
                    prompt, llm_cfg, gen_fn, system_prompt=analyst_system,
                    round_no=rnd.round_no, debug=debug_enabled, seg=idx,
                    max_tokens=int(llm_cfg.get("max_tokens", 256)),
                )
                if raw == _ANALYST_FAILED:
                    failed_windows += 1
                    neutral = fallback_neutral(plan)
                else:
                    parsed = _parse_window_neutral_response(raw, debug=debug_enabled)
                    if parsed is None:
                        failed_windows += 1
                        neutral = fallback_neutral(plan)
                    else:
                        neutral = parsed[:120]
                        if neutral:
                            accepted_response = raw
                            accepted_output = json.dumps(
                                {"neutral": neutral}, ensure_ascii=False, separators=(",", ":"),
                            )

            if not neutral.strip() and plan.get("selected_actions"):
                neutral = fallback_neutral(plan)

            if not dry_run and accepted_response is not None and accepted_output is not None:
                accepted_samples.append((accepted_response, accepted_output))

            scenes_out.append({
                "t_start": t_start, "t_end": t_end,
                "context_start": window.context_start, "context_end": window.context_end,
                "scene": scene_label,
                "actions": [action.get("type") for action in plan.get("selected_actions", [])],
                "commentary_plan": plan,
                "neutral": neutral, "hype": sc_hype,
                "scream_eligible": _scene_scream_eligible(beats, t_start, t_end),
                "char_budget": char_budget,
            })

        analyst_failed = bool(scenes_out) and failed_windows == len(scenes_out) and not any(
            str(scene.get("neutral", "")).strip() for scene in scenes_out
        )
        result = {
            "round_no": rnd.round_no,
            "start_sec": rnd.start_sec,
            "end_sec": rnd.end_sec,
            "demo_round_hint": rnd.demo_round_hint,
            "round_emotion": round_emotion,
            "peak_hype": peak_hype,
            "avg_hype": avg_hype,
            "analyst_failed": analyst_failed,
            "scenes": scenes_out,
        }
        llma_input_round = {"round_no": rnd.round_no, "windows": llma_inputs}
        return result, (accepted_samples if not analyst_failed else []), llma_input_round

    result_rounds = []
    llma_input_rounds = []
    accepted_run_samples: list[tuple[object, str]] = []
    with ThreadPoolExecutor(max_workers=concurrent_rounds) as pool:
        futures = {pool.submit(_process_round, rnd): rnd.round_no for rnd in match.rounds}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Phase3a llma_analyze", unit="round"):
            result, accepted_samples, llma_input_round = fut.result()
            result_rounds.append(result)
            accepted_run_samples.extend(accepted_samples)
            llma_input_rounds.append(llma_input_round)
    result_rounds.sort(key=lambda r: r["round_no"])
    llma_input_rounds.sort(key=lambda r: r["round_no"])

    # LLM-A 输入中间产物：规则集 plan + 增量状态报告 + 氛围帧的整合定版，供审计与复跑。
    llma_input_path = resolve_path(
        config.get("paths", {}).get("llma_input_json")
    ) or output_path.with_name("llma_input.json")
    write_json(llma_input_path, {"rounds": llma_input_rounds})

    if result_rounds and not dry_run:
        total = len(result_rounds)
        failed = sum(1 for r in result_rounds if r.get("analyst_failed"))
        print(f"[phase3a] analyst success {total - failed}/{total} rounds", file=sys.stderr)
        if failed == total or failed / total > 0.5:
            print(
                f"[phase3a] FATAL: {failed}/{total} rounds failed. Check LLM endpoint / API key.",
                file=sys.stderr,
            )
            sys.exit(1)

    manifest = {
        **new_manifest_metadata(rounds_path),
        "video_path": match.video_path,
        "map_name": match.map_name,
        "model": llm_cfg.get("model", ""),
        "rounds": result_rounds,
    }
    write_json(output_path, manifest)
    for response, output in accepted_run_samples:
        accept_api_response(response, output=output)
    return manifest
