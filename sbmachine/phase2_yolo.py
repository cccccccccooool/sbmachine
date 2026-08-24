"""Phase 2：为每个回合构建 DEM 时间线，并附加 YOLO/OCR 事实。"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from sbmachine.common import load_config, resolve_path
from sbmachine.demo_query import DemoQuery
from sbmachine.phase2_background import _demo_round_hint, build_background_info, resolve_demo_round_hints
from tools.debug.phase2 import DebugWriter
from sbmachine.phase2_ocr import (
    OcrConsensus,
    ScorePairConsensus,
    _detect_pov_ocr,
    _detect_score_ocr,
    _first_timer_region,
    _resolve_ocr_box,
    _score_regions,
    normalize_score_observation,
    parse_timer_observation,
    read_ocr_text,
    unsampled_timer_observation,
)
from sbmachine.phase2_quality import OcrBudget, build_alignment_warning, coalesce_yolo_gaps
from sbmachine.phase2_timeline import build_timeline
from sbmachine.phase2_yolo_gate import YoloUiDetector
from sbmachine.schemas import KeyFrame, YoloData, load_match, save_match
from sbmachine.time_align import C4VisualTracker, RoundTimeAlign


def should_sample_alignment_ocr(
    time_sec: float,
    start_sec: float,
    end_sec: float,
    *,
    initial_sec: float,
    period_sec: float,
    window_sec: float,
) -> bool:
    """先采样初始窗口，之后每隔一个周期采样一个完整窗口。"""
    elapsed = float(time_sec) - float(start_sec)
    if elapsed < 0 or time_sec > end_sec:
        return False
    if elapsed < initial_sec:
        return True
    if period_sec <= 0 or window_sec <= 0:
        return False
    offset = elapsed - initial_sec - period_sec
    if offset < 0:
        return False
    window_offset = offset % period_sec
    return window_offset < window_sec and time_sec + (window_sec - window_offset) <= end_sec + 1e-6


def should_sample_score_ocr(
    time_sec: float,
    start_sec: float,
    *,
    consensus_frames: int,
    reset_frames_remaining: int = 0,
) -> bool:
    """Sample score only for initial consensus and a short confirmed-reset window."""
    elapsed = int(round(float(time_sec) - float(start_sec)))
    return 0 <= elapsed < max(0, int(consensus_frames)) or int(reset_frames_remaining) > 0


def _empty_score_observation(video_time: float, status: str) -> dict:
    pair_status = "incomplete" if status == "no_region" else status
    return normalize_score_observation({
        "kind": "score_observation", "video_time": round(float(video_time), 3),
        "left": None, "right": None, "pair_status": pair_status,
        "observation_status": status,
        "ct": None, "t": None, "source": status, "confidence": 0.0,
        "variant_inference_calls": 0,
    }, video_time=video_time)


def _baseline_roi_calls(
    rounds: list, *, interval_sec: float, initial_sec: float, period_sec: float, window_sec: float,
) -> int:
    sampled = 0
    for record in rounds:
        for time_sec, _ in build_timeline(record.start_sec, record.end_sec, interval_sec=interval_sec):
            sampled += int(should_sample_alignment_ocr(
                time_sec, record.start_sec, record.end_sec,
                initial_sec=initial_sec, period_sec=period_sec, window_sec=window_sec,
            ))
    return sampled * 2


def run_phase2(
    *,
    rounds_path: Path,
    output_path: Path,
    config_path: Path,
    video_path: Path | None = None,
    dry_run: bool = False,
    debug_dir: Path | None = None,
    semantic_output_path: Path | None = None,
    progress_sink=None,
) -> None:
    config = load_config(config_path)
    match = load_match(rounds_path)
    actual_video = video_path or resolve_path(match.video_path)
    if actual_video is None and not dry_run:
        raise ValueError("video path is required")

    yolo_root = config.get("yolo", {})
    yolo_config = yolo_root.get("yolo", {})
    demo_config = config.get("demo", {})
    pov_ocr_config = yolo_root.get("pov_ocr", {})
    timer_ocr_config = yolo_root.get("timer_ocr", {})
    score_ocr_config = yolo_root.get("score_ocr", {})
    sampling_config = yolo_root.get("sampling", {})
    phase2_interval_sec = float(sampling_config.get("phase2_interval_sec", 1.0))
    alignment_initial_sec = float(sampling_config.get("alignment_initial_sec", 10.0))
    alignment_period_sec = float(sampling_config.get("alignment_period_sec", 20.0))
    alignment_window_sec = float(sampling_config.get("alignment_window_sec", 5.0))
    budget_ratio = float(sampling_config.get("alignment_ocr_budget_ratio", 1.0))
    degraded_extra_ratio = float(sampling_config.get("alignment_degraded_extra_ratio", 0.2))
    yolo_enabled = bool(yolo_config.get("enabled", True))
    crop_padding = int(yolo_root.get("crop_padding_px", 4))
    pov_match_min_score = float(pov_ocr_config.get("min_match_score", 0.6))
    spectator_min_frames = int(pov_ocr_config.get("spectator_min_frames", 3))
    timer_min_confidence = float(timer_ocr_config.get("min_confidence", 0.35))
    lock_min_samples = int(timer_ocr_config.get("lock_min_samples", 3))
    degraded_burst_frames = int(timer_ocr_config.get("degraded_burst_frames", 5))
    score_consensus_frames = int(score_ocr_config.get("consensus_frames", 5))
    score_reset_frames = int(score_ocr_config.get("reset_confirm_frames", 3))
    c4_config = yolo_root.get("c4_visual", {})

    dbg = DebugWriter(debug_dir)
    dbg.open()
    parsed_demo_dir = resolve_path(demo_config.get("parsed_dir", "output/demo"))
    demo = None if dry_run else DemoQuery.load(parsed_demo_dir or Path("output/demo"))
    if demo is not None:
        resolve_demo_round_hints(match.rounds, demo.rounds)
    baseline_calls = _baseline_roi_calls(
        match.rounds,
        interval_sec=phase2_interval_sec,
        initial_sec=alignment_initial_sec,
        period_sec=alignment_period_sec,
        window_sec=alignment_window_sec,
    )
    ocr_budget = OcrBudget(
        baseline_roi_calls=baseline_calls,
        normal_ratio=budget_ratio,
        degraded_extra_ratio=degraded_extra_ratio,
    )

    yolo = None
    try:
        for completed, round_record in enumerate(tqdm(match.rounds, desc="Phase2 YOLO", unit="round"), start=1):
            if dry_run:
                round_record.phase2_yolo = YoloData(
                    key_frames=[KeyFrame(
                        time_sec=round(round_record.start_sec, 3),
                        gate_reason="dry_run",
                        yolo_tags=["dry_run"],
                        has_frame=False,
                    )],
                    yolo_required=yolo_enabled,
                    yolo_model=str(yolo_config.get("model_path", "")),
                    sample_interval_sec=phase2_interval_sec,
                )
                if progress_sink is not None:
                    try:
                        progress_sink(completed, len(match.rounds), "round", None)
                    except Exception:
                        pass
                continue

            if actual_video is None or demo is None:
                raise ValueError("video path and parsed demo are required")
            demo_round_no = _demo_round_hint(round_record)
            round_meta = demo.round_by_no(demo_round_no)
            provisional_offset = round_record.align_offset
            if provisional_offset is None:
                freeze_end_tick = int(round_meta.get("freeze_end_tick", round_meta.get("start_tick", 0)))
                provisional_offset = float(freeze_end_tick) - round_record.start_sec * demo.tick_rate
            align = RoundTimeAlign(
                round_meta,
                demo.tick_rate,
                anchor_tolerance_sec=float(demo_config.get("anchor_tolerance_sec", 2.0)),
                provisional_offset=provisional_offset,
                lock_min_samples=lock_min_samples,
                residual_tolerance_sec=float(timer_ocr_config.get("residual_tolerance_sec", 2.0)),
                max_drift_ratio=float(timer_ocr_config.get("max_drift_ratio", 0.05)),
                reset_low_sec=float(timer_ocr_config.get("reset_low_sec", 10.0)),
                reset_high_sec=float(timer_ocr_config.get("reset_high_sec", 100.0)),
                reset_confirm_samples=int(timer_ocr_config.get("reset_confirm_samples", 3)),
            )
            timeline = build_timeline(
                round_record.start_sec,
                round_record.end_sec,
                interval_sec=phase2_interval_sec,
            )
            pov_consensus = OcrConsensus(
                window=int(pov_ocr_config.get("consensus_window", 3)),
                min_confidence=float(pov_ocr_config.get("min_confidence", 0.35)),
            )
            score_consensus = ScorePairConsensus(window=score_consensus_frames)
            c4_tracker = C4VisualTracker(
                window=int(c4_config.get("window_frames", 3)),
                min_present=int(c4_config.get("min_present_frames", 2)),
            )
            key_frames: list[KeyFrame] = []
            background: list[dict] = []
            frame_observations: list[dict] = []
            timer_observations: list[dict] = []
            score_observations: list[dict] = []
            c4_observations: list[dict] = []
            yolo_missing_times: list[float] = []
            total_yolo_frames = 0
            reset_score_remaining = 0
            degraded_remaining = 0
            scheduled_failures = 0
            last_timer_candidate: dict | None = None

            import cv2
            cap = cv2.VideoCapture(str(actual_video))
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, round_record.start_sec * 1000)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_step_sec = 1.0 / fps if fps > 0 else 0.0
                decoded_time = float(round_record.start_sec) - frame_step_sec
                for ts, requires_frame in timeline:
                    gate_reason = "demo_only"
                    yolo_tags: list[str] = []
                    yolo_confidence = 0.0
                    regions: list[dict] = []
                    pov_ocr = {"raw_text": "", "engine": "demo_only"}
                    timer_ocr = {"value": "", "raw_text": ""}
                    score_ocr = None
                    pov_crop_source = "demo_only"
                    timer_crop_source = ""
                    decoded = False
                    pov_region = None
                    timer_region = None
                    frame = None
                    yolo_bg = None

                    if requires_frame:
                        while decoded_time + 1e-6 < ts:
                            decoded = cap.grab()
                            if not decoded:
                                break
                            position_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                            decoded_time = position_msec / 1000.0 if position_msec > 0 else decoded_time + frame_step_sec
                        if decoded:
                            decoded, frame = cap.retrieve()
                    if requires_frame and decoded:
                        if yolo_enabled:
                            if yolo is None:
                                yolo = YoloUiDetector(yolo_config)
                            total_yolo_frames += 1
                            decision = yolo.decide(frame)
                            gate_reason = decision.reason
                            yolo_tags = decision.tags
                            yolo_confidence = decision.confidence
                            yolo_bg = dict(decision.background) if decision.background else None
                            regions = list((yolo_bg or {}).get("regions") or [])
                            timer_region = _first_timer_region(yolo_bg)
                            if decision.reason == "no_ui_yolo_signal":
                                yolo_missing_times.append(float(ts))
                            if yolo_bg:
                                yolo_bg["time_sec"] = round(ts, 3)
                        else:
                            yolo_bg = None
                            gate_reason = "fixed_roi_only"

                        base_timer_sample = should_sample_alignment_ocr(
                            ts, round_record.start_sec, round_record.end_sec,
                            initial_sec=alignment_initial_sec,
                            period_sec=alignment_period_sec,
                            window_sec=alignment_window_sec,
                        )
                        degraded_sample = degraded_remaining > 0
                        if base_timer_sample or degraded_sample:
                            timer_region, timer_crop_source = _resolve_ocr_box(
                                timer_region, timer_ocr_config, frame.shape, "yolo_timer_region"
                            )
                            if timer_region is None:
                                timer_ocr = unsampled_timer_observation(ts, "no_region", source=timer_crop_source)
                            elif ocr_budget.can_consume(1, degraded=degraded_sample):
                                raw_timer = read_ocr_text(
                                    frame, timer_region, padding=crop_padding,
                                    accept_pattern=r"^\s*\d{1,2}\s*[:\uFF1A]\s*\d{2}\s*$",
                                    min_accept_confidence=timer_min_confidence,
                                )
                                timer_ocr = parse_timer_observation(
                                    raw_timer.get("raw_text", ""), video_time=ts,
                                    confidence=float(raw_timer.get("confidence", 0.0)),
                                    roi_source=timer_crop_source,
                                    variant=str(raw_timer.get("variant", "none")),
                                    min_confidence=timer_min_confidence,
                                )
                                timer_ocr["variant_inference_calls"] = int(raw_timer.get("variant_inference_calls", 0))
                                timer_ocr["engine"] = raw_timer.get("engine", "")
                                ocr_budget.consume("timer", 1, variant_calls=timer_ocr["variant_inference_calls"])
                            else:
                                timer_ocr = unsampled_timer_observation(ts, "budget_exhausted")
                            align.observe_timer(timer_ocr)
                            timer_observations.append(timer_ocr)
                            if timer_ocr.get("parse_status") == "parsed" and timer_ocr.get("alignment_status") == "pending":
                                if (last_timer_candidate is not None
                                        and float(last_timer_candidate["timer_sec"]) <= align.reset_low_sec
                                        and float(timer_ocr["timer_sec"]) >= align.reset_high_sec):
                                    reset_score_remaining = max(reset_score_remaining, score_reset_frames)
                                    score_consensus = ScorePairConsensus(window=score_reset_frames)
                                last_timer_candidate = timer_ocr
                                scheduled_failures = 0
                            else:
                                scheduled_failures += 1
                                if base_timer_sample and scheduled_failures >= lock_min_samples:
                                    degraded_remaining = max(degraded_remaining, degraded_burst_frames)
                            if degraded_sample:
                                degraded_remaining = max(0, degraded_remaining - 1)
                        else:
                            timer_ocr = unsampled_timer_observation(ts, "not_scheduled")
                            timer_observations.append(timer_ocr)

                        score_sample = should_sample_score_ocr(
                            ts, round_record.start_sec,
                            consensus_frames=score_consensus_frames,
                            reset_frames_remaining=reset_score_remaining,
                        )
                        if score_sample:
                            score_roi_count = min(2, len(_score_regions(yolo_bg or {})))
                            if score_roi_count == 0:
                                score_ocr = _empty_score_observation(ts, "no_region")
                            elif ocr_budget.can_consume(score_roi_count, degraded=False):
                                score_raw = _detect_score_ocr(
                                    frame, yolo_bg, score_ocr_config, crop_padding, video_time=ts,
                                )
                                ocr_budget.consume(
                                    "score", score_roi_count,
                                    variant_calls=int(score_raw.get("variant_inference_calls", 0)),
                                    score_sides=tuple(score_raw.get("_regions", {}).keys()),
                                )
                                score_ocr = score_consensus.update(score_raw)
                            else:
                                score_ocr = _empty_score_observation(ts, "budget_exhausted")
                            if reset_score_remaining > 0:
                                reset_score_remaining -= 1
                        else:
                            score_ocr = _empty_score_observation(ts, "not_scheduled")
                        score_observations.append(score_ocr)
                        c4_regions = yolo_bg.get("c4_regions") if isinstance(yolo_bg, dict) else None
                        c4_observation = c4_tracker.update(ts, c4_regions)
                        c4_observations.append(c4_observation)
                        pov_sample, pov_crop_source, pov_region = _detect_pov_ocr(
                            frame, yolo_bg, pov_ocr_config, crop_padding
                        )
                        pov_ocr = pov_consensus.update(pov_sample)

                        if dbg.enabled:
                            fdir = dbg.frame_dir(demo_round_no)
                            stem = f"frame_{ts:.3f}"
                            dbg.save_crop(fdir, f"{stem}_pov_crop.png", dbg.crop_image(frame, pov_region, crop_padding))
                            dbg.save_crop(fdir, f"{stem}_timer_crop.png", dbg.crop_image(frame, timer_region, crop_padding))
                            score_regions = score_ocr.get("_regions", {}) if isinstance(score_ocr, dict) else {}
                            dbg.save_crop(
                                fdir, f"{stem}_score_left_crop.png",
                                dbg.crop_image(frame, score_regions.get("left"), crop_padding),
                            )
                            dbg.save_crop(
                                fdir, f"{stem}_score_right_crop.png",
                                dbg.crop_image(frame, score_regions.get("right"), crop_padding),
                            )
                    elif requires_frame:
                        gate_reason = "decode_failed"

                    if not (requires_frame and decoded):
                        timer_ocr = unsampled_timer_observation(ts, "no_region", source="decode_failed")
                        score_ocr = _empty_score_observation(ts, "no_region")
                        c4_observation = c4_tracker.update(ts, None)
                        timer_observations.append(timer_ocr)
                        score_observations.append(score_ocr)
                        c4_observations.append(c4_observation)
                    frame_observations.append({
                        "time_sec": float(ts),
                        "gate_reason": gate_reason,
                        "yolo_tags": list(yolo_tags),
                        "yolo_confidence": float(yolo_confidence),
                        "regions": list(regions),
                        "yolo_background": yolo_bg,
                        "pov_ocr": dict(pov_ocr),
                        "pov_crop_source": pov_crop_source,
                        "timer_ocr": timer_ocr,
                        "timer_crop_source": timer_crop_source,
                        "score_ocr": score_ocr,
                        "has_frame": bool(requires_frame and decoded),
                    })
            finally:
                cap.release()

            alignment_result = align.solve(
                source_start_sec=round_record.start_sec,
                source_end_sec=round_record.end_sec,
                score_observations=score_observations,
                c4_observations=c4_observations,
            )
            if alignment_result.get("status") != "locked":
                raise ValueError(
                    f"Phase2 round {round_record.round_no} alignment failed closed: "
                    f"{alignment_result.get('status', 'alignment_unresolved')}"
                )
            effective_start = float(alignment_result["effective_start_sec"])
            round_record.start_sec = max(float(round_record.start_sec), effective_start)
            filtered_observations = [
                item for item in frame_observations
                if float(item["time_sec"]) + 1e-6 >= round_record.start_sec
            ]
            background = [
                dict(item["yolo_background"])
                for item in filtered_observations
                if isinstance(item.get("yolo_background"), dict)
            ]
            prev_tick: int | None = None
            consecutive_unmatched = 0
            for item in filtered_observations:
                bg_info, tick = build_background_info(
                    demo=demo,
                    round_meta=round_meta,
                    align=align,
                    video_time=float(item["time_sec"]),
                    pov_ocr_result=item["pov_ocr"],
                    timer_ocr_result=item["timer_ocr"],
                    score_ocr_result=item["score_ocr"],
                    prev_tick=prev_tick,
                    pov_crop_source=str(item["pov_crop_source"]),
                    consecutive_unmatched=consecutive_unmatched,
                    spectator_min_frames=spectator_min_frames,
                    pov_match_min_score=pov_match_min_score,
                    timer_crop_source=str(item["timer_crop_source"]),
                    align_warnings=[],
                )
                if prev_tick is not None and tick < prev_tick:
                    raise ValueError(f"Phase2 round {round_record.round_no} produced non-monotone ticks")
                prev_tick = tick
                if item["has_frame"] and bg_info["who"]["pov_source"] in ("unmatched", "spectator"):
                    consecutive_unmatched += 1
                elif item["has_frame"]:
                    consecutive_unmatched = 0
                key_frames.append(KeyFrame(
                    time_sec=round(float(item["time_sec"]), 3),
                    gate_reason=str(item["gate_reason"]),
                    yolo_tags=list(item["yolo_tags"]),
                    yolo_confidence=float(item["yolo_confidence"]),
                    ui_regions=list(item["regions"]),
                    background_info=bg_info,
                    has_frame=bool(item["has_frame"]),
                ))
                if dbg.enabled:
                    dbg.write_frame({
                        "round_no": round_record.round_no,
                        "video_time": item["time_sec"],
                        "timer_observation": item["timer_ocr"],
                        "score_observation": item["score_ocr"],
                        "alignment_status": alignment_result["status"],
                    })

            detection_warnings = coalesce_yolo_gaps(
                [value for value in yolo_missing_times if value + 1e-6 >= round_record.start_sec],
                max_gap_sec=max(1.0, phase2_interval_sec * 1.25),
            )
            foreign_tail = alignment_result.get("foreign_tail")
            if foreign_tail:
                detection_warnings.append({
                    "type": "foreign_round_tail",
                    "source_start_sec": foreign_tail["start_sec"],
                    "effective_start_sec": round_record.start_sec,
                    "end_sec": foreign_tail["end_sec"],
                    "reason": "confirmed_timer_reset",
                })
            detection_warnings.append(build_alignment_warning(
                alignment_result,
                timer_observations,
                score_observations,
                ocr_budget.summary(),
            ))
            if alignment_result.get("score_fact_support") == "unsupported":
                detection_warnings.append({
                    "type": "unsupported_score_fact",
                    "reason": "parsed_demo_has_no_explicit_score",
                })
            if alignment_result.get("c4_evidence") == "unsupported":
                detection_warnings.append({
                    "type": "unsupported_c4_visual",
                    "reason": alignment_result.get("c4_support_reason", "c4_regions_not_available"),
                })
            if ocr_budget.budget_exhausted:
                detection_warnings.append({
                    "type": "budget_exhausted",
                    "actual_roi_calls": ocr_budget.actual_roi_calls,
                    "hard_limit": ocr_budget.hard_limit,
                })
            round_record.phase2_yolo = YoloData(
                background=background,
                key_frames=key_frames,
                yolo_required=yolo_enabled,
                yolo_model=str(yolo_config.get("model_path", "")),
                detector_mode="demo_timeline_yolo_ocr",
                sample_interval_sec=phase2_interval_sec,
                total_yolo_frames=total_yolo_frames,
                detection_warnings=detection_warnings,
            )
            if progress_sink is not None:
                try:
                    progress_sink(completed, len(match.rounds), "round", None)
                except Exception:
                    pass
    finally:
        dbg.close()
    final_budget_summary = ocr_budget.summary()
    for round_record in match.rounds:
        phase2 = getattr(round_record, "phase2_yolo", None)
        if phase2 is None:
            continue
        for warning in phase2.detection_warnings:
            if isinstance(warning, dict) and warning.get("type") == "ocr_alignment":
                warning["ocr_calls"] = dict(final_budget_summary)
    save_match(output_path, match)
    if semantic_output_path is not None:
        write_semantic_frames(match, semantic_output_path, demo=demo)


def _score_before_round(round_meta: dict) -> dict[str, int] | None:
    try:
        ct_after = int(round_meta["ct_score"])
        t_after = int(round_meta["t_score"])
    except (KeyError, TypeError, ValueError):
        return None
    winner = str(round_meta.get("winner") or "").upper()
    if winner not in {"CT", "T"}:
        return None
    return {
        "ct": max(0, ct_after - int(winner == "CT")),
        "t": max(0, t_after - int(winner == "T")),
    }


def build_semantic_frames(match, demo=None) -> list[dict]:
    """按回合导出精简的 DEM 事实时间线（不含 YOLO 检测框/标签）。

    每帧即某个关键帧的 ``background_info``（``when``/``who``/``where``/
    ``events``）。检测内部细节（``ui_regions``、``yolo_tags``、
    ``yolo_confidence``、``background``）保留在 rounds_with_yolo.json 中，
    不属于本产物。
    """
    rounds_out: list[dict] = []
    for round_record in match.rounds:
        demo_round_no = None
        round_meta: dict = {}
        score_before = None
        if demo is not None:
            demo_round_no = _demo_round_hint(round_record)
            round_meta = demo.round_by_no(demo_round_no)
            score_before = _score_before_round(round_meta)
        frames: list[dict] = []
        phase2 = getattr(round_record, "phase2_yolo", None)
        if phase2 is not None:
            for frame in phase2.key_frames:
                if frame.background_info:
                    bg = dict(frame.background_info)
                    events = dict(bg.get("events") or {})
                    events.pop("score_ocr", None)
                    bg["events"] = events
                    if score_before is not None:
                        when = dict(bg.get("when") or {})
                        when["score_before"] = dict(score_before)
                        if str(when.get("phase") or "") == "post_round":
                            when["winner"] = round_meta.get("winner")
                            when["reason"] = round_meta.get("reason")
                            when["score_after"] = {
                                "ct": round_meta.get("ct_score"),
                                "t": round_meta.get("t_score"),
                            }
                        bg["when"] = when
                    bg["has_frame"] = bool(getattr(frame, "has_frame", True))
                    frames.append(bg)
        semantic_round = {"round_no": round_record.round_no, "frames": frames}
        if demo is not None:
            semantic_round.update(
                {
                    "demo_round_no": demo_round_no,
                    "map_name": demo.map_name,
                    "capabilities": dict(demo.capabilities),
                    "round_result": {
                        key: round_meta.get(key)
                        for key in (
                            "start_tick",
                            "freeze_end_tick",
                            "end_tick",
                            "bomb_planted_tick",
                            "bomb_exploded_tick",
                            "bomb_defused_tick",
                            "bomb_begin_defuse_tick",
                            "defuser_has_kit",
                            "winner",
                            "reason",
                            "bomb_site",
                            "ct_alive_end",
                            "t_alive_end",
                            "ct_score",
                            "t_score",
                        )
                    },
                }
            )
        rounds_out.append(semantic_round)
    return rounds_out


def write_semantic_frames(match, semantic_path: Path, demo=None) -> None:
    """将精简的 DEM 事实时间线写为独立的 list[round] 产物。"""
    import json

    semantic_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_path.write_text(
        json.dumps(build_semantic_frames(match, demo=demo), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
