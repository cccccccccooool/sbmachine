"""Phase 2：为每个回合构建 DEM 时间线，并附加 YOLO/OCR 事实。"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import re
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
from sbmachine.phase2_ocr import OcrConsensus, _detect_pov_ocr, _detect_score_ocr, _first_timer_region, _resolve_ocr_box, read_ocr_text
from sbmachine.phase2_quality import coalesce_yolo_gaps
from sbmachine.phase2_timeline import build_timeline
from sbmachine.phase2_yolo_gate import YoloUiDetector
from sbmachine.schemas import KeyFrame, YoloData, load_match, save_match
from sbmachine.time_align import RoundTimeAlign


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


def run_phase2(
    *,
    rounds_path: Path,
    output_path: Path,
    config_path: Path,
    video_path: Path | None = None,
    dry_run: bool = False,
    debug_dir: Path | None = None,
    semantic_output_path: Path | None = None,
) -> None:
    config = load_config(config_path)
    match = load_match(rounds_path)
    actual_video = video_path or resolve_path(match.video_path)
    if actual_video is None and not dry_run:
        raise ValueError("video path is required")

    vision_config = config.get("vision", {})
    yolo_config = vision_config.get("yolo", {})
    demo_config = config.get("demo", {})
    pov_ocr_config = vision_config.get("pov_ocr", {})
    timer_ocr_config = vision_config.get("timer_ocr", {})
    score_ocr_config = vision_config.get("score_ocr", {})
    sampling_config = vision_config.get("sampling", {})
    phase2_interval_sec = float(sampling_config.get("phase2_interval_sec", 1.0))
    alignment_initial_sec = float(sampling_config.get("alignment_initial_sec", 10.0))
    alignment_period_sec = float(sampling_config.get("alignment_period_sec", 20.0))
    alignment_window_sec = float(sampling_config.get("alignment_window_sec", 5.0))
    yolo_enabled = bool(yolo_config.get("enabled", True))
    crop_padding = int(vision_config.get("crop_padding_px", 4))
    plant_empty_timer_frames = int(demo_config.get("plant_empty_timer_frames", 3))
    pov_match_min_score = float(pov_ocr_config.get("min_match_score", 0.6))
    spectator_min_frames = int(pov_ocr_config.get("spectator_min_frames", 3))

    dbg = DebugWriter(debug_dir)
    dbg.open()
    parsed_demo_dir = resolve_path(demo_config.get("parsed_dir", "output/demo"))
    demo = None if dry_run else DemoQuery.load(parsed_demo_dir or Path("output/demo"))
    if demo is not None:
        resolve_demo_round_hints(match.rounds, demo.rounds)

    yolo = None
    try:
        for round_record in tqdm(match.rounds, desc="Phase2 YOLO", unit="round"):
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
            key_frames: list[KeyFrame] = []
            background: list[dict] = []
            yolo_missing_times: list[float] = []
            total_yolo_frames = 0
            prev_tick: int | None = None
            consecutive_unmatched = 0
            empty_timer_count = 0

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
                    alignment_ocr_sampled = False

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
                                background.append(yolo_bg)
                        else:
                            yolo_bg = None
                            gate_reason = "fixed_roi_only"

                        if should_sample_alignment_ocr(
                            ts,
                            round_record.start_sec,
                            round_record.end_sec,
                            initial_sec=alignment_initial_sec,
                            period_sec=alignment_period_sec,
                            window_sec=alignment_window_sec,
                        ):
                            alignment_ocr_sampled = True
                            timer_region, timer_crop_source = _resolve_ocr_box(
                                timer_region, timer_ocr_config, frame.shape, "yolo_timer_region"
                            )
                            timer_ocr = (
                                read_ocr_text(
                                    frame,
                                    timer_region,
                                    padding=crop_padding,
                                    accept_pattern=r"\d{1,2}\s*[:：]\s*\d{2}",
                                )
                                if timer_region
                                else {"raw_text": "", "engine": f"no_region:{timer_crop_source}", "region": None}
                            )
                            timer_match = re.search(r"(\d{1,2})\s*[::]\s*(\d{2})", str(timer_ocr.get("raw_text", "")))
                            timer_ocr["value"] = f"{int(timer_match.group(1))}:{timer_match.group(2)}" if timer_match else ""
                            score_ocr = _detect_score_ocr(frame, yolo_bg, score_ocr_config, crop_padding)
                        else:
                            timer_ocr = {"value": "", "raw_text": "", "engine": "skipped:alignment_schedule"}
                            score_ocr = {"ct": None, "t": None, "raw": "", "source": "skipped:alignment_schedule", "confidence": 0.0}
                        pov_sample, pov_crop_source, pov_region = _detect_pov_ocr(
                            frame, yolo_bg, pov_ocr_config, crop_padding
                        )
                        pov_ocr = pov_consensus.update(pov_sample)

                        if dbg.enabled:
                            fdir = dbg.frame_dir(demo_round_no)
                            stem = f"frame_{ts:.3f}"
                            dbg.save_crop(fdir, f"{stem}_pov_crop.png", dbg.crop_image(frame, pov_region, crop_padding))
                            dbg.save_crop(fdir, f"{stem}_timer_crop.png", dbg.crop_image(frame, timer_region, crop_padding))
                    elif requires_frame:
                        gate_reason = "decode_failed"

                    if timer_ocr.get("value"):
                        empty_timer_count = 0
                    elif requires_frame and decoded and alignment_ocr_sampled:
                        empty_timer_count += 1
                    if (
                        empty_timer_count >= plant_empty_timer_frames
                        and round_meta.get("bomb_planted_tick") is not None
                        and not align.is_frozen
                        and align.offsets
                    ):
                        plant_tick = int(round_meta["bomb_planted_tick"])
                        plant_video_time = align.to_video_time(plant_tick)
                        if abs(plant_video_time - float(ts)) <= float(demo_config.get("anchor_tolerance_sec", 2.0)):
                            align.freeze(ts, event_tick=plant_tick)

                    bg_info, tick = build_background_info(
                        demo=demo,
                        round_meta=round_meta,
                        align=align,
                        video_time=ts,
                        pov_ocr_result=pov_ocr,
                        timer_ocr_result=timer_ocr,
                        score_ocr_result=score_ocr,
                        prev_tick=prev_tick,
                        pov_crop_source=pov_crop_source,
                        consecutive_unmatched=consecutive_unmatched,
                        spectator_min_frames=spectator_min_frames,
                        pov_match_min_score=pov_match_min_score,
                        timer_crop_source=timer_crop_source,
                        align_warnings=align.take_new_warnings(),
                    )
                    prev_tick = tick
                    if requires_frame and decoded:
                        if bg_info["who"]["pov_source"] in ("unmatched", "spectator"):
                            consecutive_unmatched += 1
                        else:
                            consecutive_unmatched = 0
                    key_frames.append(KeyFrame(
                        time_sec=round(ts, 3),
                        gate_reason=gate_reason,
                        yolo_tags=yolo_tags,
                        yolo_confidence=yolo_confidence,
                        ui_regions=regions,
                        background_info=bg_info,
                        has_frame=bool(requires_frame and decoded),
                    ))
            finally:
                cap.release()

            detection_warnings = coalesce_yolo_gaps(
                yolo_missing_times,
                max_gap_sec=max(1.0, phase2_interval_sec * 1.25),
            )
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
    finally:
        dbg.close()
    save_match(output_path, match)
    if semantic_output_path is not None:
        write_semantic_frames(match, semantic_output_path)


def build_semantic_frames(match) -> list[dict]:
    """按回合导出精简的 DEM 事实时间线（不含 YOLO 检测框/标签）。

    每帧即某个关键帧的 ``background_info``（``when``/``who``/``where``/
    ``events``）。检测内部细节（``ui_regions``、``yolo_tags``、
    ``yolo_confidence``、``background``）保留在 rounds_with_yolo.json 中，
    不属于本产物。
    """
    rounds_out: list[dict] = []
    for round_record in match.rounds:
        frames: list[dict] = []
        phase2 = getattr(round_record, "phase2_yolo", None)
        if phase2 is not None:
            for frame in phase2.key_frames:
                if frame.background_info:
                    bg = dict(frame.background_info)
                    bg["has_frame"] = bool(getattr(frame, "has_frame", True))
                    frames.append(bg)
        rounds_out.append({"round_no": round_record.round_no, "frames": frames})
    return rounds_out


def write_semantic_frames(match, semantic_path: Path) -> None:
    """将精简的 DEM 事实时间线写为独立的 list[round] 产物。"""
    import json

    semantic_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_path.write_text(
        json.dumps(build_semantic_frames(match), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
