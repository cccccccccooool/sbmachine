"""在确定性随机抽取的视频帧上，评估 UI YOLO 与 HUD OCR 的覆盖率。"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sbmachine.phase2_ocr import _detect_pov_ocr, _detect_score_ocr, _first_timer_region, read_ocr_text
from sbmachine.phase2_yolo import should_sample_alignment_ocr
from sbmachine.phase2_yolo_gate import YoloUiDetector


def diagnose(video: Path, model: Path, *, samples: int, seeds: list[int]) -> list[dict]:
    """按多个随机种子抽帧，统计各帧的 HUD/计时器/比分 OCR 命中情况。"""
    config = yaml.safe_load(Path("config/yolo.yaml").read_text(encoding="utf-8"))["yolo"]
    yolo_config = dict(config["yolo"])
    yolo_config["model_path"] = str(model)
    detector = YoloUiDetector(yolo_config)
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    if not capture.isOpened() or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"cannot read video metadata: {video}")
    duration = frame_count / fps
    reports = []
    try:
        for seed in seeds:
            rng = random.Random(seed)
            times = sorted(rng.uniform(0.0, duration - 1.0) for _ in range(samples))
            rows = []
            for video_time in times:
                capture.set(cv2.CAP_PROP_POS_MSEC, video_time * 1000.0)
                ok, frame = capture.read()
                background = detector.decide(frame).background if ok else {}
                timer_region = _first_timer_region(background)
                timer_ocr = (
                    read_ocr_text(frame, timer_region, padding=4)
                    if ok and timer_region is not None
                    else {"raw_text": "", "confidence": 0.0}
                )
                match = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})", str(timer_ocr.get("raw_text", "")))
                timer = f"{int(match.group(1))}:{match.group(2)}" if match else ""
                score = _detect_score_ocr(frame, background, config["score_ocr"], 4) if ok else {}
                score_ct, score_t = score.get("ct"), score.get("t")
                score_raw = str(score.get("raw") or "").strip()
                if not (
                    isinstance(score_ct, int)
                    and isinstance(score_t, int)
                    and 0 <= score_ct <= 30
                    and 0 <= score_t <= 30
                    and re.fullmatch(r"\d{1,2}\s*[:\-]\s*\d{1,2}", score_raw)
                ):
                    score_ct = score_t = None
                rows.append({
                    "video_time": round(video_time, 3),
                    "hud_detected": bool(background.get("regions")),
                    "timer_region": timer_region is not None,
                    "timer": timer,
                    "timer_raw": str(timer_ocr.get("raw_text", "")),
                    "score_ct": score_ct,
                    "score_t": score_t,
                })
            reports.append({
                "seed": seed,
                "samples": len(rows),
                "hud_detected": sum(row["hud_detected"] for row in rows),
                "timer_region": sum(row["timer_region"] for row in rows),
                "timer_value": sum(bool(row["timer"]) for row in rows),
                "score_value": sum(row["score_ct"] is not None for row in rows),
                "hits": [row for row in rows if row["timer"] or row["score_ct"] is not None],
            })
    finally:
        capture.release()
    return reports


def diagnose_sequential(video: Path, model: Path) -> dict:
    """在单个视频上按生产环境 1 FPS 调度顺序跑一遍，不依赖任何 DEM 输入。"""
    config = yaml.safe_load(Path("config/yolo.yaml").read_text(encoding="utf-8"))["yolo"]
    sampling = config.get("sampling", {})
    yolo_config = dict(config["yolo"])
    yolo_config["model_path"] = str(model)
    detector = YoloUiDetector(yolo_config)
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    if not capture.isOpened() or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"cannot read video metadata: {video}")
    duration = frame_count / fps
    interval = max(0.1, float(sampling.get("phase2_interval_sec", 1.0)))
    initial = float(sampling.get("alignment_initial_sec", 10.0))
    period = float(sampling.get("alignment_period_sec", 20.0))
    window = float(sampling.get("alignment_window_sec", 5.0))
    totals = {
        "decode_sec": 0.0,
        "yolo_sec": 0.0,
        "alignment_ocr_sec": 0.0,
        "pov_ocr_sec": 0.0,
        "samples": 0,
        "alignment_ocr_frames": 0,
        "timer_values": 0,
        "score_values": 0,
        "pov_boxes": 0,
        "pov_white_gate_passes": 0,
        "pov_values": 0,
    }
    started = time.perf_counter()
    decoded_time = -1.0 / fps
    sample_time = 0.0
    try:
        while sample_time <= duration + 1e-6:
            step_started = time.perf_counter()
            decoded = False
            while decoded_time + 1e-6 < sample_time:
                decoded = capture.grab()
                if not decoded:
                    break
                position_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                decoded_time = position_msec / 1000.0 if position_msec > 0 else decoded_time + 1.0 / fps
            if decoded:
                decoded, frame = capture.retrieve()
            totals["decode_sec"] += time.perf_counter() - step_started
            if not decoded:
                break
            totals["samples"] += 1

            yolo_started = time.perf_counter()
            background = detector.decide(frame).background
            totals["yolo_sec"] += time.perf_counter() - yolo_started

            if should_sample_alignment_ocr(
                sample_time,
                0.0,
                duration,
                initial_sec=initial,
                period_sec=period,
                window_sec=window,
            ):
                totals["alignment_ocr_frames"] += 1
                ocr_started = time.perf_counter()
                timer_region = _first_timer_region(background)
                timer_ocr = (
                    read_ocr_text(
                        frame,
                        timer_region,
                        padding=int(config.get("crop_padding_px", 4)),
                        accept_pattern=r"\d{1,2}\s*[:：]\s*\d{2}",
                    )
                    if timer_region is not None
                    else {"raw_text": ""}
                )
                timer_match = re.search(r"\d{1,2}\s*[:：]\s*\d{2}", str(timer_ocr.get("raw_text", "")))
                score = _detect_score_ocr(frame, background, config["score_ocr"], int(config.get("crop_padding_px", 4)))
                totals["alignment_ocr_sec"] += time.perf_counter() - ocr_started
                totals["timer_values"] += int(timer_match is not None)
                totals["score_values"] += int(score.get("ct") is not None and score.get("t") is not None)

            pov_started = time.perf_counter()
            pov, source, region = _detect_pov_ocr(
                frame,
                background,
                config["pov_ocr"],
                int(config.get("crop_padding_px", 4)),
            )
            totals["pov_ocr_sec"] += time.perf_counter() - pov_started
            totals["pov_boxes"] += int(region is not None)
            totals["pov_white_gate_passes"] += int(source == "yolo_pov_region")
            totals["pov_values"] += int(bool(pov.get("raw_text")))
            sample_time += interval
    finally:
        capture.release()
    totals["duration_sec"] = round(duration, 3)
    totals["total_sec"] = round(time.perf_counter() - started, 3)
    for key in ("decode_sec", "yolo_sec", "alignment_ocr_sec", "pov_ocr_sec"):
        totals[key] = round(float(totals[key]), 3)
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True, help="HUD YOLO path configured under yolo.yolo.model_path")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seeds", default="6657,20260713,42")
    parser.add_argument("--sequential", action="store_true", help="Run the production 1 FPS schedule sequentially")
    args = parser.parse_args()
    if args.sequential:
        print(json.dumps(diagnose_sequential(Path(args.video), Path(args.model)), ensure_ascii=False, indent=2))
        return 0
    reports = diagnose(
        Path(args.video),
        Path(args.model),
        samples=max(1, args.samples),
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
