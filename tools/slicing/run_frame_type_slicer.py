"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：使用 frame_type 分类模型对视频进行自动检测并切片。

启动方式：
  - GUI 调试：python tools/slicing/run_frame_type_slicer.py --gui
  - 命令行：python tools/slicing/run_frame_type_slicer.py --video <video_path> [选项]
输入数据流：输入录像路径和 frame_type 分类模型权重。
输出数据流：逐帧预测结果 JSONL 和连续 game 片段切片 JSON。
用法用途：识别连续游戏正片片段并输出切片描述结构，支持可视化 GUI 调试和命令行批量处理。

=========================================
    它识别出连续的视频游戏正片片段并输出切片 segment_output.json 描述结构，支持可视化 GUI 调试界面以及命令行批量处理。
使用方法 (Usage):
    1. 启动本地 GUI 调试界面：
        python tools/slicing/run_frame_type_slicer.py --gui
    2. 命令行批量调用：
        python tools/slicing/run_frame_type_slicer.py --video <video_path> [选项]
        参数说明:
          --video: 输入录像路径 (必填)
          --model: 游戏画面粗切模型 pt 路径 (默认: models/frame_type_game_break.pt)
          --replay-model: 局部 replay_marker 二分类模型 pt 路径
          --replay-roi: replay 标识区域 x1,y1,x2,y2
          --replay-threshold: replay 置信度阈值 (默认: 0.65)
          --frame-output: 逐帧预测结果输出路径 (默认: output/frame_type_rows.jsonl)
          --segment-output: 连续 game 粗片段切片结果输出路径 (默认: output/frame_type_segments.json)
          --markers: 可选，人工 markers.json。若提供，将按人工 start/end 点生成最终切片段
          --interval-sec: 采样间隔秒数 (默认: 1.0)
          --smooth-window: 标签平滑滑动窗口大小 (默认: 5)
          --min-live-sec: 最小有效小局秒数 (默认: 20.0)
          --bridge-gap-sec: 允许桥接合并的最大间隔秒数 (默认: 3.0)
=========================================
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
from collections import Counter, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid WinError 1114 DLL load failure on Windows when importing torch after PyQt5.
# Importing torch before PyQt5 initializes resolves this conflict.
try:
    import torch
except ImportError:
    pass

from vision_service.frame_type_model import load_checkpoint, predict_frame, resolve_device


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def to_relative_path(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def iter_video_frames(video_path: Path, interval_sec: float, start_sec: float, end_sec: float | None):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        ts = float(start_sec)
        while end_sec is None or ts <= end_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            yield round(ts, 3), frame
            ts += max(0.1, interval_sec)
    finally:
        cap.release()


def get_video_duration_sec(video_path: Path) -> float | None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0.0 or frame_count <= 0.0:
            return None
        return frame_count / fps
    finally:
        cap.release()


def estimate_probe_count(video_path: Path, interval_sec: float, start_sec: float, end_sec: float | None) -> int | None:
    duration = get_video_duration_sec(video_path)
    if duration is None:
        return None
    effective_end = min(duration, end_sec) if end_sec is not None else duration
    if effective_end < start_sec:
        return 0
    return int(math.floor((effective_end - start_sec) / max(0.1, interval_sec))) + 1


def smooth_rows(rows: list[dict], window: int) -> list[dict]:
    if window <= 1:
        for row in rows:
            row["smooth_label"] = row["label"]
        return rows
    labels = deque()
    for row in rows:
        labels.append(row["label"])
        if len(labels) > window:
            labels.popleft()
        row["smooth_label"] = Counter(labels).most_common(1)[0][0]
    return rows


def parse_roi(value: str) -> tuple[float, float, float, float] | None:
    if not value.strip():
        return None
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--replay-roi 必须是 x1,y1,x2,y2")
    return parts[0], parts[1], parts[2], parts[3]


def crop_normalized(frame, roi: tuple[float, float, float, float] | None):
    if roi is None:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    if max(roi) <= 1.0:
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return frame
    return frame[iy1:iy2, ix1:ix2]


_rapid_ocr_engine = None

def get_rapid_ocr():
    global _rapid_ocr_engine
    if _rapid_ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_ocr_engine = RapidOCR()
        except ImportError:
            pass
    return _rapid_ocr_engine


def read_timer_value(frame, roi: tuple[float, float, float, float] | None) -> str:
    if roi is None:
        return ""
    crop = crop_normalized(frame, roi)
    raw_text = ""
    try:
        ocr = get_rapid_ocr()
        if ocr:
            result, _ = ocr(crop)
            if result:
                raw_text = " ".join(str(item[1]) for item in result if len(item) >= 2)
    except Exception:
        return ""
    match = __import__("re").search(r"(\d{1,2})\s*[:：]\s*(\d{2})", raw_text)
    if not match:
        return ""
    return f"{int(match.group(1))}:{match.group(2)}"


def timer_to_seconds(value: str) -> float | None:
    text = str(value or "").strip().replace("：", ":")
    if ":" not in text:
        return None
    left, right = text.split(":", 1)
    try:
        minutes = int(left)
        seconds = int(right[:2])
    except ValueError:
        return None
    if not 0 <= seconds < 60:
        return None
    return float(minutes * 60 + seconds)


def score_signature(frame, roi: tuple[float, float, float, float] | None, *, grid: tuple[int, int] = (16, 8)) -> list[float]:
    if roi is None:
        return []
    try:
        import cv2
        import numpy as np

        crop = crop_normalized(frame, roi)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, grid, interpolation=cv2.INTER_AREA)
        return [round(float(v) / 255.0, 4) for v in resized.flatten().tolist()]
    except Exception:
        return []


def score_sig_distance(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def first_valid_timer(rows: list[dict]) -> tuple[float, float] | None:
    for row in rows:
        sec = timer_to_seconds(str(row.get("timer_value", "")))
        if sec is not None:
            return float(row.get("time_sec", 0.0)), sec
    return None


def last_valid_timer(rows: list[dict]) -> tuple[float, float] | None:
    for row in reversed(rows):
        sec = timer_to_seconds(str(row.get("timer_value", "")))
        if sec is not None:
            return float(row.get("time_sec", 0.0)), sec
    return None


def segment_rows(rows: list[dict], segment: dict) -> list[dict]:
    start = float(segment["start_sec"])
    end = float(segment["end_sec"])
    return [row for row in rows if start <= float(row.get("time_sec", 0.0)) <= end]


def should_bridge_segments(
    left: dict,
    right: dict,
    rows: list[dict],
    *,
    bridge_gap_sec: float,
    bridge_gap_max_sec: float,
    timer_tolerance_sec: float,
    score_sig_threshold: float,
) -> tuple[bool, dict]:
    gap = float(right["start_sec"]) - float(left["end_sec"])
    left_rows = segment_rows(rows, left)
    right_rows = segment_rows(rows, right)
    left_timer = last_valid_timer(left_rows)
    right_timer = first_valid_timer(right_rows)
    decision = {"gap_sec": round(gap, 3), "bridge": False, "reason": "fallback_no_bridge"}
    if right_timer and right_timer[1] >= 110.0:
        decision.update({"reason": "timer_new_round", "right_timer_sec": right_timer[1]})
        return False, decision
    if left_timer and right_timer and gap <= bridge_gap_max_sec:
        elapsed = float(right["start_sec"]) - float(left["end_sec"])
        expected = max(0.0, left_timer[1] - elapsed)
        error = abs(right_timer[1] - expected)
        if error <= timer_tolerance_sec:
            decision.update(
                {
                    "bridge": True,
                    "reason": "timer_continuity",
                    "left_timer_sec": left_timer[1],
                    "right_timer_sec": right_timer[1],
                    "timer_error_sec": round(error, 3),
                }
            )
            return True, decision
    left_sig = next((row.get("score_sig", []) for row in reversed(left_rows) if row.get("score_sig")), [])
    right_sig = next((row.get("score_sig", []) for row in right_rows if row.get("score_sig")), [])
    sig_dist = score_sig_distance(left_sig, right_sig)
    if sig_dist > score_sig_threshold:
        decision.update({"reason": "score_signature_changed", "score_sig_distance": round(sig_dist, 4)})
        return False, decision
    if gap <= bridge_gap_sec:
        decision.update({"bridge": True, "reason": "fallback_short_gap", "score_sig_distance": round(sig_dist, 4)})
        return True, decision
    decision.update({"reason": "fallback_gap_too_large", "score_sig_distance": round(sig_dist, 4)})
    return False, decision




def build_segments_v2(
    rows: list[dict],
    *,
    live_label: str,
    min_live_sec: float,
    bridge_gap_sec: float,
    bridge_gap_max_sec: float,
    timer_tolerance_sec: float = 3.0,
    score_sig_threshold: float = 0.12,
) -> list[dict]:
    raw_segments = []
    active = None
    for row in rows:
        ts = float(row["time_sec"])
        is_live = row.get("smooth_label") == live_label
        if is_live and active is None:
            active = {"start_sec": ts, "end_sec": ts, "frames": 0, "bridge_decisions": []}
        if is_live and active is not None:
            active["end_sec"] = ts
            active["frames"] += 1
        if not is_live and active is not None:
            raw_segments.append(active)
            active = None
    if active is not None:
        raw_segments.append(active)

    merged = []
    for segment in raw_segments:
        if not merged:
            merged.append(segment)
            continue
        bridge, decision = should_bridge_segments(
            merged[-1],
            segment,
            rows,
            bridge_gap_sec=bridge_gap_sec,
            bridge_gap_max_sec=bridge_gap_max_sec,
            timer_tolerance_sec=timer_tolerance_sec,
            score_sig_threshold=score_sig_threshold,
        )
        merged[-1].setdefault("bridge_decisions", []).append(decision)
        if bridge:
            merged[-1]["end_sec"] = segment["end_sec"]
            merged[-1]["frames"] += segment["frames"]
        else:
            merged.append(segment)

    return [
        {**segment, "duration_sec": round(segment["end_sec"] - segment["start_sec"], 3)}
        for segment in merged
        if segment["end_sec"] - segment["start_sec"] >= min_live_sec
    ]


def validate_segments_with_demo(segments: list[dict], rows: list[dict], demo_rounds_path: Path | None) -> list[dict]:
    if demo_rounds_path is None or not demo_rounds_path.exists():
        for segment in segments:
            segment.setdefault("demo_round_hint", "unmatched")
        return segments
    rounds = json.loads(demo_rounds_path.read_text(encoding="utf-8"))

    # Use duration-DP alignment instead of positional mapping.
    # Positional mapping (rounds[i-1]) silently drifts when segments < demo rounds.
    try:
        import sys
        from pathlib import Path as _Path
        _root = str(_Path(__file__).resolve().parents[2])
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from sbmachine.round_aligner import align_segments, apply_align_results
        results = align_segments(segments, rounds, tick_rate=64.0)
        apply_align_results(segments, results)
    except Exception as exc:
        # Fallback: mark all unmatched — never silently use position mapping
        for segment in segments:
            segment["demo_round_hint"] = "unmatched"
            segment["align_method"] = f"error:{type(exc).__name__}"
    return segments


def build_segments_from_markers(markers_path: Path) -> list[dict]:
    payload = json.loads(markers_path.read_text(encoding="utf-8"))
    markers = sorted(payload.get("markers", []) or [], key=lambda item: float(item.get("time_sec", 0)))
    segments = []
    active = None
    tags = []
    for marker in markers:
        marker_type = str(marker.get("type", "note"))
        time_sec = float(marker.get("time_sec", 0))
        if marker_type == "start":
            if active is not None:
                active["end_sec"] = time_sec
                active["duration_sec"] = round(active["end_sec"] - active["start_sec"], 3)
                active["markers"] = tags
                segments.append(active)
                tags = []
            active = {"kind": "live_round", "start_sec": time_sec, "end_sec": time_sec, "reason": "manual_marker"}
            continue
        if marker_type == "end":
            if active is not None:
                active["end_sec"] = time_sec
                active["duration_sec"] = round(active["end_sec"] - active["start_sec"], 3)
                active["markers"] = tags
                segments.append(active)
                active = None
                tags = []
            continue
        tags.append(marker)
    return [segment for segment in segments if float(segment.get("duration_sec", 0)) > 0]
def process_video_chunk_with_queue(kwargs: dict, queue) -> None:
    worker_id = kwargs["worker_id"]
    video_path = kwargs["video_path"]
    model_path = kwargs["model_path"]
    device = kwargs["device"]
    
    try:
        model, labels, img_size, _ = load_checkpoint(model_path, device)
        
        replay_model = None
        replay_labels = []
        replay_img_size = 224
        if kwargs.get("replay_model_path"):
            replay_model, replay_labels, replay_img_size, _ = load_checkpoint(kwargs["replay_model_path"], device)
            
        rows = []
        index = 0
        for ts, frame in iter_video_frames(video_path, kwargs["interval_sec"], kwargs["start_sec"], kwargs["end_sec"]):
            pred = predict_frame(model, labels, frame, img_size, device)
            row = {"time_sec": ts, **pred}
            if row["label"] == kwargs["game_label"] and replay_model is not None:
                replay_frame = crop_normalized(frame, kwargs["replay_roi"])
                replay_pred = predict_frame(replay_model, replay_labels, replay_frame, replay_img_size, device)
                row["replay_marker"] = replay_pred
                if replay_pred["label"] == kwargs["replay_label"] and replay_pred["confidence"] >= kwargs["replay_threshold"]:
                    row["is_replay"] = True
                    row["frame_type_source"] = "replay_marker_model"
                else:
                    row["is_replay"] = False
                    row["frame_type_source"] = "game_break_model"
            else:
                row["is_replay"] = False
                row["frame_type_source"] = "game_break_model"
            if row["label"] == kwargs["game_label"]:
                row["timer_value"] = read_timer_value(frame, kwargs["timer_roi"])
                row["score_sig"] = score_signature(frame, kwargs["score_roi"])
            else:
                row["timer_value"] = ""
                row["score_sig"] = []
            rows.append(row)
            index += 1
            if index % 10 == 0:
                queue.put((worker_id, index, None))
        
        queue.put((worker_id, index, rows))
    except Exception as e:
        queue.put((worker_id, -1, str(e)))


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    device = resolve_device(args.device)
    video_path = resolve_path(args.video)
    model_path = resolve_path(args.model)
    model, labels, img_size, _checkpoint = load_checkpoint(model_path, device)
    replay_model = None
    replay_labels = []
    replay_img_size = 224
    replay_roi = parse_roi(args.replay_roi)
    timer_roi = parse_roi(args.timer_roi)
    score_roi = parse_roi(args.score_roi)
    if args.replay_model:
        replay_model, replay_labels, replay_img_size, _replay_checkpoint = load_checkpoint(resolve_path(args.replay_model), device)
    rows = []
    total = estimate_probe_count(video_path, args.interval_sec, args.start_sec, args.end_sec)
    print(f"model: {model_path}")
    if replay_model is not None:
        print(f"replay_model: {resolve_path(args.replay_model)}")
        print(f"replay_roi: {args.replay_roi or 'full_frame'}")
    print(f"video: {video_path}")
    print(f"labels: {', '.join(labels)}")
    print(f"estimated_frames: {total if total is not None else 'unknown'}")
    workers = getattr(args, "workers", 1)
    if workers <= 1:
        for index, (ts, frame) in enumerate(iter_video_frames(video_path, args.interval_sec, args.start_sec, args.end_sec), start=1):
            pred = predict_frame(model, labels, frame, img_size, device)
            row = {"time_sec": ts, **pred}
            if row["label"] == args.game_label and replay_model is not None:
                replay_frame = crop_normalized(frame, replay_roi)
                replay_pred = predict_frame(replay_model, replay_labels, replay_frame, replay_img_size, device)
                row["replay_marker"] = replay_pred
                if replay_pred["label"] == args.replay_label and replay_pred["confidence"] >= args.replay_threshold:
                    row["is_replay"] = True
                    row["frame_type_source"] = "replay_marker_model"
                else:
                    row["is_replay"] = False
                    row["frame_type_source"] = "game_break_model"
            else:
                row["is_replay"] = False
                row["frame_type_source"] = "game_break_model"
            if row["label"] == args.game_label:
                row["timer_value"] = read_timer_value(frame, timer_roi)
                row["score_sig"] = score_signature(frame, score_roi)
            else:
                row["timer_value"] = ""
                row["score_sig"] = []
            rows.append(row)
            should_report = index == 1 or index % max(1, args.progress_every) == 0
            if total is not None:
                should_report = should_report or index >= total
            if should_report:
                if total:
                    percent = min(100.0, index / total * 100.0)
                    print(f"progress: {index}/{total} ({percent:.1f}%) @ {ts:.1f}s", flush=True)
                else:
                    print(f"progress: {index} frames @ {ts:.1f}s", flush=True)
    else:
        duration = get_video_duration_sec(video_path) or 0.0
        eff_start = args.start_sec
        eff_end = args.end_sec if args.end_sec is not None else duration
        if eff_end <= eff_start:
            eff_end = eff_start + 1.0
        chunk_sec = (eff_end - eff_start) / workers
        
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        
        processes = []
        for i in range(workers):
            c_start = eff_start + i * chunk_sec
            c_end = eff_start + (i + 1) * chunk_sec if i < workers - 1 else eff_end
            task_args = {
                "worker_id": i,
                "video_path": video_path,
                "model_path": model_path,
                "device": device,
                "replay_model_path": resolve_path(args.replay_model) if args.replay_model else None,
                "interval_sec": args.interval_sec,
                "start_sec": c_start,
                "end_sec": c_end,
                "game_label": args.game_label,
                "replay_label": args.replay_label,
                "replay_threshold": args.replay_threshold,
                "replay_roi": replay_roi,
                "timer_roi": timer_roi,
                "score_roi": score_roi,
            }
            p = ctx.Process(target=process_video_chunk_with_queue, args=(task_args, queue))
            p.start()
            processes.append(p)
            
        finished_workers = 0
        worker_progress = {i: 0 for i in range(workers)}
        worker_results = {}
        
        print(f"Processing video with {workers} workers using spawn context...", flush=True)
        
        while finished_workers < workers:
            try:
                item = queue.get(timeout=1.0)
                worker_id, progress, payload = item
                if progress == -1:
                    raise RuntimeError(f"Worker {worker_id} crashed: {payload}")
                
                if payload is not None:
                    worker_results[worker_id] = payload
                    worker_progress[worker_id] = progress
                    finished_workers += 1
                else:
                    worker_progress[worker_id] = progress
                
                current_total = sum(worker_progress.values())
                if total:
                    percent = min(100.0, current_total / total * 100.0)
                    print(f"progress: {current_total}/{total} ({percent:.1f}%)", flush=True)
                else:
                    print(f"progress: {current_total} frames processed", flush=True)
            except queue.Empty:
                for i, p in enumerate(processes):
                    if not p.is_alive() and i not in worker_results:
                        print(f"Worker process {i} died unexpectedly with exitcode {p.exitcode}", flush=True)
                        finished_workers = workers
                        break
                continue
            except Exception as e:
                print(f"Error during progress tracking: {e}", flush=True)
                break
        
        for p in processes:
            p.join()
            
        for i in range(workers):
            if i in worker_results:
                rows.extend(worker_results[i])
        rows.sort(key=lambda r: float(r["time_sec"]))
    rows = smooth_rows(rows, args.smooth_window)
    if args.markers:
        segments = build_segments_from_markers(resolve_path(args.markers))
        segment_mode = "manual_markers"
    else:
        segments = build_segments_v2(
            rows,
            live_label=args.live_label,
            min_live_sec=args.min_live_sec,
            bridge_gap_sec=args.bridge_gap_sec,
            bridge_gap_max_sec=args.bridge_gap_max_sec,
            timer_tolerance_sec=args.timer_tolerance_sec,
            score_sig_threshold=args.score_sig_threshold,
        )
        segments = validate_segments_with_demo(segments, rows, resolve_path(args.demo_rounds) if args.demo_rounds else None)
        segment_mode = "model_game_segments_v2"

    frame_output = resolve_path(args.frame_output)
    segment_output = resolve_path(args.segment_output)
    frame_output.parent.mkdir(parents=True, exist_ok=True)
    segment_output.parent.mkdir(parents=True, exist_ok=True)
    with frame_output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    segment_output.write_text(
        json.dumps(
            {
                "video": to_relative_path(resolve_path(args.video)),
                "model": to_relative_path(resolve_path(args.model)),
                "replay_model": to_relative_path(resolve_path(args.replay_model)) if args.replay_model else "",
                "replay_roi": args.replay_roi,
                "timer_roi": args.timer_roi,
                "score_roi": args.score_roi,
                "interval_sec": args.interval_sec,
                "bridge_gap_max_sec": args.bridge_gap_max_sec,
                "timer_tolerance_sec": args.timer_tolerance_sec,
                "score_sig_threshold": args.score_sig_threshold,
                "demo_rounds": to_relative_path(resolve_path(args.demo_rounds)) if args.demo_rounds else "",
                "labels": labels,
                "segment_mode": segment_mode,
                "markers": to_relative_path(resolve_path(args.markers)) if args.markers else "",
                "frame_count": len(rows),
                "segment_count": len(segments),
                "total_live_sec": round(sum(float(segment.get("duration_sec", 0.0)) for segment in segments), 3),
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return frame_output, segment_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 frame_type 模型粗切游戏内容；可叠加人工 markers 生成最终片段。")
    parser.add_argument("--gui", action="store_true", help="启动本地可视化调试页面")
    parser.add_argument("--video", default="")
    parser.add_argument("--model", default="models/qiepian/frame_type_classifier.pt")
    parser.add_argument("--replay-model", default="", help="可选：局部 replay_marker 二分类模型")
    parser.add_argument("--replay-roi", default="", help="可选：replay 标记区域 x1,y1,x2,y2；支持 0-1 归一化或像素坐标")
    parser.add_argument("--replay-threshold", type=float, default=0.65)
    parser.add_argument("--timer-roi", default="0.46,0.0,0.54,0.06", help="回合计时器 ROI x1,y1,x2,y2；支持归一化或像素坐标")
    parser.add_argument("--score-roi", default="0.42,0.0,0.58,0.08", help="比分区域 ROI x1,y1,x2,y2；用于灰度签名差分")
    parser.add_argument("--frame-output", default="output/frame_type_rows.jsonl")
    parser.add_argument("--segment-output", default="output/frame_type_segments.json")
    parser.add_argument("--demo-rounds", default="", help="可选：tools/parse_demo.py 输出的 rounds.json，用于切片后校验")
    parser.add_argument("--markers", default="", help="manual_segment_marker.py 输出的 markers.json；提供后按人工 start/end 点生成片段")
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--game-label", default="game")
    parser.add_argument("--live-label", default="game")
    parser.add_argument("--replay-label", default="replay_marker")
    parser.add_argument("--min-live-sec", type=float, default=20.0)
    parser.add_argument("--bridge-gap-sec", type=float, default=3.0)
    parser.add_argument("--bridge-gap-max-sec", type=float, default=20.0)
    parser.add_argument("--timer-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--score-sig-threshold", type=float, default=0.12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=1, help="多进程并行数量，提升处理速度")
    parser.add_argument("--progress-every", type=int, default=25, help="命令行进度输出间隔，按采样帧数计算")
    return parser.parse_args()


class SlicerWindow:
    def __init__(self) -> None:
        try:
            from PyQt5 import QtCore, QtWidgets
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(f"Missing dependency: {exc}. Please run: pip install PyQt5") from exc

        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.process: QtCore.QProcess | None = None

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Frame Type Slicer")
        self.window.resize(980, 720)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        form = QtWidgets.QGridLayout()
        root.addLayout(form)
        self.video_edit = self._path_row(form, 0, "视频文件", "选择视频", self.choose_video)
        self.model_edit = self._path_row(form, 1, "game/break 模型", "选择模型", self.choose_model, "models/frame_type_game_break.pt")
        self.replay_model_edit = self._path_row(form, 2, "replay 标记模型", "选择模型", self.choose_replay_model, "")
        self.frame_output_edit = self._path_row(form, 3, "逐帧输出 JSONL", "选择输出", self.choose_frame_output, "output/frame_type_rows.jsonl")
        self.segment_output_edit = self._path_row(form, 4, "切片输出 JSON", "选择输出", self.choose_segment_output, "output/frame_type_segments.json")
        self.markers_edit = self._path_row(form, 5, "人工 markers", "选择JSON", self.choose_markers, "")
        self.demo_rounds_edit = self._path_row(form, 6, "demo rounds", "选择JSON", self.choose_demo_rounds, "")

        params = QtWidgets.QGroupBox("参数")
        root.addWidget(params)
        params_layout = QtWidgets.QGridLayout(params)
        self.interval_spin = self._double_spin(params_layout, 0, "采样间隔秒", 1.0, 0.1, 60.0, 1)
        self.start_spin = self._double_spin(params_layout, 1, "开始秒", 0.0, 0.0, 999999.0, 1)
        self.end_spin = self._double_spin(params_layout, 2, "结束秒(0=不限制)", 0.0, 0.0, 999999.0, 1)
        self.smooth_spin = self._int_spin(params_layout, 3, "平滑窗口", 5, 1, 99)
        self.min_live_spin = self._double_spin(params_layout, 4, "最短 live 秒", 20.0, 0.0, 999999.0, 1)
        self.bridge_gap_spin = self._double_spin(params_layout, 5, "合并间隔秒", 3.0, 0.0, 999999.0, 1)
        self.bridge_gap_max_spin = self._double_spin(params_layout, 6, "timer连续最大间隔秒", 20.0, 0.0, 999999.0, 1)
        self.live_label_edit = QtWidgets.QLineEdit("game")
        params_layout.addWidget(QtWidgets.QLabel("live 标签"), 7, 0)
        params_layout.addWidget(self.live_label_edit, 7, 1)
        self.device_edit = QtWidgets.QLineEdit("auto")
        params_layout.addWidget(QtWidgets.QLabel("设备"), 8, 0)
        params_layout.addWidget(self.device_edit, 8, 1)
        self.replay_roi_edit = QtWidgets.QLineEdit("0,0,0.32,0.18")
        params_layout.addWidget(QtWidgets.QLabel("replay ROI"), 9, 0)
        params_layout.addWidget(self.replay_roi_edit, 9, 1)
        self.timer_roi_edit = QtWidgets.QLineEdit("0.46,0.0,0.54,0.06")
        params_layout.addWidget(QtWidgets.QLabel("timer ROI"), 10, 0)
        params_layout.addWidget(self.timer_roi_edit, 10, 1)
        self.score_roi_edit = QtWidgets.QLineEdit("0.42,0.0,0.58,0.08")
        params_layout.addWidget(QtWidgets.QLabel("score ROI"), 11, 0)
        params_layout.addWidget(self.score_roi_edit, 11, 1)
        self.replay_threshold_spin = self._double_spin(params_layout, 12, "replay 阈值", 0.65, 0.0, 1.0, 2)

        actions = QtWidgets.QHBoxLayout()
        root.addLayout(actions)
        self.run_button = QtWidgets.QPushButton("开始切片")
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.command_button = QtWidgets.QPushButton("复制命令到日志")
        actions.addWidget(self.run_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.command_button)
        actions.addStretch(1)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)
        root.addWidget(self.log, 1)

        self.run_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.command_button.clicked.connect(lambda: self.append_log(" ".join(self.build_command())))

    def _path_row(self, form, row: int, label: str, button_text: str, handler, default: str = ""):
        edit = self.QtWidgets.QLineEdit(default)
        button = self.QtWidgets.QPushButton(button_text)
        button.clicked.connect(handler)
        form.addWidget(self.QtWidgets.QLabel(label), row, 0)
        form.addWidget(edit, row, 1)
        form.addWidget(button, row, 2)
        return edit

    def _double_spin(self, layout, row: int, label: str, value: float, minimum: float, maximum: float, decimals: int):
        spin = self.QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        layout.addWidget(self.QtWidgets.QLabel(label), row, 0)
        layout.addWidget(spin, row, 1)
        return spin

    def _int_spin(self, layout, row: int, label: str, value: int, minimum: int, maximum: int):
        spin = self.QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        layout.addWidget(self.QtWidgets.QLabel(label), row, 0)
        layout.addWidget(spin, row, 1)
        return spin

    def choose_video(self) -> None:
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(self.window, "选择视频", str(PROJECT_ROOT), "Video (*.mp4 *.mkv *.avi *.mov *.flv *.webm);;All files (*)")
        if path:
            self.video_edit.setText(path)

    def choose_model(self) -> None:
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(self.window, "选择模型", str(PROJECT_ROOT), "PyTorch (*.pt *.pth);;All files (*)")
        if path:
            self.model_edit.setText(path)

    def choose_replay_model(self) -> None:
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(self.window, "选择 replay 标记模型", str(PROJECT_ROOT), "PyTorch (*.pt *.pth);;All files (*)")
        if path:
            self.replay_model_edit.setText(path)

    def choose_frame_output(self) -> None:
        path, _ = self.QtWidgets.QFileDialog.getSaveFileName(self.window, "选择逐帧输出", self.frame_output_edit.text(), "JSONL (*.jsonl);;All files (*)")
        if path:
            self.frame_output_edit.setText(path)

    def choose_segment_output(self) -> None:
        path, _ = self.QtWidgets.QFileDialog.getSaveFileName(self.window, "选择切片输出", self.segment_output_edit.text(), "JSON (*.json);;All files (*)")
        if path:
            self.segment_output_edit.setText(path)

    def choose_markers(self) -> None:
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(self.window, "选择人工 markers", str(PROJECT_ROOT), "JSON (*.json);;All files (*)")
        if path:
            self.markers_edit.setText(path)

    def choose_demo_rounds(self) -> None:
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(self.window, "选择 demo rounds.json", str(PROJECT_ROOT), "JSON (*.json);;All files (*)")
        if path:
            self.demo_rounds_edit.setText(path)

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--video",
            self.video_edit.text(),
            "--model",
            self.model_edit.text(),
            "--replay-model",
            self.replay_model_edit.text(),
            "--replay-roi",
            self.replay_roi_edit.text(),
            "--replay-threshold",
            str(self.replay_threshold_spin.value()),
            "--timer-roi",
            self.timer_roi_edit.text(),
            "--score-roi",
            self.score_roi_edit.text(),
            "--frame-output",
            self.frame_output_edit.text(),
            "--segment-output",
            self.segment_output_edit.text(),
            "--markers",
            self.markers_edit.text(),
            "--demo-rounds",
            self.demo_rounds_edit.text(),
            "--interval-sec",
            str(self.interval_spin.value()),
            "--start-sec",
            str(self.start_spin.value()),
            "--smooth-window",
            str(self.smooth_spin.value()),
            "--live-label",
            self.live_label_edit.text(),
            "--min-live-sec",
            str(self.min_live_spin.value()),
            "--bridge-gap-sec",
            str(self.bridge_gap_spin.value()),
            "--bridge-gap-max-sec",
            str(self.bridge_gap_max_spin.value()),
            "--device",
            self.device_edit.text(),
            "--progress-every",
            "5",
        ]
        if self.end_spin.value() > 0:
            cmd.extend(["--end-sec", str(self.end_spin.value())])
        return cmd

    def start(self) -> None:
        if not self.video_edit.text().strip():
            self.append_log("请先选择视频文件。")
            return
        if self.process is not None:
            self.append_log("已有任务正在运行。")
            return
        cmd = self.build_command()
        self.append_log("启动命令：")
        self.append_log(" ".join(cmd))
        self.progress_bar.setValue(0)
        self.process = self.QtCore.QProcess(self.window)
        self.process.setProgram(cmd[0])
        self.process.setArguments(cmd[1:])
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.finished)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.process.start()

    def stop(self) -> None:
        if self.process is not None:
            self.process.kill()

    def read_stdout(self) -> None:
        if self.process is not None:
            text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace").rstrip()
            self.update_progress_from_text(text)
            self.append_log(text)

    def read_stderr(self) -> None:
        if self.process is not None:
            self.append_log(bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").rstrip())

    def finished(self, exit_code: int, _status) -> None:
        if exit_code == 0:
            self.progress_bar.setValue(100)
        self.append_log(f"任务结束，exit_code={exit_code}")
        self.process = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def append_log(self, text: str) -> None:
        if text:
            self.log.appendPlainText(text)

    def update_progress_from_text(self, text: str) -> None:
        for line in text.splitlines():
            if not line.startswith("progress:") or "%" not in line:
                continue
            try:
                percent_text = line.rsplit("(", 1)[1].split("%", 1)[0]
                self.progress_bar.setValue(max(0, min(100, int(float(percent_text)))))
            except (IndexError, ValueError):
                continue


def run_gui() -> int:
    try:
        from PyQt5 import QtWidgets
    except ImportError as exc:  # pragma: no cover
        print(f"Missing dependency: {exc}. Please run: pip install PyQt5")
        return 1
    app = QtWidgets.QApplication(sys.argv)
    ui = SlicerWindow()
    ui.window.show()
    return app.exec_()


def main() -> int:
    args = parse_args()
    if args.gui or not args.video:
        return run_gui()
    frame_output, segment_output = run(args)
    print(f"frames: {frame_output}")
    print(f"segments: {segment_output}")
    payload = json.loads(segment_output.read_text(encoding="utf-8"))
    print(f"summary: probed {payload.get('frame_count', 0)} frames, cut {payload.get('segment_count', 0)} live_game segments, total_live_sec={payload.get('total_live_sec', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
