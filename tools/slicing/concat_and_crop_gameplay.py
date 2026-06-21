"""
本文件功能：视频缝合、游戏画面分类与裁剪。

启动方式：python tools/slicing/concat_and_crop_gameplay.py -i <input_dir> [选项]
输入数据流：包含输入视频的文件夹路径和 frame_type 分类模型权重。
输出数据流：缝合后的视频、分类 JSON 和仅包含游戏画面的最终视频。
用法用途：将多段视频缝合后用分类器自动识别 game 片段，裁剪并拼接为仅含游戏画面的视频。

=========================================
  2. 使用 frame_type_classifier.pt 分类器模型对缝合视频进行自动预测，生成保存分类信息的 JSON 文件。
  3. 根据 JSON 文件中的 "game" 标记片段，裁剪并拼接视频，生成仅包含游戏画面的最终视频。
  4. 保留 JSON 描述文件。

使用方法 (Usage):
  python tools/slicing/concat_and_crop_gameplay.py -i <input_dir> [-o <output_dir>] [-m <model_path>] [--reencode]

参数说明:
  -i, --input-dir: 包含输入视频的文件夹路径 (必填)
  -o, --output-dir: 结果输出文件夹路径 (默认: output/gameplay_process)
  -m, --model: 模型权重路径 (默认: models/qiepian/frame_type_classifier.pt)
  --interval-sec: 采样间隔秒数 (默认: 1.0)
  --smooth-window: 标签平滑滑动窗口大小 (默认: 5)
  --min-live-sec: 最小有效游戏片段秒数 (默认: 10.0)
  --bridge-gap-sec: 允许桥接合并的最大间隔秒数 (默认: 3.0)
  --reencode: 是否重新编码切片 (建议开启以获得更精准的切点，不开启则进行快速复制)
  --device: 设备类型 (auto/cuda/cpu) (默认: auto)
  --batch-size: 批量推理大小，GPU 建议 32-64 (默认: 32)
  --cut-workers: 并行裁剪线程数 (默认: 4)
=========================================
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PyQt5 / PyTorch WinError DLL load workaround
try:
    import torch
except ImportError:
    pass

from vision_service.frame_type_model import load_checkpoint, resolve_device

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".ts", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="视频缝合、游戏画面分类与裁剪脚本。")
    parser.add_argument("--input-dir", "-i", required=True, help="包含输入视频的文件夹路径")
    parser.add_argument("--output-dir", "-o", default="output/gameplay_process", help="输出结果的保存目录")
    parser.add_argument("--model", "-m", default="models/qiepian/frame_type_classifier.pt", help="模型权重路径")
    parser.add_argument("--interval-sec", type=float, default=1.0, help="采样间隔秒数")
    parser.add_argument("--smooth-window", type=int, default=5, help="标签平滑滑动窗口大小")
    parser.add_argument("--min-live-sec", type=float, default=10.0, help="最小有效游戏片段秒数")
    parser.add_argument("--bridge-gap-sec", type=float, default=3.0, help="允许桥接合并的最大间隔秒数")
    parser.add_argument("--reencode", action="store_true",
                        help="重新编码切片 (精准切口，较慢)；不开启则 copy 极快但切口在关键帧处")
    parser.add_argument("--device", default="auto", help="推理设备 (auto/cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批量推理大小，GPU 建议 32-64，CPU 建议 4-8")
    parser.add_argument("--cut-workers", type=int, default=4, help="并行裁剪线程数")
    return parser.parse_args()


# ── 帧提取：seek 直跳，省去中间帧解码 ─────────────────────────────────────────

def get_video_info(video_path: Path) -> tuple[float, float, int]:
    """返回 (duration_sec, fps, total_frames)"""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if fps > 0 else 0.0
        return duration, fps, total
    finally:
        cap.release()


def iter_sampled_frames(video_path: Path, interval_sec: float):
    """用 ffmpeg pipe 输出采样帧（fps=1/interval_sec），比 OpenCV seek/grab 都快。
    ffmpeg 内部用高效 skip 逻辑，不走 Python 逐帧循环。"""
    import numpy as np
    import cv2

    # 先拿宽高
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_bytes = w * h * 3
    # fps=1/N 让 ffmpeg 每 interval_sec 秒输出一帧
    vf = f"fps=1/{interval_sec}" if interval_sec >= 1.0 else f"fps={1.0/interval_sec:.6f}"
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", vf,
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    ts = 0.0
    try:
        while True:
            data = proc.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()
            yield round(ts, 3), frame
            ts += interval_sec
    finally:
        proc.stdout.close()
        proc.wait()


# ── 批量推理 ──────────────────────────────────────────────────────────────────

def predict_batch(model, labels: list[str], frames_bgr: list, img_size: int, device) -> list[dict]:
    import numpy as np
    import torch
    from vision_service.frame_type_model import frame_to_tensor

    tensors = [frame_to_tensor(f, img_size).squeeze(0) for f in frames_bgr]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
    out = []
    for p in probs:
        idx = int(np.argmax(p))
        out.append({"label": labels[idx], "confidence": round(float(p[idx]), 6)})
    return out


# ── 工具 ──────────────────────────────────────────────────────────────────────

def to_relative_path(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def safe_unlink(path: Path):
    try:
        if path.exists():
            path.unlink()
    except OSError:
        print(f"Warning: 无法删除临时文件 {path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    def resolve(p: str, default_root=PROJECT_ROOT) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (default_root / path).resolve()

    input_dir  = resolve(args.input_dir)
    output_dir = input_dir
    model_path = resolve(args.model)

    if not input_dir.exists():
        print(f"Error: 输入目录不存在: {input_dir}"); return 1
    if not model_path.exists():
        print(f"Error: 模型文件不存在: {model_path}"); return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 步骤 1: 扫描 & 缝合 ──────────────────────────────────────────────────
    print(f"扫描目录中的视频: {input_dir}")
    video_files = sorted(p for p in input_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not video_files:
        print("未在输入目录中找到任何视频文件。"); return 1

    print(f"找到 {len(video_files)} 个视频文件:")
    for v in video_files:
        print(f"  - {v.name}")

    print(f"\n[步骤 1/3] 缝合视频...")
    if len(video_files) == 1:
        # 单文件直接使用，省去拷贝开销
        merged_video_path = video_files[0]
        print(f"单文件，直接使用: {merged_video_path.name}")
    else:
        merged_video_path = output_dir / "merged_output.mp4"
        concat_list = output_dir / "temp_concat_list.txt"
        with concat_list.open("w", encoding="utf-8") as f:
            for v in video_files:
                f.write(f"file '{str(v.resolve()).replace(chr(92), '/')}'\n")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(concat_list), "-c", "copy", str(merged_video_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            print("缝合完成。")
        except subprocess.CalledProcessError as e:
            print(f"缝合失败: {e.stderr.decode(errors='replace')}"); return 1
        finally:
            safe_unlink(concat_list)

    # ── 步骤 2: 批量推理 ──────────────────────────────────────────────────────
    print(f"\n[步骤 2/3] 加载模型并预测...")
    device = resolve_device(args.device)
    print(f"设备: {device}")

    try:
        model, labels, img_size, _ = load_checkpoint(model_path, device)
    except Exception as e:
        print(f"加载模型失败: {e}"); return 1
    print(f"类别: {labels}")

    duration, fps, total_frames = get_video_info(merged_video_path)
    print(f"视频: {duration:.1f}s  {fps:.2f}fps  {total_frames}帧  "
          f"→ 每 {args.interval_sec}s 采样一帧，共约 {int(duration / args.interval_sec)} 点")

    frame_iter = iter_sampled_frames(merged_video_path, args.interval_sec)
    try:
        from tqdm import tqdm
        frame_iter = tqdm(frame_iter, total=int(duration / args.interval_sec), desc="推理")
    except ImportError:
        pass

    predictions: list[dict] = []
    buf_frames: list = []
    buf_ts:     list[float] = []

    def flush_batch():
        if not buf_frames:
            return
        preds = predict_batch(model, labels, buf_frames, img_size, device)
        for ts, pred in zip(buf_ts, preds):
            predictions.append({"time_sec": ts, **pred})
        buf_frames.clear()
        buf_ts.clear()

    for ts, frame in frame_iter:
        buf_frames.append(frame)
        buf_ts.append(ts)
        if len(buf_frames) >= args.batch_size:
            flush_batch()
    flush_batch()

    print(f"预测完成，共 {len(predictions)} 个采样点。")

    # 标签平滑
    window: deque[str] = deque()
    smoothed: list[str] = []
    for p in predictions:
        window.append(p["label"])
        if len(window) > args.smooth_window:
            window.popleft()
        smoothed.append(Counter(window).most_common(1)[0][0])
    for p, s in zip(predictions, smoothed):
        p["smooth_label"] = s

    # 提取片段
    raw_segments: list[dict] = []
    active: dict | None = None
    for p in predictions:
        ts = p["time_sec"]
        if p["smooth_label"] == "game":
            if active is None:
                active = {"start_sec": ts, "end_sec": ts}
            else:
                active["end_sec"] = ts
        elif active is not None:
            raw_segments.append(active)
            active = None
    if active is not None:
        raw_segments.append(active)

    # 桥接
    merged_segs: list[dict] = []
    for seg in raw_segments:
        if merged_segs and seg["start_sec"] - merged_segs[-1]["end_sec"] <= args.bridge_gap_sec:
            merged_segs[-1]["end_sec"] = seg["end_sec"]
        else:
            merged_segs.append(seg)

    # 过滤短片段
    final_segments = [
        {"start_sec": s["start_sec"], "end_sec": s["end_sec"],
         "duration_sec": round(s["end_sec"] - s["start_sec"], 3), "kind": "live_round"}
        for s in merged_segs
        if s["end_sec"] - s["start_sec"] >= args.min_live_sec
    ]
    print(f"提取到 {len(final_segments)} 个游戏片段。")

    # 写 JSON
    json_path = output_dir / "classification.json"
    json_path.write_text(json.dumps({
        "video": to_relative_path(merged_video_path),
        "model": to_relative_path(model_path),
        "interval_sec": args.interval_sec,
        "smooth_window": args.smooth_window,
        "min_live_sec": args.min_live_sec,
        "bridge_gap_sec": args.bridge_gap_sec,
        "segment_count": len(final_segments),
        "segments": final_segments,
        "predictions": predictions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON 已保存: {json_path}")

    # ── 步骤 3: 并行裁剪 & 合并 ───────────────────────────────────────────────
    print(f"\n[步骤 3/3] 裁剪游戏片段...")
    if not final_segments:
        print("无符合条件的片段，跳过裁剪。"); return 0

    final_video_path = output_dir / "gameplay_only.mp4"
    temp_clips: list[Path] = []

    def cut_clip(idx: int, seg: dict) -> tuple[int, Path]:
        out = output_dir / f"temp_clip_{idx:03d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{seg['start_sec']:.3f}",
            "-i", str(merged_video_path),
            "-t", f"{seg['duration_sec']:.3f}",
        ]
        if args.reencode:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"]
        else:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        cmd.append(str(out))
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return idx, out

    try:
        clip_map: dict[int, Path] = {}
        with ThreadPoolExecutor(max_workers=args.cut_workers) as ex:
            futs = {ex.submit(cut_clip, i, seg): i for i, seg in enumerate(final_segments, 1)}
            for fut in as_completed(futs):
                idx, path = fut.result()
                clip_map[idx] = path
                seg = final_segments[idx - 1]
                print(f"  片段 {idx}/{len(final_segments)}: {seg['start_sec']:.1f}s-{seg['end_sec']:.1f}s 完成")

        temp_clips = [clip_map[i] for i in sorted(clip_map)]

        clips_list = output_dir / "temp_clips_list.txt"
        with clips_list.open("w", encoding="utf-8") as f:
            for clip in temp_clips:
                f.write(f"file '{str(clip.resolve()).replace(chr(92), '/')}'\n")

        print(f"合并所有片段 → {final_video_path}...")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(clips_list), "-c", "copy", str(final_video_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        print(f"完成: {final_video_path}")

    except subprocess.CalledProcessError as e:
        print(f"裁剪/合并失败: {e}"); return 1
    finally:
        for clip in temp_clips:
            safe_unlink(clip)
        if "clips_list" in locals():
            safe_unlink(clips_list)

    print("\n全部完成:")
    print(f"  缝合视频  : {merged_video_path}")
    print(f"  分类 JSON : {json_path}")
    print(f"  游戏视频  : {final_video_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
