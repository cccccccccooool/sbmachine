"""
本文件功能：根据 segments.json 调用 ffmpeg 对视频进行无损或重编码剪切。

启动方式：python tools/slicing/cut_segments_ffmpeg.py --segments <segments_json> [选项]
输入数据流：包含片段信息的 segments.json 文件和视频源文件。
输出数据流：按片段裁剪后的 mp4 视频文件。
用法用途：读取 segments.json 中的时间段，调用 ffmpeg 逐段切剪视频，支持无损 copy 和重编码两种模式。

=========================================
使用方法 (Usage):
    python tools/slicing/cut_segments_ffmpeg.py --segments <segments_json> [选项]
    参数说明:
      --segments: 输入的 segments.json 路径 (必填)
      --video: 可选，覆盖 segments.json 中的视频源路径
      --output-dir: 剪辑输出保存文件夹 (默认: output/manual_clips)
      --prefix: 输出文件名的前缀 (默认: segment)
      --reencode: 是否重新编码，若开启可获得更准的帧切点，但速度稍慢 (命令行标志)
=========================================
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_segments(path: Path) -> tuple[str, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return str(payload.get("video", "")), payload.get("segments", []) or []
    if isinstance(payload, list):
        return "", payload
    raise ValueError("segments JSON must be a list or an object with segments.")


def cut_clip(video: Path, output: Path, start_sec: float, end_sec: float, reencode: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.05, end_sec - start_sec)
    cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", str(video), "-t", f"{duration:.3f}"]
    if reencode:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"])
    else:
        cmd.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])
    cmd.append(str(output))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 segments.json 调用 ffmpeg 切视频")
    parser.add_argument("--segments", required=True)
    parser.add_argument("--video", default="", help="可选：覆盖 segments JSON 里的 video 字段")
    parser.add_argument("--output-dir", default="output/manual_clips")
    parser.add_argument("--prefix", default="segment")
    parser.add_argument("--reencode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embedded_video, segments = load_segments(resolve_path(args.segments))
    video = resolve_path(args.video or embedded_video)
    output_dir = resolve_path(args.output_dir)
    for index, segment in enumerate(segments, start=1):
        start_sec = float(segment["start_sec"])
        end_sec = float(segment["end_sec"])
        kind = str(segment.get("kind", "clip"))
        output = output_dir / f"{args.prefix}_{index:03d}_{kind}_{start_sec:.3f}-{end_sec:.3f}.mp4"
        cut_clip(video, output, start_sec, end_sec, args.reencode)
        print(f"cut: {output}")
    print(f"done: {len(segments)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
