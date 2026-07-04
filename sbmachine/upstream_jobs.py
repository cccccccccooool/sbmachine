"""Upstream subprocess jobs used by the pipeline skeleton."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sbmachine.common import resolve_path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _call_gpu_guard(action: str, use_gpu_guard: bool) -> None:
    """调用 gpu_guard 守护进程进行显存释放或恢复霸占"""
    if not use_gpu_guard:
        return
    script = PACKAGE_ROOT / "tools" / "gpu_guard.py"
    if script.exists():
        try:
            print(f"[gpu_guard] {action}...")
            subprocess.run([sys.executable, str(script), action], check=False)
        except Exception as e:
            print(f"[gpu_guard] error: {e}")


def _run_demo_parse(paths: dict) -> None:
    """调 tools/parse_demo.py(Go 解析器)把 .dem 解析成 output/demo 工件。"""
    demo = resolve_path(paths.get("demo"))
    if demo is None:
        raise ValueError("phases.demo_parse 已开启,但 paths.demo 未配置 .dem 路径")
    out_dir = str(paths.get("demo_output_dir", "output/demo"))
    script = PACKAGE_ROOT / "tools" / "parse_demo.py"
    print(f"[demo_parse] {demo} → {out_dir}")
    result = subprocess.run([sys.executable, str(script), "--demo", str(demo), "--output-dir", out_dir])
    if result.returncode != 0:
        raise RuntimeError(f"parse_demo 失败 (exit {result.returncode})")


def _run_video_marking(paths: dict, slicer_config: dict, use_gpu_guard: bool) -> Path:
    """自动调用 tools/slicing/run_frame_type_slicer.py 进行画面预测和标记。"""
    video = resolve_path(paths.get("video"))
    if video is None:
        raise ValueError("phases.video_marking 已开启,但 paths.video 未配置")

    model = resolve_path(slicer_config.get("model", "models/qiepian/frame_type_classifier.pt"))
    if model is None or not model.exists():
        raise FileNotFoundError(f"未找到视频分类模型:{model}")

    out_jsonl = PACKAGE_ROOT / "output" / "sbmachine" / "detector_rows.jsonl"
    out_segments = PACKAGE_ROOT / "output" / "sbmachine" / "segments.json"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    script = PACKAGE_ROOT / "tools" / "slicing" / "run_frame_type_slicer.py"
    print(f"[video_marking] 开始预测视频帧分类: {video} → {out_jsonl}")

    cmd = [
        sys.executable,
        str(script),
        "--video", str(video),
        "--model", str(model),
        "--frame-output", str(out_jsonl),
        "--segment-output", str(out_segments),
        "--interval-sec", str(slicer_config.get("interval_sec", 1.0)),
        "--smooth-window", str(slicer_config.get("smooth_window", 5)),
        "--min-live-sec", str(slicer_config.get("min_live_sec", 20.0)),
        "--bridge-gap-sec", str(slicer_config.get("bridge_gap_sec", 3.0)),
    ]

    demo_rounds = resolve_path(paths.get("demo_output_dir", "output/demo")) / "rounds.json"
    if demo_rounds.exists():
        cmd.extend(["--demo-rounds", str(demo_rounds)])

    _call_gpu_guard("release", use_gpu_guard)
    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"run_frame_type_slicer 运行失败 (exit {result.returncode})")
    finally:
        _call_gpu_guard("resume", use_gpu_guard)

    return out_jsonl

