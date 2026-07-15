"""流水线骨架使用的上游子进程任务（demo 解析、视频打标）。"""
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
    script = PACKAGE_ROOT / "tools" / "start" / "gpu_guard.py"
    if script.exists():
        try:
            print(f"[gpu_guard] {action}...")
            subprocess.run([sys.executable, str(script), action], check=False)
        except Exception as exc:
            print(f"[gpu_guard] error: {exc}")


def _run_demo_parse(paths: dict) -> None:
    """调 tools/demo/parse_demo.py(Go 解析器)把 .dem 解析成 output/demo 工件。"""
    demo = resolve_path(paths.get("demo"))
    if demo is None:
        raise ValueError("phases.demo_parse 已开启,但 paths.demo 未配置 .dem 路径")
    out_dir = str(paths.get("demo_output_dir", "output/demo"))
    script = PACKAGE_ROOT / "tools" / "demo" / "parse_demo.py"
    print(f"[demo_parse] {demo} → {out_dir}")
    result = subprocess.run([sys.executable, str(script), "--demo", str(demo), "--output-dir", out_dir])
    if result.returncode != 0:
        raise RuntimeError(f"parse_demo 失败 (exit {result.returncode})")
