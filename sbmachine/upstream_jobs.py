"""流水线骨架使用的上游子进程任务（demo 解析、视频打标）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sbmachine.common import resolve_path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _call_gpu_guard(action: str, use_gpu_guard: bool, log_path: Path | None = None) -> None:
    """调用 gpu_guard 守护进程进行显存释放或恢复霸占

    log_path 非 None 时,子进程输出与本函数的日志行一并追加写入该文件,
    避免直写 fd 穿透 rich Live 造成进度条渲染撕裂。
    """
    if not use_gpu_guard:
        return
    script = PACKAGE_ROOT / "tools" / "start" / "gpu_guard.py"
    if script.exists():
        cmd = [sys.executable, str(script), action]
        try:
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                # 追加模式:同一次运行内 release/resume 会被多个阶段反复调用,
                # 用 "a" 保留完整时序,避免 "w" 互相覆盖只剩最后一次。
                # diagnostics_dir 每次运行都是新建目录,不会残留上一轮内容。
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"[gpu_guard] {action}...\n")
                    log_file.flush()
                    res = subprocess.run(cmd, check=False, stdout=log_file, stderr=subprocess.STDOUT)
                    if res.returncode != 0:
                        log_file.write(f"[gpu_guard] warning: {action} failed with returncode {res.returncode}\n")
            else:
                print(f"[gpu_guard] {action}...")
                res = subprocess.run(cmd, check=False)
                if res.returncode != 0:
                    print(f"[gpu_guard] warning: {action} failed with returncode {res.returncode}")
        except Exception as exc:
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"[gpu_guard] error: {exc}\n")
            else:
                print(f"[gpu_guard] error: {exc}")


def _run_demo_parse(paths: dict, log_path: Path | None = None) -> None:
    """调 tools/demo/parse_demo.py(Go 解析器)把 .dem 解析成 output/demo 工件。

    log_path 非 None 时,Go 解析器的全部输出与起始日志行一并写入该文件,
    避免直写 fd 穿透 rich Live 造成进度条渲染撕裂。
    """
    demo = resolve_path(paths.get("demo"))
    if demo is None:
        raise ValueError("phases.demo_parse 已开启,但 paths.demo 未配置 .dem 路径")
    out_dir = str(paths.get("demo_output_dir", "output/demo"))
    script = PACKAGE_ROOT / "tools" / "demo" / "parse_demo.py"
    cmd = [sys.executable, str(script), "--demo", str(demo), "--output-dir", out_dir]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"[demo_parse] {demo} → {out_dir}\n")
            log_file.flush()
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    else:
        print(f"[demo_parse] {demo} → {out_dir}")
        result = subprocess.run(cmd)
    if result.returncode != 0:
        detail = f" (日志: {log_path})" if log_path is not None else ""
        raise RuntimeError(f"parse_demo 失败 (exit {result.returncode}){detail}")
