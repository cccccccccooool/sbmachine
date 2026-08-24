#!/usr/bin/env python3
"""6657 离线录像解说流水线 —— 唯一启动入口。

用法：
  python run.py                  # 读 config/（默认），终端 GUI 模式
  python run.py --config config/ # 同上，显式指定
  python run.py --dry-run        # 只跑 JSON 链路，不调任何 AI 模型
  python run.py --debug          # 调试模式：透传 print，末尾输出原始 JSON

AI 服务（vLLM / SoVITS）生命周期由 config/pipeline.yaml 控制：
  runtime.manage_services: false（默认）→ 用户手动启动各服务后再运行此脚本
  runtime.manage_services: true         → run.py 自动拉起、健康检查、结束后关闭
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_loader import ConfigError
from sbmachine.common import require_path
from sbmachine.run_all import run_all


def _build_error_result(exc: BaseException, dry_run: bool) -> dict:
    """把启动期异常包装成与 run_all() 同形态的结果字典。"""
    if dry_run:
        return {
            "config_valid": False,
            "enabled_phases": [],
            "required_inputs": [],
            "services_started": [],
            "writes_performed": False,
            "errors": [str(exc)],
        }
    return {
        "status": "failed",
        "publishable": False,
        "failed_stage": "config",
        "error": str(exc),
        "exit_code": 2,
    }


def _run_text_mode(config_arg: str, dry_run: bool, *, debug_mode: bool, turbo: bool) -> int:
    """纯文本模式：直通 print，末尾输出机器可解析的原始 JSON，返回 exit_code。

    既是 --debug 的执行体，也是 rich 缺失时默认模式的降级目标。两者只差
    debug_mode：--debug 显式要调试（会开 phase3 调试落盘），而降级只是丢了
    渲染层，必须保持默认模式的语义，不能凭空打开调试落盘。
    """
    try:
        config_path = require_path(config_arg, "--config")
        result = run_all(config_path, dry_run=dry_run, debug_mode=debug_mode, turbo=turbo)
    except (ConfigError, OSError, ValueError) as exc:
        result = _build_error_result(exc, dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return 0 if result.get("config_valid", False) else 2
    return int(result.get("exit_code", 1))


def main() -> int:
    ap = argparse.ArgumentParser(description="6657 解说流水线（config 驱动，一键运行）")
    ap.add_argument("--config", default="config/", help="配置目录或文件（默认 config/）")
    ap.add_argument("--dry-run", action="store_true", help="不调 AI，只跑 JSON 链路自检")
    ap.add_argument(
        "--turbo",
        action="store_true",
        help="性能模式：放开资源锁（slicer workers 全量、关闭 CPU/内存节流），适合独占机器",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="调试模式：透传所有 print，末尾输出原始 JSON（机器可解析）",
    )
    args = ap.parse_args()

    if args.debug:
        return _run_text_mode(args.config, args.dry_run, debug_mode=True, turbo=args.turbo)

    try:
        from sbmachine.display import run_with_display
    except ImportError as exc:
        print(
            f"[run.py] 终端 GUI 渲染依赖 rich 不可用（{exc}）。\n"
            f"[run.py] 请执行：pip install rich==13.9.4\n"
            f"[run.py] 或改用 python run.py --debug 走纯文本模式。\n"
            f"[run.py] 本次已自动降级为纯文本模式继续运行。",
            file=sys.stderr,
            flush=True,
        )
        # debug_mode 保持 False：用户并未要求调试，不应因缺渲染依赖而附带触发 phase3 落盘。
        return _run_text_mode(args.config, args.dry_run, debug_mode=False, turbo=args.turbo)

    return run_with_display(args.config, dry_run=args.dry_run, turbo=args.turbo)


if __name__ == "__main__":
    raise SystemExit(main())
