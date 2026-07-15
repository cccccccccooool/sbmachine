#!/usr/bin/env python3
"""6657 离线录像解说流水线 —— 唯一启动入口。

用法：
  python run.py                  # 读 config/（默认）
  python run.py --config config/ # 同上，显式指定
  python run.py --dry-run        # 只跑 JSON 链路，不调任何 AI 模型

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


def main() -> int:
    ap = argparse.ArgumentParser(description="6657 解说流水线（config 驱动，一键运行）")
    ap.add_argument("--config", default="config/", help="配置目录或文件（默认 config/）")
    ap.add_argument("--dry-run", action="store_true", help="不调 AI，只跑 JSON 链路自检")
    args = ap.parse_args()

    try:
        config_path = require_path(args.config, "--config")
        result = run_all(config_path, dry_run=args.dry_run)
    except (ConfigError, OSError, ValueError) as exc:
        if args.dry_run:
            result = {
                "config_valid": False,
                "enabled_phases": [],
                "required_inputs": [],
                "services_started": [],
                "writes_performed": False,
                "errors": [str(exc)],
            }
        else:
            result = {
                "status": "failed",
                "publishable": False,
                "failed_stage": "config",
                "error": str(exc),
                "exit_code": 2,
            }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0 if result.get("config_valid", False) else 2
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
