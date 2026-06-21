#!/usr/bin/env python3
"""6657 离线录像解说流水线 —— 唯一启动入口。

用法：
  python run.py                  # 读 config/（默认）
  python run.py --config config/ # 同上，显式指定
  python run.py --dry-run        # 只跑 JSON 链路，不调任何 AI 模型

AI 服务（Ollama / VLM / SoVITS）生命周期由 config/pipeline.yaml 控制：
  runtime.manage_services: false（默认）→ 用户手动启动各服务后再运行此脚本
  runtime.manage_services: true         → run.py 自动拉起、健康检查、结束后关闭
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbmachine.common import load_config, require_path
from sbmachine.run_all import run_all


def main() -> int:
    ap = argparse.ArgumentParser(description="6657 解说流水线（config 驱动，一键运行）")
    ap.add_argument("--config", default="config/", help="配置目录或文件（默认 config/）")
    ap.add_argument("--dry-run", action="store_true", help="不调 AI，只跑 JSON 链路自检")
    args = ap.parse_args()

    config_path = require_path(args.config, "--config")
    config = load_config(config_path)
    manage = bool(config.get("runtime", {}).get("manage_services", False))

    print(f"[run.py] config={config_path}  dry_run={args.dry_run}  manage_services={manage}", flush=True)

    run_all(config_path, dry_run=args.dry_run)
    print("[run.py] 全部阶段完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
