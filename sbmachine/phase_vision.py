"""phase2_vision 子进程入口。加载 → 运行 → 退出（操作系统将在退出时回收显存）。

用法：python -m sbmachine.phase_vision --config config/ [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import load_config, require_path, resolve_path
from sbmachine.phase2_vision import run_phase2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config_path = require_path(args.config, "--config")
    config = load_config(config_path)
    paths = config.get("paths", {})

    rounds_p1 = require_path(paths.get("rounds_json", "output/sbmachine/rounds.json"), "paths.rounds_json")
    rounds_p2 = require_path(paths.get("rounds_with_vision_json", "output/sbmachine/rounds_with_vision.json"), "paths.rounds_with_vision_json")

    run_phase2(rounds_path=rounds_p1, output_path=rounds_p2, config_path=config_path, dry_run=args.dry_run)
    print("[phase_vision] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
