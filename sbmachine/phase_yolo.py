"""第二阶段（YOLO / 时间线处理）子进程入口。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import load_config, require_debug_output, require_path
from sbmachine.phase2_yolo import run_phase2
from sbmachine.preflight import preflight_config, require_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = require_path(args.config, "--config")
    config = load_config(config_path)
    paths = config.get("paths", {})
    if args.dry_run:
        report = preflight_config(config, root=PACKAGE_ROOT, only={"phase2"})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["config_valid"] else 2
    rounds_p1 = require_path(paths.get("rounds_json", "output/sbmachine/rounds.json"), "paths.rounds_json")
    rounds_p2 = require_path(paths.get("rounds_with_yolo_json", "output/sbmachine/rounds_with_yolo.json"), "paths.rounds_with_yolo_json")
    semantic_p2 = require_path(
        paths.get("rounds_with_yolo_semantic_json", "output/sbmachine/rounds_with_yolo_semantic.json"),
        "paths.rounds_with_yolo_semantic_json",
    )
    require_debug_output(rounds_p2, "paths.rounds_with_yolo_json")
    if rounds_p2.exists():
        rounds_p2.unlink()
    if semantic_p2.exists():
        semantic_p2.unlink()
    run_phase2(rounds_path=rounds_p1, output_path=rounds_p2, config_path=config_path, semantic_output_path=semantic_p2)
    require_outputs("phase2", [rounds_p2, semantic_p2])
    print("[phase_yolo] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
