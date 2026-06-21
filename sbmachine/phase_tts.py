"""phase4_assemble (TTS) 子进程入口。加载 → 运行 → 退出（操作系统将在退出时回收显存）。

用法：python -m sbmachine.phase_tts --config config/ [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import load_config, require_path
from sbmachine.phase4_assemble import run_phase4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config_path = require_path(args.config, "--config")
    config = load_config(config_path)
    paths = config.get("paths", {})

    rounds_p3 = require_path(paths.get("rounds_with_commentary_json", "output/sbmachine/rounds_with_commentary.json"), "paths.rounds_with_commentary_json")
    rounds_p4 = require_path(paths.get("rounds_final_json", "output/sbmachine/rounds_final.json"), "paths.rounds_final_json")
    manifest = require_path(paths.get("assemble_manifest_json", "output/sbmachine/assemble_manifest.json"), "paths.assemble_manifest_json")

    run_phase4(rounds_path=rounds_p3, output_rounds_path=rounds_p4, manifest_path=manifest, config_path=config_path, dry_run=args.dry_run)
    print("[phase_tts] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
