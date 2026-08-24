"""phase4_assemble (TTS) 子进程入口。加载 → 运行 → 退出（操作系统将在退出时回收显存）。

用法：python -m sbmachine.phase_tts --config config/ [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import ensure_output_paths, load_config, require_debug_output, require_path, resolve_path
from sbmachine.file_lock import FileLock
from sbmachine.phase4_assemble import run_phase4
from sbmachine.progress_events import ProgressEventWriter
from sbmachine.preflight import preflight_config, require_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-events")
    parser.add_argument("--progress-run-id")
    args = parser.parse_args()

    config_path = require_path(args.config, "--config")
    config = ensure_output_paths(load_config(config_path))
    paths = config.get("paths", {})

    if args.dry_run:
        report = preflight_config(config, root=PACKAGE_ROOT, only={"phase4"})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["config_valid"] else 2

    writer = (
        ProgressEventWriter(Path(args.progress_events), run_id=args.progress_run_id)
        if args.progress_events and args.progress_run_id else None
    )
    def progress_sink(completed, total, unit, detail):
        if writer is not None:
            writer.emit(event="stage_progress", stage="phase4", completed=completed, total=total, unit=unit, detail=detail)
    rounds_p3 = require_path(paths.get("rounds_with_commentary_json", "output/sbmachine/rounds_with_commentary.json"), "paths.rounds_with_commentary_json")
    render_package = (
        require_path(paths.get("commentary_render_package_json"), "paths.commentary_render_package_json")
        if (config.get("phases", {}) or {}).get("phase3c_render", False) else None
    )
    commentary = (
        None
        if render_package is not None
        else require_path(paths.get("commentary_json", "output/sbmachine/commentary.json"), "paths.commentary_json")
    )
    rounds_p4 = require_path(paths.get("rounds_final_json", "output/sbmachine/rounds_final.json"), "paths.rounds_final_json")
    manifest = require_path(paths.get("assemble_manifest_json", "output/sbmachine/assemble_manifest.json"), "paths.assemble_manifest_json")
    require_debug_output(rounds_p4, "paths.rounds_final_json")
    require_debug_output(manifest, "paths.assemble_manifest_json")

    for output in (rounds_p4, manifest):
        if output.exists():
            output.unlink()
    output_dir = resolve_path(config.get("phase4", {}).get("output_dir"))
    if output_dir is not None:
        require_debug_output(output_dir, "phase4.output_dir")
    if output_dir is not None and output_dir.exists():
        shutil.rmtree(output_dir)
    with FileLock(PACKAGE_ROOT / "output" / ".sovits.lock"):
        run_phase4(
            rounds_path=rounds_p3,
            commentary_path=commentary,
            render_package_path=render_package,
            output_rounds_path=rounds_p4,
            manifest_path=manifest,
            config_path=config_path,
            progress_sink=progress_sink,
        )
    if writer is not None:
        writer.emit(event="stage_work_complete", stage="phase4")
    require_outputs("phase4", [rounds_p4, manifest])
    print("[phase_tts] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
