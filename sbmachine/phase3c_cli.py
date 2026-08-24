"""Phase3c（LLM-C 独立阶段）子进程入口：加载 → 运行 → 退出。

用法：python -m sbmachine.phase3c_cli --config config/ [--dry-run]
对齐 sbmachine/phase_tts.py 的子进程模式；产物按输入合同版本输出 v1 或 v2。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import ensure_output_paths, load_config, require_debug_output, require_path
from sbmachine.phase3c_llmc import run_phase3c
from sbmachine.preflight import preflight_config, require_outputs
from sbmachine.progress_events import ProgressEventWriter


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
        report = preflight_config(config, root=PACKAGE_ROOT, only={"phase3c"})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["config_valid"] else 2

    writer = (
        ProgressEventWriter(Path(args.progress_events), run_id=args.progress_run_id)
        if args.progress_events and args.progress_run_id else None
    )
    def progress_sink(completed, total, unit, detail):
        if writer is not None:
            writer.emit(event="stage_progress", stage="phase3c", completed=completed, total=total, unit=unit, detail=detail)

    draft = require_path(paths.get("llmb_draft_package_json", "output/sbmachine/llmb_draft_package.json"), "paths.llmb_draft_package_json")
    output = require_path(paths.get("commentary_render_package_json", "output/sbmachine/commentary_render_package.json"), "paths.commentary_render_package_json")
    require_debug_output(output, "paths.commentary_render_package_json")
    if output.exists():
        output.unlink()

    result = run_phase3c(
        draft_package_path=draft,
        output_render_path=output,
        config_path=config_path,
        progress_sink=progress_sink,
    )
    if writer is not None:
        writer.emit(event="stage_work_complete", stage="phase3c")
    require_outputs("phase3c", [output])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("[phase3c_cli] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
