"""phase3a + phase3b 子进程入口。两者在同一进程运行（a→b 共享模型加载），然后退出。

用法：python -m sbmachine.phase_semantic --config config/ [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import ensure_output_paths, load_config, require_debug_output, require_path, resolve_path
from sbmachine.phase3a_analyst import run_phase3a
from sbmachine.phase3b_style import run_phase3b
from sbmachine.progress_events import ProgressEventWriter
from sbmachine.neutral_contract import validate_neutral_manifest
from sbmachine.preflight import preflight_config, validate_commentary_publishable, validate_neutral_publishable


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

    rounds_p2 = require_path(paths.get("rounds_with_yolo_json", "output/sbmachine/rounds_with_yolo.json"), "paths.rounds_with_yolo_json")
    rounds_neutral = require_path(paths.get("rounds_with_neutral_json", "output/sbmachine/rounds_with_neutral.json"), "paths.rounds_with_neutral_json")
    rounds_p3 = require_path(paths.get("rounds_with_commentary_json", "output/sbmachine/rounds_with_commentary.json"), "paths.rounds_with_commentary_json")
    commentary = require_path(paths.get("commentary_json", "output/sbmachine/commentary.json"), "paths.commentary_json")
    draft_package = resolve_path(paths.get("llmb_draft_package_json"))  # Phase3b 出口封存；None=不导出

    phases = config.get("phases", {})
    p3a = bool(phases.get("phase3a_semantic", phases.get("phase3_semantic", True)))
    p3b = bool(phases.get("phase3b_semantic", phases.get("phase3_semantic", True)))

    if args.dry_run:
        active = {name for name, enabled in (("phase3a", p3a), ("phase3b", p3b)) if enabled}
        report = preflight_config(config, root=PACKAGE_ROOT, only=active)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["config_valid"] else 2

    writer = (
        ProgressEventWriter(Path(args.progress_events), run_id=args.progress_run_id)
        if args.progress_events and args.progress_run_id else None
    )
    def progress_sink(stage):
        def emit(completed, total, unit, detail):
            if writer is not None:
                writer.emit(event="stage_progress", stage=stage, completed=completed, total=total, unit=unit, detail=detail)
        return emit

    if p3a or p3b:
        # 云端 API 连通性预检（对齐本地 vLLM 服务的 startup_timeout 语义）：
        # 3 次尝试，全部失败立即中止，避免整场阶段在无效请求上白跑。
        from sbmachine.llm_shim import probe_api_connectivity

        semantic_cfg = config.get("semantic", {}) if isinstance(config.get("semantic", {}), dict) else {}
        probe_cfg = dict(config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {})
        probe_model = (
            semantic_cfg.get("analyst_model")
            or semantic_cfg.get("style_model")
            or semantic_cfg.get("model")
        )
        if probe_model:
            probe_cfg["model"] = probe_model
        try:
            probe_api_connectivity(probe_cfg, attempts=3, timeout_sec=30)
        except RuntimeError as exc:
            print(f"[phase_semantic] LLM API connectivity probe failed: {exc}", file=sys.stderr)
            return 2

    if p3a:
        require_debug_output(rounds_neutral, "paths.rounds_with_neutral_json")
        if rounds_neutral.exists():
            rounds_neutral.unlink()
        run_phase3a(rounds_path=rounds_p2, output_path=rounds_neutral, config_path=config_path, progress_sink=progress_sink("phase3a"))
        if writer is not None:
            writer.emit(event="stage_work_complete", stage="phase3a")
        validate_neutral_publishable(rounds_neutral)
    if p3b:
        if p3a and writer is not None:
            writer.emit(event="stage_start", stage="phase3b")
        if not rounds_neutral.exists() and not p3a:
            print(f"[phase_semantic] missing neutral input for phase3b: {rounds_neutral}", file=sys.stderr)
            return 2
        validate_neutral_manifest(
            json.loads(rounds_neutral.read_text(encoding="utf-8")), rounds_p2,
        )
        validate_neutral_publishable(rounds_neutral)
        require_debug_output(rounds_p3, "paths.rounds_with_commentary_json")
        require_debug_output(commentary, "paths.commentary_json")
        if rounds_p3.exists():
            rounds_p3.unlink()
        if commentary.exists():
            commentary.unlink()
        run_phase3b(
            neutral_path=rounds_neutral,
            rounds_path=rounds_p2,
            output_rounds_path=rounds_p3,
            commentary_path=commentary,
            config_path=config_path,
            draft_package_path=draft_package,
            progress_sink=progress_sink("phase3b"),
        )
        if writer is not None:
            writer.emit(event="stage_work_complete", stage="phase3b")
        validate_commentary_publishable(commentary)
    print("[phase_semantic] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
