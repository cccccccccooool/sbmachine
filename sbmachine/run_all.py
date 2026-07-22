"""事务化流水线总入口；阶段编排委派给独立的 pipeline_interface。"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.config_loader import ConfigError
from sbmachine.common import load_config, require_path
from sbmachine.file_lock import FileLock, FileLockUnavailable
from sbmachine.preflight import (
    enabled_phases,
    preflight_config,
    validate_demo_publishable,
    validate_phase1_publishable,
    validate_phase2_publishable,
    validate_phase4_publishable,
)
from sbmachine.run_context import RunContext


class PreflightFailure(RuntimeError):
    pass

def _validate_enabled_outputs(config: dict) -> None:
    paths = config.get("paths", {})
    phases = config.get("phases", {})
    if phases.get("demo_parse", False):
        validate_demo_publishable(require_path(paths.get("demo_output_dir"), "paths.demo_output_dir"))
    if phases.get("preprocess_slice", phases.get("phase1_slice", True)):
        validate_phase1_publishable(
            require_path(paths.get("rounds_json"), "paths.rounds_json"),
            require_path(paths.get("round_list_json"), "paths.round_list_json"),
            require_path(paths.get("segments_out_json"), "paths.segments_out_json"),
        )
    if phases.get("phase2_yolo", True):
        validate_phase2_publishable(
            require_path(paths.get("rounds_with_yolo_json"), "paths.rounds_with_yolo_json")
        )
    if phases.get("phase4_assemble", True):
        validate_phase4_publishable(
            require_path(paths.get("rounds_final_json"), "paths.rounds_final_json"),
            require_path(paths.get("assemble_manifest_json"), "paths.assemble_manifest_json"),
        )

def _execute_pipeline(config_path: Path, config: dict, context: RunContext) -> None:
    """兼容旧调用点；实际编排由独立接口选择器承担。"""
    from sbmachine.pipeline_interface import select_pipeline_interface

    select_pipeline_interface(config_path, config, context).run()

def run_all(config_path, *, dry_run: bool = False) -> dict:
    """运行流水线，返回如实反映状态的预检或最终状态对象。"""
    try:
        config = load_config(config_path)
    except (ConfigError, OSError) as exc:
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
            "previous_success_preserved": True,
            "exit_code": 2,
        }
    report = preflight_config(config, root=PACKAGE_ROOT)
    if dry_run:
        return report

    output_root = PACKAGE_ROOT / "output"
    context = RunContext(output_root)
    try:
        with FileLock(output_root / ".run.lock"):
            try:
                effective, effective_path = context.prepare(config)
                context.current_stage = "preflight"
                context.write_diagnostic("preflight.json", report)
                if not report["config_valid"]:
                    raise PreflightFailure("; ".join(report["errors"]))
                _execute_pipeline(effective_path, effective, context)
                _validate_enabled_outputs(effective)
                manifest = {
                    "run_id": context.run_id,
                    "status": "complete",
                    "publishable": True,
                    "enabled_phases": enabled_phases(effective),
                    "checkpointed_stages": list(context.checkpointed_stages),
                }
                context.current_stage = "publish"
                context.complete(manifest)
                manifest["exit_code"] = 0
                return manifest
            except BaseException as exc:
                stage = getattr(exc, "stage", context.current_stage)
                result = context.fail(
                    str(stage),
                    str(exc),
                    extra={
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
                result["exit_code"] = 2 if isinstance(exc, PreflightFailure) else 1
                return result
    except FileLockUnavailable as exc:
        return {
            "run_id": context.run_id,
            "status": "failed",
            "publishable": False,
            "failed_stage": "lock",
            "error": str(exc),
            "previous_success_preserved": True,
            "exit_code": 3,
        }
