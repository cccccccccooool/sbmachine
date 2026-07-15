"""事务化的全流水线总编排。唯一的 CLI 入口是 run.py。"""
from __future__ import annotations

import shutil
import subprocess
import sys
import traceback
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.config_loader import ConfigError
from sbmachine.common import load_config, require_path, resolve_backend, resolve_path
from sbmachine.file_lock import FileLock, FileLockUnavailable
from sbmachine.phase1_preprocess_slice import run_preprocess_slice
from sbmachine.phase2_yolo import run_phase2
from sbmachine.phase3a_analyst import run_phase3a
from sbmachine.phase3b_style import run_phase3b
from sbmachine.phase4_assemble import run_phase4
from sbmachine.preflight import (
    enabled_phases,
    phase_enabled,
    preflight_config,
    require_outputs,
    validate_commentary_publishable,
    validate_demo_publishable,
    validate_neutral_publishable,
    validate_phase1_publishable,
    validate_phase2_publishable,
    validate_phase4_publishable,
)
from sbmachine.run_context import RunContext
from sbmachine.upstream_jobs import _call_gpu_guard, _run_demo_parse


class PreflightFailure(RuntimeError):
    pass


def _phase_enabled(phases: dict, new_key: str, old_key: str, default: bool = True) -> bool:
    return phase_enabled(phases, new_key, old_key, default)


def _phase3_enabled(phases: dict) -> tuple[bool, bool]:
    return (
        _phase_enabled(phases, "phase3a_semantic", "phase3_semantic", True),
        _phase_enabled(phases, "phase3b_semantic", "phase3_semantic", True),
    )


def _phase3_active_backends(phases: dict, config: dict) -> list[str]:
    p3a, p3b = _phase3_enabled(phases)
    backends: list[str] = []
    if p3a:
        backends.append(resolve_backend(config, "analyst"))
    if p3b:
        backends.append(resolve_backend(config, "style"))
    return backends


def _phase3_local_service_name(phases: dict, config: dict) -> str | None:
    return "vllm" if "vllm" in _phase3_active_backends(phases, config) else None


def _select_preprocess_segments(slicer_segments_path: Path | None, configured_segments_path: Path | None) -> Path | None:
    if slicer_segments_path is not None and slicer_segments_path.exists():
        print(f"[preprocess_slice] use slicer segments: {slicer_segments_path}")
        return slicer_segments_path
    if configured_segments_path is not None:
        print(f"[preprocess_slice] use configured segments: {configured_segments_path}")
        return configured_segments_path
    return None


def _run_video_marking(
    paths: dict,
    slicer_config: dict,
    use_gpu_guard: bool,
) -> tuple[Path, Path]:
    """用事务托管的输出路径运行现有的切片器（slicer）。"""
    video = require_path(paths.get("video"), "paths.video")
    model = require_path(slicer_config.get("model", "models/qiepian/frame_type_classifier.pt"), "slicer.model")
    out_jsonl = require_path(paths.get("hud_detections_jsonl"), "paths.hud_detections_jsonl")
    out_segments = require_path(paths.get("segments_out_json"), "paths.segments_out_json")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    script = PACKAGE_ROOT / "tools" / "slicing" / "run_frame_type_slicer.py"
    cmd = [
        sys.executable,
        str(script),
        "--video", str(video),
        "--model", str(model),
        "--frame-output", str(out_jsonl),
        "--segment-output", str(out_segments),
        "--interval-sec", str(slicer_config.get("interval_sec", 1.0)),
        "--smooth-window", str(slicer_config.get("smooth_window", 5)),
        "--min-live-sec", str(slicer_config.get("min_live_sec", 20.0)),
        "--bridge-gap-sec", str(slicer_config.get("bridge_gap_sec", 3.0)),
        "--device", str(slicer_config.get("device", "auto")),
        "--workers", str(slicer_config.get("workers", 1)),
    ]
    replay_model = resolve_path(slicer_config.get("replay_model", ""))
    if replay_model is not None and replay_model.exists():
        cmd.extend(["--replay-model", str(replay_model)])
        if slicer_config.get("replay_roi"):
            cmd.extend(["--replay-roi", str(slicer_config["replay_roi"])])
        if slicer_config.get("replay_threshold") is not None:
            cmd.extend(["--replay-threshold", str(slicer_config["replay_threshold"])])
    demo_rounds = require_path(paths.get("demo_output_dir"), "paths.demo_output_dir") / "rounds.json"
    if demo_rounds.exists():
        cmd.extend(["--demo-rounds", str(demo_rounds)])

    _call_gpu_guard("release", use_gpu_guard)
    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"run_frame_type_slicer failed with code {result.returncode}")
    finally:
        _call_gpu_guard("resume", use_gpu_guard)
    return out_jsonl, out_segments


def _remove_file(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()


def _remove_phase4_output_dir(config: dict) -> None:
    output_dir = resolve_path(config.get("phase4", {}).get("output_dir"))
    if output_dir is not None and output_dir.exists():
        shutil.rmtree(output_dir)


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


def _execute_pipeline(config_path: Path, config: dict, context: RunContext) -> None:
    paths = config.get("paths", {})
    phases = config.get("phases", {})
    runtime = config.get("runtime", {})
    single_container = bool(runtime.get("manage_services", False))
    use_gpu_guard = bool(runtime.get("use_gpu_guard", False))

    rounds_p1 = require_path(paths.get("rounds_json"), "paths.rounds_json")
    rounds_p2 = require_path(paths.get("rounds_with_yolo_json"), "paths.rounds_with_yolo_json")
    rounds_neutral = require_path(paths.get("rounds_with_neutral_json"), "paths.rounds_with_neutral_json")
    rounds_p3 = require_path(paths.get("rounds_with_commentary_json"), "paths.rounds_with_commentary_json")
    rounds_p4 = require_path(paths.get("rounds_final_json"), "paths.rounds_final_json")

    if phases.get("demo_parse", False):
        context.current_stage = "demo_parse"
        demo_dir = require_path(paths.get("demo_output_dir"), "paths.demo_output_dir")
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
        _run_demo_parse(paths)
        validate_demo_publishable(demo_dir)
        context.checkpoint("demo_parse")

    detections_path = resolve_path(paths.get("hud_detections_jsonl"))
    slicer_segments_path = None
    if phases.get("video_marking", False):
        context.current_stage = "video_marking"
        slicer_config = config.get("slicer", {})
        detections_path, slicer_segments_path = _run_video_marking(
            paths, slicer_config, use_gpu_guard
        )
        require_outputs("video_marking", [detections_path, slicer_segments_path])
        context.checkpoint("video_marking")

    if phases.get("preprocess_slice", phases.get("phase1_slice", True)):
        context.current_stage = "phase1"
        outputs = [
            rounds_p1,
            require_path(paths.get("round_list_json"), "paths.round_list_json"),
            require_path(paths.get("segments_out_json"), "paths.segments_out_json"),
        ]
        for output in outputs[:2]:
            _remove_file(output)
        if slicer_segments_path is None or slicer_segments_path != outputs[2]:
            _remove_file(outputs[2])
        run_preprocess_slice(
            video_path=require_path(paths.get("video"), "paths.video"),
            output_rounds_path=rounds_p1,
            output_list_path=outputs[1],
            output_segments_path=outputs[2],
            detections_path=detections_path,
            segments_path=_select_preprocess_segments(slicer_segments_path, resolve_path(paths.get("segments_json"))),
            clip_dir=resolve_path(paths.get("clip_dir")),
            map_name=str(paths.get("map_name", "Unknown")),
        )
        validate_phase1_publishable(outputs[0], outputs[1], outputs[2])
        context.checkpoint("phase1")

    if single_container:
        _run_phases_subprocess(config_path, phases, config, context, use_gpu_guard)
    else:
        _run_phases_multi_container(
            config_path, phases, config, paths,
            rounds_p1, rounds_p2, rounds_neutral, rounds_p3, rounds_p4,
            context, use_gpu_guard,
        )


def _spawn(module: str, config_path: Path, log_path: Path) -> None:
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    print(f"[run_all] spawn {module}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"{module} exited with code {result.returncode}")


def _run_phases_subprocess(
    config_path: Path,
    phases: dict,
    config: dict,
    context: RunContext,
    use_gpu_guard: bool,
) -> None:
    from sbmachine.service_manager import ServiceManager

    one_at_a_time = bool(config.get("runtime", {}).get("one_model_at_a_time", True))
    mgr = ServiceManager(config)
    try:
        if not one_at_a_time:
            phase3_service = _phase3_local_service_name(phases, config)
            if any(_phase3_enabled(phases)) and phase3_service:
                mgr.start(phase3_service)
            if phases.get("phase4_assemble", True):
                mgr.start("sovits")

        if phases.get("phase2_yolo", True):
            context.current_stage = "phase2"
            _call_gpu_guard("release", use_gpu_guard)
            try:
                _spawn("sbmachine.phase_yolo", config_path, context.diagnostics_dir / "phase2.log")
                validate_phase2_publishable(
                    require_path(
                        config.get("paths", {}).get("rounds_with_yolo_json"),
                        "paths.rounds_with_yolo_json",
                    )
                )
                context.checkpoint("phase2")
            finally:
                _call_gpu_guard("resume", use_gpu_guard)

        p3a, p3b = _phase3_enabled(phases)
        if p3a or p3b:
            context.current_stage = "phase3a" if p3a else "phase3b"
            _call_gpu_guard("release", use_gpu_guard)
            phase3_service = _phase3_local_service_name(phases, config)
            try:
                if one_at_a_time and phase3_service:
                    mgr.start(phase3_service)
                try:
                    _spawn("sbmachine.phase_semantic", config_path, context.diagnostics_dir / "phase3.log")
                except BaseException:
                    if p3a:
                        neutral = require_path(
                            config.get("paths", {}).get("rounds_with_neutral_json"),
                            "paths.rounds_with_neutral_json",
                        )
                        try:
                            validate_neutral_publishable(neutral)
                        except BaseException:
                            context.current_stage = "phase3a"
                        else:
                            context.current_stage = "phase3b" if p3b else "phase3a"
                    else:
                        context.current_stage = "phase3b"
                    raise
            finally:
                if one_at_a_time and phase3_service:
                    mgr.stop(phase3_service)
                _call_gpu_guard("resume", use_gpu_guard)

        if phases.get("phase4_assemble", True):
            context.current_stage = "phase4"
            _call_gpu_guard("release", use_gpu_guard)
            try:
                if one_at_a_time:
                    mgr.start("sovits")
                _spawn("sbmachine.phase_tts", config_path, context.diagnostics_dir / "phase4.log")
                validate_phase4_publishable(
                    require_path(config.get("paths", {}).get("rounds_final_json"), "paths.rounds_final_json"),
                    require_path(config.get("paths", {}).get("assemble_manifest_json"), "paths.assemble_manifest_json"),
                )
            finally:
                if one_at_a_time:
                    mgr.stop("sovits")
                _call_gpu_guard("resume", use_gpu_guard)
    finally:
        mgr.stop_all()


def _run_phases_multi_container(
    config_path: Path,
    phases: dict,
    config: dict,
    paths: dict,
    rounds_p1: Path,
    rounds_p2: Path,
    rounds_neutral: Path,
    rounds_p3: Path,
    rounds_p4: Path,
    context: RunContext,
    use_gpu_guard: bool,
) -> None:
    from sbmachine.compose_manager import ComposeManager

    compose_file = str(config.get("runtime", {}).get("compose_file", "docker-compose.yml"))
    mgr = ComposeManager(config, compose_file=compose_file)
    try:
        if phases.get("phase2_yolo", True):
            context.current_stage = "phase2"
            semantic_p2 = require_path(
                paths.get("rounds_with_yolo_semantic_json"),
                "paths.rounds_with_yolo_semantic_json",
            )
            _remove_file(rounds_p2)
            _remove_file(semantic_p2)
            _call_gpu_guard("release", use_gpu_guard)
            try:
                run_phase2(
                    rounds_path=rounds_p1,
                    output_path=rounds_p2,
                    config_path=config_path,
                    semantic_output_path=semantic_p2,
                )
                require_outputs("phase2", [rounds_p2, semantic_p2])
                validate_phase2_publishable(rounds_p2)
                context.checkpoint("phase2")
            finally:
                _call_gpu_guard("resume", use_gpu_guard)

        p3a, p3b = _phase3_enabled(phases)
        if p3a or p3b:
            context.current_stage = "phase3a" if p3a else "phase3b"
            _call_gpu_guard("release", use_gpu_guard)
            phase3_service = _phase3_local_service_name(phases, config)
            try:
                if phase3_service:
                    mgr.up_one("talk_service")
                if p3a:
                    _remove_file(rounds_neutral)
                    run_phase3a(rounds_path=rounds_p2, output_path=rounds_neutral, config_path=config_path)
                    validate_neutral_publishable(rounds_neutral)
                if p3b:
                    context.current_stage = "phase3b"
                    commentary = require_path(paths.get("commentary_json"), "paths.commentary_json")
                    _remove_file(rounds_p3)
                    _remove_file(commentary)
                    run_phase3b(
                        neutral_path=rounds_neutral,
                        rounds_path=rounds_p2,
                        output_rounds_path=rounds_p3,
                        commentary_path=commentary,
                        config_path=config_path,
                    )
                    validate_commentary_publishable(commentary)
            finally:
                if phase3_service:
                    mgr.down_one("talk_service")
                _call_gpu_guard("resume", use_gpu_guard)

        if phases.get("phase4_assemble", True):
            context.current_stage = "phase4"
            manifest = require_path(paths.get("assemble_manifest_json"), "paths.assemble_manifest_json")
            _remove_file(rounds_p4)
            _remove_file(manifest)
            _remove_phase4_output_dir(config)
            _call_gpu_guard("release", use_gpu_guard)
            try:
                mgr.up_one("audio_service")
                with FileLock(PACKAGE_ROOT / "output" / ".sovits.lock"):
                    run_phase4(
                        rounds_path=rounds_p3,
                        commentary_path=require_path(paths.get("commentary_json"), "paths.commentary_json"),
                        output_rounds_path=rounds_p4,
                        manifest_path=manifest,
                        config_path=config_path,
                    )
                validate_phase4_publishable(rounds_p4, manifest)
            finally:
                mgr.down_one("audio_service")
                _call_gpu_guard("resume", use_gpu_guard)
    finally:
        mgr.down_all()


# 多容器模式 → _run_phases_multi_container。
# 单容器模式 → _run_phases_subprocess。
