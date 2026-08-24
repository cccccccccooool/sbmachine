"""事务化的全流水线总编排。唯一的 CLI 入口是 run.py。"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.config_loader import ConfigError
from sbmachine.common import ensure_output_paths, load_config, require_path, resolve_backend, resolve_path
from sbmachine.file_lock import FileLock, FileLockUnavailable
from sbmachine.phase1_preprocess_slice import run_preprocess_slice
from sbmachine.phase2_yolo import run_phase2
from sbmachine.phase3a_analyst import run_phase3a
from sbmachine.phase3b_style import run_phase3b
from sbmachine.phase3c_llmc import run_phase3c
from sbmachine.phase4_assemble import run_phase4
from sbmachine.preflight import (
    enabled_phases,
    phase_enabled,
    preflight_config,
    require_outputs,
    validate_commentary_publishable,
    validate_demo_publishable,
    validate_llmb_draft_package,
    validate_neutral_publishable,
    validate_phase1_publishable,
    validate_phase2_publishable,
    validate_phase4_publishable,
    validate_render_package,
)
from sbmachine.run_context import RunContext
from sbmachine.upstream_jobs import _call_gpu_guard, _run_demo_parse


class PreflightFailure(RuntimeError):
    pass


def _cb(callbacks, key, *args):
    fn = (callbacks or {}).get(key)
    if fn:
        try:
            fn(*args)
        except Exception:
            pass


def _forward_progress_event(callbacks: dict | None, event) -> None:
    """把已校验的子进程瞬态事件映射为可选 UI 回调。

    子进程永远不能发送权威 done/canceled；work_complete 仅告知 UI 进入等待
    validator/checkpoint 的状态。
    """
    if event.event == "stage_start":
        _cb(callbacks, "on_stage_start", event.stage)
    elif event.event == "stage_progress":
        _cb(
            callbacks,
            "on_stage_progress",
            event.stage,
            event.completed,
            event.total,
            event.unit,
            event.detail,
        )
    elif event.event == "stage_work_complete":
        _cb(
            callbacks,
            "on_stage_progress",
            event.stage,
            event.completed,
            event.total,
            event.unit,
            event.detail or "处理完成，等待门禁",
        )
    elif event.event == "stage_error":
        _cb(callbacks, "on_error", event.stage, event.detail or "child stage error")


def _cb_paths(config: dict, *keys: str) -> list[Path]:
    """从 config['paths'] 读取产物路径，仅供回调展示使用（解析失败即跳过）。"""
    paths = config.get("paths", {})
    resolved: list[Path] = []
    for key in keys:
        path = resolve_path(paths.get(key))
        if path is not None:
            resolved.append(path)
    return resolved


def _decide_empty_rounds_from_commentary(config: dict, callbacks: dict | None = None) -> str:
    """Phase3b 产物存在空回合时，在进入 Phase4 前向用户做三选一决策。

    返回 "continue" / "retry" / "cancel"。默认（产物无空回合或当前
    决策模块无输出）返回 "continue"。retry/cancel 的重跑/退出逻辑由
    empty_round_decision 的调用方后续接入。
    """
    from sbmachine.empty_round_decision import decide_empty_rounds

    commentary_path = resolve_path(config.get("paths", {}).get("commentary_json"))
    if commentary_path is None or not commentary_path.is_file():
        return "continue"
    try:
        import json

        manifest = json.loads(commentary_path.read_text(encoding="utf-8"))
    except Exception:
        return "continue"
    prompt = (callbacks or {}).get("prompt_empty_rounds")
    return decide_empty_rounds(manifest, prompt=prompt if callable(prompt) else None)


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


def _select_preprocess_segments(
    slicer_segments_path: Path | None,
    configured_segments_path: Path | None,
    debug_mode: bool = False,
) -> Path | None:
    if slicer_segments_path is not None and slicer_segments_path.exists():
        if debug_mode:
            print(f"[preprocess_slice] use slicer segments: {slicer_segments_path}")
        return slicer_segments_path
    if configured_segments_path is not None and configured_segments_path.exists():
        if debug_mode:
            print(f"[preprocess_slice] use configured segments: {configured_segments_path}")
        return configured_segments_path
    return None


def _slicer_perf_args(
    slicer_config: dict,
    perf: dict | None,
    turbo: bool,
) -> dict:
    """按设备与 turbo 模式计算 slicer 的 workers/batch/节流参数。

    - GPU：单进程批推理最优（16 进程抢卡反而慢且显存碎片化），强制 workers=1。
    - CPU 非 turbo：workers 收敛到 min(配置, 4, cpu_count//4)，并启用节流。
    - turbo：完全放开（workers=配置值），关闭节流。
    """
    perf = perf or {}
    configured_workers = max(1, int(slicer_config.get("workers", 4)))
    device = str(slicer_config.get("device", "auto") or "auto")
    cpu_ceiling = float(perf.get("cpu_ceiling", 0.8))
    mem_ceiling = float(perf.get("mem_ceiling", 0.8))
    throttle_sec = float(perf.get("throttle_sec", 1.0))
    enabled = bool(perf.get("enabled", True))
    gpu = False
    if device != "cpu":
        try:
            import torch  # noqa: PLC0415

            gpu = torch.cuda.is_available() if device in ("auto", "") else str(device).startswith("cuda")
        except Exception:
            gpu = False
    if gpu:
        workers = 1
        batch_size = max(1, int(slicer_config.get("gpu_batch_size", 32)))
        throttle = enabled and not turbo
    else:
        batch_size = max(1, int(slicer_config.get("batch_size", 1)))
        if turbo:
            workers = configured_workers
        else:
            workers = min(configured_workers, max(1, min(4, os.cpu_count() // 4)))
        throttle = enabled and not turbo
    return {
        "workers": workers,
        "batch_size": batch_size,
        "throttle": throttle,
        "cpu_ceiling": cpu_ceiling,
        "mem_ceiling": mem_ceiling,
        "throttle_sec": throttle_sec,
        "device": device,
    }


def _run_video_marking(
    paths: dict,
    slicer_config: dict,
    use_gpu_guard: bool,
    log_path: Path | None = None,
    *,
    progress_events_path: Path | None = None,
    progress_run_id: str | None = None,
    callbacks: dict | None = None,
    perf: dict | None = None,
    turbo: bool = False,
) -> tuple[Path, Path]:
    """用事务托管的输出路径运行现有切片器，并尽力消费其独占进度文件。"""
    video = require_path(paths.get("video"), "paths.video")
    model = require_path(slicer_config.get("model", "models/qiepian/frame_type_classifier.pt"), "slicer.model")
    out_jsonl = require_path(paths.get("hud_detections_jsonl"), "paths.hud_detections_jsonl")
    out_segments = require_path(paths.get("segments_out_json"), "paths.segments_out_json")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    script = PACKAGE_ROOT / "tools" / "slicing" / "run_frame_type_slicer.py"
    perf_args = _slicer_perf_args(slicer_config, perf, turbo)
    cmd = [
        sys.executable, str(script), "--video", str(video), "--model", str(model),
        "--frame-output", str(out_jsonl), "--segment-output", str(out_segments),
        "--interval-sec", str(slicer_config.get("interval_sec", 1.0)),
        "--smooth-window", str(slicer_config.get("smooth_window", 5)),
        "--min-live-sec", str(slicer_config.get("min_live_sec", 20.0)),
        "--bridge-gap-sec", str(slicer_config.get("bridge_gap_sec", 3.0)),
        "--device", str(perf_args["device"]),
        "--workers", str(perf_args["workers"]),
        "--batch-size", str(perf_args["batch_size"]),
    ]
    if perf_args["throttle"]:
        cmd.extend([
            "--throttle",
            "--throttle-cpu", str(perf_args["cpu_ceiling"]),
            "--throttle-mem", str(perf_args["mem_ceiling"]),
            "--throttle-sec", str(perf_args["throttle_sec"]),
        ])
    if progress_events_path is not None and progress_run_id:
        cmd.extend(["--progress-events", str(progress_events_path), "--progress-run-id", progress_run_id])
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

    reader = None
    if progress_events_path is not None and progress_run_id:
        from sbmachine.progress_events import ProgressEventReader
        reader = ProgressEventReader(progress_events_path, run_id=progress_run_id)
    guard_log = log_path.parent / "gpu_guard.log" if log_path is not None else None
    _call_gpu_guard("release", use_gpu_guard, log_path=guard_log)
    log_file = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        else:
            process = subprocess.Popen(cmd)
        while process.poll() is None:
            if reader is not None:
                for event in reader.read_available():
                    _forward_progress_event(callbacks, event)
            time.sleep(0.05)
        if reader is not None:
            for event in reader.read_available():
                _forward_progress_event(callbacks, event)
        if process.returncode != 0:
            raise RuntimeError(f"run_frame_type_slicer failed with code {process.returncode}")
    finally:
        if log_file is not None:
            log_file.close()
        _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)
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
    if phases.get("phase3c_render", False):
        validate_render_package(
            require_path(paths.get("commentary_render_package_json"), "paths.commentary_render_package_json"),
            require_path(paths.get("llmb_draft_package_json"), "paths.llmb_draft_package_json"),
        )
    if phases.get("phase4_assemble", True):
        validate_phase4_publishable(
            require_path(paths.get("rounds_final_json"), "paths.rounds_final_json"),
            require_path(paths.get("assemble_manifest_json"), "paths.assemble_manifest_json"),
            require_path(paths.get("commentary_render_package_json"), "paths.commentary_render_package_json")
            if phases.get("phase3c_render", False) else None,
        )


def run_all(
    config_path,
    *,
    dry_run: bool = False,
    callbacks: dict | None = None,
    debug_mode: bool = False,
    turbo: bool = False,
) -> dict:
    """运行流水线，返回如实反映状态的预检或最终状态对象。

    turbo=True 时放开性能锁（slicer workers 全量、关闭 CPU/内存节流），
    用于独占机器快速出片；默认模式保持资源安全阀。
    """
    if debug_mode:
        os.environ["AI6657_DEBUG_PHASE3"] = "1"
    try:
        config = ensure_output_paths(load_config(config_path))
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
                _execute_pipeline(
                    effective_path, effective, context,
                    callbacks=callbacks, debug_mode=debug_mode, turbo=turbo,
                )
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
                _cb(callbacks, "on_error", str(stage), str(exc))
                try:
                    active_stages = enabled_phases(effective)
                    failed_at = active_stages.index(str(stage))
                except (NameError, ValueError, TypeError):
                    active_stages = []
                    failed_at = -1
                for canceled_stage in active_stages[failed_at + 1:]:
                    _cb(callbacks, "on_stage_canceled", canceled_stage, f"canceled by {stage} failure")
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


def _execute_pipeline(
    config_path: Path,
    config: dict,
    context: RunContext,
    callbacks: dict | None = None,
    debug_mode: bool = False,
    turbo: bool = False,
) -> None:
    paths = config.get("paths", {})
    phases = config.get("phases", {})
    runtime = config.get("runtime", {})
    single_container = bool(runtime.get("manage_services", False))
    use_gpu_guard = bool(runtime.get("use_gpu_guard", False))
    perf = runtime.get("perf", {}) if isinstance(runtime, dict) else {}

    rounds_p1 = require_path(paths.get("rounds_json"), "paths.rounds_json")
    rounds_p2 = require_path(paths.get("rounds_with_yolo_json"), "paths.rounds_with_yolo_json")
    rounds_neutral = require_path(paths.get("rounds_with_neutral_json"), "paths.rounds_with_neutral_json")
    rounds_p3 = require_path(paths.get("rounds_with_commentary_json"), "paths.rounds_with_commentary_json")
    rounds_p4 = require_path(paths.get("rounds_final_json"), "paths.rounds_final_json")

    if phases.get("demo_parse", False):
        context.current_stage = "demo_parse"
        _cb(callbacks, "on_stage_start", "demo_parse")
        demo_dir = require_path(paths.get("demo_output_dir"), "paths.demo_output_dir")
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
        _run_demo_parse(paths, log_path=context.diagnostics_dir / "demo_parse.log")
        validate_demo_publishable(demo_dir)
        context.checkpoint("demo_parse")
        _cb(callbacks, "on_stage_done", "demo_parse", [demo_dir])

    detections_path = resolve_path(paths.get("hud_detections_jsonl"))
    slicer_segments_path = None
    if phases.get("video_marking", False):
        context.current_stage = "video_marking"
        _cb(callbacks, "on_stage_start", "video_marking")
        slicer_config = config.get("slicer", {})
        detections_path, slicer_segments_path = _run_video_marking(
            paths, slicer_config, use_gpu_guard,
            log_path=context.diagnostics_dir / "video_marking.log",
            progress_events_path=context.diagnostics_dir / "progress" / "video_marking.jsonl",
            progress_run_id=context.run_id,
            callbacks=callbacks,
            perf=perf,
            turbo=turbo,
        )
        require_outputs("video_marking", [detections_path, slicer_segments_path])
        context.checkpoint("video_marking")
        _cb(callbacks, "on_stage_done", "video_marking", [detections_path, slicer_segments_path])

    if phases.get("preprocess_slice", phases.get("phase1_slice", True)):
        context.current_stage = "phase1"
        _cb(callbacks, "on_stage_start", "phase1")
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
            segments_path=_select_preprocess_segments(
                slicer_segments_path, resolve_path(paths.get("segments_json")), debug_mode=debug_mode
            ),
            clip_dir=resolve_path(paths.get("clip_dir")),
            map_name=str(paths.get("map_name", "Unknown")),
        )
        validate_phase1_publishable(outputs[0], outputs[1], outputs[2])
        context.checkpoint("phase1")
        _cb(callbacks, "on_stage_done", "phase1", [outputs[0], outputs[1], outputs[2]])

    if single_container:
        _run_phases_subprocess(
            config_path, phases, config, context, use_gpu_guard,
            callbacks=callbacks, debug_mode=debug_mode,
        )
    else:
        _run_phases_multi_container(
            config_path, phases, config, paths,
            rounds_p1, rounds_p2, rounds_neutral, rounds_p3, rounds_p4,
            context, use_gpu_guard,
            callbacks=callbacks, debug_mode=debug_mode,
        )


def _spawn(
    module: str,
    config_path: Path,
    log_path: Path,
    debug_mode: bool = False,
    *,
    progress_events_path: Path | None = None,
    progress_run_id: str | None = None,
    callbacks: dict | None = None,
) -> dict | None:
    """执行受管子进程，同时尽力消费其独占 JSONL 进度通道。"""
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    reader = None
    if progress_events_path is not None and progress_run_id:
        from sbmachine.progress_events import ProgressEventReader

        cmd.extend(["--progress-events", str(progress_events_path), "--progress-run-id", progress_run_id])
        reader = ProgressEventReader(progress_events_path, run_id=progress_run_id)
    log_file = None
    try:
        if debug_mode:
            print(f"[run_all] spawn {module}")
            process = subprocess.Popen(cmd)
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if reader is not None:
                for event in reader.read_available():
                    _forward_progress_event(callbacks, event)
            time.sleep(0.05)
        if reader is not None:
            for event in reader.read_available():
                _forward_progress_event(callbacks, event)
        if process.returncode != 0:
            raise RuntimeError(f"{module} exited with code {process.returncode}")
        return reader.summary if reader is not None else None
    finally:
        if log_file is not None:
            log_file.close()


def _run_phases_subprocess(
    config_path: Path,
    phases: dict,
    config: dict,
    context: RunContext,
    use_gpu_guard: bool,
    callbacks: dict | None = None,
    debug_mode: bool = False,
) -> None:
    from sbmachine.service_manager import ServiceManager

    one_at_a_time = bool(config.get("runtime", {}).get("one_model_at_a_time", True))
    mgr = ServiceManager(config)
    guard_log = context.diagnostics_dir / "gpu_guard.log"
    try:
        if not one_at_a_time:
            phase3_service = _phase3_local_service_name(phases, config)
            if any(_phase3_enabled(phases)) and phase3_service:
                mgr.start(phase3_service)
                _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)
            if phases.get("phase4_assemble", True):
                mgr.start("sovits")

        if phases.get("phase2_yolo", True):
            context.current_stage = "phase2"
            _cb(callbacks, "on_stage_start", "phase2")
            _call_gpu_guard("release", use_gpu_guard, log_path=guard_log)
            try:
                phase2_summary = _spawn(
                    "sbmachine.phase_yolo",
                    config_path,
                    context.diagnostics_dir / "phase2.log",
                    debug_mode=debug_mode,
                    progress_events_path=context.diagnostics_dir / "progress" / "phase2.jsonl",
                    progress_run_id=context.run_id,
                    callbacks=callbacks,
                )
                if phase2_summary is not None:
                    context.write_diagnostic("progress/phase2.summary.json", phase2_summary)
                validate_phase2_publishable(
                    require_path(
                        config.get("paths", {}).get("rounds_with_yolo_json"),
                        "paths.rounds_with_yolo_json",
                    )
                )
                context.checkpoint("phase2")
                _cb(
                    callbacks,
                    "on_stage_done",
                    "phase2",
                    _cb_paths(config, "rounds_with_yolo_json", "rounds_with_yolo_semantic_json"),
                )
            finally:
                _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)

        p3a, p3b = _phase3_enabled(phases)
        if p3a or p3b:
            context.current_stage = "phase3a" if p3a else "phase3b"
            if p3a:
                _cb(callbacks, "on_stage_start", "phase3a")
            elif p3b:
                _cb(callbacks, "on_stage_start", "phase3b")
            _call_gpu_guard("release", use_gpu_guard, log_path=guard_log)
            phase3_service = _phase3_local_service_name(phases, config)
            try:
                if one_at_a_time and phase3_service:
                    mgr.start(phase3_service)
                    _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)
                try:
                    phase3_summary = _spawn(
                        "sbmachine.phase_semantic",
                        config_path,
                        context.diagnostics_dir / "phase3.log",
                        debug_mode=debug_mode,
                        progress_events_path=context.diagnostics_dir / "progress" / "phase3.jsonl",
                        progress_run_id=context.run_id,
                        callbacks=callbacks,
                    )
                    if phase3_summary is not None:
                        context.write_diagnostic("progress/phase3.summary.json", phase3_summary)
                    if p3a:
                        neutral = require_path(
                            (config.get("paths") or {}).get("rounds_with_neutral_json"),
                            "paths.rounds_with_neutral_json",
                        )
                        validate_neutral_publishable(neutral)
                        context.checkpoint("phase3a")
                        _cb(
                            callbacks,
                            "on_stage_done",
                            "phase3a",
                            _cb_paths(config, "rounds_with_neutral_json"),
                        )
                    if p3b:
                        commentary = require_path(
                            (config.get("paths") or {}).get("commentary_json"),
                            "paths.commentary_json",
                        )
                        validate_commentary_publishable(commentary)
                        draft_package_value = (config.get("paths") or {}).get("llmb_draft_package_json")
                        if phases.get("phase3c_render", False):
                            draft_package = require_path(draft_package_value, "paths.llmb_draft_package_json")
                            validate_llmb_draft_package(draft_package)
                        elif draft_package_value:
                            validate_llmb_draft_package(require_path(draft_package_value, "paths.llmb_draft_package_json"))
                        context.checkpoint("phase3b")
                        _cb(
                            callbacks,
                            "on_stage_done",
                            "phase3b",
                            _cb_paths(config, "rounds_with_commentary_json", "commentary_json"),
                        )
                        # 空回合决策：进入 Phase4 前向用户三选一（continue/retry/cancel）。
                        # retry/cancel 的重跑/退出逻辑暂由后续接入，当前默认 continue 放行。
                        empty_action = _decide_empty_rounds_from_commentary(config, callbacks)
                        if empty_action == "cancel":
                            print("[run_all] 空回合决策：cancel——保留当前产物，终止流水线")
                            raise RuntimeError("empty-round decision: cancel (artifacts preserved)")
                        _cb(callbacks, "on_empty_rounds_decision", empty_action)
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
                            if p3b:
                                _cb(callbacks, "on_stage_canceled", "phase3b", "canceled by phase3a failure")
                        else:
                            _cb(
                                callbacks,
                                "on_stage_done",
                                "phase3a",
                                _cb_paths(config, "rounds_with_neutral_json"),
                            )
                            context.current_stage = "phase3b" if p3b else "phase3a"
                    else:
                        context.current_stage = "phase3b"
                    raise
            finally:
                if one_at_a_time and phase3_service:
                    mgr.stop(phase3_service)
                _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)

        if phases.get("phase3c_render", False):
            # Phase3c / LLM-C 独立阶段：消费 B 封存包，发布 render package（云端调用，无本地服务）。
            context.current_stage = "phase3c"
            _cb(callbacks, "on_stage_start", "phase3c")
            try:
                phase3c_summary = _spawn(
                    "sbmachine.phase3c_cli",
                    config_path,
                    context.diagnostics_dir / "phase3c.log",
                    debug_mode=debug_mode,
                    progress_events_path=context.diagnostics_dir / "progress" / "phase3c.jsonl",
                    progress_run_id=context.run_id,
                    callbacks=callbacks,
                )
                if phase3c_summary is not None:
                    context.write_diagnostic("progress/phase3c.summary.json", phase3c_summary)
                render_pkg = require_path(
                    (config.get("paths") or {}).get("commentary_render_package_json"),
                    "paths.commentary_render_package_json",
                )
                validate_render_package(
                    render_pkg,
                    require_path(
                        (config.get("paths") or {}).get("llmb_draft_package_json"),
                        "paths.llmb_draft_package_json",
                    ),
                )
                context.checkpoint("phase3c")
                _cb(
                    callbacks,
                    "on_stage_done",
                    "phase3c",
                    _cb_paths(config, "commentary_render_package_json"),
                )
            finally:
                _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)

        if phases.get("phase4_assemble", True):
            context.current_stage = "phase4"
            _cb(callbacks, "on_stage_start", "phase4")
            _call_gpu_guard("release", use_gpu_guard, log_path=guard_log)
            try:
                if one_at_a_time:
                    mgr.start("sovits")
                phase4_summary = _spawn(
                    "sbmachine.phase_tts",
                    config_path,
                    context.diagnostics_dir / "phase4.log",
                    debug_mode=debug_mode,
                    progress_events_path=context.diagnostics_dir / "progress" / "phase4.jsonl",
                    progress_run_id=context.run_id,
                    callbacks=callbacks,
                )
                if phase4_summary is not None:
                    context.write_diagnostic("progress/phase4.summary.json", phase4_summary)
                validate_phase4_publishable(
                    require_path((config.get("paths") or {}).get("rounds_final_json"), "paths.rounds_final_json"),
                    require_path((config.get("paths") or {}).get("assemble_manifest_json"), "paths.assemble_manifest_json"),
                    require_path((config.get("paths") or {}).get("commentary_render_package_json"), "paths.commentary_render_package_json")
                    if phases.get("phase3c_render", False) else None,
                )
                context.checkpoint("phase4")
                _cb(
                    callbacks,
                    "on_stage_done",
                    "phase4",
                    _cb_paths(config, "rounds_final_json", "assemble_manifest_json"),
                )
            finally:
                if one_at_a_time:
                    mgr.stop("sovits")
                _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)
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
    callbacks: dict | None = None,
    debug_mode: bool = False,
) -> None:
    from sbmachine.compose_manager import ComposeManager

    compose_file = str(config.get("runtime", {}).get("compose_file", "docker-compose.yml"))
    mgr = ComposeManager(config, compose_file=compose_file)
    guard_log = context.diagnostics_dir / "gpu_guard.log"
    try:
        if phases.get("phase2_yolo", True):
            context.current_stage = "phase2"
            _cb(callbacks, "on_stage_start", "phase2")
            semantic_p2 = require_path(
                paths.get("rounds_with_yolo_semantic_json"),
                "paths.rounds_with_yolo_semantic_json",
            )
            _remove_file(rounds_p2)
            _remove_file(semantic_p2)
            _call_gpu_guard("release", use_gpu_guard, log_path=guard_log)
            try:
                run_phase2(
                    rounds_path=rounds_p1,
                    output_path=rounds_p2,
                    config_path=config_path,
                    semantic_output_path=semantic_p2,
                    progress_sink=lambda completed, total, unit, detail: _cb(
                        callbacks, "on_stage_progress", "phase2", completed, total, unit, detail,
                    ),
                )
                require_outputs("phase2", [rounds_p2, semantic_p2])
                validate_phase2_publishable(rounds_p2)
                context.checkpoint("phase2")
                _cb(callbacks, "on_stage_done", "phase2", [rounds_p2, semantic_p2])
            finally:
                _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)

        p3a, p3b = _phase3_enabled(phases)
        if p3a or p3b:
            context.current_stage = "phase3a" if p3a else "phase3b"
            if p3a:
                _cb(callbacks, "on_stage_start", "phase3a")
            _call_gpu_guard("release", use_gpu_guard, log_path=guard_log)
            phase3_service = _phase3_local_service_name(phases, config)
            try:
                if phase3_service:
                    mgr.up_one("talk_service")
                if p3a:
                    _remove_file(rounds_neutral)
                    run_phase3a(
                        rounds_path=rounds_p2,
                        output_path=rounds_neutral,
                        config_path=config_path,
                        progress_sink=lambda completed, total, unit, detail: _cb(
                            callbacks, "on_stage_progress", "phase3a", completed, total, unit, detail,
                        ),
                    )
                    validate_neutral_publishable(rounds_neutral)
                    context.checkpoint("phase3a")
                    _cb(callbacks, "on_stage_done", "phase3a", [rounds_neutral])
                if p3b:
                    validate_neutral_publishable(rounds_neutral)
                    context.current_stage = "phase3b"
                    _cb(callbacks, "on_stage_start", "phase3b")
                    commentary = require_path(paths.get("commentary_json"), "paths.commentary_json")
                    _remove_file(rounds_p3)
                    _remove_file(commentary)
                    run_phase3b(
                        neutral_path=rounds_neutral,
                        rounds_path=rounds_p2,
                        output_rounds_path=rounds_p3,
                        commentary_path=commentary,
                        config_path=config_path,
                        draft_package_path=resolve_path(paths.get("llmb_draft_package_json")),
                        progress_sink=lambda completed, total, unit, detail: _cb(
                            callbacks, "on_stage_progress", "phase3b", completed, total, unit, detail,
                        ),
                    )
                    validate_commentary_publishable(commentary)
                    context.checkpoint("phase3b")
                    _cb(callbacks, "on_stage_done", "phase3b", [rounds_p3, commentary])
                    # 空回合决策：进入 Phase4 前向用户三选一（continue/retry/cancel）。
                    empty_action = _decide_empty_rounds_from_commentary(config, callbacks)
                    if empty_action == "cancel":
                        print("[run_all] 空回合决策：cancel——保留当前产物，终止流水线")
                        raise RuntimeError("empty-round decision: cancel (artifacts preserved)")
                    _cb(callbacks, "on_empty_rounds_decision", empty_action)
            finally:
                if phase3_service:
                    mgr.down_one("talk_service")
                _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)

        if phases.get("phase3c_render", False):
            context.current_stage = "phase3c"
            _cb(callbacks, "on_stage_start", "phase3c")
            draft_pkg = require_path(paths.get("llmb_draft_package_json"), "paths.llmb_draft_package_json")
            render_pkg = require_path(paths.get("commentary_render_package_json"), "paths.commentary_render_package_json")
            _remove_file(render_pkg)
            try:
                run_phase3c(
                    draft_package_path=draft_pkg,
                    output_render_path=render_pkg,
                    config_path=config_path,
                    progress_sink=lambda completed, total, unit, detail: _cb(
                        callbacks, "on_stage_progress", "phase3c", completed, total, unit, detail,
                    ),
                )
                validate_render_package(render_pkg, draft_pkg)
                context.checkpoint("phase3c")
                _cb(callbacks, "on_stage_done", "phase3c", [render_pkg])
            finally:
                _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)

        if phases.get("phase4_assemble", True):
            context.current_stage = "phase4"
            _cb(callbacks, "on_stage_start", "phase4")
            manifest = require_path(paths.get("assemble_manifest_json"), "paths.assemble_manifest_json")
            _remove_file(rounds_p4)
            _remove_file(manifest)
            _remove_phase4_output_dir(config)
            _call_gpu_guard("release", use_gpu_guard, log_path=guard_log)
            try:
                mgr.up_one("audio_service")
                with FileLock(PACKAGE_ROOT / "output" / ".sovits.lock"):
                    run_phase4(
                        rounds_path=rounds_p3,
                        commentary_path=(
                            require_path(paths.get("commentary_json"), "paths.commentary_json")
                            if not phases.get("phase3c_render", False) else None
                        ),
                        render_package_path=require_path(paths.get("commentary_render_package_json"), "paths.commentary_render_package_json")
                        if phases.get("phase3c_render", False) else None,
                        output_rounds_path=rounds_p4,
                        manifest_path=manifest,
                        config_path=config_path,
                        progress_sink=lambda completed, total, unit, detail: _cb(
                            callbacks, "on_stage_progress", "phase4", completed, total, unit, detail,
                        ),
                    )
                validate_phase4_publishable(
                    rounds_p4,
                    manifest,
                    require_path(paths.get("commentary_render_package_json"), "paths.commentary_render_package_json")
                    if phases.get("phase3c_render", False) else None,
                )
                context.checkpoint("phase4")
                _cb(callbacks, "on_stage_done", "phase4", [rounds_p4, manifest])
            finally:
                mgr.down_one("audio_service")
                _call_gpu_guard("resume", use_gpu_guard, log_path=guard_log)
    finally:
        mgr.down_all()


# 多容器模式 → _run_phases_multi_container。
# 单容器模式 → _run_phases_subprocess。
