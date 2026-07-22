"""独立流水线抽象接口层。

run_all 只负责预检与事务；此模块负责选择具体编排接口、服务后端和阶段执行策略。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import require_path, resolve_backend, resolve_path
from sbmachine.file_lock import FileLock
from sbmachine.phase1_preprocess_slice import run_preprocess_slice
from sbmachine.phase2_yolo import run_phase2
from sbmachine.phase3a_analyst import run_phase3a
from sbmachine.phase3b_style import run_phase3b
from sbmachine.phase4_assemble import run_phase4
from sbmachine.preflight import (
    phase_enabled,
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

@dataclass(frozen=True)
class StageSpec:
    """Internal description of one ordered pipeline stage."""

    key: str
    enabled: Callable[["PipelineInterface"], bool]
    current_stage: Callable[["PipelineInterface"], str]
    gpu_bound: bool = False
    service_group: str | None = None
    prepares_runtime: bool = False


class ServiceBackend:
    """Logical service lifecycle interface used by every stage executor."""

    def prepare_runtime(self, runner: "PipelineInterface") -> None:
        return None

    def start_group(self, group: str, runner: "PipelineInterface") -> None:
        raise NotImplementedError

    def stop_group(self, group: str, runner: "PipelineInterface") -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class LocalServiceBackend(ServiceBackend):
    """ServiceManager adapter for the existing local subprocess mode."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._one_at_a_time = bool(config.get("runtime", {}).get("one_model_at_a_time", True))
        self._manager = None

    def _manager_or_create(self):
        if self._manager is None:
            from sbmachine.service_manager import ServiceManager

            self._manager = ServiceManager(self._config)
        return self._manager

    @staticmethod
    def _service_for(group: str, runner: "PipelineInterface") -> str | None:
        if group == "semantic":
            return runner.phase3_service
        if group == "audio":
            return "sovits"
        raise ValueError(f"unknown local service group: {group}")

    def prepare_runtime(self, runner: "PipelineInterface") -> None:
        manager = self._manager_or_create()
        if self._one_at_a_time:
            return
        if runner.is_enabled("phase3_semantic") and runner.phase3_service:
            manager.start(runner.phase3_service)
        if runner.is_enabled("phase4"):
            manager.start("sovits")

    def start_group(self, group: str, runner: "PipelineInterface") -> None:
        if not self._one_at_a_time:
            return
        service = self._service_for(group, runner)
        if service:
            self._manager_or_create().start(service)

    def stop_group(self, group: str, runner: "PipelineInterface") -> None:
        if not self._one_at_a_time:
            return
        service = self._service_for(group, runner)
        if service:
            self._manager_or_create().stop(service)

    def close(self) -> None:
        if self._manager is not None:
            self._manager.stop_all()


class ComposeServiceBackend(ServiceBackend):
    """ComposeManager adapter; logical names stay independent from compose names."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._compose_file = str(config.get("runtime", {}).get("compose_file", "docker-compose.yml"))
        self._manager = None

    def _manager_or_create(self):
        if self._manager is None:
            from sbmachine.compose_manager import ComposeManager

            self._manager = ComposeManager(self._config, compose_file=self._compose_file)
        return self._manager

    @staticmethod
    def _service_for(group: str, runner: "PipelineInterface") -> str | None:
        if group == "semantic":
            return "talk_service" if runner.phase3_service else None
        if group == "audio":
            return "audio_service"
        raise ValueError(f"unknown compose service group: {group}")

    def prepare_runtime(self, runner: "PipelineInterface") -> None:
        self._manager_or_create()

    def start_group(self, group: str, runner: "PipelineInterface") -> None:
        service = self._service_for(group, runner)
        if service:
            self._manager_or_create().up_one(service)

    def stop_group(self, group: str, runner: "PipelineInterface") -> None:
        service = self._service_for(group, runner)
        if service:
            self._manager_or_create().down_one(service)

    def close(self) -> None:
        if self._manager is not None:
            self._manager.down_all()


class StageExecutor:
    """Execution strategy paired with a service backend."""

    def __init__(self, services: ServiceBackend) -> None:
        self.services = services

    def prepare_runtime(self, runner: "PipelineInterface") -> None:
        self.services.prepare_runtime(runner)

    def prepare_stage(self, stage: StageSpec, runner: "PipelineInterface") -> None:
        return None

    def execute(self, stage: StageSpec, runner: "PipelineInterface") -> None:
        handler = getattr(self, f"_execute_{stage.key}", None)
        if handler is None:
            raise RuntimeError(f"{type(self).__name__} cannot execute stage {stage.key}")
        handler(runner)

    def close(self) -> None:
        self.services.close()

    def _execute_demo_parse(self, runner: "PipelineInterface") -> None:
        demo_dir = require_path(runner.paths.get("demo_output_dir"), "paths.demo_output_dir")
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
        _run_demo_parse(runner.paths)
        validate_demo_publishable(demo_dir)
        runner.context.checkpoint("demo_parse")

    def _execute_video_marking(self, runner: "PipelineInterface") -> None:
        slicer_config = runner.config.get("slicer", {})
        runner.detections_path, runner.slicer_segments_path = _run_video_marking(
            runner.paths, slicer_config, runner.use_gpu_guard
        )
        require_outputs("video_marking", [runner.detections_path, runner.slicer_segments_path])
        runner.context.checkpoint("video_marking")

    def _execute_phase1(self, runner: "PipelineInterface") -> None:
        outputs = [
            runner.rounds_p1,
            require_path(runner.paths.get("round_list_json"), "paths.round_list_json"),
            require_path(runner.paths.get("segments_out_json"), "paths.segments_out_json"),
        ]
        for output in outputs[:2]:
            _remove_file(output)
        if runner.slicer_segments_path is None or runner.slicer_segments_path != outputs[2]:
            _remove_file(outputs[2])
        run_preprocess_slice(
            video_path=require_path(runner.paths.get("video"), "paths.video"),
            output_rounds_path=runner.rounds_p1,
            output_list_path=outputs[1],
            output_segments_path=outputs[2],
            detections_path=runner.detections_path,
            segments_path=_select_preprocess_segments(
                runner.slicer_segments_path,
                resolve_path(runner.paths.get("segments_json")),
            ),
            clip_dir=resolve_path(runner.paths.get("clip_dir")),
            map_name=str(runner.paths.get("map_name", "Unknown")),
        )
        validate_phase1_publishable(outputs[0], outputs[1], outputs[2])
        runner.context.checkpoint("phase1")


def _spawn(module: str, config_path: Path, log_path: Path) -> None:
    """通过独立 Python 进程执行本地阶段，并把输出写入本次事务诊断目录。"""
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    print(f"[pipeline_interface] spawn {module}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"{module} exited with code {result.returncode}")

class LocalStageExecutor(StageExecutor):
    """Keep the existing local module entry points behind the common stage interface."""

    def _execute_phase2(self, runner: "PipelineInterface") -> None:
        _spawn("sbmachine.phase_yolo", runner.config_path, runner.context.diagnostics_dir / "phase2.log")
        validate_phase2_publishable(
            require_path(
                runner.paths.get("rounds_with_yolo_json"),
                "paths.rounds_with_yolo_json",
            )
        )
        runner.context.checkpoint("phase2")

    def _execute_phase3_semantic(self, runner: "PipelineInterface") -> None:
        p3a, p3b = _phase3_enabled(runner.phases)
        try:
            _spawn("sbmachine.phase_semantic", runner.config_path, runner.context.diagnostics_dir / "phase3.log")
        except BaseException:
            if p3a:
                neutral = require_path(
                    runner.paths.get("rounds_with_neutral_json"),
                    "paths.rounds_with_neutral_json",
                )
                try:
                    validate_neutral_publishable(neutral)
                except BaseException:
                    runner.context.current_stage = "phase3a"
                else:
                    runner.context.current_stage = "phase3b" if p3b else "phase3a"
            else:
                runner.context.current_stage = "phase3b"
            raise

    def _execute_phase4(self, runner: "PipelineInterface") -> None:
        _spawn("sbmachine.phase_tts", runner.config_path, runner.context.diagnostics_dir / "phase4.log")
        validate_phase4_publishable(
            require_path(runner.paths.get("rounds_final_json"), "paths.rounds_final_json"),
            require_path(runner.paths.get("assemble_manifest_json"), "paths.assemble_manifest_json"),
        )


class ComposeStageExecutor(StageExecutor):
    """Keep host-side phase functions while Compose owns talk/audio service lifecycle."""

    def prepare_stage(self, stage: StageSpec, runner: "PipelineInterface") -> None:
        if stage.key == "phase2":
            semantic_p2 = require_path(
                runner.paths.get("rounds_with_yolo_semantic_json"),
                "paths.rounds_with_yolo_semantic_json",
            )
            _remove_file(runner.rounds_p2)
            _remove_file(semantic_p2)
        elif stage.key == "phase4":
            manifest = require_path(
                runner.paths.get("assemble_manifest_json"),
                "paths.assemble_manifest_json",
            )
            _remove_file(runner.rounds_p4)
            _remove_file(manifest)
            _remove_phase4_output_dir(runner.config)

    def _execute_phase2(self, runner: "PipelineInterface") -> None:
        semantic_p2 = require_path(
            runner.paths.get("rounds_with_yolo_semantic_json"),
            "paths.rounds_with_yolo_semantic_json",
        )
        run_phase2(
            rounds_path=runner.rounds_p1,
            output_path=runner.rounds_p2,
            config_path=runner.config_path,
            semantic_output_path=semantic_p2,
        )
        require_outputs("phase2", [runner.rounds_p2, semantic_p2])
        validate_phase2_publishable(runner.rounds_p2)
        runner.context.checkpoint("phase2")

    def _execute_phase3_semantic(self, runner: "PipelineInterface") -> None:
        p3a, p3b = _phase3_enabled(runner.phases)
        if p3a:
            _remove_file(runner.rounds_neutral)
            run_phase3a(
                rounds_path=runner.rounds_p2,
                output_path=runner.rounds_neutral,
                config_path=runner.config_path,
            )
            validate_neutral_publishable(runner.rounds_neutral)
        if p3b:
            runner.context.current_stage = "phase3b"
            commentary = require_path(runner.paths.get("commentary_json"), "paths.commentary_json")
            _remove_file(runner.rounds_p3)
            _remove_file(commentary)
            run_phase3b(
                neutral_path=runner.rounds_neutral,
                rounds_path=runner.rounds_p2,
                output_rounds_path=runner.rounds_p3,
                commentary_path=commentary,
                config_path=runner.config_path,
            )
            validate_commentary_publishable(commentary)

    def _execute_phase4(self, runner: "PipelineInterface") -> None:
        manifest = require_path(
            runner.paths.get("assemble_manifest_json"),
            "paths.assemble_manifest_json",
        )
        with FileLock(PACKAGE_ROOT / "output" / ".sovits.lock"):
            run_phase4(
                rounds_path=runner.rounds_p3,
                commentary_path=require_path(runner.paths.get("commentary_json"), "paths.commentary_json"),
                output_rounds_path=runner.rounds_p4,
                manifest_path=manifest,
                config_path=runner.config_path,
            )
        validate_phase4_publishable(runner.rounds_p4, manifest)


class PipelineInterface(ABC):
    """Ordered stage runner; its executor is the only local/Compose selection point."""

    def __init__(self, config_path: Path, config: dict, context: RunContext) -> None:
        self.config_path = config_path
        self.config = config
        self.context = context
        self.paths = config.get("paths", {})
        self.phases = config.get("phases", {})
        self.runtime = config.get("runtime", {})
        self.use_gpu_guard = bool(self.runtime.get("use_gpu_guard", False))
        self.phase3_service = _phase3_local_service_name(self.phases, self.config)

        self.rounds_p1 = require_path(self.paths.get("rounds_json"), "paths.rounds_json")
        self.rounds_p2 = require_path(
            self.paths.get("rounds_with_yolo_json"),
            "paths.rounds_with_yolo_json",
        )
        self.rounds_neutral = require_path(
            self.paths.get("rounds_with_neutral_json"),
            "paths.rounds_with_neutral_json",
        )
        self.rounds_p3 = require_path(
            self.paths.get("rounds_with_commentary_json"),
            "paths.rounds_with_commentary_json",
        )
        self.rounds_p4 = require_path(
            self.paths.get("rounds_final_json"),
            "paths.rounds_final_json",
        )
        self.detections_path = resolve_path(self.paths.get("hud_detections_jsonl"))
        self.slicer_segments_path: Path | None = None

        self.executor: StageExecutor = self._create_executor()

        self.stages = (
            StageSpec(
                "demo_parse",
                lambda runner: bool(runner.phases.get("demo_parse", False)),
                lambda runner: "demo_parse",
            ),
            StageSpec(
                "video_marking",
                lambda runner: bool(runner.phases.get("video_marking", False)),
                lambda runner: "video_marking",
            ),
            StageSpec(
                "phase1",
                lambda runner: bool(
                    runner.phases.get("preprocess_slice", runner.phases.get("phase1_slice", True))
                ),
                lambda runner: "phase1",
            ),
            StageSpec(
                "phase2",
                lambda runner: bool(runner.phases.get("phase2_yolo", True)),
                lambda runner: "phase2",
                gpu_bound=True,
                prepares_runtime=True,
            ),
            StageSpec(
                "phase3_semantic",
                lambda runner: any(_phase3_enabled(runner.phases)),
                lambda runner: "phase3a" if _phase3_enabled(runner.phases)[0] else "phase3b",
                gpu_bound=True,
                service_group="semantic",
                prepares_runtime=True,
            ),
            StageSpec(
                "phase4",
                lambda runner: bool(runner.phases.get("phase4_assemble", True)),
                lambda runner: "phase4",
                gpu_bound=True,
                service_group="audio",
                prepares_runtime=True,
            ),
        )
        self._stages_by_key = {stage.key: stage for stage in self.stages}

    @abstractmethod
    def _create_executor(self) -> StageExecutor:
        """由具体接口决定阶段执行与服务生命周期后端。"""
        raise NotImplementedError

    def is_enabled(self, key: str) -> bool:
        try:
            stage = self._stages_by_key[key]
        except KeyError as exc:
            raise ValueError(f"unknown stage: {key}") from exc
        return stage.enabled(self)

    def run(self) -> None:
        runtime_prepared = False
        try:
            for stage in self.stages:
                if not stage.enabled(self):
                    continue
                if stage.prepares_runtime and not runtime_prepared:
                    self.executor.prepare_runtime(self)
                    runtime_prepared = True
                self._execute_stage(stage)
            # Legacy runners always constructed their service manager after the
            # pre-service stages, even when every service-bound phase was disabled.
            # Preserve that cleanup boundary without starting services unnecessarily.
            if not runtime_prepared:
                self.executor.prepare_runtime(self)
        finally:
            self.executor.close()

    def _execute_stage(self, stage: StageSpec) -> None:
        self.context.current_stage = stage.current_stage(self)
        self.executor.prepare_stage(stage, self)
        gpu_released = False
        if stage.gpu_bound:
            _call_gpu_guard("release", self.use_gpu_guard)
            gpu_released = True
        try:
            if stage.service_group is not None:
                self.executor.services.start_group(stage.service_group, self)
            self.executor.execute(stage, self)
        finally:
            try:
                if stage.service_group is not None:
                    self.executor.services.stop_group(stage.service_group, self)
            finally:
                if gpu_released:
                    _call_gpu_guard("resume", self.use_gpu_guard)


class LocalPipelineInterface(PipelineInterface):
    """本地子进程 + ServiceManager 的独立编排接口。"""

    def _create_executor(self) -> StageExecutor:
        return LocalStageExecutor(LocalServiceBackend(self.config))


class ComposePipelineInterface(PipelineInterface):
    """宿主阶段函数 + Compose 服务生命周期的独立编排接口。"""

    def _create_executor(self) -> StageExecutor:
        return ComposeStageExecutor(ComposeServiceBackend(self.config))


def select_pipeline_interface(
    config_path: Path,
    config: dict,
    context: RunContext,
) -> PipelineInterface:
    """根据运行配置选择独立编排接口；run_all 不接触具体后端。"""
    if bool(config.get("runtime", {}).get("manage_services", False)):
        return LocalPipelineInterface(config_path, config, context)
    return ComposePipelineInterface(config_path, config, context)
