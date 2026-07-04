"""全流程编排（库函数，无命令行入口）。唯一启动项见仓库根 run.py。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import load_config, require_path, resolve_backend, resolve_path
from sbmachine.phase1_preprocess_slice import run_preprocess_slice
from sbmachine.phase2_vision import run_phase2
from sbmachine.phase3a_analyst import run_phase3a
from sbmachine.phase3b_style import run_phase3b
from sbmachine.phase4_assemble import run_phase4
from sbmachine.upstream_jobs import _call_gpu_guard, _run_demo_parse, _run_video_marking


def _phase_enabled(phases: dict, new_key: str, old_key: str, default: bool = True) -> bool:
    return bool(phases.get(new_key, phases.get(old_key, default)))


def _phase2_needs_service(config: dict) -> bool:
    return str(config.get("vision", {}).get("vlm", {}).get("backend", "local")).lower() != "api"


def _phase3_enabled(phases: dict) -> tuple[bool, bool]:
    return (
        _phase_enabled(phases, "phase3a_semantic", "phase3_semantic", True),
        _phase_enabled(phases, "phase3b_semantic", "phase3_semantic", True),
    )


def _phase3_needs_ollama(phases: dict, config: dict) -> bool:
    p3a, p3b = _phase3_enabled(phases)
    return (p3a and resolve_backend(config, "analyst") != "api") or (p3b and resolve_backend(config, "style") != "api")



def run_all(config_path, *, dry_run: bool = False) -> None:
    config = load_config(config_path)
    paths = config.get("paths", {})
    phases = config.get("phases", {})

    runtime = config.get("runtime", {})
    single_container = bool(runtime.get("manage_services", False))
    use_gpu_guard = bool(runtime.get("use_gpu_guard", False))

    rounds_p1 = require_path(paths.get("rounds_json", "output/sbmachine/rounds.json"), "paths.rounds_json")
    rounds_p2 = require_path(paths.get("rounds_with_vision_json", "output/sbmachine/rounds_with_vision.json"), "paths.rounds_with_vision_json")
    rounds_neutral = require_path(paths.get("rounds_with_neutral_json", "output/sbmachine/rounds_with_neutral.json"), "paths.rounds_with_neutral_json")
    rounds_p3 = require_path(paths.get("rounds_with_commentary_json", "output/sbmachine/rounds_with_commentary.json"), "paths.rounds_with_commentary_json")
    rounds_p4 = require_path(paths.get("rounds_final_json", "output/sbmachine/rounds_final.json"), "paths.rounds_final_json")

    # ── 上游 1:demo 数据获取(Go 解析器 → output/demo) ──
    if phases.get("demo_parse", False):
        _run_demo_parse(paths)

    # ── 上游 1.5:视频标记(预测 frame_type 并导出 detector_rows.jsonl) ──
    detections_path = resolve_path(paths.get("hud_detections_jsonl"))
    if phases.get("video_marking", False):
        slicer_config = config.get("slicer", {})
        detections_path = _run_video_marking(paths, slicer_config, use_gpu_guard)

    # ── 上游 2:视频标记 + 视频切片(检测/片段 → 小局 rounds.json,可选切小片) ──
    if phases.get("preprocess_slice", phases.get("phase1_slice", True)):
        run_preprocess_slice(
            video_path=require_path(paths.get("video"), "paths.video"),
            output_rounds_path=rounds_p1,
            output_list_path=require_path(paths.get("round_list_json", "output/sbmachine/round_list.json"), "paths.round_list_json"),
            output_segments_path=resolve_path(paths.get("segments_out_json", "output/sbmachine/segments.json")),
            detections_path=detections_path,
            segments_path=resolve_path(paths.get("segments_json")),
            clip_dir=resolve_path(paths.get("clip_dir")),
            map_name=str(paths.get("map_name", "Unknown")),
        )

    # manage_services 现在是「单容器 / 多容器」开关，两种模式 run.py 都全程管生命周期：
    #   true  → 单容器：本容器内逐阶段起/停服务进程（ServiceManager）。
    #   false → 多容器：run.py 自己 docker compose up 三个后端容器，跑完 down（ComposeManager）。
    if single_container:
        _run_phases_subprocess(config_path, phases, config, dry_run, use_gpu_guard)
    else:
        _run_phases_multi_container(
            config_path, phases, config, paths,
            rounds_p1, rounds_p2, rounds_neutral, rounds_p3, rounds_p4,
            dry_run, use_gpu_guard,
        )


def _spawn(module: str, config_path, dry_run: bool) -> None:
    """Spawn a phase subprocess and wait; raises on nonzero exit."""
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    if dry_run:
        cmd.append("--dry-run")
    print(f"[run_all] spawn {module}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{module} exited with code {result.returncode}")


def _run_phases_subprocess(config_path, phases: dict, config: dict, dry_run: bool, use_gpu_guard: bool) -> None:
    """manage_services=true: per-phase service lifecycle + subprocess spawn for VRAM isolation."""
    from sbmachine.service_manager import ServiceManager

    one_at_a_time = bool(config.get("runtime", {}).get("one_model_at_a_time", True))
    mgr = ServiceManager(config)

    # 确保 tmp/ 存在（服务日志写入）
    (PACKAGE_ROOT / "tmp").mkdir(exist_ok=True)

    try:
        if not one_at_a_time:
            # 不错峰：全部服务一次性拉起
            if phases.get("phase2_vision", True) and _phase2_needs_service(config): mgr.start("vlm")
            if any(_phase3_enabled(phases)) and _phase3_needs_ollama(phases, config): mgr.start("ollama")
            if phases.get("phase4_assemble", True): mgr.start("sovits")

        if phases.get("phase2_vision", True):
            _call_gpu_guard("release", use_gpu_guard)
            try:
                if one_at_a_time and _phase2_needs_service(config):
                    mgr.start("vlm")
                _spawn("sbmachine.phase_vision", config_path, dry_run)
            finally:
                if one_at_a_time and _phase2_needs_service(config):
                    mgr.stop("vlm")
                _call_gpu_guard("resume", use_gpu_guard)

        p3a, p3b = _phase3_enabled(phases)
        if p3a or p3b:
            _call_gpu_guard("release", use_gpu_guard)
            try:
                need_ollama = _phase3_needs_ollama(phases, config)
                if one_at_a_time and need_ollama:
                    mgr.start("ollama")
                _spawn("sbmachine.phase_semantic", config_path, dry_run)
            finally:
                if one_at_a_time and _phase3_needs_ollama(phases, config):
                    mgr.stop("ollama")
                _call_gpu_guard("resume", use_gpu_guard)

        if phases.get("phase4_assemble", True):
            _call_gpu_guard("release", use_gpu_guard)
            try:
                if one_at_a_time:
                    mgr.start("sovits")
                _spawn("sbmachine.phase_tts", config_path, dry_run)
            finally:
                if one_at_a_time:
                    mgr.stop("sovits")
                _call_gpu_guard("resume", use_gpu_guard)

    finally:
        mgr.stop_all()


def _run_phases_multi_container(config_path, phases: dict, config: dict, paths: dict,
                                rounds_p1, rounds_p2, rounds_neutral, rounds_p3, rounds_p4,
                                dry_run: bool, use_gpu_guard: bool) -> None:
    """多容器模式：严格单容器错峰，最大化 8-12G 显存利用。

    用到哪个阶段才 up 对应容器，阶段一跑完立刻 stop 释放整张卡，再 up 下一个。
    任意时刻卡上只有一个模型。dry_run 不碰容器，直接走 inline 自检。
    """
    if dry_run:
        _run_phases_inline(config_path, phases, paths, rounds_p1, rounds_p2, rounds_neutral, rounds_p3, rounds_p4, dry_run, use_gpu_guard)
        return

    from sbmachine.compose_manager import ComposeManager

    compose_file = str(config.get("runtime", {}).get("compose_file", "docker-compose.yml"))
    mgr = ComposeManager(config, compose_file=compose_file)

    try:
        if phases.get("phase2_vision", True):
            _call_gpu_guard("release", use_gpu_guard)
            try:
                if _phase2_needs_service(config):
                    mgr.up_one("vision_service")
                run_phase2(rounds_path=rounds_p1, output_path=rounds_p2, config_path=config_path, dry_run=dry_run)
            finally:
                if _phase2_needs_service(config):
                    mgr.down_one("vision_service")
                _call_gpu_guard("resume", use_gpu_guard)

        p3a, p3b = _phase3_enabled(phases)
        if p3a or p3b:
            _call_gpu_guard("release", use_gpu_guard)
            try:
                need_ollama = _phase3_needs_ollama(phases, config)
                if need_ollama:
                    mgr.up_one("talk_service")
                if p3a:
                    run_phase3a(rounds_path=rounds_p2, output_path=rounds_neutral, config_path=config_path, dry_run=dry_run)
                if p3b:
                    run_phase3b(
                        neutral_path=rounds_neutral,
                        rounds_path=rounds_p2,
                        output_rounds_path=rounds_p3,
                        commentary_path=require_path(paths.get("commentary_json", "output/sbmachine/commentary.json"), "paths.commentary_json"),
                        config_path=config_path,
                        dry_run=dry_run,
                    )
            finally:
                if _phase3_needs_ollama(phases, config):
                    mgr.down_one("talk_service")
                _call_gpu_guard("resume", use_gpu_guard)

        if phases.get("phase4_assemble", True):
            _call_gpu_guard("release", use_gpu_guard)
            try:
                mgr.up_one("audio_service")
                run_phase4(
                    rounds_path=rounds_p3,
                    output_rounds_path=rounds_p4,
                    manifest_path=require_path(paths.get("assemble_manifest_json", "output/sbmachine/assemble_manifest.json"), "paths.assemble_manifest_json"),
                    config_path=config_path,
                    dry_run=dry_run,
                )
            finally:
                mgr.down_one("audio_service")
                _call_gpu_guard("resume", use_gpu_guard)
    finally:
        mgr.down_all()


def _run_phases_inline(config_path, phases: dict, paths: dict,
                       rounds_p1, rounds_p2, rounds_neutral, rounds_p3, rounds_p4,
                       dry_run: bool, use_gpu_guard: bool) -> None:
    """同进程逐阶段调用（被单容器/多容器两种模式复用为实际跑阶段的内核）。"""
    if phases.get("phase2_vision", True):
        _call_gpu_guard("release", use_gpu_guard)
        try:
            run_phase2(rounds_path=rounds_p1, output_path=rounds_p2, config_path=config_path, dry_run=dry_run)
        finally:
            _call_gpu_guard("resume", use_gpu_guard)

    p3a, p3b = _phase3_enabled(phases)
    if p3a or p3b:
        _call_gpu_guard("release", use_gpu_guard)
        try:
            if p3a:
                run_phase3a(
                    rounds_path=rounds_p2,
                    output_path=rounds_neutral,
                    config_path=config_path,
                    dry_run=dry_run,
                )
            if p3b:
                run_phase3b(
                    neutral_path=rounds_neutral,
                    rounds_path=rounds_p2,
                    output_rounds_path=rounds_p3,
                    commentary_path=require_path(paths.get("commentary_json", "output/sbmachine/commentary.json"), "paths.commentary_json"),
                    config_path=config_path,
                    dry_run=dry_run,
                )
        finally:
            _call_gpu_guard("resume", use_gpu_guard)

    if phases.get("phase4_assemble", True):
        _call_gpu_guard("release", use_gpu_guard)
        try:
            run_phase4(
                rounds_path=rounds_p3,
                output_rounds_path=rounds_p4,
                manifest_path=require_path(paths.get("assemble_manifest_json", "output/sbmachine/assemble_manifest.json"), "paths.assemble_manifest_json"),
                config_path=config_path,
                dry_run=dry_run,
            )
        finally:
            _call_gpu_guard("resume", use_gpu_guard)
