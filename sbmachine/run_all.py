"""全流程编排（库函数，无命令行入口）。唯一启动项见仓库根 run.py。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import load_config, require_path, resolve_path
from sbmachine.phase1_preprocess_slice import run_preprocess_slice
from sbmachine.phase2_vision import run_phase2
from sbmachine.phase3a_analyst import run_phase3a
from sbmachine.phase3b_style import run_phase3b
from sbmachine.phase4_assemble import run_phase4


def _call_gpu_guard(action: str, use_gpu_guard: bool) -> None:
    """调用 gpu_guard """
    if not use_gpu_guard:
        return
    script = PACKAGE_ROOT / "tools" / "gpu_guard.py"
    if script.exists():
        try:
            print(f"[gpu_guard] {action}...")
            subprocess.run([sys.executable, str(script), action], check=False)
        except Exception as e:
            print(f"[gpu_guard] error: {e}")


def _run_demo_parse(paths: dict) -> None:
    """调 tools/parse_demo.py(Go 解析器)把 .dem 解析成 output/demo 工件。"""
    demo = resolve_path(paths.get("demo"))
    if demo is None:
        raise ValueError("phases.demo_parse 已开启,但 paths.demo 未配置 .dem 路径")
    out_dir = str(paths.get("demo_output_dir", "output/demo"))
    script = PACKAGE_ROOT / "tools" / "parse_demo.py"
    print(f"[demo_parse] {demo} → {out_dir}")
    result = subprocess.run([sys.executable, str(script), "--demo", str(demo), "--output-dir", out_dir])
    if result.returncode != 0:
        raise RuntimeError(f"parse_demo 失败 (exit {result.returncode})")


def _run_video_marking(paths: dict, slicer_config: dict, use_gpu_guard: bool) -> Path:
    """自动调用 tools/slicing/run_frame_type_slicer.py 进行画面预测和标记。"""
    video = resolve_path(paths.get("video"))
    if video is None:
        raise ValueError("phases.video_marking 已开启,但 paths.video 未配置")

    model = resolve_path(slicer_config.get("model", "models/qiepian/frame_type_classifier.pt"))
    if model is None or not model.exists():
        raise FileNotFoundError(f"未找到视频分类模型:{model}")

    out_jsonl = PACKAGE_ROOT / "output" / "sbmachine" / "detector_rows.jsonl"
    out_segments = PACKAGE_ROOT / "output" / "sbmachine" / "segments.json"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    script = PACKAGE_ROOT / "tools" / "slicing" / "run_frame_type_slicer.py"
    print(f"[video_marking] 开始预测视频帧分类: {video} → {out_jsonl}")

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
    ]

    demo_rounds = resolve_path(paths.get("demo_output_dir", "output/demo")) / "rounds.json"
    if demo_rounds.exists():
        cmd.extend(["--demo-rounds", str(demo_rounds)])

    _call_gpu_guard("release", use_gpu_guard)
    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"run_frame_type_slicer 运行失败 (exit {result.returncode})")
    finally:
        _call_gpu_guard("resume", use_gpu_guard)

    return out_jsonl


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
            if phases.get("phase2_vision", True):   mgr.start("vlm")
            if phases.get("phase4_assemble", True): mgr.start("sovits")

        if phases.get("phase2_vision", True):
            _call_gpu_guard("release", use_gpu_guard)
            try:
                if one_at_a_time:
                    mgr.start("vlm")
                _spawn("sbmachine.phase_vision", config_path, dry_run)
            finally:
                if one_at_a_time:
                    mgr.stop("vlm")
                _call_gpu_guard("resume", use_gpu_guard)

        if phases.get("phase3_semantic", True):
            _call_gpu_guard("release", use_gpu_guard)
            try:
                _spawn("sbmachine.phase_semantic", config_path, dry_run)
            finally:
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
                mgr.up_one("vision_service")
                run_phase2(rounds_path=rounds_p1, output_path=rounds_p2, config_path=config_path, dry_run=dry_run)
            finally:
                mgr.down_one("vision_service")
                _call_gpu_guard("resume", use_gpu_guard)

        if phases.get("phase3_semantic", True):
            _call_gpu_guard("release", use_gpu_guard)
            try:
                mgr.up_one("talk_service")
                run_phase3a(rounds_path=rounds_p2, output_path=rounds_neutral, config_path=config_path, dry_run=dry_run)
                run_phase3b(
                    neutral_path=rounds_neutral,
                    rounds_path=rounds_p2,
                    output_rounds_path=rounds_p3,
                    commentary_path=require_path(paths.get("commentary_json", "output/sbmachine/commentary.json"), "paths.commentary_json"),
                    config_path=config_path,
                    dry_run=dry_run,
                )
            finally:
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

    if phases.get("phase3_semantic", True):
        _call_gpu_guard("release", use_gpu_guard)
        try:
            run_phase3a(
                rounds_path=rounds_p2,
                output_path=rounds_neutral,
                config_path=config_path,
                dry_run=dry_run,
            )
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
