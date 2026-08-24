"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：收集 AI-6657 快速训练/试跑的客观和主观指标。

在小批次快速跑通全链路之后，此脚本会读取各类产出物（小局、切片、视觉、语义、语音等数据），
并交互式询问用户主观体验评分，最后输出 JSON 和 Markdown 复盘报告。

启动方式：python tools/utils/collect_quick_run_metrics.py [选项]
输入数据流：各类产出物 JSON (rounds.json, segments.json, assemble_manifest.json 等)。
输出数据流：JSON 和 Markdown 复盘报告到指定输出目录。
用法用途：在小批次快速跑通全链路后，读取各类产出物并交互式询问主观评分，生成复盘报告。

参数说明：
  --rounds: rounds.json 路径 (默认: output/sbmachine/rounds.json)
  --rounds-with-yolo: rounds_with_yolo.json 路径 (默认: output/sbmachine/rounds_with_yolo.json)
  --rounds-with-commentary: rounds_with_commentary.json 路径 (默认: output/sbmachine/rounds_with_commentary.json)
  --rounds-final: rounds_final.json 路径 (默认: output/sbmachine/rounds_final.json)
  --segments: segments.json 路径 (默认: output/sbmachine/segments.json)
  --detector-rows: detector_rows.jsonl 路径 (默认: output/detector_rows.jsonl)
  --assemble-manifest: assemble_manifest.json 路径 (默认: output/sbmachine/assemble_manifest.json)
  --no-interactive: 不询问主观问题，全部填默认值 (命令行非交互模式)
  --output-dir: 报告输出目录 (默认: output/quick_run_review)
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_EMOTIONS = {"平述", "激动", "惊叹"}

MINIMUM_SAMPLE_GUIDE = {
    "quick_direction_check": {
        "raw_videos": "1 到 2 段完整录像，优先选择同一赛事模板、同一 HUD 风格",
        "rounds": 20,
        "replay_or_pause_cases": 3,
        "slice_roi_images": 1,
        "yolo_frames": 80,
        "vlm_caption_frames": 60,
        "semantic_examples": 20,
        "audio_segments": 20,
        "note": "这是验证方向的最低量，不是最终训练量。低于该量也能跑通流程，但很难判断问题来自模型还是样本偶然性。",
    },
    "preferred_quick_run": {
        "rounds": "30 到 50 小局",
        "yolo_frames": "150 到 300 帧",
        "vlm_caption_frames": "100 到 200 帧",
        "semantic_examples": "30 到 50 条",
        "audio_segments": "30 到 50 条",
        "note": "如果时间允许，建议用这一档做第一轮有效判断。",
    },
}


def resolve_path(value: str | Path | None) -> Path | None:
    """将相对路径解析为基于项目根目录的绝对路径，空值返回 None。"""
    if value is None or value == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def to_relative_path(path: Path | str | None) -> str:
    """把路径转成相对项目根的 POSIX 字符串，无法相对化则返回原始 POSIX 路径。"""
    if not path:
        return ""
    target = Path(path)
    try:
        return target.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return target.as_posix()


def read_json(path: Path | None) -> Any:
    """读取 JSON 文件，路径不存在时返回 None。"""
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path | None) -> list[dict]:
    """逐行读取 JSONL，解析失败的行以 _parse_error 记录而不中断整体读取。"""
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"_parse_error": str(exc), "_line_no": line_no})
    return rows


def count_jsonl(path: Path | None) -> int:
    """统计 JSONL 的非空行数，路径不存在时返回 0。"""
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def exists_info(path: Path | None) -> dict:
    """汇总某产出物的存在性信息：相对路径、是否存在、字节数。"""
    if path is None:
        return {"path": "", "exists": False, "bytes": 0}
    return {
        "path": to_relative_path(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def pct(part: int | float, total: int | float) -> float:
    """计算百分比并保留两位小数，分母为 0 时返回 0.0。"""
    if not total:
        return 0.0
    return round(float(part) * 100.0 / float(total), 2)


def summarize_rounds(payload: dict | None) -> dict:
    """汇总 rounds.json：小局数、时长分布、kind/原因分布与切片视频存在情况。"""
    if not isinstance(payload, dict):
        return {"exists": False}
    rounds = payload.get("rounds", []) or []
    durations = [
        max(0.0, float(item.get("end_sec", 0)) - float(item.get("start_sec", 0)))
        for item in rounds
        if isinstance(item, dict)
    ]
    kinds = Counter(str(item.get("kind", "")) for item in rounds if isinstance(item, dict))
    reasons = Counter(
        str(item.get("source_reason", item.get("reason", "")))
        for item in rounds
        if isinstance(item, dict) and (item.get("source_reason") or item.get("reason"))
    )
    clip_paths = [item.get("segment_video") for item in rounds if isinstance(item, dict) and item.get("segment_video")]
    clip_exists = sum(1 for raw in clip_paths if resolve_path(raw) and resolve_path(raw).exists())
    return {
        "exists": True,
        "total_rounds": len(rounds),
        "map_name": payload.get("map_name", ""),
        "video_path": payload.get("video_path", ""),
        "score_final": payload.get("score_final", {}),
        "duration_total_sec": round(sum(durations), 3),
        "duration_min_sec": round(min(durations), 3) if durations else 0,
        "duration_max_sec": round(max(durations), 3) if durations else 0,
        "duration_avg_sec": round(statistics.mean(durations), 3) if durations else 0,
        "short_rounds_lt_10s": sum(1 for value in durations if value < 10),
        "kind_counts": dict(kinds),
        "top_reasons": dict(reasons.most_common(8)),
        "segment_video_count": len(clip_paths),
        "segment_video_exists": clip_exists,
    }


def summarize_segments(payload: Any) -> dict:
    """汇总 segments.json：切片总数、kind/原因分布与回放段数。"""
    if payload is None:
        return {"exists": False}
    segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
    if not isinstance(segments, list):
        return {"exists": True, "valid": False}
    kinds = Counter(str(item.get("kind", "")) for item in segments if isinstance(item, dict))
    reasons = Counter(str(item.get("reason", "")) for item in segments if isinstance(item, dict) and item.get("reason"))
    return {
        "exists": True,
        "valid": True,
        "total_segments": len(segments),
        "kind_counts": dict(kinds),
        "top_reasons": dict(reasons.most_common(8)),
        "replay_segments": kinds.get("replay", 0),
        "dead_to_alive_replay_segments": reasons.get("player_dead_to_alive_without_new_round", 0),
    }


def summarize_detector_rows(rows: list[dict]) -> dict:
    """汇总 detector_rows.jsonl：行数、解析错误及各类检测标记命中数。"""
    parse_errors = sum(1 for row in rows if row.get("_parse_error"))
    valid_rows = [row for row in rows if not row.get("_parse_error")]
    return {
        "exists": bool(rows),
        "row_count": len(rows),
        "parse_errors": parse_errors,
        "hud_present_rows": sum(1 for row in valid_rows if row.get("hud_present")),
        "replay_marker_rows": sum(1 for row in valid_rows if row.get("replay_marker")),
        "bomb_indicator_rows": sum(1 for row in valid_rows if row.get("bomb_indicator")),
        "rows_with_players": sum(1 for row in valid_rows if row.get("players")),
        "rows_with_dead_state": sum(
            1
            for row in valid_rows
            if any(isinstance(player, dict) and player.get("dead") is not None for player in row.get("players", []))
        ),
    }


def summarize_vision(payload: dict | None) -> dict:
    """汇总视觉阶段产物：背景行、关键帧、解码率、YOLO 帧数与门控原因分布。"""
    if not isinstance(payload, dict):
        return {"exists": False}
    rounds = payload.get("rounds", []) or []
    total_background = 0
    total_keyframes = 0
    decoded_frames = 0
    total_yolo_frames = 0
    detector_modes = Counter()
    yolo_required = Counter()
    gate_reasons = Counter()
    for item in rounds:
        if not isinstance(item, dict):
            continue
        vision = (
            item.get("_phase2_yolo")
            or item.get("phase2_yolo")
            or {}
        )
        if not isinstance(vision, dict):
            continue
        background = vision.get("background", []) or []
        frames = vision.get("key_frames", []) or []
        total_background += len(background)
        total_keyframes += len(frames)
        total_yolo_frames += int(vision.get("total_yolo_frames", 0) or 0)
        detector_modes[str(vision.get("detector_mode", ""))] += 1
        yolo_required[str(bool(vision.get("yolo_required", False)))] += 1
        for frame in frames:
            if isinstance(frame, dict):
                gate_reasons[str(frame.get("gate_reason", ""))] += 1
                if frame.get("has_frame", frame.get("has_vlm", False)):
                    decoded_frames += 1
    return {
        "exists": True,
        "rounds_with_yolo": sum(
            1 for item in rounds if isinstance(item, dict) and (
                item.get("_phase2_yolo") or item.get("phase2_yolo")
            )
        ),
        "total_background_rows": total_background,
        "total_keyframes": total_keyframes,
        "decoded_frames": decoded_frames,
        "decoded_frame_rate_pct": pct(decoded_frames, total_keyframes),
        "total_yolo_frames": total_yolo_frames,
        "detector_modes": dict(detector_modes),
        "yolo_required_counts": dict(yolo_required),
        "gate_reasons": dict(gate_reasons),
    }


def summarize_semantic(payload: dict | None) -> dict:
    """汇总语义阶段产物：解说文本长度、空文本数与情绪标签分布（含未知情绪）。"""
    if not isinstance(payload, dict):
        return {"exists": False}
    rounds = payload.get("rounds", []) or []
    empty = 0
    lengths = []
    emotions = Counter()
    unknown_emotions = Counter()
    for item in rounds:
        if not isinstance(item, dict):
            continue
        semantic = item.get("_phase3_semantic") or item.get("phase3_semantic") or {}
        text = ""
        segments = []
        if isinstance(semantic, dict):
            text = str(semantic.get("commentary_text", ""))
            segments = semantic.get("emotion_segments", []) or []
        elif item.get("commentary_text"):
            text = str(item.get("commentary_text", ""))
            segments = item.get("emotion_segments", []) or []
        if not text.strip():
            empty += 1
        lengths.append(len(text))
        for segment in segments:
            emotion = str(segment.get("emotion", "") if isinstance(segment, dict) else "")
            emotions[emotion] += 1
            if emotion and emotion not in ALLOWED_EMOTIONS:
                unknown_emotions[emotion] += 1
    return {
        "exists": True,
        "rounds_with_semantic": len(rounds) - empty,
        "empty_commentary": empty,
        "commentary_length_avg": round(statistics.mean(lengths), 2) if lengths else 0,
        "commentary_length_min": min(lengths) if lengths else 0,
        "commentary_length_max": max(lengths) if lengths else 0,
        "emotion_counts": dict(emotions),
        "unknown_emotion_counts": dict(unknown_emotions),
    }


def summarize_audio(rounds_payload: dict | None, manifest: dict | None) -> dict:
    """汇总语音阶段产物：优先读取拼接清单，回退到旧版 rounds_final 的嵌套音频字段。"""
    manifest_rounds = manifest.get("rounds", []) if isinstance(manifest, dict) else []

    audio_paths: list[str] = []
    video_paths: list[str] = []
    for item in manifest_rounds:
        if not isinstance(item, dict):
            continue
        ap = item.get("audio_path")
        if ap:
            audio_paths.append(ap)
        vp = item.get("video_path")
        if vp:
            video_paths.append(vp)

    # 回退：旧版 rounds_final 格式（phase4_audio 嵌套在每个 round 内）
    if not audio_paths and isinstance(rounds_payload, dict):
        for item in rounds_payload.get("rounds", []):
            if not isinstance(item, dict):
                continue
            audio = item.get("_phase4_audio") or item.get("phase4_audio") or {}
            if isinstance(audio, dict) and audio.get("audio_path"):
                audio_paths.append(audio["audio_path"])

    audio_exists = sum(1 for raw in audio_paths if resolve_path(raw) and resolve_path(raw).exists())
    video_exists = sum(1 for raw in video_paths if resolve_path(raw) and resolve_path(raw).exists())
    skipped = manifest.get("skipped_rounds", 0) if isinstance(manifest, dict) else 0
    return {
        "round_audio_count": len(audio_paths),
        "round_audio_exists": audio_exists,
        "round_audio_missing": len(audio_paths) - audio_exists,
        "round_video_count": len(video_paths),
        "round_video_exists": video_exists,
        "skipped_rounds": skipped,
    }


def summarize_templates(root: Path | None) -> dict:
    """汇总切片 ROI 模板：统计每个模板缺失的 ROI、选手槽位和状态取色样本数量。"""
    if root is None or not root.exists():
        return {"exists": False}
    templates = list(root.glob("*/slice_roi_template.json"))
    required_roi = {"hud_present", "round_timer", "score_left", "score_right", "bomb_indicator", "replay_marker"}
    rows = []
    for path in templates:
        payload = read_json(path) or {}
        roi = set((payload.get("roi") or {}).keys())
        if "score_ct" in roi:
            roi.add("score_left")
        if "score_t" in roi:
            roi.add("score_right")
        players = payload.get("player_slots", []) or []
        samples = payload.get("state_color_samples", {}) or {}
        rows.append(
            {
                "template": to_relative_path(path),
                "event_name": payload.get("event_name", ""),
                "missing_roi": sorted(required_roi - roi),
                "player_slots": len(players),
                "alive_color_samples": len(samples.get("alive", []) or []),
                "dead_color_samples": len(samples.get("dead", []) or []),
            }
        )
    return {"exists": True, "template_count": len(templates), "templates": rows}


def summarize_yolo(root: Path | None) -> dict:
    """汇总 YOLO 训练数据集：统计每个数据集的训练/验证图片和标注数量。"""
    if root is None or not root.exists():
        return {"exists": False}
    datasets = []
    for data_yaml in root.glob("*/data.yaml"):
        dataset_dir = data_yaml.parent
        train_images = list((dataset_dir / "images" / "train").glob("*")) if (dataset_dir / "images" / "train").exists() else []
        val_images = list((dataset_dir / "images" / "val").glob("*")) if (dataset_dir / "images" / "val").exists() else []
        train_labels = list((dataset_dir / "labels" / "train").glob("*.txt")) if (dataset_dir / "labels" / "train").exists() else []
        val_labels = list((dataset_dir / "labels" / "val").glob("*.txt")) if (dataset_dir / "labels" / "val").exists() else []
        classes = read_json(dataset_dir / "classes.json") or {}
        datasets.append(
            {
                "dataset_dir": to_relative_path(dataset_dir),
                "data_yaml": to_relative_path(data_yaml),
                "train_images": len(train_images),
                "val_images": len(val_images),
                "train_labels": len(train_labels),
                "val_labels": len(val_labels),
                "class_count": len(classes.get("classes", []) or []),
            }
        )
    return {"exists": True, "dataset_count": len(datasets), "datasets": datasets}


def ask(prompt: str, default: str = "", *, no_interactive: bool = False) -> str:
    """交互式询问一行文本；非交互模式直接返回默认值。"""
    if no_interactive:
        return default
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def ask_int(prompt: str, default: int = 0, *, no_interactive: bool = False) -> int:
    """交互式询问一个整数；无法解析时回退到默认值。"""
    value = ask(prompt, str(default), no_interactive=no_interactive)
    try:
        return int(value)
    except ValueError:
        return default


def ask_score(prompt: str, default: int = 3, *, no_interactive: bool = False) -> int:
    """交互式询问一个 1-5 分的评分，并夹紧到合法区间。"""
    return max(1, min(5, ask_int(prompt + " (1-5)", default, no_interactive=no_interactive)))


def collect_subjective(no_interactive: bool) -> dict:
    """采集切片/视觉/语义/语音/总体五个维度的主观复盘评分与备注。"""
    print("\n=== 主观复盘采集 ===")
    print("不知道就直接回车。评分含义：1=很差，3=可用但要改，5=很好。")
    return {
        "slice": {
            "expected_rounds": ask_int("你人工预期应该切出多少小局", 0, no_interactive=no_interactive),
            "wrong_slice_count": ask_int("切片错误总数，含漏切/多切/边界错误", 0, no_interactive=no_interactive),
            "missed_round_count": ask_int("漏切小局数量", 0, no_interactive=no_interactive),
            "extra_round_count": ask_int("多切小局数量", 0, no_interactive=no_interactive),
            "boundary_error_count": ask_int("边界明显错误数量", 0, no_interactive=no_interactive),
            "replay_misclassified_count": ask_int("回放误判或漏判数量", 0, no_interactive=no_interactive),
            "notes": ask("切片错在哪里，举例时间点/原因", no_interactive=no_interactive),
        },
        "vision": {
            "score": ask_score("视觉信息整体质量", 3, no_interactive=no_interactive),
            "ocr_error_count": ask_int("你观察到的 OCR/ROI 硬事实错误数量", 0, no_interactive=no_interactive),
            "vlm_hallucination_count": ask_int("VLM 明显幻觉/乱说数量", 0, no_interactive=no_interactive),
            "yolo_error_count": ask_int("YOLO 漏检/误检数量", 0, no_interactive=no_interactive),
            "notes": ask("视觉阶段主要问题", no_interactive=no_interactive),
        },
        "semantic": {
            "score": ask_score("解说文本可用性", 3, no_interactive=no_interactive),
            "fact_error_count": ask_int("解说事实错误数量", 0, no_interactive=no_interactive),
            "emotion_error_count": ask_int("情绪标签不合适数量", 0, no_interactive=no_interactive),
            "style_issue_count": ask_int("风格不自然/不像目标解说数量", 0, no_interactive=no_interactive),
            "notes": ask("语义模型主要问题", no_interactive=no_interactive),
        },
        "audio": {
            "score": ask_score("语音自然度/情绪表现", 3, no_interactive=no_interactive),
            "bad_audio_count": ask_int("明显坏音频数量，含爆音/断句/错情绪", 0, no_interactive=no_interactive),
            "sync_issue_count": ask_int("音频节奏或拼接问题数量", 0, no_interactive=no_interactive),
            "notes": ask("语音阶段主要问题", no_interactive=no_interactive),
        },
        "overall": {
            "score": ask_score("这次快训方向总体可信度", 3, no_interactive=no_interactive),
            "next_priority": ask("你认为下一步最该改什么", no_interactive=no_interactive),
        },
    }


def make_recommendations(metrics: dict) -> list[str]:
    """基于客观产物与主观评分给出下一轮优先修复方向的建议。"""
    recs = []
    artifacts = metrics["artifacts"]
    rounds = artifacts["rounds"].get("total_rounds", 0)
    subjective = metrics.get("subjective", {})
    slice_data = subjective.get("slice", {})
    expected = slice_data.get("expected_rounds", 0)
    wrong_slice = slice_data.get("wrong_slice_count", 0)
    replay_wrong = slice_data.get("replay_misclassified_count", 0)
    vision = subjective.get("vision", {})
    semantic = subjective.get("semantic", {})
    audio = subjective.get("audio", {})

    if rounds < MINIMUM_SAMPLE_GUIDE["quick_direction_check"]["rounds"]:
        recs.append("本次小局数低于快速方向验证下限，报告只能用于排查流程通不通，不建议据此判断最终效果。")
    if expected and rounds and abs(rounds - expected) > 1:
        recs.append("切片数量与人工预期差距较大，优先复查 ROI 模板、timer/score OCR 和 replay 判断。")
    if wrong_slice > max(1, rounds * 0.1):
        recs.append("切片错误率超过 10%，下一轮先修切片，不建议继续扩大语义/语音训练。")
    if replay_wrong:
        recs.append("存在回放误判/漏判，优先检查 replay_marker ROI 和玩家死转活规则的阈值。")
    if vision.get("ocr_error_count", 0) > 0:
        recs.append("存在 OCR/ROI 硬事实错误，应先校验切片 ROI 与视觉输入字段，而不是调语义模型。")
    if vision.get("vlm_hallucination_count", 0) > 0:
        recs.append("VLM 有幻觉，收紧 prompt 或增加 VLM caption 快训样本，并强调不读取 HUD 数字。")
    if vision.get("yolo_error_count", 0) > max(2, rounds * 0.15):
        recs.append("YOLO UI 定位错误偏多，先补充 left/right player_hud_group 样本；逐个选手装备和击杀栏交给 local VLM。")
    if semantic.get("fact_error_count", 0) > 0:
        recs.append("语义事实错误时，先判断错误来自视觉 JSON 还是 LLM 自行编造；后者需要加 SFT 反例和提示约束。")
    if semantic.get("emotion_error_count", 0) > max(1, rounds * 0.1):
        recs.append("情绪标签错误率偏高，应先修情绪词映射和语义 SFT，再扩语音参考音频。")
    if audio.get("bad_audio_count", 0) > 0:
        recs.append("语音坏段较多时，先检查情绪参考音频、文本长度和 GPT-SoVITS 分段策略。")
    if not recs:
        recs.append("本次快训未暴露明显阻塞项，可以扩大样本量并进入下一轮对比实验。")
    return recs


def quality_gate(metrics: dict) -> dict:
    """把各阶段是否达标折算成一组布尔门槛，便于快速判断本轮是否可信。"""
    artifacts = metrics["artifacts"]
    subjective = metrics["subjective"]
    rounds = artifacts["rounds"].get("total_rounds", 0)
    wrong_slice = subjective["slice"].get("wrong_slice_count", 0)
    replay_wrong = subjective["slice"].get("replay_misclassified_count", 0)
    decoded_frame_rate = artifacts["vision"].get("decoded_frame_rate_pct", 0)
    empty_commentary = artifacts["semantic"].get("empty_commentary", 0)
    bad_audio = subjective["audio"].get("bad_audio_count", 0)
    return {
        "sample_size_pass": rounds >= MINIMUM_SAMPLE_GUIDE["quick_direction_check"]["rounds"],
        "slice_pass": wrong_slice <= max(1, rounds * 0.1),
        "replay_pass": replay_wrong <= 1,
        "vision_pass": subjective["vision"].get("score", 0) >= 3 and decoded_frame_rate > 0,
        "semantic_pass": subjective["semantic"].get("score", 0) >= 3 and empty_commentary == 0,
        "audio_pass": subjective["audio"].get("score", 0) >= 3 and bad_audio <= max(1, rounds * 0.1),
        "overall_pass": subjective["overall"].get("score", 0) >= 3,
    }


def write_markdown(path: Path, metrics: dict) -> None:
    """把指标结构渲染成人类可读的 Markdown 复盘报告并写盘。"""
    artifacts = metrics["artifacts"]
    subjective = metrics["subjective"]
    gate = metrics["quality_gate"]
    min_sample = metrics["minimum_sample_guide"]["quick_direction_check"]
    lines = [
        "# AI-6657 快速训练/试跑复盘报告",
        "",
        f"- 生成时间: {metrics['created_at']}",
        f"- 项目根目录: {metrics['project_root']}",
        "",
        "## 最低样本量",
        "",
        f"- 原始录像: {min_sample['raw_videos']}",
        f"- 有效小局: 至少 {min_sample['rounds']} 局",
        f"- 回放/暂停/异常案例: 至少 {min_sample['replay_or_pause_cases']} 个",
        f"- 切片 ROI 模板截图: 至少 {min_sample['slice_roi_images']} 张",
        f"- YOLO 选手 HUD 组标注帧: 至少 {min_sample['yolo_frames']} 帧",
        f"- VLM caption 帧: 至少 {min_sample['vlm_caption_frames']} 帧",
        f"- 语义样本: 至少 {min_sample['semantic_examples']} 条",
        f"- 语音合成段: 至少 {min_sample['audio_segments']} 段",
        f"- 说明: {min_sample['note']}",
        "",
        "## 客观指标",
        "",
        f"- 小局数: {artifacts['rounds'].get('total_rounds', 0)}",
        f"- 小局总时长: {artifacts['rounds'].get('duration_total_sec', 0)}s",
        f"- 短小局(<10s): {artifacts['rounds'].get('short_rounds_lt_10s', 0)}",
        f"- segment 类型: `{json.dumps(artifacts['segments'].get('kind_counts', {}), ensure_ascii=False)}`",
        f"- replay 段数: {artifacts['segments'].get('replay_segments', 0)}",
        f"- 死转活 replay 段数: {artifacts['segments'].get('dead_to_alive_replay_segments', 0)}",
        f"- detector 行数: {artifacts['detector_rows'].get('row_count', 0)}",
        f"- detector 解析错误: {artifacts['detector_rows'].get('parse_errors', 0)}",
        f"- 视觉背景行: {artifacts['vision'].get('total_background_rows', 0)}",
        f"- 解码视觉帧: {artifacts['vision'].get('decoded_frames', 0)}",
        f"- 视觉帧解码率: {artifacts['vision'].get('decoded_frame_rate_pct', 0)}%",
        f"- 解说空文本数: {artifacts['semantic'].get('empty_commentary', 0)}",
        f"- 情绪标签分布: `{json.dumps(artifacts['semantic'].get('emotion_counts', {}), ensure_ascii=False)}`",
        f"- 未知情绪标签: `{json.dumps(artifacts['semantic'].get('unknown_emotion_counts', {}), ensure_ascii=False)}`",
        f"- 小段音频存在数: {artifacts['audio'].get('round_audio_exists', 0)}/{artifacts['audio'].get('round_audio_count', 0)}",
        f"- 小段视频存在数: {artifacts['audio'].get('round_video_exists', 0)}/{artifacts['audio'].get('round_video_count', 0)}",
        f"- 跳过局数: {artifacts['audio'].get('skipped_rounds', 0)}",
        "",
        "## 主观复盘",
        "",
        f"- 人工预期小局数: {subjective['slice'].get('expected_rounds', 0)}",
        f"- 切片错误数: {subjective['slice'].get('wrong_slice_count', 0)}",
        f"- 回放误判/漏判: {subjective['slice'].get('replay_misclassified_count', 0)}",
        f"- 视觉质量评分: {subjective['vision'].get('score', 0)}/5",
        f"- 语义文本评分: {subjective['semantic'].get('score', 0)}/5",
        f"- 语音评分: {subjective['audio'].get('score', 0)}/5",
        f"- 总体方向评分: {subjective['overall'].get('score', 0)}/5",
        "",
        "## 合格门槛",
        "",
        f"- 样本量达标: {gate['sample_size_pass']}",
        f"- 切片达标: {gate['slice_pass']}",
        f"- 回放判断达标: {gate['replay_pass']}",
        f"- 视觉达标: {gate['vision_pass']}",
        f"- 语义达标: {gate['semantic_pass']}",
        f"- 语音达标: {gate['audio_pass']}",
        f"- 总体达标: {gate['overall_pass']}",
        "",
        "## 建议",
        "",
    ]
    lines.extend(f"- {item}" for item in metrics["recommendations"])
    lines.extend(
        [
            "",
            "## 原始备注",
            "",
            f"- 切片: {subjective['slice'].get('notes', '')}",
            f"- 视觉: {subjective['vision'].get('notes', '')}",
            f"- 语义: {subjective['semantic'].get('notes', '')}",
            f"- 语音: {subjective['audio'].get('notes', '')}",
            f"- 下一优先级: {subjective['overall'].get('next_priority', '')}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect(args: argparse.Namespace) -> dict:
    """读取全链路各阶段产物，汇总客观指标并采集主观评分，返回完整复盘指标字典。"""
    rounds = read_json(resolve_path(args.rounds))
    rounds_with_yolo = read_json(resolve_path(args.rounds_with_yolo))
    rounds_with_commentary = read_json(resolve_path(args.rounds_with_commentary))
    rounds_final = read_json(resolve_path(args.rounds_final))
    segments = read_json(resolve_path(args.segments))
    assemble_manifest = read_json(resolve_path(args.assemble_manifest))
    detector_rows = read_jsonl(resolve_path(args.detector_rows))

    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": to_relative_path(PROJECT_ROOT),
        "minimum_sample_guide": MINIMUM_SAMPLE_GUIDE,
        "inputs": {
            "rounds": exists_info(resolve_path(args.rounds)),
            "rounds_with_yolo": exists_info(resolve_path(args.rounds_with_yolo)),
            "rounds_with_commentary": exists_info(resolve_path(args.rounds_with_commentary)),
            "rounds_final": exists_info(resolve_path(args.rounds_final)),
            "segments": exists_info(resolve_path(args.segments)),
            "detector_rows": exists_info(resolve_path(args.detector_rows)),
            "assemble_manifest": exists_info(resolve_path(args.assemble_manifest)),
        },
        "artifacts": {
            "rounds": summarize_rounds(rounds),
            "segments": summarize_segments(segments),
            "vision": summarize_vision(rounds_with_yolo or rounds_final),
            "semantic": summarize_semantic(rounds_with_commentary or rounds_final),
            "audio": summarize_audio(rounds_final, assemble_manifest),
            "detector_rows": summarize_detector_rows(detector_rows),
            "slice_templates": summarize_templates(resolve_path(args.slice_template_root)),
            "yolo_datasets": summarize_yolo(resolve_path(args.yolo_root)),
            "vlm": {
                "frames_metadata_rows": count_jsonl(resolve_path(args.vlm_metadata)),
                "caption_count": len(read_json(resolve_path(args.vlm_captions)) or {}),
                "sft_rows": count_jsonl(resolve_path(args.vlm_sft)),
            },
            "llm_sft": {"visual_sft_rows": count_jsonl(resolve_path(args.visual_sft))},
        },
        "subjective": collect_subjective(args.no_interactive),
    }
    metrics["quality_gate"] = quality_gate(metrics)
    metrics["recommendations"] = make_recommendations(metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    """解析命令行参数：各产出物路径、输出目录与是否非交互。"""
    parser = argparse.ArgumentParser(description="收集 AI-6657 快速训练/试跑复盘指标")
    parser.add_argument("--rounds", default="output/sbmachine/rounds.json")
    parser.add_argument(
        "--rounds-with-yolo",
        "--rounds-with-vision",
        dest="rounds_with_yolo",
        default="output/sbmachine/rounds_with_yolo.json",
    )
    parser.add_argument("--rounds-with-commentary", default="output/sbmachine/rounds_with_commentary.json")
    parser.add_argument("--rounds-final", default="output/sbmachine/rounds_final.json")
    parser.add_argument("--segments", default="output/sbmachine/segments.json")
    parser.add_argument("--detector-rows", default="output/detector_rows.jsonl")
    parser.add_argument("--assemble-manifest", default="output/sbmachine/assemble_manifest.json")
    parser.add_argument("--slice-template-root", default="data/vision/slice_roi_templates")
    parser.add_argument("--yolo-root", default="data/vision/yolo_background")
    parser.add_argument("--vlm-metadata", default="data/vision/vlm_frames/metadata.jsonl")
    parser.add_argument("--vlm-captions", default="data/vision/vlm_captions.json")
    parser.add_argument("--vlm-sft", default="data/vision/vlm_sft.jsonl")
    parser.add_argument("--visual-sft", default="data/sft/visual_aligned_segments.jsonl")
    parser.add_argument("--output-dir", default="output/quick_run_review")
    parser.add_argument("--no-interactive", action="store_true", help="不询问主观问题，全部填默认值")
    return parser.parse_args()


def main() -> int:
    """入口：采集指标后写出带时间戳的 JSON 与 Markdown 复盘报告。"""
    args = parse_args()
    metrics = collect(args)
    output_dir = resolve_path(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"quick_run_metrics_{stamp}.json"
    md_path = output_dir / f"quick_run_metrics_{stamp}.md"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, metrics)
    print(f"已写入 JSON: {json_path}")
    print(f"已写入 Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
