"""从 ASR 片段清单切出 GPT-SoVITS 训练素材。

启动方式：python audio_service/prepare_gpt_sovits_dataset.py --config ... （独立运行）。
输入数据流：源音频文件 + ASR 片段清单 JSONL（含 start/end/text/keep_ai 字段）。
输出数据流：写出切片音频文件和 GPT-SoVITS 训练 list 文件（格式：音频路径|说话人|语种|文本）。
用法用途：从已清洗的 ASR 片段中切出符合时长要求的音频切片，生成 GPT-SoVITS 训练所需的 list 格式。

输出的 list 格式为：
    音频路径|说话人|语种|文本
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("缺少 pyyaml，请先执行：pip install pyyaml")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_ffmpeg() -> str:
    """优先使用 imageio_ffmpeg 自带的 ffmpeg，缺失时回退到系统 ffmpeg。"""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def read_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_segments(jsonl_path: Path, only_kept: bool) -> list[dict]:
    segments = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            if only_kept and item.get("keep_ai") is False:
                continue
            if not isinstance(item.get("start"), (int, float)) or not isinstance(item.get("end"), (int, float)):
                continue
            text = str(item.get("text", "")).strip()
            if text:
                item["text"] = clean_text(text)
                segments.append(item)
    return segments


def clean_text(text: str) -> str:
    """清理 ASR 文本里对训练无益的空白和异常符号。"""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "，")
    return text


def cut_segment(source_audio: Path, output_file: Path, start: float, end: float) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(source_audio),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "32000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(output_file),
    ]
    subprocess.run(command, check=True, timeout=120)


def build_dataset(
    source_audio: Path,
    segments_jsonl: Path,
    clip_dir: Path,
    list_path: Path,
    speaker: str,
    language: str,
    min_duration: float,
    max_duration: float,
    only_kept: bool,
) -> None:
    if not source_audio.exists():
        raise FileNotFoundError(f"找不到源音频：{source_audio}")
    if not segments_jsonl.exists():
        raise FileNotFoundError(f"找不到片段清单：{segments_jsonl}")

    segments = read_segments(segments_jsonl, only_kept)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for index, segment in enumerate(segments, start=1):
        start = float(segment["start"])
        end = float(segment["end"])
        duration = end - start
        if duration < min_duration or duration > max_duration:
            continue
        clip_path = clip_dir / f"{index:06d}.wav"
        try:
            cut_segment(source_audio, clip_path, start, end)
        except subprocess.TimeoutExpired:
            print(f"[prepare_dataset] 跳过片段 {index:06d}（ffmpeg 超时，start={start:.3f} end={end:.3f}）")
            continue
        lines.append(f"{clip_path.resolve()}|{speaker}|{language}|{segment['text']}")

    with list_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))
        if lines:
            file.write("\n")

    print(f"已生成 {len(lines)} 条 GPT-SoVITS 训练素材：{list_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 GPT-SoVITS 训练 list 和切片音频")
    parser.add_argument("--config", default="voice/gpt_sovits_runtime.yaml", help="运行配置文件")
    parser.add_argument("--source-audio", help="源音频，默认读配置")
    parser.add_argument("--segments-jsonl", help="ASR 片段清单，默认读配置")
    parser.add_argument("--clip-dir", help="输出切片目录，默认读配置")
    parser.add_argument("--list-path", help="输出 list 文件，默认读配置")
    parser.add_argument("--speaker", help="说话人名，默认读配置")
    parser.add_argument("--language", help="语种，默认 zh")
    parser.add_argument("--min-duration", type=float, help="最短片段秒数")
    parser.add_argument("--max-duration", type=float, help="最长片段秒数")
    parser.add_argument("--include-deleted", action="store_true", help="包含 AI 粗筛标记为删除的片段")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = read_config(resolve_path(args.config))
    dataset_config = config.get("dataset", {})

    try:
        build_dataset(
            source_audio=resolve_path(args.source_audio or dataset_config.get("source_audio", "data/audio/6657_ai_filtered.wav")),
            segments_jsonl=resolve_path(args.segments_jsonl or dataset_config.get("segments_jsonl", "data/audio/6657_segments_ai.jsonl")),
            clip_dir=resolve_path(args.clip_dir or dataset_config.get("clip_dir", "data/voice/clips")),
            list_path=resolve_path(args.list_path or dataset_config.get("list_path", "data/voice/6657.list")),
            speaker=args.speaker or dataset_config.get("speaker", "6657"),
            language=args.language or dataset_config.get("language", "zh"),
            min_duration=args.min_duration if args.min_duration is not None else float(dataset_config.get("min_duration", 2.0)),
            max_duration=args.max_duration if args.max_duration is not None else float(dataset_config.get("max_duration", 12.0)),
            only_kept=not args.include_deleted,
        )
    except Exception as exc:
        print(f"生成失败：{exc}")
        sys.exit(1)
