"""第四阶段（TTS 语音合成与最终音频拼接）。读取前一阶段带情感标签的解说文本，调用 TTS 客户端分段合成语音，最后将它们拼接成全场完整音频，并可选与切片视频合成最终带解说的视频。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from sbmachine.common import load_config, require_path, resolve_path, write_json
from sbmachine.schemas import AudioData, load_match, save_match


def _tagged_text(round_record) -> str:
    semantic = round_record.phase3_semantic
    if semantic is None:
        return ""
    if semantic.emotion_segments:
        return "".join(f"[{segment.emotion}]{segment.text}" for segment in semantic.emotion_segments)
    return semantic.commentary_text


def _concat_audio(parts: list[Path], output_path: Path) -> Path | None:
    if not parts:
        return None
    try:
        from pydub import AudioSegment

        combined = AudioSegment.empty()
        for part in parts:
            combined += AudioSegment.from_file(part)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(output_path, format=output_path.suffix.lower().lstrip(".") or "wav")
        return output_path
    except ImportError:
        from audio_service.gpt_sovits_client import _concat_wav_stdlib

        _concat_wav_stdlib(parts, output_path)
        return output_path


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", *args], check=True)


def _make_video_clips(match, source_video: Path, clip_dir: Path, final_audio: Path | None, final_video: Path | None) -> Path | None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for round_record in match.rounds:
        clip = clip_dir / f"round_{round_record.round_no:03d}.mp4"
        _run_ffmpeg(
            [
                "-ss",
                str(round_record.start_sec),
                "-to",
                str(round_record.end_sec),
                "-i",
                str(source_video),
                "-c",
                "copy",
                str(clip),
            ]
        )
        clips.append(clip)

    concat_list = clip_dir / "concat.txt"
    concat_list.write_text("".join(f"file '{clip.as_posix()}'\n" for clip in clips), encoding="utf-8")
    filtered_video = clip_dir / "filtered_video.mp4"
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(filtered_video)])
    if final_audio and final_video:
        final_video.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(["-i", str(filtered_video), "-i", str(final_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-shortest", str(final_video)])
        return final_video
    return filtered_video


def run_phase4(
    *,
    rounds_path: Path,
    output_rounds_path: Path,
    manifest_path: Path,
    config_path: Path,
    dry_run: bool = False,
) -> dict:
    from audio_service.gpt_sovits_client import read_config as read_tts_config
    from audio_service.gpt_sovits_client import synthesize_emotional

    config = load_config(config_path)
    match = load_match(rounds_path)
    tts_config = config.get("tts", {})
    tts_runtime_path = require_path(tts_config.get("config", "audio_service/gpt_sovits_runtime.yaml"), "tts.config")
    tts_runtime = read_tts_config(tts_runtime_path) if not dry_run else {}
    audio_dir = require_path(tts_config.get("output_dir", "output/sbmachine/audio"), "tts.output_dir")
    final_audio = require_path(tts_config.get("final_audio", "output/sbmachine/final_commentary.wav"), "tts.final_audio")
    audio_parts: list[Path] = []
    records = []

    for round_record in tqdm(match.rounds, desc="Phase4 TTS", unit="round"):
        text = _tagged_text(round_record)
        audio_path = audio_dir / f"round_{round_record.round_no:03d}.wav"
        if text and not dry_run:
            synthesize_emotional(tts_runtime, text, audio_path)
            audio_parts.append(audio_path)
            round_record.phase4_audio = AudioData(audio_path=str(audio_path))
        elif text:
            round_record.phase4_audio = AudioData(audio_path=str(audio_path))
        records.append(
            {
                "round_no": round_record.round_no,
                "audio_path": str(audio_path),
                "order_manifest": str(audio_path.with_name(f"{audio_path.stem}_order.json")),
                "text": text,
            }
        )

    stitched_audio = None if dry_run else _concat_audio(audio_parts, final_audio)

    video_manifest = None
    video_config = config.get("video", {})
    if video_config.get("make_filtered_video", False) and not dry_run:
        source_video = resolve_path(match.video_path)
        if source_video is not None:
            video_manifest = _make_video_clips(
                match,
                source_video,
                require_path(video_config.get("clip_dir", "output/sbmachine/video_clips"), "video.clip_dir"),
                stitched_audio,
                resolve_path(video_config.get("final_video", "output/sbmachine/final_video.mp4")),
            )

    save_match(output_rounds_path, match)
    manifest = {
        "rounds": records,
        "final_audio": str(stitched_audio) if stitched_audio else None,
        "final_video": str(video_manifest) if video_manifest else None,
    }
    write_json(manifest_path, manifest)
    return manifest
