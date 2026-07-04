"""Phase 4 audio/video helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _tagged_text(round_record) -> str:
    semantic = round_record.phase3_semantic
    if semantic is None:
        return ""
    if semantic.emotion_segments:
        return "".join(f"[{segment.emotion}]{segment.text}" for segment in semantic.emotion_segments)
    return semantic.commentary_text


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", *args], check=True)


def _mux_round_video(
    clip_path: Path,
    audio_path: Path,
    output_path: Path,
    game_vol: float = 0.25,
    comm_vol: float = 1.0,
) -> Path:
    """将单局视频片段与解说音轨混音，游戏原声降至 game_vol，解说保持 comm_vol。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-i", str(clip_path),
            "-i", str(audio_path),
            "-filter_complex",
            f"[0:a]volume={game_vol}[bg];[1:a]volume={comm_vol}[sp];[bg][sp]amix=inputs=2:duration=first[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-shortest",
            str(output_path),
        ]
    )
    return output_path


_DUMMY_PLACEHOLDERS = (
    "中性稿缺失",
    "暂无解说",
    "跳过解说",
)


def _is_dummy_round(text: str) -> bool:
    """哑局判定：commentary_text 为空、全为 [style error:、或含占位串。"""
    if not text:
        return True
    stripped = text.strip()
    # 占位串：来自 phase3b 的「中性稿缺失/暂无解说」分支
    if any(ph in stripped for ph in _DUMMY_PLACEHOLDERS):
        return True
    lines = [seg.strip() for seg in stripped.split("[") if seg.strip()]
    return all(seg.startswith("style error:") for seg in lines) if lines else True

