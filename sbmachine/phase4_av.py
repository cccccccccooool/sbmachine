"""Phase 4 音视频辅助函数：TTS 拼接到静音时间轴、逐局视频混音、WAV/媒体探测。"""
from __future__ import annotations

import json
import math
import subprocess
import wave
from pathlib import Path


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", *args], check=True)


def _probe_media(path: Path) -> tuple[bool, float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    raw_duration = video.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot determine video duration: {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"invalid video duration {duration!r}: {path}")
    return has_audio, duration


def _mux_round_video(
    clip_path: Path,
    audio_path: Path,
    output_path: Path,
    game_vol: float = 0.25,
    comm_vol: float = 1.0,
) -> Path:
    """混入解说音轨，不缩短视频；兼容无游戏原声的 clip。

    刻意不用 -shortest：解说比画面短时也要保留完整视频时长，
    因此对两路音频都做 apad 补齐再 atrim 到画面时长。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_game_audio, duration = _probe_media(clip_path)
    duration_arg = f"{duration:.6f}"
    if has_game_audio:
        audio_filter = (
            f"[0:a:0]volume={game_vol},apad,atrim=duration={duration_arg}[bg];"
            f"[1:a:0]volume={comm_vol},apad,atrim=duration={duration_arg}[sp];"
            "[bg][sp]amix=inputs=2:duration=longest[aout]"
        )
    else:
        audio_filter = f"[1:a:0]volume={comm_vol},apad,atrim=duration={duration_arg}[aout]"
    _run_ffmpeg(
        [
            "-i", str(clip_path),
            "-i", str(audio_path),
            "-filter_complex", audio_filter,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            str(output_path),
        ]
    )
    return output_path


def _wav_info(path: Path) -> tuple[wave._wave_params, int, float]:
    try:
        with wave.open(str(path), "rb") as wav:
            params = wav.getparams()
            if params.comptype != "NONE":
                raise ValueError(f"unsupported WAV compression {params.comptype}: {path}")
            if params.nchannels <= 0 or params.sampwidth <= 0 or params.framerate <= 0 or params.nframes <= 0:
                raise ValueError(f"invalid or empty WAV: {path}")
            frames = wav.readframes(params.nframes)
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"invalid WAV: {path}") from exc
    expected_size = params.nframes * params.nchannels * params.sampwidth
    if len(frames) != expected_size:
        raise ValueError(f"truncated WAV: {path}")
    return params, params.nframes, params.nframes / params.framerate


def _write_silence_wav(
    output_path: Path,
    duration_sec: float,
    *,
    sample_rate: int = 32000,
    channels: int = 1,
    sample_width: int = 2,
) -> Path:
    if duration_sec <= 0 or sample_rate <= 0 or channels <= 0 or sample_width <= 0:
        raise ValueError("silence WAV parameters must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(duration_sec * sample_rate)
    silence_byte = b"\x80" if sample_width == 1 else b"\x00"
    silence_frame = silence_byte * channels * sample_width
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(silence_frame * frame_count)
    return output_path


def _assemble_scene_wav(
    scenes: list[tuple[Path, float, float]],
    output_path: Path,
    round_duration_sec: float,
    *,
    default_sample_rate: int = 32000,
) -> list[float]:
    """把各场景 WAV 铺到以本局起点为零的静音 PCM 时间轴上。"""
    if not scenes:
        _write_silence_wav(output_path, round_duration_sec, sample_rate=default_sample_rate)
        return []

    params: wave._wave_params | None = None
    scene_frames: list[tuple[bytes, int, float]] = []
    durations: list[float] = []
    for path, relative_start, window_duration in scenes:
        current_params, frame_count, audio_duration = _wav_info(path)
        if relative_start < 0 or window_duration <= 0:
            raise ValueError(f"invalid scene time window for {path}")
        if audio_duration > window_duration + 1e-9:
            raise RuntimeError(
                f"scene audio exceeds its window: {path} is {audio_duration:.3f}s, window is {window_duration:.3f}s"
            )
        if params is None:
            params = current_params
        elif current_params[:3] != params[:3] or current_params.comptype != params.comptype:
            raise RuntimeError(
                f"WAV format mismatch: expected {params[:3]}, got {current_params[:3]} for {path}"
            )
        with wave.open(str(path), "rb") as wav:
            frames = wav.readframes(frame_count)
        scene_frames.append((frames, frame_count, relative_start))
        durations.append(audio_duration)

    assert params is not None
    canvas_frames = round(round_duration_sec * params.framerate)
    if canvas_frames <= 0:
        raise ValueError("round duration must be positive")
    bytes_per_frame = params.nchannels * params.sampwidth
    silence_byte = b"\x80" if params.sampwidth == 1 else b"\x00"
    canvas = bytearray(silence_byte * bytes_per_frame * canvas_frames)
    for frames, frame_count, relative_start in scene_frames:
        start_frame = round(relative_start * params.framerate)
        end_frame = start_frame + frame_count
        if end_frame > canvas_frames:
            raise RuntimeError(
                f"scene audio exceeds round window: ends at {end_frame / params.framerate:.3f}s, "
                f"round is {round_duration_sec:.3f}s"
            )
        start_byte = start_frame * bytes_per_frame
        canvas[start_byte:start_byte + len(frames)] = frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(canvas)
    return durations

