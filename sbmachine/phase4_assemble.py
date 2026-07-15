"""Phase 4：逐场景合成解说 TTS，并与每局音频/视频对齐混流。"""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from sbmachine.common import load_config, read_json, require_path, resolve_path, write_json
from sbmachine.phase4_av import _assemble_scene_wav, _mux_round_video, _run_ffmpeg, _wav_info
from sbmachine.schemas import AudioData, load_match, save_match


_EMOTIONS = {"平述", "激动", "惊叹"}


def _tts_cache_key(text: str, fingerprint: str) -> str:
    """由运行时指纹加文本算出 TTS 缓存文件名；指纹变则缓存自动失效。"""
    return hashlib.sha256(f"{fingerprint}\0{text}".encode("utf-8")).hexdigest()


def _synthesize_with_cache(
    tts_runtime: dict,
    text: str,
    fingerprint: str,
    audio_path: Path,
    cache_dir: Path,
    synthesize_emotional,
) -> Path:
    """命中缓存则复用，否则合成到临时文件校验通过后再落缓存，最后拷到目标路径。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_tts_cache_key(text, fingerprint)}.wav"
    if cache_path.exists():
        try:
            _wav_info(cache_path)
        except ValueError:
            cache_path.unlink()
        else:
            print(f"[phase4] 复用 TTS 缓存: {cache_path}")
            shutil.copy2(cache_path, audio_path)
            return audio_path

    print(f"[phase4] 生成 TTS 缓存: {cache_path}")
    fd, tmp_name = tempfile.mkstemp(prefix="tts_", suffix=".wav", dir=str(cache_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        synthesize_emotional(tts_runtime, text, tmp_path)
        _wav_info(tmp_path)
        shutil.move(str(tmp_path), str(cache_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    shutil.copy2(cache_path, audio_path)
    return audio_path


def _load_commentary_scenes(commentary_path: Path, match) -> dict[int, list[dict]]:
    payload = read_json(commentary_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("rounds"), list):
        raise ValueError("commentary.json must be an object containing a rounds array")
    if payload.get("video_path") != match.video_path:
        raise ValueError("commentary.json and rounds_with_commentary must have the same video_path")
    if payload.get("map_name") != match.map_name:
        raise ValueError("commentary.json and rounds_with_commentary must have the same map_name")

    rounds_by_no = {round_record.round_no: round_record for round_record in match.rounds}
    scenes_by_round: dict[int, list[dict]] = {}
    for round_index, item in enumerate(payload["rounds"]):
        label = f"commentary.rounds[{round_index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        if "round_no" not in item:
            raise ValueError(f"{label}.round_no is required")
        if not isinstance(item["round_no"], int) or isinstance(item["round_no"], bool):
            raise ValueError(f"{label}.round_no must be an integer")
        round_no = item["round_no"]
        if round_no in scenes_by_round:
            raise ValueError(f"duplicate commentary round_no: {round_no}")
        round_record = rounds_by_no.get(round_no)
        if round_record is None:
            raise ValueError(f"commentary contains unknown round_no: {round_no}")
        status = item.get("status")
        if status not in {"ok", "silent"}:
            raise ValueError(f"{label}.status must be ok or silent, got {status!r}")
        for field, expected in (("start_sec", round_record.start_sec), ("end_sec", round_record.end_sec)):
            value = item.get(field)
            if isinstance(value, bool):
                raise ValueError(f"{label}.{field} must match rounds_with_commentary")
            try:
                source_time = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}.{field} must match rounds_with_commentary") from exc
            if not math.isfinite(source_time) or not math.isclose(source_time, expected, abs_tol=1e-6):
                raise ValueError(f"{label}.{field} does not match rounds_with_commentary")
        if not isinstance(item.get("commentary_text"), str):
            raise ValueError(f"{label}.commentary_text must be a string")
        semantic = round_record.phase3_semantic
        if semantic is None:
            raise ValueError(f"rounds_with_commentary round {round_no} is missing phase3 commentary")
        if item["commentary_text"] != semantic.commentary_text:
            raise ValueError(f"{label}.commentary_text does not match rounds_with_commentary")
        if not isinstance(item.get("emotion_segments"), list):
            raise ValueError(f"{label}.emotion_segments must be an array")
        manifest_segments = []
        for segment_index, segment in enumerate(item["emotion_segments"]):
            if not isinstance(segment, dict):
                raise ValueError(f"{label}.emotion_segments[{segment_index}] must be an object")
            manifest_segments.append((segment.get("emotion"), segment.get("text"), segment.get("order")))
        semantic_segments = [
            (segment.emotion, segment.text, segment.order)
            for segment in semantic.emotion_segments
        ]
        if manifest_segments != semantic_segments:
            raise ValueError(f"{label}.emotion_segments do not match rounds_with_commentary")
        if "scenes" not in item:
            raise ValueError(f"{label}.scenes is required; legacy round-level commentary is unsupported")
        if not isinstance(item["scenes"], list):
            raise ValueError(f"{label}.scenes must be an array")
        if bool(item["scenes"]) != bool(item["commentary_text"].strip()):
            raise ValueError(f"{label}.commentary_text and scenes must both be empty or both be populated")
        if bool(item["emotion_segments"]) != bool(item["commentary_text"].strip()):
            raise ValueError(f"{label}.commentary_text and emotion_segments must both be empty or both be populated")
        if status == "ok" and not item["commentary_text"].strip():
            raise ValueError(f"{label}.status ok requires commentary text")
        if status == "silent" and item["commentary_text"].strip():
            raise ValueError(f"{label}.status silent requires empty commentary text")

        normalized: list[dict] = []
        last_end: float | None = None
        for scene_index, scene in enumerate(item["scenes"]):
            scene_label = f"{label}.scenes[{scene_index}]"
            if not isinstance(scene, dict):
                raise ValueError(f"{scene_label} must be an object")
            missing = {"t_start", "t_end", "text", "emotion"} - scene.keys()
            if missing:
                raise ValueError(f"{scene_label} is missing fields: {', '.join(sorted(missing))}")
            if isinstance(scene["t_start"], bool) or isinstance(scene["t_end"], bool):
                raise ValueError(f"{scene_label} times must be numeric")
            try:
                t_start = float(scene["t_start"])
                t_end = float(scene["t_end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{scene_label} times must be numeric") from exc
            if not math.isfinite(t_start) or not math.isfinite(t_end) or t_start >= t_end:
                raise ValueError(f"{scene_label} must have finite t_start < t_end")
            if t_start < round_record.start_sec - 1e-6 or t_end > round_record.end_sec + 1e-6:
                raise ValueError(f"{scene_label} lies outside round {round_no}")
            if last_end is not None and t_start < last_end - 1e-6:
                raise ValueError(f"{scene_label} overlaps the previous scene")
            text = scene["text"]
            emotion = scene["emotion"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{scene_label}.text must be a non-empty string")
            if not isinstance(emotion, str) or emotion not in _EMOTIONS:
                raise ValueError(f"{scene_label}.emotion must be one of {sorted(_EMOTIONS)}")
            normalized.append({
                "t_start": t_start,
                "t_end": t_end,
                "text": text.strip(),
                "emotion": emotion,
            })
            last_end = t_end
        rendered_text = "".join(f"[{scene['emotion']}]{scene['text']}" for scene in normalized)
        if rendered_text != item["commentary_text"]:
            raise ValueError(f"{label}.scenes text does not match commentary_text")
        scenes_by_round[round_no] = normalized

    missing_rounds = sorted(set(rounds_by_no) - set(scenes_by_round))
    if missing_rounds:
        raise ValueError(f"commentary.json is missing rounds: {missing_rounds}")
    return scenes_by_round


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip_cache_fingerprint(source_sha256: str, start_sec: float, end_sec: float) -> str:
    """clip 缓存指纹：源视频内容哈希 + 起止时间戳，任一变化即视为新 clip。"""
    identity = f"{source_sha256}\0{float(start_sec):.17g}\0{float(end_sec):.17g}"
    return hashlib.sha256(identity.encode("ascii")).hexdigest()


def _resolve_existing_clip(round_record, clip_cache_dir: Path, fingerprint: str | None = None) -> Path | None:
    """优先复用 phase1 切好的 segment_video，其次命中指纹缓存，都没有返回 None。"""
    segment_video = resolve_path(getattr(round_record, "segment_video", ""))
    if segment_video is not None and segment_video.exists():
        return segment_video
    if fingerprint is not None:
        candidate = clip_cache_dir / f"clip_{fingerprint}.mp4"
        if candidate.exists():
            return candidate
    return None


def _clip_for_round(
    round_record,
    source_video: Path,
    clip_cache_dir: Path,
    source_sha256: str | None = None,
) -> Path:
    existing = _resolve_existing_clip(round_record, clip_cache_dir)
    if existing is not None:
        print(f"[phase4] 复用 clip: {existing}")
        return existing

    source_sha256 = source_sha256 or _sha256_file(source_video)
    fingerprint = _clip_cache_fingerprint(source_sha256, round_record.start_sec, round_record.end_sec)
    existing = _resolve_existing_clip(round_record, clip_cache_dir, fingerprint)
    if existing is not None:
        print(f"[phase4] 复用 clip: {existing}")
        return existing

    clip_cache_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_cache_dir / f"clip_{fingerprint}.mp4"
    print(f"[phase4] 切 clip: {clip_path}")
    _run_ffmpeg([
        "-ss", str(round_record.start_sec),
        "-to", str(round_record.end_sec),
        "-i", str(source_video),
        "-c", "copy",
        str(clip_path),
    ])
    return clip_path


def run_phase4(
    *,
    rounds_path: Path,
    commentary_path: Path,
    output_rounds_path: Path,
    manifest_path: Path,
    config_path: Path,
    dry_run: bool = False,
) -> dict:
    from audio_service.gpt_sovits_client import read_config as read_tts_config
    from audio_service.gpt_sovits_client import synthesize_emotional
    from audio_service.gpt_sovits_client import tts_cache_fingerprint

    config = load_config(config_path)
    match = load_match(rounds_path)
    scenes_by_round = _load_commentary_scenes(commentary_path, match)

    # ── TTS 配置 ──
    tts_config = config.get("tts", {})
    tts_runtime_path = require_path(tts_config.get("config", "audio_service/gpt_sovits_runtime.yaml"), "tts.config")
    tts_runtime = read_tts_config(tts_runtime_path) if not dry_run else {}

    # ── Phase4 配置（从 pipeline.yaml phase4 节读取，缺则用默认值）──
    p4 = config.get("phase4", {})
    output_dir = resolve_path(p4.get("output_dir", "output/sbmachine/rounds")) or Path("output/sbmachine/rounds")
    tts_cache_dir = resolve_path(p4.get("tts_cache_dir", "output/tts_cache")) or Path("output/tts_cache")
    clip_cache_dir = resolve_path(p4.get("clip_cache_dir", "output/clips")) or Path("output/clips")
    comm_vol = float(p4.get("commentary_volume", 1.0))
    game_vol = float(p4.get("game_audio_volume", 0.25))
    sample_rate = int(p4.get("sample_rate", 32000))
    if sample_rate <= 0:
        raise ValueError("phase4.sample_rate must be positive")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── 源视频（可选，用于逐局视频混音）──
    source_video = resolve_path(getattr(match, "video_path", None))

    records = []
    skipped_count = 0
    source_video_sha256: str | None = None

    for round_record in tqdm(match.rounds, desc="Phase4 TTS+Mux", unit="round"):
        rno = round_record.round_no
        scenes = scenes_by_round[rno]
        audio_path = output_dir / f"round_{rno:03d}.wav"
        round_duration = round_record.end_sec - round_record.start_sec
        if round_duration <= 0:
            raise ValueError(f"round {rno} must have start_sec < end_sec")
        skipped = not scenes
        if skipped:
            skipped_count += 1

        scene_inputs: list[tuple[Path, float, float]] = []
        scene_records: list[dict] = []
        for scene_index, scene in enumerate(scenes, start=1):
            relative_start = scene["t_start"] - round_record.start_sec
            window_duration = scene["t_end"] - scene["t_start"]
            scene_audio_path = output_dir / f"round_{rno:03d}_scene_{scene_index:03d}.wav"
            fingerprint = tts_cache_fingerprint(tts_runtime, f"[{scene['emotion']}]{scene['text']}") if not dry_run else None
            if not dry_run:
                _synthesize_with_cache(
                    tts_runtime,
                    f"[{scene['emotion']}]{scene['text']}",
                    fingerprint,
                    scene_audio_path,
                    tts_cache_dir,
                    synthesize_emotional,
                )
                scene_inputs.append((scene_audio_path, relative_start, window_duration))
            scene_records.append({
                "source_start_sec": scene["t_start"],
                "source_end_sec": scene["t_end"],
                "relative_start_sec": relative_start,
                "audio_duration_sec": None,
                "audio_path": str(scene_audio_path),
                "cache_fingerprint": fingerprint,
            })

        if not dry_run:
            durations = _assemble_scene_wav(
                scene_inputs,
                audio_path,
                round_duration,
                default_sample_rate=sample_rate,
            )
            for scene_record, duration in zip(scene_records, durations):
                scene_record["audio_duration_sec"] = duration
            _, _, round_audio_duration = _wav_info(audio_path)
        else:
            round_audio_duration = round_duration
        round_record.phase4_audio = AudioData(audio_path=str(audio_path), duration_sec=round_audio_duration)

        # 逐局视频混音（需要 make_video=true、源视频和该局时间戳）
        video_path: str | None = None
        if p4.get("make_video", False) and source_video is not None and scenes and not dry_run:
            existing_clip = _resolve_existing_clip(round_record, clip_cache_dir)
            if existing_clip is not None:
                print(f"[phase4] 复用 clip: {existing_clip}")
                clip_path = existing_clip
            else:
                if source_video_sha256 is None:
                    source_video_sha256 = _sha256_file(source_video)
                clip_path = _clip_for_round(
                    round_record,
                    source_video,
                    clip_cache_dir,
                    source_sha256=source_video_sha256,
                )
            out_mp4 = output_dir / f"round_{rno:03d}.mp4"
            _mux_round_video(clip_path, audio_path, out_mp4, game_vol=game_vol, comm_vol=comm_vol)
            video_path = str(out_mp4)

        records.append({
            "round_no": rno,
            "audio_path": str(audio_path),
            "video_path": video_path,
            "skipped": skipped,
            "aligned": True,
            "segments": scene_records,
        })

    manifest = {
        "rounds": records,
        "total_rounds": len(match.rounds),
        "skipped_rounds": skipped_count,
        "output_dir": str(output_dir),
        "commentary_path": str(commentary_path),
    }
    if not dry_run:
        save_match(output_rounds_path, match)
        write_json(manifest_path, manifest)
    return manifest
