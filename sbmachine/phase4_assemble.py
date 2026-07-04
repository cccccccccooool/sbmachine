"""第四阶段（TTS 语音合成与逐局音画合成）。读取前一阶段带情感标签的解说文本，调用 TTS 客户端逐局合成语音，再用 ffmpeg 将每局解说音轨与游戏原声混音，产出逐局 mp4。"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from sbmachine.common import load_config, require_path, resolve_path, write_json
from sbmachine.phase4_av import _is_dummy_round, _mux_round_video, _run_ffmpeg, _tagged_text
from sbmachine.schemas import AudioData, load_match, save_match





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

    # ── TTS 配置 ──
    tts_config = config.get("tts", {})
    tts_runtime_path = require_path(tts_config.get("config", "audio_service/gpt_sovits_runtime.yaml"), "tts.config")
    tts_runtime = read_tts_config(tts_runtime_path) if not dry_run else {}

    # ── Phase4 配置（从 pipeline.yaml phase4 节读取，缺则用默认值）──
    p4 = config.get("phase4", {})
    output_dir = Path(p4.get("output_dir", "output/sbmachine/rounds"))
    comm_vol = float(p4.get("commentary_volume", 1.0))
    game_vol = float(p4.get("game_audio_volume", 0.25))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 源视频（可选，用于逐局视频混音）──
    source_video = resolve_path(getattr(match, "video_path", None))

    records = []
    skipped_count = 0

    for round_record in tqdm(match.rounds, desc="Phase4 TTS+Mux", unit="round"):
        rno = round_record.round_no
        text = _tagged_text(round_record)
        audio_path = output_dir / f"round_{rno:03d}.wav"
        dummy = _is_dummy_round(text)

        if dummy:
            # 哑局：跳过 TTS 和视频混音
            skipped_count += 1
            records.append({
                "round_no": rno,
                "audio_path": str(audio_path),
                "video_path": None,
                "skipped": True,
            })
            continue

        # TTS 合成
        if not dry_run:
            synthesize_emotional(tts_runtime, text, audio_path)
        round_record.phase4_audio = AudioData(audio_path=str(audio_path))

        # 逐局视频混音（需要 make_video=true、源视频和该局时间戳）
        video_path: str | None = None
        if p4.get("make_video", False) and source_video is not None and not dry_run:
            clip_path = output_dir / f"clip_{rno:03d}.mp4"
            # 先切出该局原始片段
            _run_ffmpeg([
                "-ss", str(round_record.start_sec),
                "-to", str(round_record.end_sec),
                "-i", str(source_video),
                "-c", "copy",
                str(clip_path),
            ])
            out_mp4 = output_dir / f"round_{rno:03d}.mp4"
            _mux_round_video(clip_path, audio_path, out_mp4, game_vol=game_vol, comm_vol=comm_vol)
            video_path = str(out_mp4)

        records.append({
            "round_no": rno,
            "audio_path": str(audio_path),
            "video_path": video_path,
            "skipped": False,
        })

    save_match(output_rounds_path, match)
    manifest = {
        "rounds": records,
        "total_rounds": len(match.rounds),
        "skipped_rounds": skipped_count,
        "output_dir": str(output_dir),
    }
    write_json(manifest_path, manifest)
    return manifest
