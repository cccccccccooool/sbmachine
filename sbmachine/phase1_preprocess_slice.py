"""第一阶段（预处理切片与特征提取）。负责读取整局录像以及原始检测/片段清单，并切分成小局，可选利用 ffmpeg 切出小局视频。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import read_json, require_path, resolve_path, write_json
from sbmachine.phase1_slice import build_rounds_from_segments
from sbmachine.schemas import MatchPackage, save_match


def load_or_build_segments(
    *,
    detections_path: Path | None,
    segments_path: Path | None,
    timer_increase_sec: float,
    stale_timer_sec: float,
    min_live_segment_sec: float,
    live_start_confirmations: int,
    max_live_other_gap_sec: float,
    terminal_grace_sec: float,
    debug_path: Path | None,
) -> list[dict]:
    if segments_path is not None:
        payload = read_json(segments_path)
        if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
            return payload["segments"]
        if isinstance(payload, list):
            return payload
        raise ValueError("segments JSON 必须是 list,或包含 segments 字段。")

    if detections_path is None:
        raise ValueError("切片预处理需要 --detections 或 --segments。")

    from sbmachine.round_segmenter import SegmenterConfig, load_observations, segment_observations

    config = SegmenterConfig(
        min_live_segment_sec=min_live_segment_sec,
    )
    return [
        segment.to_dict()
        for segment in segment_observations(load_observations(detections_path), config, debug_path=debug_path)
    ]


def cut_video_clip(video_path: Path, output_path: Path, start_sec: float, end_sec: float, *, reencode: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.05, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
    ]
    if reencode:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"])
    else:
        cmd.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)


def attach_round_clips(match: MatchPackage, video_path: Path, clip_dir: Path | None, *, reencode: bool) -> None:
    if clip_dir is None:
        return
    for round_record in match.rounds:
        clip_path = clip_dir / f"round_{round_record.round_no:03d}.mp4"
        cut_video_clip(video_path, clip_path, round_record.start_sec, round_record.end_sec, reencode=reencode)
        round_record.segment_video = str(clip_path)


def write_round_list(path: Path, match: MatchPackage) -> Path:
    payload = {
        "video_path": match.video_path,
        "map_name": match.map_name,
        "total_rounds": len(match.rounds),
        "score_final": {"ct": match.score_final.ct, "t": match.score_final.t},
        "rounds": [
            {
                "round_no": item.round_no,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "score_before": {"ct": item.score_before.ct, "t": item.score_before.t},
                "score_after": {"ct": item.score_after.ct, "t": item.score_after.t},
                "segment_video": item.segment_video,
                "source_reason": item.source_reason,
            }
            for item in match.rounds
        ],
    }
    return write_json(path, payload)


def run_preprocess_slice(
    *,
    video_path: Path,
    output_rounds_path: Path,
    output_list_path: Path,
    output_segments_path: Path | None = None,
    detections_path: Path | None = None,
    segments_path: Path | None = None,
    clip_dir: Path | None = None,
    map_name: str = "Unknown",
    timer_increase_sec: float = 3.0,
    stale_timer_sec: float = 2.5,
    min_live_segment_sec: float = 6.0,
    live_start_confirmations: int = 2,
    max_live_other_gap_sec: float = 12.0,
    terminal_grace_sec: float = 3.0,
    debug_path: Path | None = None,
    reencode_clips: bool = False,
) -> MatchPackage:
    segments = load_or_build_segments(
        detections_path=detections_path,
        segments_path=segments_path,
        timer_increase_sec=timer_increase_sec,
        stale_timer_sec=stale_timer_sec,
        min_live_segment_sec=min_live_segment_sec,
        live_start_confirmations=live_start_confirmations,
        max_live_other_gap_sec=max_live_other_gap_sec,
        terminal_grace_sec=terminal_grace_sec,
        debug_path=debug_path,
    )
    if output_segments_path is not None:
        write_json(output_segments_path, segments)

    match = build_rounds_from_segments(video_path, segments, map_name)
    attach_round_clips(match, video_path, clip_dir, reencode=reencode_clips)
    save_match(output_rounds_path, match)
    write_round_list(output_list_path, match)
    return match
