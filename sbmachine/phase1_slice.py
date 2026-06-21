"""第一阶段（视频粗切分）。根据比分变化与 HUD 时间戳，对整局录像进行逻辑上的粗切分，生成以回合为单位的片段结构。"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.common import read_json, require_path, resolve_path
from sbmachine.schemas import MatchPackage, RoundRecord, Score, save_match


def build_rounds_from_segments(video_path: Path, segments: list[dict], map_name: str) -> MatchPackage:
    """将 segments 列表转换为 MatchPackage。"""
    rounds: list[RoundRecord] = []
    previous_score = Score()
    for item in segments:
        kind = str(item.get("kind") or "live_round")
        if kind not in {"live_round", "live_postplant"}:
            continue
        score_after = Score.from_any(item.get("score")) if item.get("score") is not None else previous_score
        if kind == "live_postplant" and rounds:
            rounds[-1].end_sec = max(rounds[-1].end_sec, float(item.get("end_sec", rounds[-1].end_sec)))
            rounds[-1].score_after = score_after
            if item.get("reason"):
                rounds[-1].source_reason = "; ".join(part for part in [rounds[-1].source_reason, str(item["reason"])] if part)
            previous_score = score_after
            continue
        rounds.append(
            RoundRecord(
                round_no=len(rounds) + 1,
                start_sec=float(item.get("start_sec", 0)),
                end_sec=float(item.get("end_sec", 0)),
                score_before=previous_score,
                score_after=score_after,
                kind=kind,
                source_reason=str(item.get("reason", "")),
            )
        )
        previous_score = score_after
    return MatchPackage(video_path=str(video_path), map_name=map_name, rounds=rounds, score_final=previous_score)


def run_phase1(
    *,
    video_path: Path,
    output_path: Path,
    detections_path: Path | None = None,
    segments_path: Path | None = None,
    map_name: str = "Unknown",
    timer_increase_sec: float = 3.0,
    stale_timer_sec: float = 2.5,
    min_live_segment_sec: float = 6.0,
    live_start_confirmations: int = 2,
    max_live_other_gap_sec: float = 12.0,
    terminal_grace_sec: float = 3.0,
    debug_path: Path | None = None,
) -> MatchPackage:
    """执行第一阶段视频粗切分。可。（旧接口），也被 phase1_preprocess_slice.py 间接使用。"""
    if segments_path is not None:
        payload = read_json(segments_path)
        if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
            segments = payload["segments"]
        elif isinstance(payload, list):
            segments = payload
        else:
            raise ValueError("segments JSON 必须是 list,或包含 segments 字段的 dict。")
    elif detections_path is not None:
        from sbmachine.round_segmenter import SegmenterConfig, load_observations, segment_observations

        config = SegmenterConfig(
            min_live_segment_sec=min_live_segment_sec,
        )
        segments = [
            segment.to_dict()
            for segment in segment_observations(load_observations(detections_path), config, debug_path=debug_path)
        ]
    else:
        raise ValueError("第一阶段需要提供 --detections 或 --segments")

    match = build_rounds_from_segments(video_path, segments, map_name)
    save_match(output_path, match)
    return match
