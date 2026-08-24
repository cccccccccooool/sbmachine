"""帧类型切片适配器。负责将 tools/run_frame_type_slicer.py 输出的帧分类结果转换为统一的 VideoSegment 回合片段对象，供后续流程使用。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class SegmentKind(str, Enum):
    LIVE_ROUND = "live_round"
    LIVE_POSTPLANT = "live_postplant"
    REPLAY = "replay"
    BREAK = "break"
    UNKNOWN_REVIEW = "unknown_review"


@dataclass
class Score:
    ct: int = 0
    t: int = 0

    @classmethod
    def from_values(cls, ct, t) -> "Score":
        return cls(ct=int(ct or 0), t=int(t or 0))


@dataclass
class VideoSegment:
    kind: SegmentKind
    start_sec: float
    end_sec: float
    score: Score | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "start_sec": round(float(self.start_sec), 3),
            "end_sec": round(float(self.end_sec), 3),
            "score": None if self.score is None else {"ct": self.score.ct, "t": self.score.t},
            "reason": self.reason,
        }


@dataclass
class SegmenterConfig:
    min_live_segment_sec: float = 6.0
    bridge_gap_sec: float = 3.0
    live_label: str = "game"


def load_observations(path: str | Path) -> list[dict]:
    observations_path = Path(path)
    if observations_path.suffix.lower() == ".jsonl":
        observations = []
        with observations_path.open("r", encoding="utf-8") as observations_file:
            for line in observations_file:
                if line.strip():
                    observations.append(json.loads(line))
        return observations
    payload = json.loads(observations_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        return payload["segments"]
    if isinstance(payload, list):
        return payload
    raise ValueError("frame_type 输入必须是 JSONL 行、segments 数组,或包含 segments 字段的 JSON。")


def _segment_from_dict(segment_data: dict) -> VideoSegment:
    kind = str(segment_data.get("kind") or segment_data.get("label") or segment_data.get("smooth_label") or SegmentKind.UNKNOWN_REVIEW.value)
    if kind == "game":
        kind = SegmentKind.LIVE_ROUND.value
    elif kind not in {item.value for item in SegmentKind}:
        kind = SegmentKind.BREAK.value
    score = segment_data.get("score")
    return VideoSegment(
        kind=SegmentKind(kind),
        start_sec=float(segment_data.get("start_sec", segment_data.get("time_sec", 0))),
        end_sec=float(segment_data.get("end_sec", segment_data.get("time_sec", 0))),
        score=Score.from_values(score.get("ct"), score.get("t")) if isinstance(score, dict) else None,
        reason=str(segment_data.get("reason", segment_data.get("smooth_label", segment_data.get("label", "frame_type")))),
    )


def _segments_from_rows(observations: list[dict], config: SegmenterConfig) -> list[VideoSegment]:
    live_segments: list[VideoSegment] = []
    active_segment: VideoSegment | None = None
    for observation in sorted(observations, key=lambda item: float(item.get("time_sec", item.get("start_sec", 0)))):
        timestamp = float(observation.get("time_sec", observation.get("start_sec", 0)))
        observation_label = str(observation.get("smooth_label", observation.get("label", "")))
        is_live_segment = observation_label == config.live_label or observation_label == SegmentKind.LIVE_ROUND.value
        if is_live_segment:
            if active_segment is None:
                active_segment = VideoSegment(SegmentKind.LIVE_ROUND, timestamp, timestamp, reason="frame_type_live_game")
            else:
                active_segment.end_sec = timestamp
            continue
        if active_segment is not None:
            if active_segment.end_sec - active_segment.start_sec >= config.min_live_segment_sec:
                live_segments.append(active_segment)
            active_segment = None
    if active_segment is not None and active_segment.end_sec - active_segment.start_sec >= config.min_live_segment_sec:
        live_segments.append(active_segment)
    return _merge_live_gaps(live_segments, config.bridge_gap_sec)


def _merge_live_gaps(live_segments: list[VideoSegment], bridge_gap_sec: float) -> list[VideoSegment]:
    merged_segments: list[VideoSegment] = []
    for segment in live_segments:
        if merged_segments and segment.start_sec - merged_segments[-1].end_sec <= bridge_gap_sec:
            merged_segments[-1].end_sec = segment.end_sec
            merged_segments[-1].reason = "; ".join(part for part in [merged_segments[-1].reason, segment.reason] if part)
        else:
            merged_segments.append(segment)
    return merged_segments


def segment_observations(
    observations: Iterable[dict],
    config: SegmenterConfig | None = None,
    debug_path: str | Path | None = None,
) -> list[VideoSegment]:
    segmenter_config = config or SegmenterConfig()
    observation_rows = list(observations)
    if not observation_rows:
        return []
    if all("start_sec" in row and "end_sec" in row for row in observation_rows):
        segments = [_segment_from_dict(row) for row in observation_rows]
    else:
        segments = _segments_from_rows(observation_rows, segmenter_config)
    if debug_path is not None:
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        Path(debug_path).write_text(
            json.dumps({"mode": "frame_type_only", "segments": [item.to_dict() for item in segments]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return segments
