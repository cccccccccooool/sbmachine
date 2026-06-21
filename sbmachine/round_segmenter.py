"""帧类型切片适配器。负责将 tools/run_frame_type_slicer.py 输出的帧分类结果转换为统一 the VideoSegment 回合片段对象，供后续流程使用。"""
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
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        return payload["segments"]
    if isinstance(payload, list):
        return payload
    raise ValueError("frame_type 输入必须是 JSONL 行、segments 数组,或包含 segments 字段的 JSON。")


def _segment_from_dict(data: dict) -> VideoSegment:
    kind = str(data.get("kind") or data.get("label") or data.get("smooth_label") or SegmentKind.UNKNOWN_REVIEW.value)
    if kind == "game":
        kind = SegmentKind.LIVE_ROUND.value
    elif kind not in {item.value for item in SegmentKind}:
        kind = SegmentKind.BREAK.value
    score = data.get("score")
    return VideoSegment(
        kind=SegmentKind(kind),
        start_sec=float(data.get("start_sec", data.get("time_sec", 0))),
        end_sec=float(data.get("end_sec", data.get("time_sec", 0))),
        score=Score.from_values(score.get("ct"), score.get("t")) if isinstance(score, dict) else None,
        reason=str(data.get("reason", data.get("smooth_label", data.get("label", "frame_type")))),
    )


def _segments_from_rows(rows: list[dict], config: SegmenterConfig) -> list[VideoSegment]:
    segments: list[VideoSegment] = []
    active: VideoSegment | None = None
    for row in sorted(rows, key=lambda item: float(item.get("time_sec", item.get("start_sec", 0)))):
        ts = float(row.get("time_sec", row.get("start_sec", 0)))
        label = str(row.get("smooth_label", row.get("label", "")))
        is_live = label == config.live_label or label == SegmentKind.LIVE_ROUND.value
        if is_live:
            if active is None:
                active = VideoSegment(SegmentKind.LIVE_ROUND, ts, ts, reason="frame_type_live_game")
            else:
                active.end_sec = ts
            continue
        if active is not None:
            if active.end_sec - active.start_sec >= config.min_live_segment_sec:
                segments.append(active)
            active = None
    if active is not None and active.end_sec - active.start_sec >= config.min_live_segment_sec:
        segments.append(active)
    return _merge_live_gaps(segments, config.bridge_gap_sec)


def _merge_live_gaps(segments: list[VideoSegment], bridge_gap_sec: float) -> list[VideoSegment]:
    merged: list[VideoSegment] = []
    for segment in segments:
        if merged and segment.start_sec - merged[-1].end_sec <= bridge_gap_sec:
            merged[-1].end_sec = segment.end_sec
            merged[-1].reason = "; ".join(part for part in [merged[-1].reason, segment.reason] if part)
        else:
            merged.append(segment)
    return merged


def segment_observations(
    observations: Iterable[dict],
    config: SegmenterConfig | None = None,
    debug_path: str | Path | None = None,
) -> list[VideoSegment]:
    cfg = config or SegmenterConfig()
    rows = list(observations)
    if not rows:
        return []
    if all("start_sec" in row and "end_sec" in row for row in rows):
        segments = [_segment_from_dict(row) for row in rows]
    else:
        segments = _segments_from_rows(rows, cfg)
    if debug_path is not None:
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        Path(debug_path).write_text(
            json.dumps({"mode": "frame_type_only", "segments": [item.to_dict() for item in segments]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return segments
