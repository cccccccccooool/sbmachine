"""数据结构定义。包含对局比分、关键帧、视觉数据、情感片段、音频数据、回合记录及整体包状态的统一数据模型定义与持久化接口。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sbmachine.common import read_json, write_json


@dataclass
class Score:
    ct: int = 0
    t: int = 0

    @classmethod
    def from_any(cls, value: Any) -> "Score":
        if isinstance(value, Score):
            return value
        if isinstance(value, dict):
            return cls(ct=int(value.get("ct", 0)), t=int(value.get("t", 0)))
        return cls()


@dataclass
class KeyFrame:
    time_sec: float
    gate_reason: str
    vlm_hint: str
    vlm_response: str = ""
    yolo_tags: list[str] = field(default_factory=list)
    yolo_confidence: float = 0.0
    global_vlm_output: str = ""
    ui_regions: list[dict] = field(default_factory=list)
    background_info: dict = field(default_factory=dict)
    has_vlm: bool = True   # False = 背景行(仅 demo 事实,无画面解码)


@dataclass
class VisionData:
    background: list[dict] = field(default_factory=list)
    key_frames: list[KeyFrame] = field(default_factory=list)
    yolo_required: bool = False
    yolo_model: str = ""
    detector_mode: str = "yolo_ui_locator_then_layered_vlm"
    sample_interval_sec: float = 1.0
    total_yolo_frames: int = 0
    total_vlm_calls: int = 0


@dataclass
class EmotionSegment:
    emotion: str
    text: str
    order: int = 0


@dataclass
class SemanticData:
    model_profile: str = "lite"
    model_name: str = ""
    commentary_text: str = ""
    emotion_segments: list[EmotionSegment] = field(default_factory=list)


@dataclass
class AudioData:
    audio_path: str = ""
    duration_sec: float | None = None


@dataclass
class RoundRecord:
    round_no: int
    start_sec: float
    end_sec: float
    score_before: Score = field(default_factory=Score)
    score_after: Score = field(default_factory=Score)
    kind: str = "live_round"
    source_reason: str = ""
    segment_video: str = ""
    demo_round_hint: int | str | None = None
    phase2_vision: VisionData | None = None
    phase3_semantic: SemanticData | None = None
    phase4_audio: AudioData | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "RoundRecord":
        return cls(
            round_no=int(data.get("round_no", 0)),
            start_sec=float(data.get("start_sec", 0)),
            end_sec=float(data.get("end_sec", 0)),
            score_before=Score.from_any(data.get("score_before")),
            score_after=Score.from_any(data.get("score_after")),
            kind=str(data.get("kind", "live_round")),
            source_reason=str(data.get("source_reason", data.get("reason", ""))),
            segment_video=str(data.get("segment_video", "")),
            demo_round_hint=data.get("demo_round_hint"),
            phase2_vision=_vision_from_dict(data.get("_phase2_vision") or data.get("phase2_vision")),
            phase3_semantic=_semantic_from_dict(data.get("_phase3_semantic") or data.get("phase3_semantic")),
            phase4_audio=_audio_from_dict(data.get("_phase4_audio") or data.get("phase4_audio")),
        )

    def to_dict(self) -> dict:
        data = {
            "round_no": self.round_no,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "score_before": asdict(self.score_before),
            "score_after": asdict(self.score_after),
            "kind": self.kind,
            "source_reason": self.source_reason,
        }
        if self.segment_video:
            data["segment_video"] = self.segment_video
        if self.demo_round_hint is not None:
            data["demo_round_hint"] = self.demo_round_hint
        if self.phase2_vision is not None:
            data["_phase2_vision"] = asdict(self.phase2_vision)
        if self.phase3_semantic is not None:
            data["_phase3_semantic"] = asdict(self.phase3_semantic)
        if self.phase4_audio is not None:
            data["_phase4_audio"] = asdict(self.phase4_audio)
        return data


@dataclass
class MatchPackage:
    video_path: str
    map_name: str = "Unknown"
    total_rounds: int = 0
    score_final: Score = field(default_factory=Score)
    rounds: list[RoundRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "MatchPackage":
        rounds = [RoundRecord.from_dict(item) for item in data.get("rounds", [])]
        score_final = Score.from_any(data.get("score_final"))
        if rounds and data.get("score_final") is None:
            score_final = rounds[-1].score_after
        return cls(
            video_path=str(data.get("video_path", "")),
            map_name=str(data.get("map_name", "Unknown")),
            total_rounds=int(data.get("total_rounds", len(rounds))),
            score_final=score_final,
            rounds=rounds,
        )

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "map_name": self.map_name,
            "total_rounds": len(self.rounds) if self.total_rounds == 0 else self.total_rounds,
            "score_final": asdict(self.score_final),
            "rounds": [item.to_dict() for item in self.rounds],
        }


def load_match(path: Path) -> MatchPackage:
    """从 JSON 文件加载 MatchPackage。"""
    return MatchPackage.from_dict(read_json(path))


def save_match(path: Path, match: MatchPackage) -> Path:
    """将 MatchPackage 保存为 JSON 文件。"""
    match.total_rounds = len(match.rounds)
    if match.rounds:
        match.score_final = match.rounds[-1].score_after
    return write_json(path, match.to_dict())


def _vision_from_dict(data: Any) -> VisionData | None:
    if not isinstance(data, dict):
        return None
    frames = [KeyFrame(**{k: v for k, v in item.items() if k in KeyFrame.__dataclass_fields__}) for item in data.get("key_frames", [])]
    copied = dict(data)
    copied["key_frames"] = frames
    return VisionData(**{k: v for k, v in copied.items() if k in VisionData.__dataclass_fields__})


def _semantic_from_dict(data: Any) -> SemanticData | None:
    if not isinstance(data, dict):
        return None
    segments = [EmotionSegment(**{k: v for k, v in item.items() if k in EmotionSegment.__dataclass_fields__}) for item in data.get("emotion_segments", [])]
    copied = dict(data)
    copied["emotion_segments"] = segments
    return SemanticData(**{k: v for k, v in copied.items() if k in SemanticData.__dataclass_fields__})


def _audio_from_dict(data: Any) -> AudioData | None:
    if not isinstance(data, dict):
        return None
    return AudioData(**{k: v for k, v in data.items() if k in AudioData.__dataclass_fields__})
