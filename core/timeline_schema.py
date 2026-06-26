"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：离线比赛时间线 Schema。

输入数据流：无文件 I/O；由上游阶段（demo 解析 + 视觉感知）组装 MatchTimeline 对象。
输出数据流：MatchTimeline.render() 返回结构化中文文本，直接喂给 LLM 作为解说证据材料。
用法用途：定义离线录像时间线数据模型，比 GameState 更高一级——一个按时间排序的回合事件集合，可渲染成解说 LLM 所需的稳定证据。

实时流水线对于单次观测状态仍然使用 ``GameState``。离线视频处理需要比它更高一级：
一个按时间排序的回合事件集合，可以渲染成解说 LLM 所需的稳定证据。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class EvidenceSource(str, Enum):
    OCR = "ocr"
    KILLFEED = "killfeed"
    AUDIO = "audio"
    YOLO = "yolo"
    VLM = "vlm"
    SCRIPT = "script"
    HUMAN = "human"
    FUSED = "fused"


class TimelineEventType(str, Enum):
    ROUND_START = "round_start"
    CONTACT = "contact"
    KILL = "kill"
    TRADE = "trade"
    UTILITY = "utility"
    BOMB_PLANT = "bomb_plant"
    BOMB_DEFUSE = "bomb_defuse"
    ROUND_END = "round_end"
    SCENE = "scene"
    NOTE = "note"


EVENT_TYPE_CN = {
    TimelineEventType.ROUND_START: "回合开始",
    TimelineEventType.CONTACT: "交火",
    TimelineEventType.KILL: "击杀",
    TimelineEventType.TRADE: "补枪/交换",
    TimelineEventType.UTILITY: "道具",
    TimelineEventType.BOMB_PLANT: "下包",
    TimelineEventType.BOMB_DEFUSE: "拆包",
    TimelineEventType.ROUND_END: "回合结束",
    TimelineEventType.SCENE: "镜头描述",
    TimelineEventType.NOTE: "备注",
}


def format_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS 字符串。

    被 TimelineEvent.render() 和 RoundTimeline.render() 调用。

    Parameters
    ----------
    seconds : float
        从视频起点开始的秒数。

    Returns
    -------
    str
        格式为 "MM:SS" 的时间字符串。
    """
    total = max(0, int(round(seconds)))
    minute, second = divmod(total, 60)
    return f"{minute:02d}:{second:02d}"


@dataclass
class TimelineEvent:
    """回合内一个由证据支持的事件。"""

    time_sec: float
    event_type: TimelineEventType = TimelineEventType.NOTE
    summary: str = ""
    game_clock: Optional[str] = None
    side: Optional[str] = None
    confidence: float = 1.0
    evidence: list[EvidenceSource] = field(default_factory=list)

    def render(self) -> str:
        """渲染单个事件为中文文本行。

        被 RoundTimeline.render() 逐事件调用。

        无参数。返回如 "- 01:30 / 局内 1:15 [击杀/CT] / 证据:killfeed 0.95: ZywOo 用 AWP 击杀 sh1ro" 的字符串。
        """
        clock = f" / 局内 {self.game_clock}" if self.game_clock else ""
        side = f" / {self.side}" if self.side else ""
        evidence = ""
        if self.evidence:
            evidence = " / 证据:" + ",".join(source.value for source in self.evidence)
        confidence = ""
        if self.confidence < 0.99:
            confidence = f" / 置信度:{self.confidence:.2f}"
        label = EVENT_TYPE_CN.get(self.event_type, self.event_type.value)
        low_conf = " / 低置信仅作参考" if self.confidence < 0.70 else ""
        return f"- {format_time(self.time_sec)}{clock} [{label}{side}]{evidence}{confidence}{low_conf}: {self.summary}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["evidence"] = [source.value for source in self.evidence]
        return data


@dataclass
class RoundTimeline:
    """离线录像(VOD)中的一个完整回合。"""

    round_no: int
    start_sec: float
    end_sec: float
    map_name: str = "Unknown"
    score_ct: int = 0
    score_t: int = 0
    winner: Optional[str] = None
    events: list[TimelineEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """渲染整回合时间线为多行中文文本。

        被 MatchTimeline.render() 逐回合调用。

        无参数。返回多行字符串，含回合头信息和所有事件行。
        """
        header = (
            f"## 第{self.round_no}回合 "
            f"({format_time(self.start_sec)}-{format_time(self.end_sec)}) "
            f"| {self.map_name} | CT {self.score_ct}-{self.score_t} T"
        )
        if self.winner:
            header += f" | 胜方:{self.winner}"

        lines = [header]
        for note in self.notes:
            lines.append(f"- 备注: {note}")
        for event in sorted(self.events, key=lambda item: item.time_sec):
            lines.append(event.render())
        if len(lines) == 1:
            lines.append("- 暂无可靠事件,等待 OCR/YOLO/VLM/音频补充。")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["events"] = [event.to_dict() for event in self.events]
        return data


@dataclass
class MatchTimeline:
    """渲染并喂给解说模型的完整离线录像时间线。"""

    video_path: str
    map_name: str = "Unknown"
    rounds: list[RoundTimeline] = field(default_factory=list)
    source_note: str = "offline_video"

    def render(self) -> str:
        """渲染整场比赛时间线为多行中文文本。

        被 talk_service/commentary_pipeline.py 的 _render_state() 调用。

        无参数。返回多行字符串，含视频信息、地图、解说要求，以及逐回合详情。
        """
        lines = [
            f"# 离线录像解说材料",
            f"视频: {self.video_path}",
            f"地图: {self.map_name}",
            "要求: 严格按时间顺序讲述整局发生了什么;硬事实以证据字段为准,VLM 只作为镜头描述参考。",
        ]
        for round_timeline in sorted(self.rounds, key=lambda item: item.round_no):
            lines.append("")
            lines.append(round_timeline.render())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "map_name": self.map_name,
            "source_note": self.source_note,
            "rounds": [round_timeline.to_dict() for round_timeline in self.rounds],
        }


def event_from_dict(data: dict) -> TimelineEvent:
    """从 dict 反序列化为 TimelineEvent 对象。

    被 round_from_dict() 逐事件调用。

    Parameters
    ----------
    data : dict
        单事件的 JSON 字典。

    Returns
    -------
    TimelineEvent
        填充好所有字段的事件对象。
    """
    evidence = []
    for value in data.get("evidence", []):
        try:
            evidence.append(EvidenceSource(str(value)))
        except ValueError:
            evidence.append(EvidenceSource.HUMAN)
    try:
        event_type = TimelineEventType(str(data.get("event_type", TimelineEventType.NOTE.value)))
    except ValueError:
        event_type = TimelineEventType.NOTE
    return TimelineEvent(
        time_sec=float(data.get("time_sec", 0)),
        event_type=event_type,
        summary=str(data.get("summary", "")),
        game_clock=data.get("game_clock"),
        side=data.get("side"),
        confidence=float(data.get("confidence", 1.0)),
        evidence=evidence,
    )


def round_from_dict(data: dict, default_map: str = "Unknown") -> RoundTimeline:
    """从 dict 反序列化为 RoundTimeline 对象。

    被 match_from_dict() 逐回合调用。

    Parameters
    ----------
    data : dict
        单回合的 JSON 字典。
    default_map : str
        默认地图名，当 data 中无 map_name 时使用。

    Returns
    -------
    RoundTimeline
        填充好所有字段的回合对象。
    """
    return RoundTimeline(
        round_no=int(data.get("round_no", 1)),
        start_sec=float(data.get("start_sec", 0)),
        end_sec=float(data.get("end_sec", data.get("start_sec", 0))),
        map_name=str(data.get("map_name", default_map)),
        score_ct=int(data.get("score_ct", 0)),
        score_t=int(data.get("score_t", 0)),
        winner=data.get("winner"),
        events=[event_from_dict(item) for item in data.get("events", [])],
        notes=[str(item) for item in data.get("notes", [])],
    )


def match_from_dict(data: dict) -> MatchTimeline:
    """从 dict 反序列化为 MatchTimeline 对象。

    被 talk_service/commentary_pipeline.py 的 _render_state() 调用。

    Parameters
    ----------
    data : dict
        整场比赛的 JSON 字典。

    Returns
    -------
    MatchTimeline
        填充好所有字段的对局时间线对象。
    """
    map_name = str(data.get("map_name", "Unknown"))
    return MatchTimeline(
        video_path=str(data.get("video_path", "")),
        map_name=map_name,
        source_note=str(data.get("source_note", "offline_video")),
        rounds=[round_from_dict(item, map_name) for item in data.get("rounds", [])],
    )
