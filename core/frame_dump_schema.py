"""Frame-level visual dump schema for offline SFT data collection.

启动方式：被 data_pipeline/build_visual_sft_dataset.py 导入调用（fragment_dump_from_dict）。
输入数据流：视觉转储 JSON（由感知模块对预切小局逐帧采集产出）。
输出数据流：FragmentVisualDump.render() 返回结构化文本，供 SFT 数据构造时与主播原话配对。
用法用途：定义帧级视觉转储的数据模型；每个采样帧携带视觉检测器读取到的所有信息，SFT builder 将这些转储与同一视频时间线上的解说转录配对。

The input videos are assumed to be pre-cut into small live-game fragments.  This
schema is deliberately simple: each sampled frame carries whatever the visual
detectors can read, and the SFT builder aligns these dumps with commentator
transcripts on the same video timeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class PlayerUiState:
    name: str = ""
    side: str = ""
    hp: Optional[int] = None
    ammo: Optional[int] = None
    weapon: str = ""
    utilities: list[str] = field(default_factory=list)
    kills: Optional[int] = None
    deaths: Optional[int] = None
    money: Optional[int] = None
    has_armor: Optional[bool] = None
    has_defuse_kit: Optional[bool] = None
    dead: Optional[bool] = None
    state_color_debug: dict = field(default_factory=dict)

    def render(self) -> str:
        """渲染单帧视觉信息为中文文本行。

        被 FragmentVisualDump.render() 逐帧调用。

        无参数。返回多行字符串，包含时间戳、帧类型、计时器、比分、击杀栏、选手状态、场景提示。
        """
        bits = []
        if self.name:
            bits.append(self.name)
        if self.side:
            bits.append(self.side)
        if self.hp is not None:
            bits.append(f"HP{self.hp}")
        if self.ammo is not None:
            bits.append(f"Ammo{self.ammo}")
        if self.weapon:
            bits.append(self.weapon)
        if self.dead is True:
            bits.append("已阵亡")
        if self.kills is not None:
            bits.append(f"K{self.kills}")
        if self.deaths is not None:
            bits.append(f"D{self.deaths}")
        if self.money is not None:
            bits.append(f"${self.money}")
        if self.has_armor is not None:
            bits.append("有甲" if self.has_armor else "无甲")
        if self.has_defuse_kit is not None:
            bits.append("有钳" if self.has_defuse_kit else "无钳")
        if self.utilities:
            bits.append("道具:" + ",".join(self.utilities))
        return " / ".join(bits) if bits else "未知选手"


@dataclass
class KillfeedRow:
    killer: str = ""
    victim: str = ""
    weapon: str = ""
    killer_side: str = ""
    victim_side: str = ""
    headshot: bool = False
    through_smoke: bool = False
    wallbang: bool = False

    def key(self) -> tuple[str, str, str, str]:
        return (self.killer, self.victim, self.killer_side, self.victim_side)

    def render(self) -> str:
        """渲染击杀栏行为中文文本。

        被 FrameDump.render() 逐行调用。

        无参数。返回如 "ZywOo 用 AWP 击杀 sh1ro(killer:CT,victim:T,爆头,穿烟)" 的字符串。
        """
        tags = []
        if self.killer_side:
            tags.append(f"killer:{self.killer_side}")
        if self.victim_side:
            tags.append(f"victim:{self.victim_side}")
        if self.headshot:
            tags.append("爆头")
        if self.through_smoke:
            tags.append("穿烟")
        if self.wallbang:
            tags.append("穿墙")
        suffix = f"({','.join(tags)})" if tags else ""
        weapon = f" 用 {self.weapon}" if self.weapon else ""
        return f"{self.killer or '?'}{weapon} 击杀 {self.victim or '?'}{suffix}"


@dataclass
class FrameDump:
    time_sec: float
    frame_type: str = "live"
    round_timer_sec: Optional[float] = None
    score_ct: Optional[int] = None
    score_t: Optional[int] = None
    players: list[PlayerUiState] = field(default_factory=list)
    killfeed: list[KillfeedRow] = field(default_factory=list)
    scene_hint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        """渲染整帧转储为多行中文文本。

        被 FragmentVisualDump.render() 逐帧调用。

        无参数。返回多行字符串，含计时器/比分/击杀栏/选手列表/场景提示。
        """
        lines = [f"@{self.time_sec:.1f}s [{self.frame_type}]"]
        if self.round_timer_sec is not None:
            lines.append(f"  timer: {self.round_timer_sec:.1f}s")
        if self.score_ct is not None and self.score_t is not None:
            lines.append(f"  score: CT {self.score_ct}-{self.score_t} T")
        if self.killfeed:
            lines.append("  killfeed:")
            for row in self.killfeed:
                lines.append(f"    - {row.render()}")
        if self.players:
            lines.append("  players:")
            for player in self.players:
                lines.append(f"    - {player.render()}")
        if self.scene_hint:
            lines.append(f"  scene: {self.scene_hint}")
        return "\n".join(lines)


@dataclass
class FragmentVisualDump:
    video_path: str
    start_sec: float
    end_sec: float
    frames: list[FrameDump] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        """渲染整个片段转储为多行中文文本。

        被 data_pipeline/build_visual_sft_dataset.py 的 build_visual_sft_dataset() 调用，
        产出文本作为 SFT 数据的 state 字段。

        无参数。返回多行字符串，含片段头信息和逐帧详情。
        """
        lines = [f"# Fragment {self.start_sec:.1f}s-{self.end_sec:.1f}s", f"video: {self.video_path}"]
        for frame in self.frames:
            lines.append("")
            lines.append(frame.render())
        return "\n".join(lines)


def frame_from_dict(data: dict) -> FrameDump:
    """从 dict 反序列化为 FrameDump 对象。

    被 fragment_dump_from_dict() 逐帧调用。

    Parameters
    ----------
    data : dict
        单帧的 JSON 字典（含 time_sec/frame_type/players/killfeed 等字段）。

    Returns
    -------
    FrameDump
        填充好所有字段的帧对象。
    """
    return FrameDump(
        time_sec=float(data.get("time_sec", 0)),
        frame_type=str(data.get("frame_type", "live")),
        round_timer_sec=None if data.get("round_timer_sec") is None else float(data["round_timer_sec"]),
        score_ct=None if data.get("score_ct") is None else int(data["score_ct"]),
        score_t=None if data.get("score_t") is None else int(data["score_t"]),
        players=[PlayerUiState(**{k: v for k, v in item.items() if k in PlayerUiState.__dataclass_fields__}) for item in data.get("players", [])],
        killfeed=[KillfeedRow(**{k: v for k, v in item.items() if k in KillfeedRow.__dataclass_fields__}) for item in data.get("killfeed", [])],
        scene_hint=str(data.get("scene_hint", "")),
    )


def fragment_dump_from_dict(data: dict) -> FragmentVisualDump:
    """从 dict 反序列化为 FragmentVisualDump 对象。

    被 data_pipeline/build_visual_sft_dataset.py 的 build_visual_sft_dataset() 调用。

    Parameters
    ----------
    data : dict
        片段级 JSON 字典（含 video_path/start_sec/end_sec/frames）。

    Returns
    -------
    FragmentVisualDump
        填充好所有字段的片段对象。
    """
    return FragmentVisualDump(
        video_path=str(data.get("video_path", "")),
        start_sec=float(data.get("start_sec", 0)),
        end_sec=float(data.get("end_sec", 0)),
        frames=[frame_from_dict(item) for item in data.get("frames", [])],
    )
