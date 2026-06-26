"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：统一对局状态 Schema。

启动方式：被 vision_service/context_assembler.py 等模块导入。
输入数据流：无文件 I/O；由上游感知模块（OCR/YOLO/VLM/音频）组装 GameState 对象。
输出数据流：GameState.render() 返回结构化中文文本，直接喂给 LLM 作为解说 prompt。
用法用途：定义 GameState 数据模型及其 render() 方法，作为感知模块与解说 LLM 之间的统一契约。
所有感知模块最终都应产出 GameState；人设 LLM 的 SFT 训练输入和线上推理输入也都应使用 GameState.render()。
这条契约一旦稳定，后续训练数据和真实推理场景就不会出现格式错位。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    CT = "CT"
    T = "T"


class Phase(str, Enum):
    FREEZE = "freeze"
    OPENING = "opening"
    MID = "mid"
    CLUTCH = "clutch"
    POST = "post"


class UtilityType(str, Enum):
    SMOKE = "smoke"
    MOLOTOV = "molotov"
    FLASH = "flash"
    HE = "he"


class EventSource(str, Enum):
    KILLFEED = "killfeed"
    AUDIO = "audio"
    FUSED = "fused"


PHASE_CN = {
    Phase.FREEZE: "准备阶段",
    Phase.OPENING: "开局",
    Phase.MID: "中期",
    Phase.CLUTCH: "残局",
    Phase.POST: "回合结束",
}

UTIL_CN = {
    UtilityType.SMOKE: "烟雾",
    UtilityType.MOLOTOV: "燃烧弹",
    UtilityType.FLASH: "闪光弹",
    UtilityType.HE: "高爆雷",
}


@dataclass
class KillEvent:
    killer: str
    victim: str
    weapon: str
    through_smoke: bool = False
    is_headshot: bool = False
    source: EventSource = EventSource.KILLFEED


@dataclass
class UtilityOnMap:
    kind: UtilityType
    callout: str
    active: bool = True


@dataclass
class DirectorView:
    """导播当前视角。

    这里只描述可观测内容,不从单帧画面臆断击杀、经济或是否真正交火。是否交火由音频事件仲裁。
    """

    focus_player: Optional[str] = None
    focus_side: Optional[Side] = None
    weapon: Optional[str] = None
    crosshair_toward: Optional[str] = None
    action: Optional[str] = None
    engaged: bool = False


@dataclass
class GameState:
    map_name: str
    round_no: int
    score_ct: int
    score_t: int
    phase: Phase = Phase.MID
    econ_ct: str = ""
    econ_t: str = ""
    alive_ct: int = 5
    alive_t: int = 5
    director: DirectorView = field(default_factory=DirectorView)
    utilities: list[UtilityOnMap] = field(default_factory=list)
    recent_kills: list[KillEvent] = field(default_factory=list)
    bomb_planted: bool = False
    bomb_site: Optional[str] = None
    audio_labels: list[str] = field(default_factory=list)
    meme_hints: list[str] = field(default_factory=list)

    def render(self) -> str:
        """渲染为喂给 LLM 的结构化中文文本。

        被 talk_service/commentary_pipeline.py 的 _render_state() 间接调用，
        也可被 __main__ 直接调用测试。

        无参数。返回多行中文字符串，包含地图、比分、经济、存活、导播视角、道具、击杀、音频事件等。
        """
        lines: list[str] = []
        lines.append(
            f"[{self.map_name} | 第{self.round_no}回合 | "
            f"CT {self.score_ct}-{self.score_t} T | {PHASE_CN[self.phase]}]"
        )

        economy = []
        if self.econ_ct:
            economy.append(f"CT {self.econ_ct}")
        if self.econ_t:
            economy.append(f"T {self.econ_t}")
        if economy:
            lines.append("经济: " + " / ".join(economy))

        lines.append(f"存活: CT {self.alive_ct} vs T {self.alive_t}")

        director = self.director
        if director.focus_player:
            prefix = f"{director.focus_side.value} " if director.focus_side else ""
            segment = f"镜头: {prefix}{director.focus_player}"
            if director.weapon:
                segment += f"({director.weapon})"
            if director.action:
                segment += f", {director.action}"
            if director.crosshair_toward:
                segment += f", 准心朝向{director.crosshair_toward}"
            if not director.engaged:
                segment += " [未交火]"
            lines.append(segment)

        if self.utilities:
            lines.append("道具(小地图):")
            for item in self.utilities:
                status = "" if item.active else "(已消散)"
                lines.append(f"  - {item.callout}: {UTIL_CN[item.kind]}{status}")

        if self.bomb_planted:
            lines.append(f"炸弹: 已安放于 {self.bomb_site or '?'}")

        if self.recent_kills:
            lines.append("最近事件:")
            for kill in self.recent_kills:
                extra = []
                if kill.through_smoke:
                    extra.append("穿烟")
                if kill.is_headshot:
                    extra.append("爆头")
                tag = f"[{','.join(extra)}]" if extra else ""
                lines.append(f"  - {kill.killer}({kill.weapon}{tag}) 击杀 {kill.victim}")

        if self.audio_labels:
            lines.append("音频事件: " + "、".join(self.audio_labels))

        if self.meme_hints:
            lines.append("[解说参考素材](可选,自然融入,勿生硬): " + "; ".join(self.meme_hints))

        return "\n".join(lines)


if __name__ == "__main__":
    demo = GameState(
        map_name="Mirage",
        round_no=14,
        score_ct=8,
        score_t=5,
        phase=Phase.CLUTCH,
        econ_ct="满经济,双狙",
        econ_t="半起强打",
        alive_ct=5,
        alive_t=3,
        director=DirectorView(
            focus_player="ZywOo",
            focus_side=Side.CT,
            weapon="AWP",
            crosshair_toward="B点窗口",
            action="架点试探",
            engaged=False,
        ),
        utilities=[UtilityOnMap(UtilityType.SMOKE, "A点连接"), UtilityOnMap(UtilityType.MOLOTOV, "中路")],
        recent_kills=[KillEvent("ZywOo", "sh1ro", "AWP", through_smoke=True, source=EventSource.FUSED)],
        audio_labels=["远处枪声"],
        meme_hints=["翻云覆雨,用于形容残局翻盘"],
    )
    print(demo.render())
