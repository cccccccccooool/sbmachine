"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：上下文融合器。负责将 OCR 数据、YOLO 检出的道具数据、VLM 画面描述及音频交火事件整合成统一对局状态 GameState。

输入数据流：ocr_data/minimap_data/scene_desc/audio_events 四条信息流（均为 dict/对象）。
输出数据流：返回 GameState 对象，可调用 .render() 输出结构化中文文本。
用法用途：通过 assemble() 函数将多条感知数据流融合为统一的 GameState。
"""
from __future__ import annotations

from core.state_schema import (
    DirectorView,
    EventSource,
    GameState,
    KillEvent,
    Phase,
    Side,
    UtilityOnMap,
    UtilityType,
)


def _safe_int(value, default: int) -> int:
    """安全转换整数，OCR 脏数据（空字符串/None/非数字字符串）时返回 default。被 assemble() 调用。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _phase(value) -> Phase:
    """将字符串/Phase 值安全转为 Phase 枚举，无法匹配时返回 Phase.MID。被 assemble() 调用。"""
    if isinstance(value, Phase):
        return value
    try:
        return Phase(str(value))
    except ValueError:
        return Phase.MID


def _side(value):
    """将字符串/Side 值安全转为 Side 枚举或 None。被 assemble() 调用。"""
    if value in (None, ""):
        return None
    if isinstance(value, Side):
        return value
    try:
        return Side(str(value))
    except ValueError:
        return None


def _utility_kind(value) -> UtilityType:
    """将字符串/UtilityType 值安全转为 UtilityType 枚举，无法匹配时返回 SMOKE。被 assemble() 调用。"""
    if isinstance(value, UtilityType):
        return value
    try:
        return UtilityType(str(value))
    except ValueError:
        return UtilityType.SMOKE


def assemble(ocr_data=None, minimap_data=None, scene_desc=None, audio_events=None, meme_hints=None) -> GameState:
    """融合四条信息流，生成 GameState。

    被 talk_service/commentary_pipeline.py 的 _render_state() 调用。

    Parameters
    ----------
    ocr_data : dict | None
        OCR 数据，含 map_name/round_no/score_ct/score_t/phase/econ/alive/kills 等。
    minimap_data : dict | None
        小地图数据，含 utilities 列表。
    scene_desc : dict | None
        VLM 画面描述，含 focus_player/focus_side/weapon/crosshair_toward/action/engaged。
    audio_events : object | None
        音频事件对象，需有 has_gunfire 属性和 labels 列表。
    meme_hints : list[str] | None
        可选的解说素材提示。

    Returns
    -------
    GameState
        融合后的对局状态对象。
    """
    ocr_data = ocr_data or {}
    minimap_data = minimap_data or {}
    scene_desc = scene_desc or {}

    audio_labels = list(getattr(audio_events, "labels", []) or [])

    director = DirectorView(
        focus_player=scene_desc.get("focus_player"),
        focus_side=_side(scene_desc.get("focus_side")),
        weapon=scene_desc.get("weapon"),
        crosshair_toward=scene_desc.get("crosshair_toward"),
        action=scene_desc.get("action"),
        engaged=arbitrate_engaged(bool(scene_desc.get("engaged")), audio_events),
    )

    utilities = [
        UtilityOnMap(kind=_utility_kind(item.get("kind")), callout=item.get("callout", "未知位置"), active=item.get("active", True))
        for item in minimap_data.get("utilities", [])
    ]

    kills = []
    for item in ocr_data.get("kills", []):
        kills.append(
            KillEvent(
                killer=item.get("killer", "未知"),
                victim=item.get("victim", "未知"),
                weapon=item.get("weapon", "未知武器"),
                through_smoke=bool(item.get("through_smoke", False)),
                is_headshot=bool(item.get("is_headshot", False)),
                source=EventSource.KILLFEED,
            )
        )

    return GameState(
        map_name=ocr_data.get("map_name", "Unknown"),
        round_no=_safe_int(ocr_data.get("round_no"), 1),
        score_ct=_safe_int(ocr_data.get("score_ct"), 0),
        score_t=_safe_int(ocr_data.get("score_t"), 0),
        phase=_phase(ocr_data.get("phase", Phase.MID)),
        econ_ct=ocr_data.get("econ_ct", ""),
        econ_t=ocr_data.get("econ_t", ""),
        alive_ct=_safe_int(ocr_data.get("alive_ct"), 5),
        alive_t=_safe_int(ocr_data.get("alive_t"), 5),
        director=director,
        utilities=utilities,
        recent_kills=kills,
        bomb_planted=bool(ocr_data.get("bomb_planted", False)),
        bomb_site=ocr_data.get("bomb_site"),
        audio_labels=audio_labels,
        meme_hints=list(meme_hints or []),
    )


def arbitrate_engaged(scene_says_engaged: bool, audio_events) -> bool:
    """判断是否真的交火。

    被 assemble() 调用。

    Parameters
    ----------
    scene_says_engaged : bool
        VLM 判定是否交火。
    audio_events : object | None
        音频事件对象，需有 has_gunfire 属性。

    Returns
    -------
    bool
        没有音频证据时保守返回 False，避免把试探/peek/架点误判成激烈交火。
    """
    if audio_events is None:
        return False
    return bool(getattr(audio_events, "has_gunfire", False))
