"""6657 风格离线录像解说 AI 项目
项目功能:搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能:解说文本情绪标签解析与参考音频映射。解析内联在解说文本中的情绪标签,并为不同情绪匹配对应的语音合成参考音频配置。

启动方式:被 sbmachine/phase3b_style.py(parse_emotional_text)和 audio_service/gpt_sovits_client.py(synthesize_emotional)导入调用。
输入数据流:带内联情绪标签的解说文本字符串(如 "[激动]漂亮![平述]这波操作...")。
输出数据流:返回 EmotionSegment 列表(情绪+文本对)或参考音频配置 dict。
用法用途:通过 parse_emotional_text() 切分情绪段,通过 resolve_emotion_ref() 查找情绪对应的参考音频。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 固定的小情绪集合。标签集越小,LLM 越好学、解析越稳。
EMOTIONS = ("平述", "激动", "惊叹")
DEFAULT_EMOTION = "平述"

# 形如 [激动] 的标签;只匹配在集合内的标签,避免把正文里的方括号误当标签。
_TAG_PATTERN = re.compile(r"\[(" + "|".join(EMOTIONS) + r")\]")


@dataclass
class EmotionSegment:
    """一段带情绪的文本。"""

    emotion: str
    text: str


def parse_emotional_text(text: str) -> list[EmotionSegment]:
    """把带内联情绪标签的文本切成有序的 (情绪, 文本) 段。被 phase3b_style.py 和 gpt_sovits_client.py 调用。

    Parameters
    ----------
    text : str
        带情绪标签的文本,如 "[激动]漂亮![平述]这波操作..."。

    Returns
    -------
    list[EmotionSegment]
        情绪段列表，相邻同情绪段已合并。
    """
    raw = (text or "").strip()
    if not raw:
        return []

    segments: list[EmotionSegment] = []
    cursor = 0
    current = DEFAULT_EMOTION

    for match in _TAG_PATTERN.finditer(raw):
        chunk = raw[cursor:match.start()].strip()
        if chunk:
            segments.append(EmotionSegment(current, chunk))
        current = match.group(1)
        cursor = match.end()

    tail = raw[cursor:].strip()
    if tail:
        segments.append(EmotionSegment(current, tail))

    # 合并相邻同情绪段。
    merged: list[EmotionSegment] = []
    for seg in segments:
        if merged and merged[-1].emotion == seg.emotion:
            merged[-1] = EmotionSegment(seg.emotion, merged[-1].text + seg.text)
        else:
            merged.append(seg)
    return merged


def resolve_emotion_ref(emotion: str, emotion_refs: dict, default_ref: dict) -> dict:
    """按情绪查参考音频配置,查不到回退到默认 ref。被 gpt_sovits_client.py 的 synthesize_emotional() 调用。

    Parameters
    ----------
    emotion : str
        情绪名称(如 "激动"、"平述")。
    emotion_refs : dict
        情绪到参考音频的映射,形如 {"激动": {"audio_path": "...", "prompt_text": "..."}}。
    default_ref : dict
        默认参考音频配置(含 audio_path/prompt_text/prompt_lang)。

    Returns
    -------
    dict
        一定含 audio_path/prompt_text/prompt_lang 三个键。
    """
    ref = dict(default_ref or {})
    override = (emotion_refs or {}).get(emotion)
    if override:
        ref.update({k: v for k, v in override.items() if v})
    ref.setdefault("prompt_lang", "zh")
    return ref
