"""6657 风格离线录像解说 AI 项目。

项目功能：搭建一个“整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 ->
GPT-SoVITS 语音”的离线生成流水线。
本文件负责解析解说文本中的内联情绪标签，并为不同情绪匹配语音合成参考音频配置。

输入是带内联情绪标签的解说文本字符串（如 "[激动]漂亮![平述]这波操作..."）。
输出是 EmotionSegment 列表（情绪与文本对）或参考音频配置字典。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 固定的小情绪集合。标签集越小，LLM 越好学，解析越稳。
EMOTIONS = ("平述", "激动", "惊叹")
DEFAULT_EMOTION = "平述"

# 只匹配集合内形如 [激动] 的标签，避免把正文中的方括号误当成标签。
_TAG_PATTERN = re.compile(r"\[(" + "|".join(EMOTIONS) + r")\]")


@dataclass
class EmotionSegment:
    """一段带情绪的文本。"""

    emotion: str
    text: str


def parse_emotional_text(text: str) -> list[EmotionSegment]:
    """把带内联情绪标签的文本切成有序的（情绪，文本）段。

    Parameters
    ----------
    text : str
        带情绪标签的文本，如 "[激动]漂亮![平述]这波操作..."。

    Returns
    -------
    list[EmotionSegment]
        情绪段列表，相邻同情绪段已合并。
    """
    normalized_text = (text or "").strip()
    if not normalized_text:
        return []

    emotion_segments: list[EmotionSegment] = []
    cursor = 0
    current_emotion = DEFAULT_EMOTION

    for match in _TAG_PATTERN.finditer(normalized_text):
        chunk = normalized_text[cursor:match.start()].strip()
        if chunk:
            emotion_segments.append(EmotionSegment(current_emotion, chunk))
        current_emotion = match.group(1)
        cursor = match.end()

    tail = normalized_text[cursor:].strip()
    if tail:
        emotion_segments.append(EmotionSegment(current_emotion, tail))

    # 合并相邻的同情绪段。
    merged_segments: list[EmotionSegment] = []
    for segment in emotion_segments:
        if merged_segments and merged_segments[-1].emotion == segment.emotion:
            merged_segments[-1] = EmotionSegment(
                segment.emotion,
                merged_segments[-1].text + segment.text,
            )
        else:
            merged_segments.append(segment)
    return merged_segments


def resolve_emotion_ref(emotion: str, emotion_refs: dict, default_ref: dict) -> dict:
    """按情绪查找参考音频配置，查不到时回退到默认配置。

    Parameters
    ----------
    emotion : str
        情绪名称（如 "激动"、"平述"）。
    emotion_refs : dict
        情绪到参考音频的映射，形如 {"激动": {"audio_path": "...", "prompt_text": "..."}}。
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
