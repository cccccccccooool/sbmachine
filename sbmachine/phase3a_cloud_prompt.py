"""云端分析单局 CS2 回合的 prompt 与响应契约。"""
from __future__ import annotations

import json


_SYSTEM = """你是 CS2 赛事中性底稿分析员。输入 JSON 的 timeline 是同一小局经过清洗和去重的事实时间线。
先按 t 字段理解事件先后、局势变化和连续片段，再自主选择本局唯一最值得解说的重点。
只能使用 timeline 中存在的 id 和事实作为证据；不要创造人物、击杀、道具作用、战术意图、路线、上下层关系。

只返回一个严格合法的 JSON 对象，恰好包含三个键：timeline_summary、evidence_ids、neutral。
所有属性名和字符串都必须用英文双引号包裹，不得缺引号、不得有多余逗号。
不要输出 Markdown、解释或额外文本，绝对不要回显或复制输入 JSON 的 round/roster/timeline/instructions 字段。

字段定义：
- timeline_summary：字符串，用一两句话概述本局关键节点，仅供审计。
- evidence_ids：字符串数组，是你为 neutral 选中的证据 id，全部取自输入 timeline 的 id 字段。
- neutral：一句自然中文中性稿，最多 100 个字符，只描述所选证据对应的事实。

没有可靠重点时进入静默：neutral 用空字符串 ""，evidence_ids 用空数组 []。

有内容示例：
{"timeline_summary":"A在3秒取得首杀后队伍占据主动。","evidence_ids":["kill-1"],"neutral":"A率先取得首杀，为队伍打开局面。"}
静默示例：
{"timeline_summary":"本局无明显可解说重点。","evidence_ids":[],"neutral":""}"""


def cloud_system_prompt() -> str:
    return _SYSTEM


def cloud_response_format() -> dict:
    return {"type": "json_object"}


def build_cloud_round_prompt(payload: dict) -> str:
    return (
        "以下是本局输入数据（仅供分析，不要复制回输出）：\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\n请只输出一个严格合法的 JSON 对象，键恰好为 timeline_summary、evidence_ids、neutral。"
    )


def _repair_missing_key_quotes(text: str) -> str:
    """部分供应商会丢掉对象键的开引号（例如 `,evidence_ids":`），这里补回。"""
    import re

    return re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)("\s*:)', r'\1"\2\3', text)


def _loads_tolerant(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return json.loads(_repair_missing_key_quotes(stripped))


def parse_cloud_response(text: str) -> dict:
    try:
        data = _loads_tolerant(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("cloud analyst response is not exact JSON") from exc
    if not isinstance(data, dict) or "neutral" not in data or "evidence_ids" not in data:
        raise ValueError("cloud analyst response has an invalid field set")
    if not isinstance(data["neutral"], str) or not isinstance(data["evidence_ids"], list):
        raise ValueError("cloud analyst response has invalid field types")
    summary = data.get("timeline_summary", "")
    if not isinstance(summary, (str, list)):
        raise ValueError("cloud analyst response has invalid field types")
    return {"timeline_summary": summary, "evidence_ids": data["evidence_ids"], "neutral": data["neutral"]}
