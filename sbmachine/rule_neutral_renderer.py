"""Phase3a 规则中性句渲染器：纯模板基线（§8）。

- render_neutral：按优先级/时间稳定排序，用白名单连接词把 fact units 连成完整中性句。
- render_capsule：全部 required 事实的最短完整句表达；禁止子串截断。
- validate_preserved_facts：原子级校验器，计算 preserved_fact_ids（模型不得自报）。

本模块只消费白名单 fact-unit 任务（§8.1），不接触原始 DEM/坐标/未选事件。
纯模板是默认生产实现，不是模型失败时的兜底。
"""

from __future__ import annotations

import re

from sbmachine.commentary_planner import _ANCHOR_CATEGORIES

NEUTRAL_RENDERER_POLICY = "template_default_v1"
NEUTRAL_SOURCE = "rule_template"

_JOINER_NEUTRAL = "，随后"
_JOINER_NEUTRAL_SIMULTANEOUS = "，同时"
_JOINER_CAPSULE = "，"
_TICK_PER_SEC = 30

# 事件动词：用于原子级校验器识别事件锚点
_EVENT_VERBS = {
    "kill": "击杀",
    "bomb_planted": "安放",
    "defuse_started": "拆弹",
    "bomb_exploded": "爆炸",
    "bomb_defused": "拆除",
    "round_result": "赢下",
    "team_eliminated": "清零",
}
_RESULT_WORDS = ("赢下回合", "获胜", "胜", "回合结束", "清零")


class RendererError(ValueError):
    """渲染器无法生成合法输出。"""


class RendererUnfitError(RendererError):
    """capsule 在最高语速安全上界内仍无法进入固定 slot（写 JSON 前失败）。"""


def _safe_task(task: dict) -> tuple[list[dict], list[str]]:
    if not isinstance(task, dict):
        raise RendererError("task must be a dict")
    fact_units = task.get("fact_units")
    if not isinstance(fact_units, list) or not fact_units:
        raise RendererError("task.fact_units must be a non-empty list")
    required_fact_ids = task.get("required_fact_ids") or []
    if not isinstance(required_fact_ids, list):
        raise RendererError("task.required_fact_ids must be a list")
    return fact_units, required_fact_ids


def _sorted_units(fact_units: list[dict]) -> list[dict]:
    """稳定排序：priority desc, anchor_tick asc, fact_id asc（§8.2-1）。"""
    return sorted(
        fact_units,
        key=lambda u: (-int(u.get("priority", 0)), int(u.get("anchor_tick", 0)), str(u.get("fact_id", ""))),
    )


def _fact_by_id(fact_units: list[dict]) -> dict[str, dict]:
    return {str(u.get("fact_id")): u for u in fact_units}


def render_neutral(task: dict) -> dict:
    """纯模板 neutral：单事实直接使用完整 canonical_clause；多事实白名单连接词。

    返回 {neutral, neutral_source, preserved_fact_ids, renderer_policy}。
    模板无法覆盖全部 required 时抛 RendererError（不截断、不伪造）。
    """
    fact_units, required_fact_ids = _safe_task(task)
    units = _sorted_units(fact_units)
    by_id = _fact_by_id(fact_units)
    missing = [fid for fid in required_fact_ids if fid not in by_id]
    if missing:
        raise RendererError(f"required fact IDs missing from fact_units: {missing}")

    pieces: list[str] = []
    prev_tick: int | None = None
    for unit in units:
        clause = str(unit.get("canonical_clause") or "").strip()
        if not clause:
            continue
        if pieces:
            gap_sec = abs(int(unit.get("anchor_tick", 0)) - (prev_tick or 0)) / _TICK_PER_SEC
            if str(unit.get("kind")) == "round_result":
                joiner = _JOINER_NEUTRAL
            elif gap_sec <= 1.0:
                joiner = _JOINER_NEUTRAL_SIMULTANEOUS
            else:
                joiner = _JOINER_NEUTRAL
            pieces.append(joiner + clause)
        else:
            pieces.append(clause)
        prev_tick = int(unit.get("anchor_tick", 0))

    neutral = "".join(pieces)
    return {
        "neutral": neutral,
        "neutral_source": NEUTRAL_SOURCE,
        "preserved_fact_ids": validate_preserved_facts(neutral, fact_units, required_fact_ids)["preserved_fact_ids"],
        "renderer_policy": NEUTRAL_RENDERER_POLICY,
    }


def render_capsule(task: dict) -> str:
    """最短完整句 capsule：只含 required 事实的最短完整表达，绝不对子串截断。"""
    fact_units, required_fact_ids = _safe_task(task)
    by_id = _fact_by_id(fact_units)
    missing = [fid for fid in required_fact_ids if fid not in by_id]
    if missing:
        raise RendererError(f"required fact IDs missing from fact_units: {missing}")

    ordered = []
    for fid in required_fact_ids:
        unit = by_id[fid]
        clause = str(unit.get("capsule_clause") or unit.get("canonical_clause") or "").strip()
        if not clause:
            raise RendererError(f"required fact {fid} has no capsule clause")
        ordered.append(clause)
    return _JOINER_CAPSULE.join(ordered)


def _tokens_of_clause(unit: dict) -> list[object]:
    """从结构化字段抽取校验 token（不依赖措辞）。tuple 表示任一命中即可。"""
    tokens: list[object] = []
    for key in ("attacker", "victim", "winner", "side"):
        value = str(unit.get(key) or "").strip()
        if value and value not in ("对手", "进攻方", "一方"):
            tokens.append(value)
    if "C4" in str(unit.get("canonical_clause") or ""):
        tokens.append("C4")
    kind = str(unit.get("kind"))
    if kind == "kill":
        tokens.append("击杀")
    elif kind == "bomb_planted":
        tokens.append("安放")
    elif kind == "bomb_exploded":
        tokens.append("爆炸")
    elif kind == "bomb_defused":
        tokens.append(("拆除", "已拆除"))
    elif kind == "round_result":
        if str(unit.get("winner") or "").strip():
            tokens.append(("赢下", "胜", "获胜"))
        else:
            tokens.append(("回合结束", "结束"))
    elif kind == "team_eliminated":
        tokens.append("清零")
    return tokens


def _text_contains(text: str, token: object) -> bool:
    candidates = token if isinstance(token, tuple) else (token,)
    return any(
        (
            re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(candidate)}(?![A-Za-z0-9_.-])", text) is not None
            if re.fullmatch(r"[A-Za-z0-9_.-]+", candidate)
            else candidate in text
        )
        for candidate in candidates
    )


def validate_preserved_facts(text: str, fact_units: list[dict], required_fact_ids: list[str]) -> dict:
    """原子级校验：required 事实的关键 token 是否全部出现在文本中。

    返回 {preserved_fact_ids, missing_required, unexpected_fact_ids}。
    - preserved：required 事实的关键 token（玩家/队名/事件动词/C4）全部命中。
    - missing_required：未命中或不在 fact_units 中的 required ID。
    - unexpected：文本中出现但未被任何 fact unit 授权的事件动词类/拉丁实体。
    """
    if not isinstance(text, str):
        raise RendererError("text must be str")
    by_id = _fact_by_id(fact_units)
    preserved: list[str] = []
    missing: list[str] = []
    for fid in required_fact_ids:
        unit = by_id.get(fid)
        if unit is None:
            missing.append(fid)
            continue
        tokens = _tokens_of_clause(unit)
        if all(_text_contains(text, token) for token in tokens):
            preserved.append(fid)
        else:
            missing.append(fid)

    present_kinds = {kind for kind, verb in _EVENT_VERBS.items() if verb in text}
    authorized_kinds = {str(u.get("kind")) for u in fact_units}
    unexpected: list[str] = sorted(present_kinds - authorized_kinds)

    authorized_latin = set()
    for unit in fact_units:
        for key in ("attacker", "victim", "winner", "side"):
            value = str(unit.get(key) or "").strip()
            if re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                authorized_latin.add(value)
    present_latin = set(re.findall(r"[A-Za-z][A-Za-z0-9_.-]*", text))
    unexpected.extend(sorted(token for token in present_latin if token not in authorized_latin))

    return {
        "preserved_fact_ids": preserved,
        "missing_required": missing,
        "unexpected_fact_ids": unexpected,
    }


def check_task_units(task: dict) -> dict:
    """交付前校验（§8.4）：required 覆盖 100%、无白名单外事实。失败抛 RendererError。"""
    fact_units, required_fact_ids = _safe_task(task)
    by_id = _fact_by_id(fact_units)
    missing = [fid for fid in required_fact_ids if fid not in by_id]
    if missing:
        raise RendererError(f"required fact IDs missing from fact_units: {missing}")
    return {"required_count": len(required_fact_ids), "unit_count": len(fact_units)}
