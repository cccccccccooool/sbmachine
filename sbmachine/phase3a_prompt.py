"""Prompt and strict response helpers for one Phase 3a rule window."""
from __future__ import annotations

import json
import re

from core.prompt_loader import load_prompt
from sbmachine.llm_projection import build_llm_window_projection
from sbmachine.phase3a_payload import _dumps_compact

_ANALYST_JSON_CONTRACT = """Return exactly {"neutral":"..."}.
Do not emit reasoning, Markdown, code fences, prefixes, suffixes, or extra fields."""


def _build_analyst_system() -> str:
    return load_prompt("analyst_system") + "\n\n" + _ANALYST_JSON_CONTRACT


def _safe_window_payload(window_payload: object) -> dict:
    if not isinstance(window_payload, dict):
        return build_llm_window_projection({})
    plan = window_payload.get("commentary_plan")
    if plan is None:
        plan = window_payload
    return build_llm_window_projection(
        plan,
        rule_state=window_payload.get("rule_state"),
        player_state=window_payload.get("player_state"),
    )


def _build_window_prompt(
    window_payload: dict,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
    scene: str | None = None,
    state_block: str = "",
) -> str:
    """Build a prompt that sends exactly one authoritative projection with a character limit.

    No second event_summary, no tactic_block outside the JSON — the model reads
    the projection directly.
    """
    del t_start, t_end, scene, state_block
    # A4/A5: 投影中包含 character_limit，Prompt 只传一份权威 JSON。
    character_limit = window_payload.get("character_limit", 100)
    # 从 window_payload 构建干净投影（不含 character_limit，它在模板中单独替换）
    public = _safe_window_payload(window_payload)
    template = load_prompt("analyst_round")
    return (template
            .replace("{character_limit}", str(character_limit))
            .replace("{json_payload}", _dumps_compact(public)))


def _first_json_obj(text: str, debug: bool = False) -> dict | None:
    """Accept exactly one JSON object; fencing or surrounding prose is invalid."""
    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _parse_window_neutral_response(text: str, debug: bool = False) -> str | None:
    data = _first_json_obj(text, debug=debug)
    if data is None:
        return None
    # 模型可能输出 JSON 级 think/reasoning 字段，在严格校验前剥离已知多余字段，
    # 避免因多字段被契约拒绝。
    for _key in ("think", "reasoning", "reasoning_content"):
        data.pop(_key, None)
    if set(data) != {"neutral"}:
        return None
    value = data.get("neutral")
    if not isinstance(value, str):
        return None
    return value.strip()


_FACT_TERMS = (
    "击杀", "双杀", "三杀", "交换", "存活", "人数", "血量", "总血量", "安装", "安放",
    "拆除", "爆炸", "淘汰", "清零", "致盲", "燃烧", "烟雾", "补枪",
    "回防", "断火", "绕后", "推进", "稳胜",
)
_LOCATION_TERMS = (
    "A区", "B区", "A点", "B点", "中路", "长廊", "包点", "门口", "楼梯",
    "下水道", "警家", "匪家", "连接", "拱门", "二楼", "地下",
)
_WEAPON_TERMS = ("步枪", "手枪", "狙击枪", "冲锋枪", "霰弹枪", "刀", "雷")
_RESERVED_LATIN = {"T", "CT", "C4", "HP"}


def _rule_state_fact_terms(projection: dict) -> set[str]:
    """把可口播的 rule_state 字段映射为中文语义词，不把英文键本身当中文证据。"""
    state = projection.get("rule_state")
    teams = state.get("teams") if isinstance(state, dict) else None
    if not isinstance(teams, dict):
        return set()
    terms: set[str] = set()
    team_rows = [row for row in teams.values() if isinstance(row, dict)]
    if any("alive_count" in row for row in team_rows):
        terms.update(("存活", "人数"))
    return terms


def validate_neutral_semantics(neutral: str, projection: object) -> tuple[str | None, str | None]:
    """Deterministically validate a parsed neutral against projection v3.

    projection_version 3 起新增 utility_state/state_delta 白名单 action；语义校验
    授权词逻辑与 v2 保持不变（只校验 vs. 旧版本仍可读）。
    """
    if not isinstance(projection, dict) or int(projection.get("projection_version") or 0) < 2:
        return "semantic_contract_error", "projection_version>=2 is required"
    topic = projection.get("main_topic")
    topic_kind = topic.get("kind") if isinstance(topic, dict) else None
    facts = projection.get("required_facts")
    if not isinstance(facts, list):
        return "semantic_contract_error", "required_facts must be a list"
    if topic_kind != "silence" and not neutral:
        return "semantic_contract_error", "neutral is empty for non-silence topic"
    if topic_kind != "silence" and not facts:
        return "semantic_contract_error", "non-silence projection has no required facts"

    canonical_texts: list[str] = []
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("required") is not True:
            return "semantic_contract_error", "required fact is malformed"
        canonical = fact.get("canonical_text")
        if not isinstance(canonical, str) or not canonical:
            return "semantic_contract_error", "required canonical_text is invalid"
        canonical_texts.append(canonical)
        canonical_pattern = re.escape(canonical)
        if canonical.startswith("T方"):
            canonical_pattern = r"(?<!C)" + canonical_pattern
        if re.search(canonical_pattern, neutral) is None:
            swapped = canonical.replace("CT方", "__SIDE__").replace("T方", "CT方").replace("__SIDE__", "T方")
            if swapped != canonical and swapped in neutral:
                return "side_mismatch", f"required fact {fact.get('fact_id')} swaps T/CT"
            return "required_fact_missing", f"required fact {fact.get('fact_id')} is missing"

    semantic_view = {
        key: projection.get(key)
        for key in ("main_topic", "selected_actions", "required_facts", "rule_state", "tactic_hint", "player_state")
        if key in projection
    }
    public_text = json.dumps(semantic_view, ensure_ascii=False, sort_keys=True)
    allowed_numbers = set(re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?", public_text))
    output_numbers = set(re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?", neutral))
    extra_numbers = sorted(output_numbers - allowed_numbers)
    if extra_numbers:
        return "unexpected_fact", f"unexpected numbers: {extra_numbers}"

    allowed_latin = set(re.findall(r"[A-Za-z][A-Za-z0-9_.-]*", public_text)) | _RESERVED_LATIN
    output_latin = set(re.findall(r"[A-Za-z][A-Za-z0-9_.-]*", neutral))
    extra_latin = sorted(token for token in output_latin if token not in allowed_latin)
    if extra_latin:
        return "unexpected_fact", f"unexpected entities: {extra_latin}"
    allowed_state_terms = _rule_state_fact_terms(projection)
    for term in _LOCATION_TERMS + _WEAPON_TERMS + _FACT_TERMS:
        if term in neutral and term not in public_text and term not in allowed_state_terms:
            return "unexpected_fact", f"unexpected fact term: {term}"
    return None, None


def compute_preserved_fact_ids(text: str, fact_units: list[dict], required_fact_ids: list[str]) -> dict:
    """原子级 preserved 计算（§8.2/§8.4）：委托 rule_neutral_renderer 校验器。

    返回 {preserved_fact_ids, missing_required, unexpected_fact_ids}。
    模型不得自报 preserved IDs；本函数是唯一权威计算入口。
    """
    from sbmachine.rule_neutral_renderer import validate_preserved_facts

    return validate_preserved_facts(text, fact_units, required_fact_ids)
