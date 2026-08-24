"""Cloud prompt and response contract for rule-layer window projections."""
from __future__ import annotations

import json


_SYSTEM = """You write one neutral CS2 commentary sentence from a complete round of
ordered rule-layer window projections. Each window's order is chronological
rule order and its scene is a verified rule classification. Compare the window
cards, then choose at most one supplied window id. Use only that selected
window's main_topic, selected_actions, optional rule_state, and optional
verified tactic_hint. Do not infer routes, raw positions, future intent,
causes, or success/failure. If no window is suitable, return silence. Return
only JSON with exactly window_id and neutral."""


def cloud_system_prompt() -> str:
    return _SYSTEM


def cloud_response_format() -> dict:
    return {"type": "json_object"}


def build_cloud_round_prompt(payload: dict) -> str:
    return "Rule-layer projections only:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\nReturn exactly {\"window_id\": string|null, \"neutral\": string}."


def parse_cloud_response(text: str) -> dict:
    try:
        data = json.loads(text.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("cloud analyst response is not exact JSON") from exc
    # Qwen3 可能输出 JSON 级 think/reasoning 字段；严格校验前剥离。
    for _key in ("think", "reasoning", "reasoning_content"):
        data.pop(_key, None)
    if not isinstance(data, dict) or set(data) != {"window_id", "neutral"}:
        raise ValueError("cloud analyst response has an invalid field set")
    if data["window_id"] is not None and not isinstance(data["window_id"], str):
        raise ValueError("cloud analyst response has invalid window_id")
    if not isinstance(data["neutral"], str):
        raise ValueError("cloud analyst response has invalid neutral")
    return {"window_id": data["window_id"], "neutral": data["neutral"]}