"""单个 Phase 3a 时间窗的 prompt 与严格响应解析辅助函数。"""
from __future__ import annotations

import json

from core.prompt_loader import load_prompt
from sbmachine.phase3a_payload import _dumps_compact

_ANALYST_JSON_CONTRACT = """Return exactly one JSON object: {\"neutral\": string}.
Describe facts from this one window only. Keep neutral under 100 Chinese characters.
Do not return scene lists, timestamps, keyframes, markdown, or extra text.
When evidence is insufficient, return an empty neutral string."""


def _build_analyst_system() -> str:
    return load_prompt("analyst_system") + "\n\n" + _ANALYST_JSON_CONTRACT


def _build_window_prompt(window_payload: dict, *, t_start: float, t_end: float, scene: str, state_block: str = "") -> str:
    template = load_prompt("analyst_round")
    plan = window_payload.get("commentary_plan") or {}
    main_topic = (plan.get("main_topic") or {}).get("summary") or "No reliable main topic; return an empty neutral string."
    return (template
            .replace("{scene}", scene)
            .replace("{t_start}", f"{t_start:.3f}")
            .replace("{t_end}", f"{t_end:.3f}")
            .replace("{event_summary}", str(main_topic))
            .replace("{state_block}", f"\n【场上状态】\n{state_block}\n" if state_block else "")
            .replace("{json_payload}", _dumps_compact(window_payload)))


def _first_json_obj(text: str, debug: bool = False) -> dict | None:
    """只接受一个精确的 JSON 对象；代码围栏和前后夹杂的自然语言均视为非法。"""
    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _parse_window_neutral_response(text: str, debug: bool = False) -> str | None:
    data = _first_json_obj(text, debug=debug)
    if data is None or set(data) != {"neutral"}:
        return None
    value = data.get("neutral")
    if not isinstance(value, str):
        return None
    return value.strip()
