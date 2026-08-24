"""Cloud payload built exclusively from strict LLM window projections."""
from __future__ import annotations

from sbmachine.llm_projection import build_llm_window_projection


def _public_window(raw: object, index: int) -> tuple[dict, dict] | None:
    if not isinstance(raw, dict):
        return None
    start, end, scene = raw.get("t_start"), raw.get("t_end"), raw.get("scene")
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        return None
    if not isinstance(end, (int, float)) or isinstance(end, bool) or start >= end:
        return None
    if not isinstance(scene, str) or not scene:
        return None

    plan = raw.get("commentary_plan")
    if plan is None:
        plan = raw
    projection = build_llm_window_projection(plan, rule_state=raw.get("rule_state"))
    if (projection.get("main_topic") or {}).get("kind") == "silence":
        return None
    if not projection.get("required_facts"):
        raise ValueError("semantic_contract_error: non-silence projection has no required facts")
    character_limit = raw.get("character_limit", 100)
    if not isinstance(character_limit, int) or isinstance(character_limit, bool) or character_limit < 1:
        raise ValueError("cloud window character_limit must be a positive integer")
    if projection.get("required_chars", 0) > character_limit:
        raise ValueError("projection_budget_error")
    public = {
        "id": f"window-{index}",
        "order": index,
        "scene": scene,
        **projection,
        "character_limit": character_limit,
    }
    # Timestamps remain in-process for manifest construction and response mapping.
    # The cloud receives only a rule-derived order and scene label.
    internal = {**public, "t_start": float(start), "t_end": float(end)}
    return public, internal


def build_cloud_round_payload(*, round_no: int, map_name: str, windows: list[dict]) -> tuple[dict, dict[str, dict]]:
    """Return ordered rule context with no roster, raw frames or timestamps."""
    if not isinstance(round_no, int) or isinstance(round_no, bool) or round_no < 1:
        raise ValueError("round_no must be a positive integer")
    if not isinstance(map_name, str) or not map_name:
        raise ValueError("map_name must be a non-empty string")
    items: list[dict] = []
    by_id: dict[str, dict] = {}
    for index, raw in enumerate(windows, start=1):
        result = _public_window(raw, index)
        if result is None:
            continue
        public, internal = result
        items.append(public)
        by_id[public["id"]] = internal
    payload = {
        "contract_version": 5,
        "round_context": {
            "round_no": round_no,
            "map_name": map_name,
            "window_count": len(items),
            "selection_policy": "at_most_one",
        },
        "windows": items,
        "instructions": {"allow_silence": True, "max_neutral_chars": 100},
    }
    return payload, by_id
