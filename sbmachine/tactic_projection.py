"""Project raw planning frames into LLM-safe window facts."""
from __future__ import annotations

from dataclasses import dataclass

from sbmachine.commentary_planner import PlannerState, plan_window
from sbmachine.scene_context import SceneWindow
from sbmachine.tactic_book import CompiledTacticBook
from sbmachine.tactic_matcher import TacticMatch, match_window


@dataclass(frozen=True)
class WindowRuleProjection:
    """Keep the public plan separate from debug-only match evidence."""

    plan: dict
    tactic_hint: dict | None
    debug: dict | None


def build_window_rule_projection(
    map_name: str | None,
    window: SceneWindow,
    ownership_frames: list[dict],
    context_frames: list[dict],
    all_round_frames: list[dict],
    state: PlannerState,
    tactic_book: CompiledTacticBook,
    *,
    is_last_window: bool = False,
    char_budget: int | None = None,
) -> WindowRuleProjection:
    """Match first, then pass only the public tactic hint to the planner."""
    match: TacticMatch | None = match_window(
        tactic_book,
        ownership_frames,
        context_frames=context_frames,
        scene=window.scene,
        active_rule_ids=state.active_tactic_rule_ids,
    )
    tactic_hint = match.to_prompt_payload() if match is not None else None
    plan = plan_window(
        map_name, window, ownership_frames, context_frames, all_round_frames, state,
        tactic_hint=tactic_hint,
        is_last_window=is_last_window,
        char_budget=char_budget,
    )
    debug = None
    if match is not None:
        debug = {"rule_id": match.rule_id, "matched_at": match.matched_at, "evidence": match.evidence}
    return WindowRuleProjection(plan=plan, tactic_hint=tactic_hint, debug=debug)
