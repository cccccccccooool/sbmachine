"""Resolve compact local player context without asking the LLM to infer a map."""
from __future__ import annotations

import json
import math

from sbmachine.common import PROJECT_ROOT, load_cs_game_rules


def _player_pos(player: dict) -> tuple[float, float, float] | None:
    try:
        return float(player["x"]), float(player["y"]), float(player.get("z", 0.0))
    except (KeyError, TypeError, ValueError):
        return None


def _distance(a: dict, b: dict) -> float:
    """Planar proximity; vertical separation is entirely a logical-level rule."""
    pa, pb = _player_pos(a), _player_pos(b)
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) if pa and pb else float("inf")


def load_map_template(map_name: str | None) -> dict:
    if not map_name:
        return {}
    safe = "".join(ch for ch in str(map_name).lower() if ch.isalnum() or ch in "_-")
    path = PROJECT_ROOT / "database" / "maps" / f"{safe}.json"
    if not path.exists():
        return {}
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    # Legacy/observed-only templates are never spatial authority. They degrade
    # to no template rather than silently receiving invented default levels.
    return template if _is_reviewed_template(template) else {}

def _alive(players: list[dict]) -> list[dict]:
    return [player for player in players if int(player.get("hp") or 0) > 0]


def _isolated(players: list[dict], side: str, ratio: float) -> dict | None:
    team = [player for player in _alive(players) if str(player.get("side", "")).upper() == side]
    if len(team) < 3:
        return None
    scored = []
    for player in team:
        nearest = min((_distance(player, other) for other in team if other is not player), default=float("inf"))
        scored.append((nearest, player))
    scored.sort(key=lambda item: item[0], reverse=True)
    baseline = sorted(item[0] for item in scored[1:])[len(scored[1:]) // 2]
    return scored[0][1] if baseline > 0 and scored[0][0] >= baseline * ratio else None


def _stable_isolated(frames: list[dict], side: str, ratio: float) -> dict | None:
    """Require the same isolated player in two latest usable samples."""
    candidates: list[dict] = []
    for frame in reversed(frames):
        players = (frame.get("where") or {}).get("players") or []
        candidate = _isolated(players, side, ratio)
        if candidate is not None:
            candidates.append(candidate)
        if len(candidates) == 2:
            break
    if len(candidates) == 2 and candidates[0].get("name") == candidates[1].get("name"):
        return candidates[0]
    return None


def _callout_node(template: dict, callout: object) -> dict:
    nodes = template.get("callouts") or {}
    node = nodes.get(str(callout or "")) if isinstance(nodes, dict) else None
    return node if isinstance(node, dict) else {}


def _callout_label(template: dict, callout: object) -> str:
    raw = str(callout or "")
    return str(_callout_node(template, raw).get("zh") or raw)


def _is_vertical_edge(edge: dict) -> bool:
    if str(edge.get("kind") or "") in {"stairs", "ladder", "ramp", "vent", "drop", "lift"}:
        return True
    return int(edge.get("level_delta") or 0) != 0


def _is_reviewed_template(template: dict) -> bool:
    """Only a completed human-reviewed layered graph can constrain locality."""
    source = template.get("source") or {}
    callouts = template.get("callouts")
    if not (
        int(template.get("version") or 0) >= 2
        and isinstance(source, dict)
        and source.get("manual_review_required") is True
        and source.get("manual_reviewed") is True
        and isinstance(callouts, dict)
        and bool(callouts)
    ):
        return False
    return all(
        isinstance(node, dict)
        and bool(str(node.get("zh") or "").strip())
        and isinstance(node.get("level"), int)
        and bool(str(node.get("layer") or "").strip())
        for node in callouts.values()
    )


def _trusted_edges(template: dict) -> list[dict]:
    """Observed tick transitions may skip rooms; only human-reviewed edges are spatial facts."""
    return [
        edge for edge in (template.get("directed_transitions") or [])
        if isinstance(edge, dict) and str(edge.get("source") or "") in {"manual", "manual+observed"}
    ]


def _callout_relation(template: dict, source: object, target: object) -> str:
    left, right = str(source or ""), str(target or "")
    if not _is_reviewed_template(template):
        return "coordinate_unverified"
    source_node, target_node = _callout_node(template, left), _callout_node(template, right)
    if not source_node or not target_node:
        return "coordinate_unverified"
    if left == right:
        return "same_callout"

    matching_edges = [
        edge for edge in _trusted_edges(template)
        if (
            str(edge.get("from")) == left
            and str(edge.get("to")) == right
        ) or (
            edge.get("one_way") is False
            and str(edge.get("from")) == right
            and str(edge.get("to")) == left
        )
    ]
    if any(_is_vertical_edge(edge) for edge in matching_edges):
        return "vertical_connected"
    if matching_edges:
        return "adjacent_callout"

    # A reviewed graph wins over coordinate coincidence. Without a declared
    # edge, a different callout is not a local tactical relationship.
    source_level, target_level = source_node.get("level"), target_node.get("level")
    if source_node and target_node and isinstance(source_level, int) and isinstance(target_level, int) and source_level != target_level:
        return "separate_level"
    if source_node and target_node:
        return "map_disconnected"
    return "coordinate_unverified"


def _map_precision(template: dict) -> str:
    return "reviewed_graph" if _is_reviewed_template(template) else "coordinate_fallback"

def _local_actions(actions: list[dict], anchor: dict | None, nearby: dict[str, list[dict]]) -> list[dict]:
    if anchor is None:
        return []
    local_names = {str(anchor.get("name") or "")}
    for group in nearby.values():
        local_names.update(str(item.get("name") or "") for item in group)
    local_names.discard("")
    selected = []
    for action in actions:
        if action.get("type") in {"bomb_planted", "defuse_started"}:
            selected.append(action)
            continue
        actors = {str(action.get(key) or "") for key in ("attacker", "victim", "thrower")}
        if actors & local_names:
            selected.append(action)
            continue
        # Coordinates alone carry no human-confirmed logical level. Do not
        # reintroduce cross-floor false locality after player filtering.
    return selected


def resolve_spatial_context(map_name: str | None, scene: str, frames: list[dict], actions: list[dict]) -> dict:
    rules = load_cs_game_rules()["spatial"]
    template = load_map_template(map_name)
    representative = next((frame for frame in reversed(frames) if (frame.get("where") or {}).get("players")), {})
    players = _alive((representative.get("where") or {}).get("players") or [])
    who = representative.get("who") or {}
    pov_name = str(who.get("pov_player") or "")
    anchor = next((player for player in players if str(player.get("name")) == pov_name), None)
    source, confidence = ("pov", 1.0) if anchor else ("none", 0.0)
    if anchor is None:
        opposite = str(rules.get("isolated_opposite_side", {}).get(scene, ""))
        anchor = _stable_isolated(frames, opposite, float(rules["isolated_ratio_threshold"])) if opposite else None
        if anchor is not None:
            source, confidence = "isolated_opposite", 0.75
    limit = int(rules["nearby_limit_per_side"])
    local_radius = float(rules.get("local_action_radius_units", 800.0))
    nearby: dict[str, list[dict]] = {"teammates": [], "enemies": []}
    if anchor is not None and _is_reviewed_template(template) and _callout_node(template, anchor.get("callout")):
        groups = {
            "teammates": (p for p in players if p is not anchor and p.get("side") == anchor.get("side")),
            "enemies": (p for p in players if p.get("side") != anchor.get("side")),
        }
        for key, group in groups.items():
            candidates = []
            for player in group:
                if not _callout_node(template, player.get("callout")):
                    continue
                distance = _distance(anchor, player)
                relation = _callout_relation(template, anchor.get("callout"), player.get("callout"))
                if relation not in {"same_callout", "adjacent_callout", "vertical_connected"} or not math.isfinite(distance) or distance > local_radius:
                    continue
                candidates.append((distance, player, relation))
            candidates.sort(key=lambda item: item[0])
            nearby[key] = [{
                "name": player.get("name"), "callout": player.get("callout"),
                "callout_zh": _callout_label(template, player.get("callout")),
                "distance_units": round(distance, 1), "weapon": player.get("weapon"), "relation": relation,
            } for distance, player, relation in candidates[:limit]]
    local_actions = _local_actions(actions, anchor, nearby)
    return {
        "map_template_available": bool(template),
        "map_precision": _map_precision(template),
        "anchor": ({"name": anchor.get("name"), "side": anchor.get("side"), "callout": anchor.get("callout"),
                    "callout_zh": _callout_label(template, anchor.get("callout")), "weapon": anchor.get("weapon")} if anchor else None),
        "anchor_source": source, "anchor_confidence": confidence, "nearby": nearby,
        "local_actions": local_actions,
    }
