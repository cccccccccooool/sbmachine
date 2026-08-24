"""以规划帧为输入的确定性、无未来泄露战术匹配。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sbmachine.tactic_book import CompiledTactic, CompiledTacticBook


_SIDES = {"T", "CT"}
_CONDITION_ORDER = {"alive_count": 0, "zone_count": 1, "bomb_planted": 1, "event_count": 2}


@dataclass(frozen=True)
class TacticMatch:
    rule_id: str
    label: str
    hint: str
    matched_at: float
    evidence: dict[str, Any]

    def to_prompt_payload(self) -> dict[str, str | float]:
        """只投影可向 LLM 公开的短提示，绝不携带规则层证据。"""
        return {
            "rule_id": self.rule_id,
            "label": self.label,
            "hint": self.hint,
            "matched_at": self.matched_at,
        }


def _mapping(value: object) -> dict | None:
    return value if isinstance(value, dict) else None


def _time(frame: object) -> float | None:
    frame_data = _mapping(frame)
    when = _mapping(frame_data.get("when")) if frame_data is not None else None
    if when is None:
        return None
    try:
        return float(when.get("video_time"))
    except (TypeError, ValueError):
        return None


def _players(frame: object) -> list[dict] | None:
    frame_data = _mapping(frame)
    where = _mapping(frame_data.get("where")) if frame_data is not None else None
    players = where.get("players") if where is not None else None
    return players if isinstance(players, list) else None


def _events(frame: object) -> dict | None:
    frame_data = _mapping(frame)
    return _mapping(frame_data.get("events")) if frame_data is not None else None


def _is_alive(player: dict) -> bool:
    hp = player.get("hp")
    return isinstance(hp, (int, float)) and not isinstance(hp, bool) and hp > 0


def _in_bbox(point: Iterable[object], bbox: dict) -> bool:
    values = list(point)
    if len(values) != 3:
        return False
    try:
        return all(
            float(bbox[axis][0]) <= float(value) <= float(bbox[axis][1])
            for axis, value in zip(("x", "y", "z"), values)
        )
    except (TypeError, ValueError, KeyError, IndexError):
        return False


def _has_zone_fields(player: dict, zone: dict) -> bool:
    if "callouts_any" in zone:
        callout = player.get("callout")
        return isinstance(callout, str) and bool(callout.strip())
    return all(axis in player for axis in ("x", "y", "z")) and _in_bbox(
        (player.get("x"), player.get("y"), player.get("z")), zone["bbox"]
    )


def _in_zone(player: dict, zone: dict) -> bool:
    if "callouts_any" in zone:
        return player.get("callout") in zone["callouts_any"]
    return _in_bbox((player.get("x"), player.get("y"), player.get("z")), zone["bbox"])


def _count_matches(value: int, bounds: list[object]) -> bool:
    lower, upper = bounds
    return value >= lower and (upper is None or value <= upper)


def _same_side_players(frame: object, side: str) -> list[dict] | None:
    players = _players(frame)
    if players is None:
        return None
    selected = []
    for player in players:
        if not isinstance(player, dict):
            return None
        player_side = player.get("side")
        if player_side not in _SIDES:
            return None
        if player_side == side:
            selected.append(player)
    return selected


def _utility_event_key(utility: dict) -> tuple | None:
    stable_id = utility.get("stable_event_id") or utility.get("dedup_key")
    if isinstance(stable_id, str) and stable_id:
        return ("utility_throw", "stable", utility.get("round_no"), stable_id)
    entity_id = utility.get("entity_id")
    if entity_id is not None:
        return ("utility_throw", "entity", entity_id)
    throw_tick = utility.get("throw_tick")
    thrower = utility.get("thrower")
    utility_type = utility.get("type")
    if (
        throw_tick is None
        or not isinstance(thrower, str)
        or not thrower
        or not isinstance(utility_type, str)
        or not utility_type
    ):
        return None
    return ("utility_throw", "throw", throw_tick, thrower, utility_type)


def _participant_event_key(event: str, row: dict) -> tuple | None:
    actor = row.get("attacker")
    victim = row.get("victim")
    if not isinstance(actor, str) or not actor or not isinstance(victim, str) or not victim:
        return None
    tick = row.get("tick")
    if tick is not None:
        return (event, tick, actor, victim)
    if event != "flash":
        return None
    duration = row.get("duration_s", row.get("duration"))
    try:
        duration_key = round(float(duration), 3)
    except (TypeError, ValueError):
        return None
    return ("flash", actor, victim, duration_key, bool(row.get("is_teammate")))


def _event_rows(frames: object, candidate_time: float, window_sec: float) -> list[dict]:
    if not isinstance(frames, list):
        return []
    rows: list[dict] = []
    seen: set[tuple] = set()
    for frame in sorted(frames, key=lambda item: (_time(item) is None, _time(item) or 0.0)):
        time = _time(frame)
        if time is None or time < candidate_time - window_sec or time > candidate_time:
            continue
        players = _players(frame)
        events = _events(frame)
        if players is None or events is None:
            continue
        lookup = {
            str(player.get("name")): player
            for player in players
            if isinstance(player, dict) and player.get("name")
        }
        utilities = events.get("utilities")
        if isinstance(utilities, list):
            for utility in utilities:
                if not isinstance(utility, dict) or utility.get("_event") != "throw":
                    continue
                key = _utility_event_key(utility)
                if key is None or key in seen:
                    continue
                seen.add(key)
                actor = utility.get("thrower")
                rows.append({
                    "event": "utility_throw",
                    "time": time,
                    "actor": actor,
                    "type": utility.get("type"),
                    "player": lookup.get(str(actor)),
                    "destination": (
                        utility.get("dest_x"), utility.get("dest_y"), utility.get("dest_z")
                    ),
                })
        kills = events.get("kills")
        if isinstance(kills, list):
            for kill in kills:
                if not isinstance(kill, dict) or kill.get("is_corpse_shoot"):
                    continue
                key = _participant_event_key("kill", kill)
                if key is None or key in seen:
                    continue
                seen.add(key)
                actor = kill.get("attacker")
                rows.append({
                    "event": "kill",
                    "time": time,
                    "actor": actor,
                    "type": kill.get("weapon"),
                    "player": lookup.get(str(actor)),
                    "destination": None,
                })
        flashes = events.get("flashes")
        if isinstance(flashes, list):
            for flash in flashes:
                if not isinstance(flash, dict):
                    continue
                key = _participant_event_key("flash", flash)
                if key is None or key in seen:
                    continue
                seen.add(key)
                actor = flash.get("attacker")
                rows.append({
                    "event": "flash",
                    "time": time,
                    "actor": actor,
                    "type": None,
                    "player": lookup.get(str(actor)),
                    "destination": None,
                })
        c4 = _mapping(events.get("c4"))
        if c4 is None:
            continue
        plant_tick = c4.get("plant_tick")
        if c4.get("planted") and plant_tick is not None:
            key = ("bomb_planted", plant_tick)
            if key not in seen:
                seen.add(key)
                rows.append({"event": "bomb_planted", "time": time, "actor": None, "type": None, "player": None, "destination": None})
        defuse_tick = c4.get("begin_defuse_tick")
        if defuse_tick is not None:
            key = ("defuse_started", defuse_tick)
            if key not in seen:
                seen.add(key)
                rows.append({"event": "defuse_started", "time": time, "actor": None, "type": None, "player": None, "destination": None})
    return rows


def _evaluate_condition(condition: dict, frame: object, candidate_time: float, context_frames: object) -> dict | None:
    kind = condition["kind"]
    if kind == "alive_count":
        players = _same_side_players(frame, condition["side"])
        if players is None or any("hp" not in player for player in players):
            return None
        count = sum(_is_alive(player) for player in players)
        return {"kind": kind, "count": count} if _count_matches(count, condition["count"]) else None
    if kind == "zone_count":
        players = _same_side_players(frame, condition["side"])
        if players is None:
            return None
        relevant = [player for player in players if _is_alive(player)]
        if any(not _has_zone_fields(player, condition["zone"]) for player in relevant):
            return None
        count = sum(_in_zone(player, condition["zone"]) for player in relevant)
        return {"kind": kind, "count": count} if _count_matches(count, condition["count"]) else None
    if kind == "bomb_planted":
        events = _events(frame)
        c4 = _mapping(events.get("c4")) if events is not None else None
        if c4 is None:
            return None
        planted = bool(c4.get("planted"))
        return {"kind": kind, "value": planted} if planted == condition["value"] else None

    rows = _event_rows(context_frames, candidate_time, float(condition.get("window_sec", candidate_time)))
    matched = []
    for row in rows:
        if row["event"] != condition["event"]:
            continue
        player = row["player"]
        if "actor_side" in condition:
            if not isinstance(player, dict) or player.get("side") != condition["actor_side"]:
                continue
        if "actor_zone" in condition:
            if not isinstance(player, dict) or not _has_zone_fields(player, condition["actor_zone"]):
                continue
            if not _in_zone(player, condition["actor_zone"]):
                continue
        if "types_any" in condition:
            event_type = row["type"]
            if not isinstance(event_type, str) or event_type not in condition["types_any"]:
                continue
        if "destination_bbox" in condition and not _in_bbox(row["destination"] or (), condition["destination_bbox"]):
            continue
        matched.append(row)
    count = len(matched)
    return {"kind": kind, "count": count, "window_sec": condition.get("window_sec")} if _count_matches(count, condition["count"]) else None


def _matches_tactic(
    tactic: CompiledTactic,
    frame: object,
    candidate_time: float,
    context_frames: object,
    scene: str | None,
) -> dict | None:
    if tactic.scene is not None and scene not in tactic.scene:
        return None
    frame_data = _mapping(frame)
    when = _mapping(frame_data.get("when")) if frame_data is not None else None
    if when is None:
        return None
    if tactic.time_window_sec is not None:
        try:
            relative = float(when.get("relative_sec"))
        except (TypeError, ValueError):
            return None
        if not tactic.time_window_sec[0] <= relative <= tactic.time_window_sec[1]:
            return None
    side_players = _same_side_players(frame, tactic.side)
    if not side_players:
        return None
    evidence = []
    for condition in sorted(tactic.when, key=lambda item: _CONDITION_ORDER[item["kind"]]):
        result = _evaluate_condition(condition, frame, candidate_time, context_frames)
        if result is None:
            return None
        evidence.append(result)
    return {"conditions": evidence}


def match_window(
    book: CompiledTacticBook,
    ownership_frames: object,
    *,
    context_frames: object | None = None,
    scene: str | None = None,
    active_rule_ids: set[str] | None = None,
) -> TacticMatch | None:
    """在一个现有窗口内选择唯一最高优先级规则，绝不读取候选时刻之后的事件。"""
    if active_rule_ids is not None and not isinstance(active_rule_ids, set):
        return None
    if not isinstance(ownership_frames, list):
        if active_rule_ids is not None:
            active_rule_ids.clear()
        return None
    context = ownership_frames if context_frames is None else context_frames
    if not isinstance(context, list):
        if active_rule_ids is not None:
            active_rule_ids.clear()
        return None
    if not book.tactics:
        if active_rule_ids is not None:
            active_rule_ids.clear()
        return None

    candidates: list[tuple[CompiledTactic, float, dict]] = []
    for tactic in book.tactics:
        for frame in sorted(ownership_frames, key=lambda item: (_time(item) is None, _time(item) or 0.0)):
            candidate_time = _time(frame)
            if candidate_time is None:
                continue
            evidence = _matches_tactic(tactic, frame, candidate_time, context, scene)
            if evidence is not None:
                candidates.append((tactic, candidate_time, evidence))
                break

    matching_rule_ids = {tactic.rule_id for tactic, _, _ in candidates}
    if active_rule_ids is not None:
        active_rule_ids.intersection_update(matching_rule_ids)
    if not candidates:
        return None
    best_priority = max(tactic.priority for tactic, _, _ in candidates)
    winners = [candidate for candidate in candidates if candidate[0].priority == best_priority]
    if len(winners) != 1:
        return None
    tactic, matched_at, evidence = winners[0]
    if active_rule_ids is not None:
        if tactic.rule_id in active_rule_ids:
            return None
        active_rule_ids.add(tactic.rule_id)
    return TacticMatch(tactic.rule_id, tactic.label, tactic.hint, matched_at, evidence)
