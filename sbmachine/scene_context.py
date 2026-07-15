"""Deterministic phase windows and multi-action extraction for Phase 3a."""
from __future__ import annotations

from dataclasses import dataclass

from sbmachine.common import load_cs_game_rules


@dataclass(frozen=True)
class SceneWindow:
    t_start: float
    t_end: float
    scene: str
    context_start: float
    context_end: float


def frame_time(frame: dict) -> float:
    return float((frame.get("when") or {}).get("video_time", 0.0))


def _is_planted(frame: dict) -> bool:
    when = frame.get("when") or {}
    c4 = (frame.get("events") or {}).get("c4") or {}
    if "planted" in c4:
        return bool(c4.get("planted"))
    tick = when.get("tick")
    plant_tick = c4.get("plant_tick")
    try:
        return tick is not None and plant_tick is not None and int(tick) >= int(plant_tick)
    except (TypeError, ValueError):
        return False


def classify_frame_scene(frame: dict, prepare_sec: float = 5.0) -> str:
    when = frame.get("when") or {}
    phase = str(when.get("phase") or "")
    relative = when.get("relative_sec")
    if phase == "post_round":
        return "收尾"
    try:
        if phase == "pre_round" or (relative is not None and float(relative) < prepare_sec):
            return "准备"
    except (TypeError, ValueError):
        if phase == "pre_round":
            return "准备"
    if _is_planted(frame):
        return "炸弹"
    if phase == "in_round" or relative is not None:
        return "未下包"
    return "默认场景"


def _split_span(lo: float, hi: float, max_sec: float, min_sec: float) -> list[tuple[float, float]]:
    if hi <= lo:
        return []
    boundaries = [lo]
    while hi - boundaries[-1] > max_sec:
        boundaries.append(round(boundaries[-1] + max_sec, 3))
    boundaries.append(hi)
    if len(boundaries) >= 3 and boundaries[-1] - boundaries[-2] < min_sec:
        boundaries.pop(-2)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _kill_times(frames: list[dict]) -> list[float]:
    values: list[float] = []
    for frame in frames:
        if any(not k.get("is_corpse_shoot") for k in ((frame.get("events") or {}).get("kills") or [])):
            values.append(frame_time(frame))
    return sorted(set(values))


def build_scene_contexts(
    frames: list[dict],
    start_sec: float,
    end_sec: float,
    *,
    window_max_sec: float | None = None,
    window_min_sec: float | None = None,
) -> list[SceneWindow]:
    rules = load_cs_game_rules()
    cfg = rules.get("scene", {})
    prepare_sec = float(cfg.get("prepare_sec", 5.0))
    max_sec = float(window_max_sec or cfg.get("window_max_sec", 10.0))
    min_sec = float(window_min_sec or cfg.get("window_min_sec", 3.0))
    context_sec = float(cfg.get("cross_stage_context_sec", 3.0))
    ordered = sorted(frames, key=frame_time)
    if not ordered:
        return [SceneWindow(start_sec, end_sec, "默认场景", start_sec, end_sec)]

    spans: list[tuple[float, float, str]] = []
    current_scene = classify_frame_scene(ordered[0], prepare_sec)
    current_start = start_sec
    for frame in ordered[1:]:
        scene = classify_frame_scene(frame, prepare_sec)
        t = max(start_sec, min(end_sec, frame_time(frame)))
        if scene != current_scene and t > current_start:
            spans.append((current_start, t, current_scene))
            current_start = t
            current_scene = scene
    if end_sec > current_start:
        spans.append((current_start, end_sec, current_scene))
    if not spans:
        spans = [(start_sec, end_sec, current_scene)]

    base: list[SceneWindow] = []
    for lo, hi, scene in spans:
        for sub_lo, sub_hi in _split_span(lo, hi, max_sec, min_sec):
            base.append(SceneWindow(sub_lo, sub_hi, scene, sub_lo, sub_hi))

    kills = _kill_times(ordered)
    result: list[SceneWindow] = []
    for idx, window in enumerate(base):
        context_start = window.t_start
        if idx > 0 and base[idx - 1].scene != window.scene:
            before = [t for t in kills if window.t_start - context_sec <= t < window.t_start]
            after = [t for t in kills if window.t_start <= t < window.t_start + context_sec]
            if before and after:
                context_start = max(start_sec, min(before))
        result.append(SceneWindow(
            window.t_start,
            window.t_end,
            window.scene,
            context_start,
            window.t_end,
        ))
    return result


def _player_lookup(frame: dict) -> dict[str, dict]:
    return {
        str(player.get("name", "")): player
        for player in ((frame.get("where") or {}).get("players") or [])
        if player.get("name")
    }


def _position(player: dict | None) -> list[float] | None:
    if not player:
        return None
    try:
        return [float(player["x"]), float(player["y"]), float(player.get("z", 0.0))]
    except (KeyError, TypeError, ValueError):
        return None


def _utility_kind(value: object) -> str:
    return str(value or "").replace("Grenade", "").replace("Incendiary", "Molotov").strip() or "道具"


def extract_actions(frames: list[dict], lo: float, hi: float) -> list[dict]:
    rules = load_cs_game_rules()
    action_cfg = rules.get("actions", {})
    utility_types = {str(v).casefold() for v in action_cfg.get("utility_throw_types", [])}
    flash_min = float(action_cfg.get("effective_flash_min_sec", 2.0))
    actions: list[dict] = []
    seen: set[tuple] = set()
    for frame in sorted(frames, key=frame_time):
        t = frame_time(frame)
        if not (lo <= t < hi):
            continue
        events = frame.get("events") or {}
        players = _player_lookup(frame)
        for kill in events.get("kills") or []:
            if kill.get("is_corpse_shoot"):
                continue
            key = ("kill", kill.get("tick"), kill.get("attacker"), kill.get("victim"))
            if key in seen:
                continue
            seen.add(key)
            attacker = players.get(str(kill.get("attacker", "")))
            victim = players.get(str(kill.get("victim", "")))
            actions.append({
                "type": "kill",
                "event_time": t,
                "event_tick": kill.get("tick"),
                "attacker": kill.get("attacker"),
                "victim": kill.get("victim"),
                "weapon": kill.get("weapon"),
                "victim_weapon": victim.get("weapon") if victim else None,
                "attacker_pos": _position(attacker),
                "victim_pos": _position(victim),
                "headshot": bool(kill.get("headshot")),
                "through_smoke": bool(kill.get("through_smoke")),
                "no_scope": bool(kill.get("no_scope")),
                "is_wallbang": bool(kill.get("is_wallbang")),
                "attacker_blind": bool(kill.get("attacker_blind")),
            })
        for utility in events.get("utilities") or []:
            if str(utility.get("_event", "")) != "throw":
                continue
            raw_type = str(utility.get("type", ""))
            if raw_type.casefold() not in utility_types:
                continue
            key = ("utility_throw", utility.get("entity_id"), utility.get("throw_tick"))
            if key in seen:
                continue
            seen.add(key)
            actions.append({
                "type": "utility_throw",
                "event_time": t,
                "event_tick": utility.get("throw_tick"),
                "entity_id": utility.get("entity_id"),
                "utility": raw_type,
                "thrower": utility.get("thrower"),
                "throw_position": [
                    utility.get("throw_pos_x"), utility.get("throw_pos_y"), utility.get("throw_pos_z")
                ],
                "destination": [utility.get("dest_x"), utility.get("dest_y"), utility.get("dest_z")],
            })
        # Active smoke/fire is a durable fallback for parser outputs that did
        # not retain a throw event. Record its first observed frame only.
        for field, default_name in (("smokes_active", "烟雾"), ("infernos_active", "燃烧瓶")):
            for utility in events.get(field) or []:
                identity = utility.get("entity_id", utility.get("start_tick", utility.get("throw_tick")))
                key = ("utility_effect", field, identity, utility.get("thrower"))
                if key in seen:
                    continue
                seen.add(key)
                actions.append({
                    "type": "utility_effect",
                    "event_time": t,
                    "event_tick": utility.get("start_tick", utility.get("throw_tick")),
                    "entity_id": utility.get("entity_id"),
                    "utility": _utility_kind(utility.get("type", utility.get("grenade_type", default_name))),
                    "thrower": utility.get("thrower"),
                    "destination": [utility.get("x", utility.get("dest_x")), utility.get("y", utility.get("dest_y")), utility.get("z", utility.get("dest_z"))],
                    "source": field,
                })
        c4 = events.get("c4") or {}
        if c4.get("planted"):
            key = ("bomb_planted", c4.get("plant_tick"))
            if key not in seen:
                seen.add(key)
                actions.append({
                    "type": "bomb_planted", "event_time": t,
                    "event_tick": c4.get("plant_tick"),
                })
        if c4.get("begin_defuse_tick"):
            key = ("defuse_started", c4.get("begin_defuse_tick"))
            if key not in seen:
                seen.add(key)
                actions.append({
                    "type": "defuse_started", "event_time": t,
                    "event_tick": c4.get("begin_defuse_tick"),
                    "defuser_has_kit": c4.get("defuser_has_kit"),
                })
        for flash in events.get("flashes") or []:
            try:
                duration = float(flash.get("duration_s", flash.get("duration", 0.0)) or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            if duration < flash_min:
                continue
            key = ("effective_flash", flash.get("tick"), flash.get("attacker"), flash.get("victim"))
            if key in seen:
                continue
            seen.add(key)
            actions.append({
                "type": "effective_flash",
                "event_time": t,
                "event_tick": flash.get("tick"),
                "attacker": flash.get("attacker"),
                "victim": flash.get("victim"),
                "duration_s": round(duration, 2),
                "is_teammate": bool(flash.get("is_teammate")),
            })
    return sorted(actions, key=lambda item: (float(item.get("event_time", 0)), str(item.get("type", ""))))
