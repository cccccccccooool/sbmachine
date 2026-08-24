"""Deterministic phase windows and multi-action extraction for Phase 3a."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sbmachine.common import load_cs_game_rules, load_hype_rules
from sbmachine.hype_score import _speech_rate_config
from sbmachine.rule_compare import enrich_kill_actions
from sbmachine.utility_projection import normalize_utility_kind


@dataclass(frozen=True)
class SceneWindow:
    t_start: float
    t_end: float
    scene: str
    context_start: float
    context_end: float


def frame_time(frame: dict) -> float:
    return float((frame.get("when") or {}).get("video_time", 0.0))


def owns_time(value: float, lo: float, hi: float, *, include_end: bool = False) -> bool:
    """统一窗口所有权：普通窗口半开，回合末窗可显式闭合右端点。"""
    return lo <= value <= hi if include_end else lo <= value < hi


def _is_planted(frame: dict) -> bool:
    when = frame.get("when") or {}
    c4 = (frame.get("events") or {}).get("c4") or {}
    if "planted" in c4:
        return bool(c4.get("planted"))
    tick = when.get("tick")
    plant_tick = c4.get("plant_tick")
    try:
        return (
            tick is not None and plant_tick is not None and int(tick) >= int(plant_tick)
        )
    except (TypeError, ValueError):
        return False


def classify_frame_scene(frame: dict, prepare_sec: float = 5.0) -> str:
    when = frame.get("when") or {}
    phase = str(when.get("phase") or "")
    relative = when.get("relative_sec")
    if phase == "post_round":
        return "收尾"
    try:
        if phase == "pre_round" or (
            relative is not None and float(relative) < prepare_sec
        ):
            return "准备"
    except (TypeError, ValueError):
        if phase == "pre_round":
            return "准备"
    if _is_planted(frame):
        return "炸弹"
    if phase == "in_round" or relative is not None:
        return "未下包"
    return "默认场景"


def _split_span(
    lo: float, hi: float, max_sec: float, min_sec: float
) -> list[tuple[float, float]]:
    if hi <= lo:
        return []
    boundaries = [lo]
    while hi - boundaries[-1] > max_sec:
        boundaries.append(round(boundaries[-1] + max_sec, 3))
    boundaries.append(hi)
    if len(boundaries) >= 3 and boundaries[-1] - boundaries[-2] < min_sec:
        boundaries.pop(-2)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


_SCENE_PRIORITY: dict[str, int] = {"收尾": 3, "炸弹": 2, "未下包": 1, "准备": 0}


def _effective_min_sec(window_min_sec: float, runtime_config: dict | None = None) -> float:
    """最小可行窗口时长 = config 切窗下限 与 语音物理极限推导值 取大。

    语音物理极限：取最慢语速档位，预算地板 8 字，加 1.5× 裕量。
    语速优先取 runtime_config 的 semantic.speech_rate，其次取 hype_rules。
    """
    speech = _speech_rate_config(runtime_config)
    base = float(speech.get("base_char_per_sec", 5.0))
    factors = speech.get("char_budget_factor", {})
    worst_factor = min((float(v) for v in factors.values()), default=1.0)
    worst_rate = base * worst_factor
    budget_floor = 8
    margin = 1.5
    speech_min = math.ceil(budget_floor * margin / worst_rate) if worst_rate > 0 else 3
    return max(window_min_sec, float(speech_min))


def _merge_short_windows(base: list[SceneWindow], effective_min_sec: float) -> list[SceneWindow]:
    """倒序扫描：时长 < effective_min_sec 的窗口向前/后合并，合并后场景按优先级取大。"""
    if not base:
        return base
    result: list[SceneWindow] = list(base)
    n = len(result)
    for i in range(n - 1, -1, -1):
        w = result[i]
        if w.t_end - w.t_start >= effective_min_sec:
            continue
        if i == 0 and n >= 2:
            nxt = result[i + 1]
            scene = nxt.scene if _SCENE_PRIORITY.get(nxt.scene, 0) >= _SCENE_PRIORITY.get(w.scene, 0) else w.scene
            result[i + 1] = SceneWindow(
                w.t_start, nxt.t_end, scene,
                w.context_start, nxt.context_end,
            )
            result.pop(i)
            n -= 1
        elif i > 0:
            prv = result[i - 1]
            scene = prv.scene if _SCENE_PRIORITY.get(prv.scene, 0) >= _SCENE_PRIORITY.get(w.scene, 0) else w.scene
            result[i - 1] = SceneWindow(
                prv.t_start, w.t_end, scene,
                min(prv.context_start, w.context_start), prv.context_end,
            )
            result.pop(i)
            n -= 1
    return result


def _kill_times(frames: list[dict]) -> list[float]:
    values: list[float] = []
    for frame in frames:
        if any(
            not k.get("is_corpse_shoot")
            for k in ((frame.get("events") or {}).get("kills") or [])
        ):
            values.append(frame_time(frame))
    return sorted(set(values))


def build_scene_contexts(
    frames: list[dict],
    start_sec: float,
    end_sec: float,
    *,
    window_max_sec: float | None = None,
    window_min_sec: float | None = None,
    runtime_config: dict | None = None,
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

    effective_min = _effective_min_sec(min_sec, runtime_config)
    base = _merge_short_windows(base, effective_min)

    kills = _kill_times(ordered)
    result: list[SceneWindow] = []
    for idx, window in enumerate(base):
        context_start = window.t_start
        if idx > 0 and base[idx - 1].scene != window.scene:
            before = [
                t for t in kills if window.t_start - context_sec <= t < window.t_start
            ]
            after = [
                t for t in kills if window.t_start <= t < window.t_start + context_sec
            ]
            if before and after:
                context_start = max(start_sec, min(before))
        result.append(
            SceneWindow(
                window.t_start,
                window.t_end,
                window.scene,
                context_start,
                window.t_end,
            )
        )
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
    return normalize_utility_kind(value)


def _event_xyz(event: dict, compact_key: str, prefix: str) -> list[float | None] | None:
    compact = event.get(compact_key)
    if isinstance(compact, list) and len(compact) == 3:
        return list(compact)
    values = [event.get(f"{prefix}_{axis}") for axis in ("x", "y", "z")]
    return values if any(value is not None for value in values) else None


def _kill_snapshot_position(kill: dict, role: str, fallback: dict | None) -> list[float] | None:
    values = [kill.get(f"{role}_{axis}") for axis in ("x", "y", "z")]
    if all(value is not None for value in values):
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError):
            pass
    return _position(fallback)


def _has_projected_effect(
    projected: list[tuple[str, str, int]], *, kind: str, thrower: object, start_tick: object
) -> bool:
    try:
        target_tick = int(start_tick)
    except (TypeError, ValueError):
        return False
    actor = str(thrower or "")
    return any(
        known_kind == kind
        and known_thrower == actor
        and abs(known_tick - target_tick) <= 128
        for known_kind, known_thrower, known_tick in projected
    )


def _tick_event_time(frames: list[dict], event_tick: object) -> float | None:
    """将 DEM tick 映射到首个不早于它的采样帧；范围外返回无穷哨兵。"""
    try:
        target = int(event_tick)
    except (TypeError, ValueError):
        return None
    samples: list[tuple[int, float]] = []
    for frame in frames:
        try:
            samples.append(
                (int((frame.get("when") or {}).get("tick")), frame_time(frame))
            )
        except (TypeError, ValueError):
            continue
    samples.sort()
    if not samples:
        return None
    if target < samples[0][0]:
        return float("-inf")
    if target > samples[-1][0]:
        return float("inf")
    return next(time for tick, time in samples if tick >= target)


def extract_actions(
    frames: list[dict],
    lo: float,
    hi: float,
    *,
    include_end: bool = False,
) -> list[dict]:
    """提取瞬时动作；include_end 仅供回合最后窗口使用。"""
    rules = load_cs_game_rules()
    action_cfg = rules.get("actions", {})
    utility_types = {
        str(v).casefold() for v in action_cfg.get("utility_throw_types", [])
    }
    flash_min = float(action_cfg.get("effective_flash_min_sec", 2.0))
    actions: list[dict] = []
    seen: set[tuple] = set()
    projected_effects: list[tuple[str, str, int]] = []
    for source_frame in frames:
        for utility in ((source_frame.get("events") or {}).get("utilities") or []):
            if not isinstance(utility, dict):
                continue
            effect_tick = utility.get("effect_tick", utility.get("det_tick"))
            try:
                effect_tick_i = int(effect_tick)
            except (TypeError, ValueError):
                continue
            projected_effects.append(
                (
                    _utility_kind(utility.get("kind", utility.get("type"))),
                    str(utility.get("thrower") or ""),
                    effect_tick_i,
                )
            )
    for frame in sorted(frames, key=frame_time):
        t = frame_time(frame)
        if not owns_time(t, lo, hi, include_end=include_end):
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
            pov_player = str((frame.get("who") or {}).get("pov_player") or "")
            actions.append(
                {
                    "type": "kill",
                    "event_id": f"kill:{kill.get('tick')}:{kill.get('attacker') or ''}:{kill.get('victim') or ''}",
                    "event_time": t,
                    "event_tick": kill.get("tick"),
                    "attacker": kill.get("attacker"),
                    "victim": kill.get("victim"),
                    "attacker_side": attacker.get("side") if attacker else None,
                    "victim_side": victim.get("side") if victim else None,
                    "weapon": kill.get("weapon"),
                    "victim_weapon": victim.get("weapon") if victim else None,
                    "attacker_pos": _kill_snapshot_position(kill, "killer", attacker),
                    "victim_pos": _kill_snapshot_position(kill, "victim", victim),
                    "headshot": bool(kill.get("headshot")),
                    "through_smoke": bool(kill.get("through_smoke")),
                    "no_scope": bool(kill.get("no_scope")),
                    "is_wallbang": bool(kill.get("is_wallbang")),
                    "attacker_blind": bool(kill.get("attacker_blind")),
                    "assisted_flash": bool(kill.get("assisted_flash")),
                    "distance": kill.get("distance"),
                    "killer_x": kill.get("killer_x"),
                    "killer_y": kill.get("killer_y"),
                    "killer_z": kill.get("killer_z"),
                    "victim_x": kill.get("victim_x"),
                    "victim_y": kill.get("victim_y"),
                    "victim_z": kill.get("victim_z"),
                    "killer_yaw": kill.get("killer_yaw"),
                    "victim_yaw": kill.get("victim_yaw"),
                    "killer_airborne": kill.get("killer_airborne"),
                    "victim_airborne": kill.get("victim_airborne"),
                    "killer_scoped": kill.get("killer_scoped"),
                    "victim_scoped": kill.get("victim_scoped"),
                    "killer_spotted_victim": kill.get("killer_spotted_victim"),
                    "victim_spotted_killer": kill.get("victim_spotted_killer"),
                    "pov_player": pov_player,
                }
            )
        for utility in events.get("utilities") or []:
            if str(utility.get("_event", "")) != "throw":
                continue
            raw_type = str(utility.get("type", ""))
            if raw_type.casefold() not in utility_types:
                continue
            stable_id = utility.get("stable_event_id") or utility.get("dedup_key")
            key = (
                "utility_throw",
                stable_id or utility.get("entity_id"),
                utility.get("throw_tick"),
                utility.get("thrower"),
                raw_type,
            )
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                {
                    "type": "utility_throw",
                    "event_id": utility.get("event_id")
                    or f"utility_throw:{utility.get('throw_tick')}:{stable_id or utility.get('entity_id')}",
                    "event_time": t,
                    "event_tick": utility.get("throw_tick"),
                    "entity_id": utility.get("entity_id"),
                    "stable_event_id": stable_id,
                    "utility": raw_type,
                    "utility_kind": utility.get("kind") or _utility_kind(raw_type),
                    "thrower": utility.get("thrower"),
                    "throw_position": _event_xyz(utility, "throw_xyz", "throw_pos"),
                    "destination": _event_xyz(utility, "landing_xyz", "dest"),
                    "effect_tick": utility.get("effect_tick", utility.get("det_tick")),
                }
            )
        # Active smoke/fire is a durable fallback for parser outputs that did
        # not retain a throw event. Record its first observed frame only.
        for field, default_name, canonical_kind in (
            ("smokes_active", "烟雾", "smoke"),
            ("infernos_active", "燃烧瓶", "molotov"),
        ):
            for utility in events.get(field) or []:
                raw_effect_kind = utility.get("type", utility.get("grenade_type"))
                effect_kind = (
                    _utility_kind(raw_effect_kind)
                    if raw_effect_kind is not None
                    else canonical_kind
                )
                if _has_projected_effect(
                    projected_effects,
                    kind=effect_kind,
                    thrower=utility.get("thrower"),
                    start_tick=utility.get("start_tick", utility.get("throw_tick")),
                ):
                    continue
                identity = utility.get(
                    "entity_id", utility.get("start_tick", utility.get("throw_tick"))
                )
                key = ("utility_effect", field, identity, utility.get("thrower"))
                if key in seen:
                    continue
                seen.add(key)
                actions.append(
                    {
                        "type": "utility_effect",
                        "event_id": f"utility_effect:{utility.get('start_tick', utility.get('throw_tick'))}:{identity}",
                        "event_time": t,
                        "event_tick": utility.get(
                            "start_tick", utility.get("throw_tick")
                        ),
                        "entity_id": utility.get("entity_id"),
                        "utility": _utility_kind(raw_effect_kind or default_name),
                        "thrower": utility.get("thrower"),
                        "destination": [
                            utility.get("x", utility.get("dest_x")),
                            utility.get("y", utility.get("dest_y")),
                            utility.get("z", utility.get("dest_z")),
                        ],
                        "source": field,
                    }
                )
        c4 = events.get("c4") or {}
        if c4.get("planted"):
            transition_time = _tick_event_time(frames, c4.get("plant_tick"))
            key = ("bomb_planted", c4.get("plant_tick"))
            if (transition_time is None or t == transition_time) and key not in seen:
                seen.add(key)
                actions.append(
                    {
                        "type": "bomb_planted",
                        "event_id": f"bomb_planted:{c4.get('plant_tick')}",
                        "event_time": t,
                        "event_tick": c4.get("plant_tick"),
                    }
                )
        if c4.get("begin_defuse_tick"):
            transition_time = _tick_event_time(frames, c4.get("begin_defuse_tick"))
            key = ("defuse_started", c4.get("begin_defuse_tick"))
            if (transition_time is None or t == transition_time) and key not in seen:
                seen.add(key)
                actions.append(
                    {
                        "type": "defuse_started",
                        "event_id": f"defuse_started:{c4.get('begin_defuse_tick')}",
                        "event_time": t,
                        "event_tick": c4.get("begin_defuse_tick"),
                        "defuser_has_kit": c4.get("defuser_has_kit"),
                    }
                )
        for terminal_type, keys in (
            ("bomb_exploded", ("bomb_exploded_tick", "explode_tick", "exploded_tick")),
            ("bomb_defused", ("bomb_defused_tick", "defuse_tick", "defused_tick")),
        ):
            terminal_tick = next(
                (c4.get(key) for key in keys if c4.get(key) is not None), None
            )
            transition_time = _tick_event_time(frames, terminal_tick)
            if transition_time is not None and t != transition_time:
                continue
            key = (terminal_type, terminal_tick)
            if terminal_tick is not None and key not in seen:
                seen.add(key)
                actions.append(
                    {
                        "type": terminal_type,
                        "event_id": f"{terminal_type}:{terminal_tick}",
                        "event_time": t,
                        "event_tick": terminal_tick,
                    }
                )
        when = frame.get("when") or {}
        if str(when.get("phase") or "") == "post_round":
            round_end_tick = when.get("tick")
            key = ("round_end",)
            if key not in seen:
                seen.add(key)
                actions.append(
                    {
                        "type": "round_end",
                        "event_id": f"round_end:{round_end_tick}",
                        "event_time": t,
                        "event_tick": round_end_tick,
                        "winner": when.get("winner"),
                    }
                )
        for flash in events.get("flashes") or []:
            try:
                duration = float(
                    flash.get("duration_s", flash.get("duration", 0.0)) or 0.0
                )
            except (TypeError, ValueError):
                duration = 0.0
            if duration < flash_min:
                continue
            key = (
                "effective_flash",
                flash.get("tick"),
                flash.get("attacker"),
                flash.get("victim"),
            )
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                {
                    "type": "effective_flash",
                    "event_id": f"effective_flash:{flash.get('tick')}:{flash.get('attacker') or ''}:{flash.get('victim') or ''}",
                    "event_time": t,
                    "event_tick": flash.get("tick"),
                    "attacker": flash.get("attacker"),
                    "victim": flash.get("victim"),
                    "duration_s": round(duration, 2),
                    "is_teammate": bool(flash.get("is_teammate")),
                }
            )
    ordered = sorted(
        actions,
        key=lambda item: (float(item.get("event_time", 0)), str(item.get("type", ""))),
    )
    enrich_kill_actions(ordered, frames)
    return ordered
