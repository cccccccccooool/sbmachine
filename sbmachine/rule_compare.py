"""Phase2 硬事实与 Phase3 选题之间的确定性规则比较层。

规则只比较 DEM 已提供的字段；缺字段即不触发。坐标、朝向和跨 tick 证据
保留在内部 ``_rule_evidence``，不会进入 LLM 投影。
"""

from __future__ import annotations

import math
from typing import Iterable

from sbmachine.common import load_cs_game_rules


RULE_LABELS = {
    "air_noscope": "空中盲狙",
    "jump_kill": "腾空击杀",
    "victim_airborne": "空中击落",
    "backstab": "背身击杀",
    "unaware_kill": "单向信息击杀",
    "blind_kill": "致盲状态击杀",
    "through_smoke": "混烟击杀",
    "wallbang": "穿墙击杀",
    "no_scope": "盲狙",
    "flick_shot": "快速转向击杀",
    "one_tap": "单发击杀",
    "moving_kill": "移动击杀",
    "point_blank": "近距离击杀",
    "long_range": "远距离击杀",
    "scoped_kill": "开镜击杀",
    "headshot": "爆头",
    "caught_switching": "切换装备时被击杀",
}

_DEFAULT_PRIORITIES = {
    "air_noscope": 0.98,
    "jump_kill": 0.88,
    "blind_kill": 0.86,
    "backstab": 0.84,
    "victim_airborne": 0.82,
    "unaware_kill": 0.80,
    "one_tap": 0.80,
    "flick_shot": 0.79,
    "caught_switching": 0.78,
    "through_smoke": 0.76,
    "wallbang": 0.76,
    "no_scope": 0.76,
    "point_blank": 0.72,
    "long_range": 0.70,
    "moving_kill": 0.68,
    "scoped_kill": 0.60,
    "headshot": 0.55,
}

_SNIPER_WEAPONS = frozenset({"awp", "ssg 08", "ssg08", "scar 20", "scar20", "g3sg1"})
_UTILITY_TOKENS = ("grenade", "flash", "smoke", "molotov", "incendiary", "decoy", "c4")
_SIDEARMS = frozenset(
    {
        "glock 18", "glock", "usp s", "usp", "p2000", "p250", "five seven",
        "tec 9", "desert eagle", "deagle", "dual berettas", "cz75 auto", "r8 revolver",
    }
)


def rule_label(rule_id: object) -> str:
    return RULE_LABELS.get(str(rule_id or ""), str(rule_id or ""))


def _normalize_weapon(value: object) -> str:
    return (
        str(value or "")
        .replace("weapon_", "")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
        .casefold()
    )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _angle_diff(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _position(record: dict, prefix: str) -> tuple[float, float, float] | None:
    values = tuple(_number(record.get(f"{prefix}_{axis}")) for axis in ("x", "y", "z"))
    return values if all(value is not None for value in values) else None  # type: ignore[return-value]


def _event_actor(row: dict, *keys: str) -> str:
    return next((str(row.get(key) or "") for key in keys if row.get(key)), "")


def _unique_rows(rows: Iterable[dict], fields: tuple[str, ...]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = tuple(row.get(field) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def pov_role(kill: dict, pov_player: object) -> str:
    """POV 是主角身份；没有可靠 POV 时显式返回 unavailable，交给空间层降级。"""
    pov = str(pov_player or "").strip()
    if not pov:
        return "unavailable"
    if pov == str(kill.get("attacker") or ""):
        return "killer"
    if pov == str(kill.get("victim") or ""):
        return "victim"
    return "observer"


def rule_priority(rule_id: object, config: dict | None = None) -> float:
    cfg = config or {}
    configured = cfg.get("priorities") if isinstance(cfg, dict) else None
    value = configured.get(str(rule_id)) if isinstance(configured, dict) else None
    try:
        return float(value) if value is not None else float(_DEFAULT_PRIORITIES.get(str(rule_id), 0.0))
    except (TypeError, ValueError):
        return float(_DEFAULT_PRIORITIES.get(str(rule_id), 0.0))


def compare_round_context(
    frames: Iterable[dict], *, config: dict | None = None
) -> dict:
    """提取无未来泄露的比分与消费上下文；消费额不冒充当前装备价值。"""
    cfg = config or {}
    ordered_frames = sorted(
        (frame for frame in frames if isinstance(frame, dict)),
        key=lambda frame: float((frame.get("when") or {}).get("video_time", 0.0) or 0.0),
    )
    tags: list[str] = []
    result: dict[str, object] = {"tags": tags}

    score_before = next(
        (
            (frame.get("when") or {}).get("score_before")
            for frame in ordered_frames
            if isinstance((frame.get("when") or {}).get("score_before"), dict)
        ),
        None,
    )
    if isinstance(score_before, dict):
        ct_score = _integer(score_before.get("ct"))
        t_score = _integer(score_before.get("t"))
        if ct_score is not None and t_score is not None and ct_score >= 0 and t_score >= 0:
            result["score_before"] = {"ct": ct_score, "t": t_score}
            match_point_base = int(cfg.get("match_point_base_score", 12))
            overtime_step = max(1, int(cfg.get("overtime_match_point_step", 3)))

            def is_match_point(score: int, other: int) -> bool:
                return (
                    score >= match_point_base
                    and (score - match_point_base) % overtime_step == 0
                    and score > other
                )

            if is_match_point(ct_score, t_score):
                tags.append("ct_match_point")
            elif is_match_point(t_score, ct_score):
                tags.append("t_match_point")
            elif ct_score == t_score and ct_score >= match_point_base:
                tags.append("overtime_tie")
            elif ct_score == t_score and ct_score >= int(cfg.get("late_tie_min_score", 10)):
                tags.append("late_tie")

    representative = next(
        (
            frame
            for frame in reversed(ordered_frames)
            if (frame.get("where") or {}).get("players")
        ),
        None,
    )
    if representative is not None:
        spending: dict[str, list[float]] = {"T": [], "CT": []}
        for player in (representative.get("where") or {}).get("players") or []:
            side = str(player.get("side") or "").upper()
            value = _number(player.get("money_spent_this_round"))
            if side in spending and value is not None and value >= 0:
                spending[side].append(value)
        minimum_players = int(cfg.get("spend_context_min_players_per_side", 3))
        if all(len(spending[side]) >= minimum_players for side in ("T", "CT")):
            totals = {side: int(round(sum(spending[side]))) for side in ("T", "CT")}
            result["team_money_spent"] = {"t": totals["T"], "ct": totals["CT"]}
            low_side = min(totals, key=totals.get)
            high_side = "CT" if low_side == "T" else "T"
            if (
                totals[low_side] <= int(cfg.get("spend_gap_low_max", 5000))
                and totals[high_side] >= int(cfg.get("spend_gap_high_min", 12000))
            ):
                tags.append("spend_gap")
                result["lower_spend_side"] = low_side
                result["spend_gap"] = totals[high_side] - totals[low_side]
    return result


def _snapshot_rows(
    snapshots: Iterable[dict], *, event_tick: int, player: str
) -> list[dict]:
    rows = [
        row
        for row in snapshots
        if _integer(row.get("event_tick")) == event_tick
        and str(row.get("name") or "") == player
        and (_integer(row.get("tick")) or -1) < event_tick
    ]
    return sorted(rows, key=lambda row: _integer(row.get("tick")) or -1)


def _movement_speed(rows: list[dict], tick_rate: float) -> float | None:
    if len(rows) < 2:
        return None
    left, right = rows[-2], rows[-1]
    lt, rt = _integer(left.get("tick")), _integer(right.get("tick"))
    lx, ly = _number(left.get("x")), _number(left.get("y"))
    rx, ry = _number(right.get("x")), _number(right.get("y"))
    if None in (lt, rt, lx, ly, rx, ry) or rt <= lt:
        return None
    return math.hypot(rx - lx, ry - ly) * float(tick_rate) / float(rt - lt)


def _is_primary_weapon(value: object) -> bool:
    weapon = _normalize_weapon(value)
    if not weapon or weapon.startswith("knife") or weapon in _SIDEARMS:
        return False
    return not any(token in weapon for token in _UTILITY_TOKENS)


def compare_kill(
    kill: dict,
    *,
    snapshots: Iterable[dict] = (),
    fires: Iterable[dict] = (),
    equips: Iterable[dict] = (),
    tick_rate: float = 64.0,
    config: dict | None = None,
) -> dict:
    """对单次击杀执行分层比较，返回稳定 tag、主 tag 与内部证据。"""
    cfg = config or {}
    tags: dict[str, float] = {}
    evidence: dict[str, object] = {}

    def add(rule_id: str, confidence: float = 1.0, **facts: object) -> None:
        tags[rule_id] = max(tags.get(rule_id, 0.0), float(confidence))
        if facts:
            evidence[rule_id] = facts

    if kill.get("headshot") is True:
        add("headshot", headshot=True)
    if kill.get("through_smoke") is True:
        add("through_smoke", through_smoke=True)
    if kill.get("no_scope") is True:
        add("no_scope", no_scope=True)
    if kill.get("is_wallbang") is True:
        add("wallbang", is_wallbang=True)
    if kill.get("attacker_blind") is True:
        add("blind_kill", attacker_blind=True)
    if kill.get("killer_scoped") is True:
        add("scoped_kill", killer_scoped=True)
    if kill.get("victim_airborne") is True:
        add("victim_airborne", victim_airborne=True)

    weapon = _normalize_weapon(kill.get("weapon"))
    killer_airborne = kill.get("killer_airborne") is True
    if killer_airborne:
        add("jump_kill", killer_airborne=True)
    if killer_airborne and kill.get("no_scope") is True and weapon in _SNIPER_WEAPONS:
        add("air_noscope", killer_airborne=True, no_scope=True, weapon=kill.get("weapon"))

    if (
        "killer_spotted_victim" in kill
        and "victim_spotted_killer" in kill
        and kill.get("killer_spotted_victim") is True
        and kill.get("victim_spotted_killer") is False
    ):
        add("unaware_kill", 0.95, killer_spotted_victim=True, victim_spotted_killer=False)

    distance = _number(kill.get("distance"))
    point_blank_max = float(cfg.get("point_blank_max_units", 120.0))
    long_range_min = float(cfg.get("long_range_min_units", 1000.0))
    if distance is not None and distance <= point_blank_max:
        add("point_blank", distance=round(distance, 1))
    elif distance is not None and distance >= long_range_min:
        add("long_range", distance=round(distance, 1))

    killer_pos, victim_pos = _position(kill, "killer"), _position(kill, "victim")
    victim_yaw = _number(kill.get("victim_yaw"))
    if killer_pos and victim_pos and victim_yaw is not None:
        toward_killer = math.degrees(
            math.atan2(killer_pos[1] - victim_pos[1], killer_pos[0] - victim_pos[0])
        )
        away = _angle_diff(victim_yaw, toward_killer)
        threshold = float(cfg.get("backstab_victim_away_min_deg", 100.0))
        if away >= threshold:
            add("backstab", 0.9, victim_away_deg=round(away, 1))

    event_tick = _integer(kill.get("event_tick", kill.get("tick")))
    attacker, victim = str(kill.get("attacker") or ""), str(kill.get("victim") or "")
    if event_tick is not None and attacker:
        attacker_snaps = _snapshot_rows(snapshots, event_tick=event_tick, player=attacker)
        if any(row.get("is_airborne") is True for row in attacker_snaps[-2:]):
            add("jump_kill", 0.95, snapshot_airborne=True)
        speed = _movement_speed(attacker_snaps, tick_rate)
        moving_min = float(cfg.get("moving_kill_min_units_per_sec", 120.0))
        if speed is not None and speed >= moving_min and "jump_kill" not in tags:
            add("moving_kill", 0.85, speed_units_per_sec=round(speed, 1))

        current_yaw = _number(kill.get("killer_yaw"))
        lookback_ticks = int(float(cfg.get("flick_lookback_sec", 0.5)) * tick_rate)
        yaw_samples = [
            _number(row.get("yaw"))
            for row in attacker_snaps
            if event_tick - (_integer(row.get("tick")) or event_tick) <= lookback_ticks
        ]
        yaw_samples = [value for value in yaw_samples if value is not None]
        if current_yaw is not None and yaw_samples:
            yaw_delta = max(_angle_diff(current_yaw, value) for value in yaw_samples)
            if yaw_delta >= float(cfg.get("flick_min_yaw_delta_deg", 25.0)):
                add("flick_shot", 0.9, max_yaw_delta_deg=round(yaw_delta, 1))

        fire_window = int(float(cfg.get("one_tap_lookback_sec", 1.5)) * tick_rate)
        matching_fires = [
            row
            for row in fires
            if event_tick - fire_window < (_integer(row.get("tick")) or -1) <= event_tick
            and _event_actor(row, "shooter", "user_name") == attacker
            and (
                not weapon
                or _normalize_weapon(row.get("weapon")) == weapon
            )
        ]
        if len(matching_fires) == 1:
            add("one_tap", 0.95, shots_in_window=1)

    if event_tick is not None and victim:
        victim_equips = sorted(
            (
                row
                for row in equips
                if _event_actor(row, "player", "user_name") == victim
                and (_integer(row.get("tick")) or -1) <= event_tick
            ),
            key=lambda row: _integer(row.get("tick")) or -1,
        )
        if len(victim_equips) >= 2:
            previous, current = victim_equips[-2], victim_equips[-1]
            switch_tick = _integer(current.get("tick"))
            previous_tick = _integer(previous.get("tick"))
            current_weapon = _normalize_weapon(current.get("weapon", current.get("item")))
            previous_weapon = previous.get("weapon", previous.get("item"))
            switch_window = int(float(cfg.get("switch_death_window_sec", 1.5)) * tick_rate)
            held_min = int(float(cfg.get("switch_primary_hold_min_sec", 10.0)) * tick_rate)
            is_utility = current_weapon.startswith("knife") or any(
                token in current_weapon for token in _UTILITY_TOKENS
            )
            if (
                switch_tick is not None
                and previous_tick is not None
                and 0 <= event_tick - switch_tick <= switch_window
                and switch_tick - previous_tick >= held_min
                and is_utility
                and _is_primary_weapon(previous_weapon)
            ):
                add(
                    "caught_switching",
                    0.9,
                    previous_weapon=previous_weapon,
                    current_weapon=current.get("weapon", current.get("item")),
                    ticks_before_death=event_tick - switch_tick,
                )

    ordered = sorted(tags, key=lambda rule_id: (-rule_priority(rule_id, cfg), rule_id))
    primary = ordered[0] if ordered else None
    return {
        "tags": ordered,
        "primary": primary,
        "confidence": round(tags.get(primary, 0.0), 3) if primary else 0.0,
        "evidence": evidence,
    }


def enrich_kill_actions(
    actions: list[dict], frames: list[dict], *, tick_rate: float = 64.0
) -> None:
    """从帧内跨 tick 事件建索引，并就地丰富 kill action。"""
    fires: list[dict] = []
    equips: list[dict] = []
    snapshots: list[dict] = []
    for frame in frames:
        events = frame.get("events") or {}
        fires.extend(events.get("weapon_fires") or events.get("fired") or [])
        equips.extend(events.get("item_equips") or events.get("equips") or [])
        snapshots.extend(events.get("event_snapshots") or [])
    fires = _unique_rows(fires, ("tick", "shooter", "weapon"))
    equips = _unique_rows(equips, ("tick", "player", "weapon"))
    snapshots = _unique_rows(snapshots, ("tick", "event_tick", "name"))

    inferred_tick_rate = next(
        (
            _number((frame.get("when") or {}).get("tick_rate"))
            for frame in frames
            if _number((frame.get("when") or {}).get("tick_rate")) is not None
        ),
        None,
    )
    rules = load_cs_game_rules()
    cfg = rules.get("rule_compare", {})
    try:
        configured_tick_rate = float(inferred_tick_rate or tick_rate)
    except (TypeError, ValueError):
        configured_tick_rate = float(tick_rate)
    round_context = compare_round_context(frames, config=cfg)

    for action in actions:
        if action.get("type") != "kill":
            continue
        comparison = compare_kill(
            action,
            snapshots=snapshots,
            fires=fires,
            equips=equips,
            tick_rate=configured_tick_rate,
            config=cfg,
        )
        action["rule_tags"] = comparison["tags"]
        action["primary_rule"] = comparison["primary"]
        action["rule_confidence"] = comparison["confidence"]
        action["pov_role"] = pov_role(action, action.get("pov_player"))
        round_tags = list(round_context.get("tags") or [])
        lower_spend_side = str(round_context.get("lower_spend_side") or "")
        if (
            lower_spend_side
            and str(action.get("attacker_side") or "").upper() == lower_spend_side
            and str(action.get("victim_side") or "").upper()
            in ({"T", "CT"} - {lower_spend_side})
        ):
            round_tags.append("lower_spend_side_kill")
        action["round_tags"] = round_tags
        action["round_context"] = dict(round_context)
        action["_rule_evidence"] = comparison["evidence"]
