"""为云端分析员构建紧凑、可按证据 id 寻址的小局时间线。"""
from __future__ import annotations

from collections.abc import Iterable


def _frame_time(frame: dict, fallback: float) -> float:
    when = frame.get("when") or {}
    value = when.get("video_time", when.get("relative_sec", fallback))
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return round(fallback, 3)


def _event_id(prefix: str, number: int) -> str:
    return f"{prefix}-{number}"


def _extract_flags(item: dict) -> list[str]:
    flags = []
    for key, label in (("headshot", "headshot"), ("through_smoke", "through_smoke"), ("no_scope", "no_scope"), ("is_wallbang", "wallbang"), ("attacker_blind", "attacker_blind")):
        if item.get(key):
            flags.append(label)
    return flags


def _extract_players(frame: dict) -> list[dict]:
    players = (frame.get("where") or {}).get("players") or []
    result = []
    for player in players:
        name = str(player.get("name") or "").strip()
        if not name:
            continue
        result.append({
            "name": name,
            "side": str(player.get("side") or ""),
            "weapon": str(player.get("weapon", player.get("active_weapon", "")) or ""),
            "callout": str(player.get("callout") or ""),
        })
    return result


def _append_state_delta(timeline: list[dict], counter: list[int], t: float, current: dict[str, dict], previous: dict[str, dict]) -> None:
    changes = []
    for name, state in current.items():
        old = previous.get(name, {})
        delta = {"player": name}
        for field in ("hp", "weapon", "callout"):
            if state.get(field) is not None and state.get(field) != old.get(field):
                delta[field] = state[field]
        if len(delta) > 1:
            changes.append(delta)
    if changes:
        counter[0] += 1
        timeline.append({"id": _event_id("state", counter[0]), "t": t, "type": "state_delta", "changes": changes})


def build_cloud_round_payload(round_record, map_name: str) -> tuple[dict, dict[str, dict]]:
    """把重复的 Phase2 帧转成紧凑、不做解读的时间线。"""
    raw_frames = getattr(getattr(round_record, "phase2_yolo", None), "key_frames", []) or []
    frames = []
    for frame in raw_frames:
        info = dict(getattr(frame, "background_info", {}) or {})
        if info:
            frames.append(info)
    frames.sort(key=lambda item: _frame_time(item, round_record.start_sec))

    initial_players = _extract_players(frames[0]) if frames else []
    roster = [{key: value for key, value in player.items() if value} for player in initial_players]
    timeline: list[dict] = []
    evidence: dict[str, dict] = {}
    counters = {name: [0] for name in ("phase", "pov", "state", "kill", "utility", "flash", "bomb")}
    previous_phase = None
    previous_pov: tuple[str, str, str] | None = None
    previous_players: dict[str, dict] = {}
    seen_kills: set[tuple] = set()
    seen_utility: set[tuple] = set()
    seen_flash: set[tuple] = set()
    seen_bomb: set[tuple] = set()

    def add(item: dict) -> None:
        timeline.append(item)
        evidence[item["id"]] = item

    for frame in frames:
        t = _frame_time(frame, round_record.start_sec)
        when = frame.get("when") or {}
        phase = str(when.get("phase") or "")
        if phase and phase != previous_phase:
            counters["phase"][0] += 1
            add({"id": _event_id("phase", counters["phase"][0]), "t": t, "type": "phase", "value": phase})
            previous_phase = phase

        who = frame.get("who") or {}
        pov = (str(who.get("pov_player") or ""), str(who.get("view") or ""), str((frame.get("where") or {}).get("pov_callout") or ""))
        if pov != previous_pov and any(pov):
            counters["pov"][0] += 1
            add({"id": _event_id("pov", counters["pov"][0]), "t": t, "type": "pov_change", "player": pov[0], "view": pov[1], "callout": pov[2]})
            previous_pov = pov

        current_players = {}
        for player in (frame.get("where") or {}).get("players") or []:
            name = str(player.get("name") or "").strip()
            if name:
                current_players[name] = {"hp": player.get("hp"), "weapon": player.get("weapon", player.get("active_weapon")), "callout": player.get("callout")}
        if previous_players:
            _append_state_delta(timeline, counters["state"], t, current_players, previous_players)
        previous_players = current_players

        events = frame.get("events") or {}
        for kill in events.get("kills") or []:
            if kill.get("is_corpse_shoot"):
                continue
            key = (kill.get("tick"), str(kill.get("attacker") or ""), str(kill.get("victim") or ""))
            if key in seen_kills:
                continue
            seen_kills.add(key)
            counters["kill"][0] += 1
            item = {"id": _event_id("kill", counters["kill"][0]), "t": t, "type": "kill", "attacker": key[1], "victim": key[2]}
            if kill.get("assister"):
                item["assister"] = str(kill["assister"])
            if kill.get("weapon"):
                item["weapon"] = str(kill["weapon"])
            flags = _extract_flags(kill)
            if flags:
                item["flags"] = flags
            add(item)

        for utility in events.get("utilities") or []:
            key = (utility.get("entity_id"), utility.get("throw_tick"), utility.get("det_tick"), utility.get("type"))
            if key in seen_utility:
                continue
            seen_utility.add(key)
            counters["utility"][0] += 1
            item = {"id": _event_id("utility", counters["utility"][0]), "t": t, "type": "grenade", "kind": str(utility.get("type") or ""), "thrower": str(utility.get("thrower") or "")}
            add(item)

        for flash in events.get("flashes") or []:
            duration = flash.get("duration_s", flash.get("duration"))
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                continue
            key = (flash.get("tick"), str(flash.get("attacker") or ""), str(flash.get("victim") or ""))
            if duration < 2 or flash.get("is_teammate") or key in seen_flash:
                continue
            seen_flash.add(key)
            counters["flash"][0] += 1
            add({"id": _event_id("flash", counters["flash"][0]), "t": t, "type": "flash", "attacker": key[1], "victim": key[2], "duration_s": round(duration, 1)})

        c4 = events.get("c4") or {}
        for event_type, source_key in (("bomb_planted", "plant_tick"), ("defuse_started", "begin_defuse_tick")):
            tick = c4.get(source_key)
            key = (event_type, tick)
            active = bool(c4.get("planted")) if event_type == "bomb_planted" else bool(tick)
            if not active or key in seen_bomb:
                continue
            seen_bomb.add(key)
            counters["bomb"][0] += 1
            item = {"id": _event_id("bomb", counters["bomb"][0]), "t": t, "type": event_type}
            if event_type == "defuse_started":
                item["has_kit"] = bool(c4.get("defuser_has_kit"))
            add(item)

    timeline.sort(key=lambda item: (float(item["t"]), item["id"]))
    payload = {
        "contract_version": 1,
        "round": {
            "round_no": round_record.round_no,
            "map_name": map_name,
            "t_start": round(round_record.start_sec, 3),
            "t_end": round(round_record.end_sec, 3),
            "score_before": {"ct": round_record.score_before.ct, "t": round_record.score_before.t},
            "score_after": {"ct": round_record.score_after.ct, "t": round_record.score_after.t},
        },
        "roster": roster,
        "timeline": timeline,
        "instructions": {"allow_silence": True, "max_neutral_chars": 100},
    }
    return payload, {item["id"]: item for item in timeline}
