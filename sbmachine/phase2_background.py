"""第二阶段：构建背景信息行的辅助函数。"""
from __future__ import annotations

from sbmachine.phase2_ocr import normalize_score_observation

def _demo_round_hint(round_record) -> int:
    """读取视频回合已解析出的 demo 回合号，缺失或非法时抛错（fail-closed）。"""
    hint = getattr(round_record, "demo_round_hint", None)
    if hint is None or str(hint).strip().casefold() == "unmatched":
        raise ValueError(f"video round {round_record.round_no} has no resolved demo round hint")
    try:
        round_no = int(hint)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid demo_round_hint for video round {round_record.round_no}: {hint!r}") from exc
    if round_no <= 0:
        raise ValueError(f"invalid demo_round_hint for video round {round_record.round_no}: {hint!r}")
    return round_no


def resolve_demo_round_hints(rounds: list, demo_rounds: list[dict]) -> None:
    """用最近的已锚定回合补齐缺失的 demo 回合号提示，无锚点时按 1->1 顺推。"""
    available = {
        int(item.get("round_no", 0))
        for item in demo_rounds
        if isinstance(item, dict) and int(item.get("round_no", 0)) > 0
    }
    if not available:
        raise ValueError("parsed demo contains no numbered rounds")

    anchors: list[tuple[int, int]] = []
    for index, round_record in enumerate(rounds):
        hint = getattr(round_record, "demo_round_hint", None)
        try:
            demo_round_no = int(hint)
        except (TypeError, ValueError):
            continue
        if demo_round_no in available:
            anchors.append((index, demo_round_no))

    first_demo_round = min(available)
    for index, round_record in enumerate(rounds):
        hint = getattr(round_record, "demo_round_hint", None)
        try:
            if int(hint) in available:
                continue
        except (TypeError, ValueError):
            pass

        if anchors:
            anchor_index, anchor_round = min(anchors, key=lambda item: abs(item[0] - index))
            guessed = anchor_round + index - anchor_index
        else:
            guessed = first_demo_round + index
        if guessed not in available:
            raise ValueError(
                f"video round {round_record.round_no} guessed demo round {guessed}, "
                "but that round does not exist"
            )
        round_record.demo_round_hint = guessed


def _player_state_with_callouts(demo: DemoQuery, tick: int, round_no: int) -> list[dict]:
    """从 demo 读取指定 tick 的玩家状态，并附带点位（callout）信息。"""
    players = []
    for row in demo.state_at(tick, round_no=round_no):
        x = float(row.get("x") or 0.0)
        y = float(row.get("y") or 0.0)
        z = float(row.get("z") or 0.0)
        # 点位 ID 只接受当前 Go parser 写入的 callout；旧产物缺失时保持为空。
        callout = str(row.get("callout") or "")
        players.append(
            {
                "steamid": str(row.get("steamid", "")),
                "name": str(row.get("name", "")),
                "side": str(row.get("side", "")),
                "hp": row.get("hp"),
                "armor": row.get("armor"),
                "helmet": bool(row.get("has_helmet", False)),
                "weapon": str(row.get("active_weapon", "")),
                "callout": callout,
                "x": x,
                "y": y,
                "z": z,
                "money": row.get("money"),
                "ammo": row.get("ammo"),
                "yaw": row.get("yaw"),
                "pitch": row.get("pitch"),
                "is_walking": bool(row.get("is_walking", False)),
                "is_airborne": bool(row.get("is_airborne", False)),
                "is_ducking": bool(row.get("is_ducking", False)),
                "is_scoped": bool(row.get("is_scoped", False)),
                "zoom_level": row.get("zoom_level"),
                "in_bomb_zone": bool(row.get("in_bomb_zone", False)),
                "money_spent_this_round": row.get("money_spent_this_round"),
            }
        )
    return players


def _c4_planted_at(round_meta: dict, tick: int) -> bool:
    """判断给定 tick 时 C4 是否处于已安放且未终结（爆炸/拆除/回合结束）的状态。"""
    plant_tick = round_meta.get("bomb_planted_tick")
    try:
        plant = int(plant_tick)
    except (TypeError, ValueError):
        return False
    terminal_ticks: list[int] = []
    for key in ("bomb_exploded_tick", "bomb_defused_tick", "end_tick"):
        try:
            value = int(round_meta.get(key))
        except (TypeError, ValueError):
            continue
        if value > plant:
            terminal_ticks.append(value)
    terminal = min(terminal_ticks) if terminal_ticks else None
    return int(tick) >= plant and (terminal is None or int(tick) < terminal)


def _pov_callout(players: list[dict], steamid: str, name: str) -> str:
    """按 steamid 优先、名字兜底的顺序，取出 POV 玩家所在点位。"""
    if steamid:
        for player in players:
            if str(player.get("steamid", "")) == str(steamid):
                return str(player.get("callout", ""))
    target = str(name).casefold()
    if target:
        for player in players:
            if str(player.get("name", "")).casefold() == target:
                return str(player.get("callout", ""))
    return ""


def build_background_info(
    *,
    demo: DemoQuery,
    round_meta: dict,
    align: RoundTimeAlign,
    video_time: float,
    pov_ocr_result: dict,
    timer_ocr_result: dict,
    score_ocr_result: dict | None = None,
    prev_tick: int | None,
    pov_crop_source: str = "unknown",
    consecutive_unmatched: int = 0,
    spectator_min_frames: int = 2,
    pov_match_min_score: float = 0.6,
    timer_crop_source: str = "yolo_timer_region",
    align_warnings: list[str] | None = None,
) -> tuple[dict, int]:
    """汇总某一帧的时间/POV/事件信息，构建单帧背景数据行，返回 (背景字典, tick)。"""
    if not align.is_locked:
        raise ValueError("Phase2 background projection requires a locked OCR alignment")
    timer = (
        str(timer_ocr_result.get("normalized", timer_ocr_result.get("value", "")) or "").strip()
        if timer_ocr_result.get("alignment_status") == "accepted"
        else ""
    )
    tick = align.to_tick(video_time)
    players = _player_state_with_callouts(
        demo,
        tick,
        round_no=int(round_meta.get("round_no", 0)),
    )
    match = demo.match_player(str(pov_ocr_result.get("raw_text", "")))

    # 应用最小匹配度分数守护:如果 OCR 结果与任何名单成员都不匹配,
    # 则视其为未匹配 / 可能是导播/观众视角画面。
    if match.score < pov_match_min_score:
        pov_name = ""
        steamid = ""
        # 在连续 N 帧未匹配后,判定为导播/观众镜头,而不是单纯的 OCR 识别失败。
        if consecutive_unmatched + 1 >= spectator_min_frames:
            pov_source = "spectator"
            view = "director"
        else:
            pov_source = "unmatched"
            view = "player"
    else:
        pov_name = match.name
        steamid = match.steamid
        pov_source = pov_crop_source
        view = "player"

    relative_sec = align.relative_sec_for_tick(tick)

    # 标记这一帧属于回合的哪一部分。
    if relative_sec < 0:
        phase = "pre_round"
    elif relative_sec <= 115.0:
        phase = "in_round"
    else:
        phase = "post_round"

    prev = prev_tick if prev_tick is not None else tick
    kills = demo.kills_between(prev, tick)
    if hasattr(demo, "utility_throws_between"):
        utilities = [
            item
            for item in demo.utility_throws_between(prev, tick)
            if int(item.get("throw_tick", -1)) > prev
        ]
    else:
        utilities = [
            item
            for item in demo.utilities_between(prev, tick)
            if int(
                item.get("throw_tick")
                if item.get("_event") == "throw"
                else item.get("det_tick", -1)
            )
            > prev
        ]
    damages = demo.damages_between(prev, tick) if hasattr(demo, "damages_between") else []
    flashes = demo.flashes_between(prev, tick) if hasattr(demo, "flashes_between") else []
    # 当前 tick 仍在生效的烟雾 / 火焰（作用域为整回合）
    smokes_now = demo.smokes_active_at(tick) if hasattr(demo, "smokes_active_at") else []
    infernos_now = demo.infernos_active_at(tick) if hasattr(demo, "infernos_active_at") else []
    weapon_fires = (
        demo.weapon_fires_between(prev, tick)
        if hasattr(demo, "weapon_fires_between")
        else []
    )
    item_equips = (
        demo.item_equips_between(prev, tick)
        if hasattr(demo, "item_equips_between")
        else []
    )
    event_snapshots = (
        demo.event_snapshots_for_kills(kills)
        if hasattr(demo, "event_snapshots_for_kills")
        else []
    )
    plant_tick = round_meta.get("bomb_planted_tick")
    effective_timer_source = timer_crop_source if timer else ""
    score_debug = normalize_score_observation(score_ocr_result, video_time=video_time)
    score_debug.pop("_regions", None)
    bg = {
        "when": {
            "video_time": round(float(video_time), 3),
            "timer": timer,
            "timer_source": effective_timer_source,
            "tick": tick,
            "tick_rate": float(getattr(demo, "tick_rate", align.tick_rate)),
            "round_no": int(round_meta.get("round_no", 0)),
            "relative_sec": round(float(relative_sec), 3),
            "phase": phase,
            "align_frozen": align.is_frozen,
            "align_warnings": list(align_warnings or []),
        },
        "who": {
            "pov_player": pov_name,
            "ocr_raw": str(pov_ocr_result.get("raw_text", "")),
            "match_score": match.score,
            "ocr_engine": str(pov_ocr_result.get("engine", "")),
            "pov_source": pov_source,
            "view": view,
        },
        "where": {
            # steamid 仅用于内部 POV 匹配(上方 _pov_callout / who 计算),不写入最终背景数据。
            "pov_callout": _pov_callout(players, steamid, pov_name),
            "players": [{k: v for k, v in p.items() if k != "steamid"} for p in players],
        },
        "events": {
            "kills":     kills,
            "utilities": utilities,
            "damages":   damages,
            "flashes":   flashes,
            "smokes_active":   smokes_now,
            "infernos_active": infernos_now,
            "weapon_fires": weapon_fires,
            "item_equips": item_equips,
            "event_snapshots": event_snapshots,
            "c4": {
                "planted":   _c4_planted_at(round_meta, tick),
                "plant_tick": plant_tick,
                "begin_defuse_tick": round_meta.get("bomb_begin_defuse_tick"),
                "defuser_has_kit":   round_meta.get("defuser_has_kit"),
                "bomb_site": round_meta.get("bomb_site"),
                "bomb_exploded_tick": round_meta.get("bomb_exploded_tick"),
                "bomb_defused_tick": round_meta.get("bomb_defused_tick"),
            },
            "score_ocr": score_debug,
        },
    }
    return bg, tick
