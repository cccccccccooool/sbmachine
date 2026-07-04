"""Phase 2 background row construction helpers."""
from __future__ import annotations

import json
import re

from sbmachine.time_align import parse_timer_seconds


def parse_vlm_json(text: str) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None

def _demo_round_hint(round_record) -> int:
    hint = getattr(round_record, "demo_round_hint", None)
    if hint is not None:
        try:
            return int(hint)
        except (TypeError, ValueError):
            pass
    return int(round_record.round_no)


def _player_state_with_callouts(demo: DemoQuery, tick: int) -> list[dict]:
    players = []
    for row in demo.state_at(tick):
        x = float(row.get("x") or 0.0)
        y = float(row.get("y") or 0.0)
        z = float(row.get("z") or 0.0)
        # Go parser 将区域名称写入 'callout' 列;旧产物无此列时降级调用 callout_of()。
        callout = str(row.get("callout") or "") or demo.callout_of(x, y, z)
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
            }
        )
    return players


def _pov_callout(players: list[dict], steamid: str, name: str) -> str:
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
    desc: str,
    vlm_response: str,
    pov_ocr_result: dict,
    timer_ocr_result: dict,
    score_ocr_result: dict | None = None,
    prev_tick: int | None,
    pov_crop_source: str = "unknown",
    consecutive_unmatched: int = 0,
    spectator_min_frames: int = 2,
    pov_match_min_score: float = 0.6,
    timer_crop_source: str = "yolo_timer_region",
) -> tuple[dict, int]:
    timer = str(timer_ocr_result.get("value", "") or "").strip()
    if timer:
        align.add_anchor(video_time, timer)
    tick = align.to_tick(video_time)
    players = _player_state_with_callouts(demo, tick)
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

    timer_seconds = parse_timer_seconds(timer)
    if timer_seconds is not None:
        relative_sec = 115.0 - timer_seconds
    else:
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
    utilities = demo.utilities_between(prev, tick)
    damages = demo.damages_between(prev, tick) if hasattr(demo, "damages_between") else []
    flashes = demo.flashes_between(prev, tick) if hasattr(demo, "flashes_between") else []
    # active smokes / infernos at this tick (full-round scope)
    smokes_now = demo.smokes_active_at(tick) if hasattr(demo, "smokes_active_at") else []
    infernos_now = demo.infernos_active_at(tick) if hasattr(demo, "infernos_active_at") else []
    plant_tick = round_meta.get("bomb_planted_tick")
    effective_timer_source = timer_crop_source if timer else ""
    bg = {
        "when": {
            "video_time": round(float(video_time), 3),
            "timer": timer,
            "timer_source": effective_timer_source,
            "tick": tick,
            "round_no": int(round_meta.get("round_no", 0)),
            "relative_sec": round(float(relative_sec), 3),
            "phase": phase,
            "align_frozen": align.is_frozen,
            "align_warnings": list(align.warnings),
        },
        "who": {
            "pov_player": pov_name,
            "ocr_raw": str(pov_ocr_result.get("raw_text", "")),
            "match_score": match.score,
            "ocr_engine": str(pov_ocr_result.get("engine", "")),
            "pov_source": pov_source,
            "view": view,
        },
        "what": {
            "desc": str(desc or "").strip(),
            "vlm_raw": vlm_response,
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
            "c4": {
                "planted":   bool(align.is_frozen and plant_tick is not None),
                "plant_tick": plant_tick,
                "begin_defuse_tick": round_meta.get("bomb_begin_defuse_tick"),
                "defuser_has_kit":   round_meta.get("defuser_has_kit"),
            },
            "score_ocr": score_ocr_result or {},
        },
    }
    return bg, tick

