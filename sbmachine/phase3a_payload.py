"""Phase 3a LLM payload helpers."""
from __future__ import annotations

import json


# ── analyst prompt 预算（压缩到小 ctx 内，落 8-12G 卡 num_ctx=8192~16384；不靠堆 num_ctx） ──
_ANALYST_PROMPT_TOKEN_BUDGET = 8000   # slim payload JSON 目标 ≤ ~8k token
_ANALYST_MAX_FRAMES = 30              # 降采样目标帧数（事件帧全留，空窗帧按间隔抽稀）
_ANALYST_MIN_FRAMES = 8               # 预算实在不够时的帧数下限
_CHARS_PER_TOKEN = 2.0               # CJK 估算 ~2 字符/token




# ── LLM payload filter ──

def _filter_payload_for_llm(keyframes: list[dict]) -> list[dict]:
    """Strip internal/noisy fields before sending to LLM. Raw data stays in JSON files."""
    import copy
    out = []
    for frame in copy.deepcopy(keyframes):
        # ── who: drop OCR internals ──
        who = frame.get("who", {})
        frame["who"] = {
            "pov_player": who.get("pov_player"),
            "view":       who.get("view"),   # player / director
        }

        # ── where.players: strip steamid/coords, keep playstate ──
        players = frame.get("where", {}).get("players", [])
        ct_money, t_money = 0, 0
        clean_players = []
        for p in players:
            side = str(p.get("side", "")).upper()
            money = int(p.get("money") or 0)
            if side == "CT":
                ct_money += money
            elif side == "T":
                t_money += money
            clean_players.append({
                "name":    p.get("name"),
                "side":    p.get("side"),
                "hp":      p.get("hp"),
                "armor":   p.get("armor"),
                "helmet":  p.get("helmet"),
                "weapon":  p.get("weapon"),
                "callout": p.get("callout"),
            })
        frame.setdefault("where", {})["players"] = clean_players

        ev = frame.setdefault("events", {})

        # ── team money totals (low priority hint for eco analysis) ──
        ev["team_money"] = {"CT": ct_money, "T": t_money}

        # ── kills: mark corpse-shoot (same victim already dead this round) ──
        dead_this_round: set[str] = set()
        clean_kills = []
        for k in ev.get("kills", []):
            victim = str(k.get("victim", ""))
            is_corpse = victim in dead_this_round
            dead_this_round.add(victim)
            entry = dict(k)
            if is_corpse:
                entry["is_corpse_shoot"] = True  # 鞭尸：victim已死，本条不算有效击杀
            clean_kills.append(entry)
        ev["kills"] = clean_kills

        # ── damages: victim + health_after only ──
        ev["damages"] = [
            {
                "attacker":    d.get("attacker"),
                "victim":      d.get("victim"),
                "health_after": d.get("health_after"),
            }
            for d in ev.get("damages", [])
        ]

        # ── flashes: keep all (no threshold), drop steamids ──
        ev["flashes"] = [
            {
                "attacker":    f.get("attacker"),
                "victim":      f.get("victim"),
                "duration":    f.get("duration"),
                "is_teammate": f.get("is_teammate"),
            }
            for f in ev.get("flashes", [])
        ]

        # ── smokes: drop raw coords, keep thrower + tick range ──
        ev["smokes_active"] = [
            {
                "thrower":    s.get("thrower"),
                "start_tick": s.get("start_tick"),
                "end_tick":   s.get("end_tick"),
            }
            for s in ev.get("smokes_active", [])
        ]

        # ── infernos: drop hull polygon, keep thrower + area ──
        ev["infernos_active"] = [
            {
                "thrower":    i.get("thrower"),
                "area_approx": i.get("area_approx"),
            }
            for i in ev.get("infernos_active", [])
        ]

        out.append(frame)
    return out


# ── prompt assembly ──

def _dumps_compact(obj: dict) -> str:
    """紧凑序列化（去 indent，省 ~30% 体积）。"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _slim_frame_for_prompt(frame: dict) -> dict:
    """瘦身单帧（仅作用于喂 LLM 的 payload，不动喂 compute_hype 的全帧 beats）。
    删调试字段(align_warnings/timer/tick/vlm_raw)、空数组、零值；只留 analyst 所需事实。
    """
    out: dict = {}
    # when：保留 video_time（下游 scene t_start/t_end 锚点，守音画同步）+ relative_sec + phase；
    #       删 align_warnings(45% 体积)/timer/tick/timer_source/align_frozen。
    when = frame.get("when", {}) or {}
    when_slim = {k: when.get(k) for k in ("video_time", "relative_sec", "phase") if when.get(k) is not None}
    if when_slim:
        out["when"] = when_slim
    # what：删 vlm_raw（= desc 的重复），只留 desc。
    desc = (frame.get("what", {}) or {}).get("desc")
    if desc:
        out["what"] = {"desc": desc}
    who = frame.get("who", {}) or {}
    who_slim = {k: who.get(k) for k in ("view", "pov_player") if who.get(k) is not None}
    if who_slim:
        out["who"] = who_slim
    players = (frame.get("where", {}) or {}).get("players", [])
    if players:
        out["players"] = [
            {k: p.get(k) for k in ("name", "side", "hp", "weapon", "callout") if p.get(k) is not None}
            for p in players
        ]
    # events：删空数组、全 null c4、{0,0} team_money、score_ocr.raw（score 只留 ct/t）。
    ev = frame.get("events", {}) or {}
    ev_slim: dict = {}
    for key in ("kills", "damages", "flashes", "smokes_active", "infernos_active"):
        if ev.get(key):
            ev_slim[key] = ev[key]
    c4 = ev.get("c4") or {}
    if c4.get("planted") or c4.get("begin_defuse_tick"):
        ev_slim["c4"] = {k: v for k, v in c4.items() if v not in (None, False)}
    tm = ev.get("team_money") or {}
    if tm.get("CT") or tm.get("T"):
        ev_slim["team_money"] = tm
    if ev_slim:
        out["events"] = ev_slim
    return out


def _frame_is_event(slim_frame: dict) -> bool:
    """是否事件帧（含击杀/伤害/炸弹）——降采样时必须保留。"""
    ev = slim_frame.get("events", {})
    return bool(ev.get("kills") or ev.get("damages") or ev.get("c4"))


def _evenly_sample(indices: list[int], k: int) -> list[int]:
    """从有序 indices 等距抽 k 个。"""
    if k <= 0 or not indices:
        return []
    if len(indices) <= k:
        return list(indices)
    step = len(indices) / k
    return [indices[int(i * step)] for i in range(k)]


def _frame_is_tactical(slim_frame: dict) -> bool:
    """战术帧：含烟雾/燃烧/闪光弹，或有非空 VLM desc。降采样时次优先保留。"""
    ev = slim_frame.get("events", {})
    return bool(
        ev.get("smokes_active") or ev.get("infernos_active") or ev.get("flashes")
        or (slim_frame.get("what", {}) or {}).get("desc")
    )


def _downsample_frames(frames: list[dict], max_frames: int) -> list[dict]:
    """降采样：事件帧全留 > 战术帧次优先 > 空窗帧抽稀，保证事实地基不丢。"""
    if len(frames) <= max_frames:
        return frames
    event_idx = [i for i, f in enumerate(frames) if _frame_is_event(f)]
    event_set = set(event_idx)
    if len(event_idx) >= max_frames:
        keep = set(_evenly_sample(event_idx, max_frames))
    else:
        tactical_idx = [i for i in range(len(frames)) if i not in event_set and _frame_is_tactical(frames[i])]
        tactical_set = set(tactical_idx)
        combined = len(event_idx) + len(tactical_idx)
        if combined >= max_frames:
            keep = event_set | set(_evenly_sample(tactical_idx, max_frames - len(event_idx)))
        else:
            non_tactical = [i for i in range(len(frames)) if i not in event_set and i not in tactical_set]
            keep = event_set | tactical_set | set(_evenly_sample(non_tactical, max_frames - combined))
    return [f for i, f in enumerate(frames) if i in keep]


def _slim_payload_for_prompt(payload: dict, downsample: bool = True) -> dict:
    """喂给 LLM 的瘦身 payload。瘦字段 + 跨帧折叠去冗余 + 紧凑序列化。
    downsample=True（默认）：超预算则降帧（保证零截断，OFF 分支二次压缩）。
    downsample=False：仅瘦字段不降帧，供估算真实体积 / 切段（segment 分支）。"""
    slim_frames = [_slim_frame_for_prompt(f) for f in payload.get("keyframes", [])]

    # 改动1：持续事件首尾折叠——同一颗烟雾/燃烧只在首现帧保留，后续帧删除该条
    seen_smokes: set[tuple] = set()
    seen_infernos: set[tuple] = set()
    # 改动2：who / when.phase 去冗余（video_time 永远保留）
    prev_who_key: tuple | None = None
    prev_phase: str | None = None

    for frame in slim_frames:
        ev = frame.get("events", {})
        if ev:
            smokes = ev.get("smokes_active")
            if smokes:
                fresh = []
                for s in smokes:
                    key = (s.get("thrower"), s.get("start_tick"), s.get("end_tick"))
                    if key not in seen_smokes:
                        seen_smokes.add(key)
                        fresh.append(s)
                if fresh:
                    ev["smokes_active"] = fresh
                else:
                    del ev["smokes_active"]

            infernos = ev.get("infernos_active")
            if infernos:
                fresh = []
                for inf in infernos:
                    key = (inf.get("thrower"), inf.get("area_approx"))
                    if key not in seen_infernos:
                        seen_infernos.add(key)
                        fresh.append(inf)
                if fresh:
                    ev["infernos_active"] = fresh
                else:
                    del ev["infernos_active"]

            if not ev:
                frame.pop("events", None)

        who = frame.get("who")
        if who is not None:
            who_key = (who.get("view"), who.get("pov_player"))
            if who_key == prev_who_key:
                del frame["who"]
            else:
                prev_who_key = who_key

        when = frame.get("when")
        if when is not None:
            cur_phase = when.get("phase")
            if cur_phase is not None and cur_phase == prev_phase:
                when.pop("phase", None)
            elif cur_phase is not None:
                prev_phase = cur_phase

    out = {k: payload[k] for k in ("round_no", "start_sec", "end_sec", "demo_round_hint") if k in payload}
    if not downsample:
        out["keyframes"] = slim_frames
        return out
    target = _ANALYST_MAX_FRAMES
    while True:
        out["keyframes"] = _downsample_frames(slim_frames, target)
        est_tok = len(_dumps_compact(out)) / _CHARS_PER_TOKEN
        if est_tok <= _ANALYST_PROMPT_TOKEN_BUDGET or target <= _ANALYST_MIN_FRAMES:
            return out
        target = max(_ANALYST_MIN_FRAMES, int(target * _ANALYST_PROMPT_TOKEN_BUDGET / est_tok))

def _semantic_payload(round_record) -> dict:
    keyframes = []
    if round_record.phase2_vision is not None:
        for frame in round_record.phase2_vision.key_frames:
            bg = dict(frame.background_info) if frame.background_info else {}
            bg["has_vlm"] = bool(getattr(frame, "has_vlm", True))
            keyframes.append(bg)
    return {
        "round_no":        round_record.round_no,
        "start_sec":       round_record.start_sec,
        "end_sec":         round_record.end_sec,
        "demo_round_hint": round_record.demo_round_hint,
        "keyframes":       _filter_payload_for_llm(keyframes),
    }
