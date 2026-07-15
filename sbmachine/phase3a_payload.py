"""Phase 3a LLM payload 辅助函数。"""
from __future__ import annotations

import json
from pathlib import Path


def load_semantic_frames(semantic_path: Path) -> dict[int, list[dict]]:
    """加载精简 DEM-fact 时间线（rounds_with_yolo_semantic.json），作为 LLM-A 帧的物理来源。

    文件缺失时（dry-run 或旧产物）由调用方在内存中重建帧作为兜底。
    """
    if not semantic_path.is_file():
        return {}
    try:
        payload = json.loads(semantic_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list):
        return {}
    frames_by_round: dict[int, list[dict]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        round_no = entry.get("round_no")
        frames = entry.get("frames")
        if isinstance(round_no, int) and not isinstance(round_no, bool) and isinstance(frames, list):
            frames_by_round[round_no] = [frame for frame in frames if isinstance(frame, dict)]
    return frames_by_round


def _normalize_planning_frames(keyframes: list[dict]) -> list[dict]:
    """归一化 DEM 帧供确定性规划使用；这不是给 LLM 的 payload。"""
    import copy
    out = []
    dead_this_round: set[str] = set()
    for frame in copy.deepcopy(keyframes):
        # ── who：丢弃 OCR 内部字段 ──
        who = frame.get("who", {})
        frame["who"] = {
            "pov_player": who.get("pov_player"),
            "view":       who.get("view"),   # player / director
        }

        # ── where.players：去掉 steamid，保留坐标供确定性空间规划 ──
        players = frame.get("where", {}).get("players", [])
        clean_players = []
        for p in players:
            clean_players.append({
                "name":    p.get("name"),
                "side":    p.get("side"),
                "hp":      p.get("hp"),
                "armor":   p.get("armor"),
                "helmet":  p.get("helmet"),
                "weapon":  p.get("weapon", p.get("active_weapon")),
                "callout": p.get("callout"),
                "ammo":    p.get("ammo"),
                "x":       p.get("x"),
                "y":       p.get("y"),
                "z":       p.get("z"),
            })
        frame.setdefault("where", {})["players"] = clean_players

        ev = frame.setdefault("events", {})

        # ── kills：标记鞭尸（同一 victim 本局已阵亡） ──
        clean_kills = []
        for k in ev.get("kills", []):
            victim = str(k.get("victim", ""))
            is_corpse = bool(victim) and victim in dead_this_round
            if victim:
                dead_this_round.add(victim)
            entry = dict(k)
            if is_corpse:
                entry["is_corpse_shoot"] = True  # 鞭尸：victim已死，本条不算有效击杀
            clean_kills.append(entry)
        ev["kills"] = clean_kills

        # ── damages：只留 victim + health_after ──
        ev["damages"] = [
            {
                "attacker":    d.get("attacker"),
                "victim":      d.get("victim"),
                "health_after": d.get("health_after"),
            }
            for d in ev.get("damages", [])
        ]

        # ── flashes：全部保留（不设阈值），丢弃 steamid ──
        flash_warn = False
        ev["flashes"] = []
        for f in ev.get("flashes", []):
            dur = f.get("duration_s", f.get("duration"))
            if dur is None:
                flash_warn = True
            ev["flashes"].append({
                "attacker":    f.get("attacker"),
                "victim":      f.get("victim"),
                "duration_s":  dur,
                "is_teammate": f.get("is_teammate"),
            })
        if flash_warn:
            import sys
            print("[phase3a_payload] WARNING: some flash events have no duration_s or duration field", file=sys.stderr)

        # ── smokes：丢弃原始坐标，保留投掷者 + tick 区间 ──
        ev["smokes_active"] = [
            {
                "thrower":    s.get("thrower"),
                "start_tick": s.get("start_tick"),
                "end_tick":   s.get("end_tick"),
            }
            for s in ev.get("smokes_active", [])
        ]

        # ── infernos：丢弃火焰包络多边形，保留投掷者 + 面积 ──
        ev["infernos_active"] = [
            {
                "thrower":    i.get("thrower"),
                "area_approx": i.get("area_approx"),
            }
            for i in ev.get("infernos_active", [])
        ]

        out.append(frame)
    return out


# ── prompt 组装 ──

def _dumps_compact(obj: dict) -> str:
    """紧凑序列化（去 indent，省 ~30% 体积）。"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _slim_frame_for_prompt(frame: dict) -> dict:
    """只向 LLMA 暴露低权重的视觉氛围信息。

    选手状态与事件留在确定性规划器内部；若在此重复它们，模型会重新做规则决策，
    把紧凑的 plan 又还原成一份帧转储。
    """
    out: dict = {}
    when = frame.get("when", {}) or {}
    when_slim = {
        key: when.get(key)
        for key in ("video_time", "relative_sec", "phase")
        if when.get(key) is not None
    }
    if when_slim:
        out["when"] = when_slim

    who = frame.get("who", {}) or {}
    who_slim = {
        key: who.get(key)
        for key in ("view", "pov_player")
        if who.get(key) is not None
    }
    if who_slim:
        out["who"] = who_slim
    return out


_STATE_FIELDS = ("hp", "weapon", "callout")


def _player_state_snapshot(frames: list[dict]) -> dict[str, dict]:
    """取窗口内最后一个带选手列表的帧，抽出 name -> {side, hp, weapon, callout}。"""
    representative = next(
        (frame for frame in reversed(frames) if (frame.get("where") or {}).get("players")),
        None,
    )
    if representative is None:
        return {}
    snapshot: dict[str, dict] = {}
    for player in (representative.get("where") or {}).get("players") or []:
        name = str(player.get("name") or "").strip()
        if not name:
            continue
        snapshot[name] = {
            "side":    player.get("side"),
            "hp":      player.get("hp"),
            "weapon":  player.get("weapon", player.get("active_weapon")),
            "callout": player.get("callout"),
        }
    return snapshot


def build_state_block(frames: list[dict], reported: dict[str, dict]) -> str:
    """构建给 LLM-A 的增量状态报告：回合首窗全量汇报一次，之后仅报变化。

    ``reported`` 是回合内跨窗口复用的可变字典（记录上次已汇报状态）。
    与云端 ``_append_state_delta`` 同一取舍：只追 hp/weapon/callout；
    ammo 与原始坐标留在确定性规划器内部，不进 prompt。
    """
    snapshot = _player_state_snapshot(frames)
    if not snapshot:
        return ""
    if not reported:
        parts = []
        for name, state in snapshot.items():
            desc = "·".join(str(state[key]) for key in ("side", "weapon", "callout") if state.get(key))
            hp = state.get("hp")
            if hp is not None:
                desc = f"{desc}·{hp}血" if desc else f"{hp}血"
            parts.append(f"{name}({desc})" if desc else name)
            reported[name] = dict(state)
        return "开局状态：" + "；".join(parts)

    changes: list[str] = []
    for name, state in snapshot.items():
        old = reported.get(name)
        if old is None:
            # 数据缺口后首次见到该选手：静默纳入跟踪，不当作事件汇报。
            reported[name] = dict(state)
            continue
        if old.get("dead"):
            continue
        hp = state.get("hp")
        if isinstance(hp, (int, float)) and hp <= 0:
            changes.append(f"{name} 阵亡")
            old["dead"] = True
            continue
        detail = []
        if state.get("hp") is not None and state.get("hp") != old.get("hp"):
            detail.append(f"血量{old.get('hp')}→{state.get('hp')}")
        if state.get("weapon") and state.get("weapon") != old.get("weapon"):
            detail.append(f"换枪{state.get('weapon')}")
        if state.get("callout") and state.get("callout") != old.get("callout"):
            detail.append(f"转移{state.get('callout')}")
        if detail:
            changes.append(f"{name} " + "、".join(detail))
        for key in _STATE_FIELDS:
            if state.get(key) is not None:
                old[key] = state.get(key)
    if not changes:
        return ""
    return "状态变化：" + "；".join(changes)


def _semantic_payload(round_record, external_frames: list[dict] | None = None) -> dict:
    """从精简 DEM 帧构建规划器 / LLM payload。

    当提供 ``external_frames``（rounds_with_yolo_semantic.json 时间线）时，它们是
    物理事实来源；否则从 ``round_record.phase2_yolo`` 在内存中重建帧，用于 dry-run
    以及早于 semantic sidecar 的旧输入。
    """
    keyframes: list[dict] = []
    if external_frames is not None:
        for frame in external_frames:
            bg = dict(frame) if isinstance(frame, dict) else {}
            bg.pop("what", None)
            bg["has_frame"] = bool(bg.get("has_frame", True))
            keyframes.append(bg)
    else:
        phase2 = getattr(round_record, "phase2_yolo", None)
        if phase2 is not None:
            for frame in phase2.key_frames:
                bg = dict(frame.background_info) if frame.background_info else {}
                bg.pop("what", None)
                bg["has_frame"] = bool(getattr(frame, "has_frame", True))
                keyframes.append(bg)
    return {
        "round_no":        round_record.round_no,
        "start_sec":       round_record.start_sec,
        "end_sec":         round_record.end_sec,
        "demo_round_hint": round_record.demo_round_hint,
        "keyframes":       _normalize_planning_frames(keyframes),
    }
