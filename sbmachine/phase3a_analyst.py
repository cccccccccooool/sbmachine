"""Phase 3a — 分析模型：事件行 → 中性解说稿 + hype 曲线。"""
from __future__ import annotations

import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from core.prompt_loader import load_prompt
from sbmachine.common import load_config, load_hype_rules, require_path, write_json
from sbmachine.schemas import load_match

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── analyst prompt 预算（压缩到小 ctx 内，落 8-12G 卡 num_ctx=8192~16384；不靠堆 num_ctx） ──
_ANALYST_PROMPT_TOKEN_BUDGET = 8000   # slim payload JSON 目标 ≤ ~8k token
_ANALYST_MAX_FRAMES = 30              # 降采样目标帧数（事件帧全留，空窗帧按间隔抽稀）
_ANALYST_MIN_FRAMES = 8               # 预算实在不够时的帧数下限
_CHARS_PER_TOKEN = 2.0               # CJK 估算 ~2 字符/token




def compute_hype(beats: list[dict], demo_rounds: list[dict], tick_rate: float = 64.0) -> list[float]:
    """Compute per-beat hype score [0,1]. All weights from hype_rules.json.

    Returns one float per beat.
    """
    rules = load_hype_rules()
    tau = float(rules["decay_tau_sec"])
    base = rules["base_scores"]
    bonuses = rules["kill_flag_bonuses"]
    long_dist = float(rules.get("long_distance_threshold", 1000))
    mp_mult = float(rules.get("match_point_multiplier", 1.4))

    def decay(dt: float) -> float:
        return math.exp(-dt / tau)

    events: list[tuple[float, float]] = []
    round_kill_count: dict[int, int] = {}

    for beat in beats:
        t = float(beat.get("when", {}).get("video_time", 0))
        rno = int(beat.get("when", {}).get("round_no", 0))

        for k in beat.get("events", {}).get("kills", []):
            round_kill_count[rno] = round_kill_count.get(rno, 0) + 1
            kc = round_kill_count[rno]
            s = float(base.get(f"kill_{kc}k", base["kill_single"])) if kc >= 3 else float(base["kill_single"])
            if k.get("through_smoke"):
                s += float(bonuses.get("through_smoke", 0))
            if k.get("no_scope"):
                s += float(bonuses.get("no_scope", 0))
            if k.get("is_wallbang"):
                s += float(bonuses.get("is_wallbang", 0))
            if k.get("attacker_blind"):
                s += float(bonuses.get("attacker_blind", 0))
            if float(k.get("distance", 0)) > long_dist:
                s += float(bonuses.get("long_distance", 0))
            events.append((t, s))

        c4 = beat.get("events", {}).get("c4", {})
        if c4.get("planted"):
            events.append((t, float(base["bomb_plant"])))
        if c4.get("begin_defuse_tick") and c4.get("defuser_has_kit") is False:
            events.append((t, float(base["no_kit_defuse"])))

        for dmg in beat.get("events", {}).get("damages", []):
            if int(dmg.get("health_after", 100)) <= 15:
                events.append((t, float(base["low_blood"])))

    # match-point rounds（CS2 MR12：先到13胜，故12胜即赛点。阈值可从 hype_rules 覆盖）
    match_point_rounds: set[int] = set()
    mp_at = int(rules.get("match_point_at_wins", 12))
    ct, t_s = 0, 0
    for rd in demo_rounds:
        rno = int(rd.get("round_no", 0))
        if max(ct, t_s) >= mp_at:
            match_point_rounds.add(rno)
        w = rd.get("winner", "")
        if w == "CT":
            ct += 1
        elif w == "T":
            t_s += 1

    scores = []
    for beat in beats:
        t = float(beat.get("when", {}).get("video_time", 0))
        rno = int(beat.get("when", {}).get("round_no", 0))
        h = sum(s * decay(abs(t - et)) for et, s in events)
        if rno in match_point_rounds:
            h *= mp_mult
        scores.append(round(min(h, 1.0), 3))
    return scores


def dominant_round_emotion(avg_hype: float) -> str:
    """Return dominant roundlevel emotion name from hype score (平淡/激动/尖叫)."""
    rules = load_hype_rules()
    em = rules["emotions"]
    # check from highest threshold down
    if avg_hype >= float(em["尖叫"]["threshold"]):
        return "尖叫"
    if avg_hype >= float(em["激动"]["threshold"]):
        return "激动"
    return "平淡"


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


_ANALYST_JSON_CONTRACT = (
    '严格输出单个 JSON 对象：{"scenes":[{"t_start":float,"t_end":float,"scene":str,"neutral":str}]}；'
    "t_start/t_end 填入 prompt 窗口列表中的对应值（不要自己算）；不加 markdown 代码块，不加任何额外文本。"
)


def build_scene_windows(
    beats: list[dict],
    start_sec: float,
    end_sec: float,
    window_max_sec: float = 10.0,
    window_min_sec: float = 3.0,
) -> list[tuple[float, float]]:
    """从 demo 事件锚点确定性切窗。

    算法：
    1. 收集所有击杀/植弹事件的 video_time 作为锚点。
    2. 锚点之间若间距 > window_max_sec，等距插补窗。
    3. 若相邻边界间距 < window_min_sec，合并到相邻窗。
    4. 返回有序、无缝覆盖 [start_sec, end_sec] 的窗口列表。
    5. beats 为空返回 [(start_sec, end_sec)]（单窗兜底）。
    """
    anchor_times: list[float] = []
    for beat in beats:
        t = float((beat.get("when") or {}).get("video_time", 0))
        ev = beat.get("events") or {}
        has_kill = any(not k.get("is_corpse_shoot") for k in (ev.get("kills") or []))
        has_bomb = bool((ev.get("c4") or {}).get("planted"))
        if has_kill or has_bomb:
            anchor_times.append(t)

    anchors: list[float] = sorted(set(
        t for t in anchor_times if start_sec <= t <= end_sec
    ))

    if not anchors:
        return [(start_sec, end_sec)]

    boundaries: list[float] = [start_sec]
    for a in anchors:
        if a > boundaries[-1]:   # 去重：锚点与前边界相同时跳过，避免零长窗
            boundaries.append(a)
    if end_sec > boundaries[-1]:
        boundaries.append(end_sec)

    # 等距插补过长间距
    expanded: list[float] = [boundaries[0]]
    for i in range(1, len(boundaries)):
        lo, hi = boundaries[i - 1], boundaries[i]
        gap = hi - lo
        if gap > window_max_sec:
            n_insert = int(gap / window_max_sec)
            step = gap / (n_insert + 1)
            for j in range(1, n_insert + 1):
                expanded.append(round(lo + j * step, 3))
        expanded.append(hi)

    # 合并过短窗（< window_min_sec）：把过短边界并入后一侧，不删边界
    merged: list[float] = [expanded[0]]
    for i in range(1, len(expanded)):
        gap = expanded[i] - merged[-1]
        is_last = (i == len(expanded) - 1)
        if gap < window_min_sec and not is_last:
            # 跳过该边界（并入后段），保证下一个更大间距消化掉当前短段
            continue
        merged.append(expanded[i])

    # 钳制 [start_sec, end_sec]，截掉超界边界，确保末边界 == end_sec
    merged = [b for b in merged if start_sec <= b <= end_sec]
    if not merged or merged[-1] != end_sec:
        merged.append(end_sec)

    windows = [(merged[i], merged[i + 1]) for i in range(len(merged) - 1)
               if merged[i] < merged[i + 1]]    # 滤掉零长/逆序对（防御）
    return windows if windows else [(start_sec, end_sec)]


def _build_analyst_system() -> str:
    """analyst 的 ollama system 字段 = 中性事实规则 + JSON 契约（非 6657 persona）。
    放进 system，ollama 截断输入时输出契约不丢（FIX-0）。"""
    return load_prompt("analyst_system") + "\n\n" + _ANALYST_JSON_CONTRACT


def _format_window_list(windows: list[tuple[float, float]], beats: list[dict]) -> str:
    """生成给 LLM 的窗口列表文本，每行标注窗内事件。"""
    lines = []
    for idx, (lo, hi) in enumerate(windows):
        events_in_window: list[str] = []
        for beat in beats:
            t = float((beat.get("when") or {}).get("video_time", 0))
            if not (lo <= t < hi):
                continue
            ev = beat.get("events") or {}
            for k in (ev.get("kills") or []):
                if k.get("is_corpse_shoot"):
                    continue
                attacker = k.get("attacker", "?")
                weapon = k.get("weapon", "?")
                victim = k.get("victim", "?")
                callout = k.get("callout") or ""
                loc = f"@{callout}" if callout else ""
                events_in_window.append(f"■击杀 {attacker}({weapon})→{victim}{loc}")
            c4 = ev.get("c4") or {}
            if c4.get("planted"):
                events_in_window.append("■C4 planted")
            if c4.get("begin_defuse_tick"):
                events_in_window.append("■拆弹开始")
        event_str = " / ".join(events_in_window) if events_in_window else "无"
        lines.append(f"窗{idx + 1}: t∈[{lo:.1f}, {hi:.1f}]  事件: {event_str}")
    return "\n".join(lines)


def _build_analyst_prompt(payload: dict, windows: list[tuple[float, float]] | None = None) -> str:
    """user prompt = 窗口列表 + 紧凑 JSON payload。"""
    template = load_prompt("analyst_round")
    beats = payload.get("keyframes", [])
    if windows is None:
        windows = [(float(payload.get("start_sec", 0)), float(payload.get("end_sec", 0)))]
    n = len(windows)
    window_list = _format_window_list(windows, beats)
    return (template
            .replace("{N}", str(n))
            .replace("{window_list}", window_list)
            .replace("{json_payload}", _dumps_compact(payload)))


# ── 切段机器（segment_long_rounds=true 时启用；切段+合并全在 phase3a 内部，下游透明） ──

def _is_cut_candidate(frames: list[dict], i: int) -> bool:
    """帧 i 是否可作段边界：phase 切换 / 空窗帧；绝不在含 kills/damages/c4 的事件帧上落刀。"""
    if _frame_is_event(frames[i]):
        return False
    prev_phase = (frames[i - 1].get("when", {}) or {}).get("phase")
    cur_phase = (frames[i].get("when", {}) or {}).get("phase")
    if cur_phase != prev_phase:
        return True
    return not frames[i].get("events")


def _segment_windows(frames: list[dict], budget: int, overlap: int) -> list[dict]:
    """贪心把帧切成 K 个归属窗。超 budget 时回退到最近候选切点落刀。
    返回 [{"lo":float,"hi":float,"frames":[...含前向 overlap 帧作上下文...]}]，
    lo/hi 为不含重叠的归属时间区间（按 when.video_time），相邻窗 [lo,hi) 无缝无叠。"""
    n = len(frames)
    if n == 0:
        return []

    def vtime(idx: int) -> float:
        return float((frames[idx].get("when", {}) or {}).get("video_time", 0.0))

    # 1) 贪心确定每段起始索引
    starts = [0]
    acc = 0.0
    last_cand: int | None = None
    i = 0
    while i < n:
        acc += len(_dumps_compact(frames[i])) / _CHARS_PER_TOKEN
        if acc > budget and i > starts[-1]:
            cut = last_cand if (last_cand is not None and last_cand > starts[-1]) else i
            starts.append(cut)
            last_cand = None
            acc = 0.0
            i = cut          # 从切点重新累计
            continue
        if i > starts[-1] and _is_cut_candidate(frames, i):
            last_cand = i
        i += 1

    # 2) 起始索引 → 归属窗（lo/hi/frames）
    bounds = starts + [n]
    windows = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        if s >= e:
            continue
        lo = vtime(s)
        hi = vtime(e) if e < n else float("inf")   # 末段 hi=inf，吃到回合结束
        ctx = max(0, s - overlap)
        windows.append({"lo": lo, "hi": hi, "frames": frames[ctx:e]})
    return windows


def _build_segment_prompt(seg_payload: dict, i: int, k: int, lo: float, hi: float,
                           windows: list[tuple[float, float]] | None = None,
                           state_so_far: str = "") -> str:
    """段 prompt = 分段元信息头（身份+时间窗）+ 跨段前情（i>1 时）+ 整局模板。
    state_so_far: _build_round_state_so_far() 输出，仅第 2 段起注入，不写入任何 JSON 产物。
    """
    hi_disp = float(seg_payload.get("end_sec", lo)) if hi == float("inf") else hi
    meta = (
        f"【分段说明】本局共切 {k} 段，当前第 {i} 段，时间窗 t∈[{lo:.0f},{hi_disp:.0f})。"
        f"只输出 t_start 落在本窗内的 scene；不写「本局开始/结束」类整局收尾语。"
    )
    if state_so_far:
        meta += f"\n\n【本局前情】（第 {i}/{k} 段，前段已发生）\n{state_so_far}"
    # 传入本段对应的子窗口（若无则用全段单窗兜底）
    seg_windows = windows or [(lo, hi_disp)]
    return meta + "\n\n" + _build_analyst_prompt(seg_payload, windows=seg_windows)


def _build_round_state_so_far(beats: list[dict]) -> str:
    """从已覆盖的 beats 提取前情摘要（击杀 + 植弹），注入分段 prompt 的跨段上下文。
    去重：同一 victim 的击杀只报告一次（防重叠帧重复报告）。
    仅注入 LLM-A 的 prompt，不出现在任何 JSON 产物中。
    """
    seen_victims: set[str] = set()
    kill_lines: list[str] = []
    bomb_planted = False

    for beat in beats:
        ev = beat.get("events") or {}
        for k in (ev.get("kills") or []):
            if k.get("is_corpse_shoot"):
                continue
            victim = str(k.get("victim", "?"))
            if victim in seen_victims:
                continue
            seen_victims.add(victim)
            attacker = k.get("attacker", "?")
            weapon = k.get("weapon", "?")
            callout = k.get("callout") or ""
            loc = f" 在 {callout}" if callout else ""
            kill_lines.append(f"- 击杀: {attacker}({weapon}){loc} 击杀 {victim}")
        c4 = ev.get("c4") or {}
        if c4.get("planted"):
            bomb_planted = True

    if not kill_lines and not bomb_planted:
        return ""

    parts = kill_lines[:]
    if bomb_planted:
        parts.append("- 炸弹已植入")
    return "\n".join(parts)


def _analyst_scenes_segmented(
    slim_full: dict,
    llm_cfg: dict,
    system: str,
    round_no: int,
    debug: bool,
    overlap: int,
    scene_windows: list[tuple[float, float]] | None = None,
) -> list[dict] | None:
    """逐段分析 + 归属窗合并。返回与 _parse_scenes_response 同形的 parsed_scenes，全段失败→None。
    scene_windows: build_scene_windows 产出的确定性窗口，用于精确分段 prompt 注入。
    """
    seg_windows = _segment_windows(slim_full.get("keyframes", []), _ANALYST_PROMPT_TOKEN_BUDGET, overlap)
    if not seg_windows:
        return None
    k = len(seg_windows)
    base = {key: slim_full[key] for key in ("round_no", "start_sec", "end_sec", "demo_round_hint") if key in slim_full}
    all_frames = slim_full.get("keyframes", [])
    merged: list[dict] = []
    covered_beats: list[dict] = []   # 已覆盖段的 beats，用于跨段前情
    for idx, w in enumerate(seg_windows):
        seg_payload = {**base, "keyframes": w["frames"]}
        # 跨段前情：第 2 段起注入前段已发生事件摘要
        state_so_far = ""
        if idx > 0 and covered_beats:
            state_so_far = _build_round_state_so_far(covered_beats)
        # 本分析段对应的确定性子窗口
        if scene_windows:
            # 只取落在本段时间范围内的子窗口
            lo_f, hi_f = w["lo"], w["hi"]
            sub_windows: list[tuple[float, float]] = [
                (a, b) for (a, b) in scene_windows if a < hi_f and b > lo_f
            ] or [(lo_f, hi_f if hi_f != float("inf") else float(base.get("end_sec", lo_f)))]
        else:
            lo_f = w["lo"]
            hi_f = w["hi"] if w["hi"] != float("inf") else float(base.get("end_sec", lo_f))
            sub_windows = [(lo_f, hi_f)]
        prompt = _build_segment_prompt(
            seg_payload, idx + 1, k, w["lo"], w["hi"],
            windows=sub_windows,
            state_so_far=state_so_far,
        )
        raw = _call_analyst(prompt, llm_cfg, system_prompt=system, round_no=round_no, debug=debug, seg=idx + 1)
        if raw == _ANALYST_FAILED:
            # 跳过失败段，但仍累计 covered_beats
            covered_beats.extend(w["frames"])
            continue
        for sc in (_parse_scenes_response(raw) or []):
            t0 = float(sc.get("t_start", -1))
            if w["lo"] <= t0 < w["hi"]:          # ★归属窗去重：重叠帧仅上下文，边界事件归前段
                merged.append(sc)
        covered_beats.extend(w["frames"])
    if not merged:
        return None
    merged.sort(key=lambda s: float(s.get("t_start", 0)))
    for a, b in zip(merged, merged[1:]):         # 单调缝合：t_end≤下个 t_start（守音画同步）
        if float(a.get("t_end", 0)) > float(b.get("t_start", 0)):
            a["t_end"] = b["t_start"]
    return merged


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


_ANALYST_FAILED = "__ANALYST_FAILED__"


# ── scene helpers ──

def _parse_scenes_response(text: str) -> list[dict] | None:
    """Try to parse LLM output as JSON scenes array. Returns None on failure."""
    stripped = text.strip()
    # 剥离 ```json ... ``` 围栏（预呓文常带）
    if "```" in stripped:
        m = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
        if m:
            stripped = m.group(1).strip()
    # 取最外层 { }（容忍"好的/我现在"类预呓文前后缀）
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        stripped = stripped[start:end + 1]
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and isinstance(data.get("scenes"), list):
            scenes = data["scenes"]
            if scenes and all(isinstance(s, dict) for s in scenes):
                return scenes
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _scene_hype(beats: list[dict], hypes: list[float], t_start: float, t_end: float) -> float:
    """Average hype for beats whose video_time falls in [t_start, t_end)."""
    vals = [
        h for b, h in zip(beats, hypes)
        if t_start <= float(b.get("when", {}).get("video_time", 0)) < t_end
    ]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _speech_rate_config() -> dict:
    rules = load_hype_rules()
    return rules.get("speech_rate", {"base_char_per_sec": 5.0, "emotion_speed_factor": {}})


_HYPE_EMOTION_TO_TTS = {"平淡": "平述", "激动": "激动", "尖叫": "惊叹"}


def _compute_char_budget(duration: float, hype_emotion: str, speech_rate: dict) -> int:
    """Chars budget = duration × base_char_per_sec × emotion_speed_factor."""
    base = float(speech_rate.get("base_char_per_sec", 5.0))
    tts_emotion = _HYPE_EMOTION_TO_TTS.get(hype_emotion, "平述")
    factor = float(speech_rate.get("emotion_speed_factor", {}).get(tts_emotion, 1.0))
    return max(20, int(duration * base * factor))


def _call_analyst(prompt: str, llm_cfg: dict, system_prompt: str | None = None, round_no: int = 0, debug: bool = False, seg: int = 0) -> str:
    try:
        from sbmachine.common import generate_commentary
        log_ctx = {"round": f"round{round_no}"}
        if seg > 0:
            log_ctx["scene"] = f"seg{seg}"
        result = generate_commentary(prompt, llm_cfg, system_prompt=system_prompt, log_ctx=log_ctx)
    except Exception as exc:
        print(f"[phase3a] round {round_no} analyst error: {exc}", file=sys.stderr)
        return _ANALYST_FAILED

    # ── debug dump ──
    if debug:
        debug_dir = _PROJECT_ROOT / "output" / "debug_phase3"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dump = {
            "round_no":  round_no,
            "seg":       seg,
            "model":     llm_cfg.get("model", ""),
            "phase":     "3a_analyst",
            "prompt":    prompt,
            "response":  result,
        }
        name = f"r{round_no:03d}_s{seg}_3a_analyst.json" if seg else f"r{round_no:03d}_3a_analyst.json"
        out = debug_dir / name
        out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    # ────────────────────────────────────────────────────────────────────────
    return result


# ── main runner ──

def run_phase3a(
    *,
    rounds_path: Path,
    output_path: Path,
    config_path: Path,
    demo_rounds_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    import os
    config = load_config(config_path)
    debug_enabled = bool(config.get("debug", {}).get("phase3", False) or os.getenv("AI6657_DEBUG_PHASE3"))
    llm_cfg = dict(config.get("llm", {}))
    analyst_model = config.get("semantic", {}).get("analyst_model") or config.get("semantic", {}).get("model", "")
    if analyst_model:
        llm_cfg["model"] = analyst_model
    # 分析模型输出 JSON scenes 数组（多 scene 较长），单独给较大上限封住失控生成。
    llm_cfg["max_tokens"] = int(config.get("semantic", {}).get("analyst_max_tokens", 3072))

    match = load_match(rounds_path)

    demo_rounds: list[dict] = []
    try:
        from sbmachine.common import resolve_path
        pd = resolve_path(config.get("demo", {}).get("parsed_dir", "output/demo"))
        if pd:
            if not demo_rounds and (pd / "rounds.json").exists():
                demo_rounds = json.loads((pd / "rounds.json").read_text(encoding="utf-8"))
    except Exception as exc:  # IO error or JSON corruption in optional demo file
        print(f"[phase3a] warning: could not load demo rounds.json: {exc}", file=sys.stderr)
    if demo_rounds_path and demo_rounds_path.exists():
        demo_rounds = json.loads(demo_rounds_path.read_text(encoding="utf-8"))

    tick_rate = 64.0
    try:
        from sbmachine.common import resolve_path
        pd = resolve_path(config.get("demo", {}).get("parsed_dir", "output/demo"))
        if pd and (pd / "demo_meta.json").exists():
            tick_rate = float(json.loads((pd / "demo_meta.json").read_text()).get("tick_rate", 64))
    except Exception as exc:  # IO error or JSON corruption in optional demo_meta.json
        print(f"[phase3a] warning: could not read demo_meta.json (tick_rate defaults to {tick_rate}): {exc}", file=sys.stderr)

    concurrent_rounds = max(1, int(config.get("semantic", {}).get("analyst_concurrent_rounds", 1)))
    analyst_system = _build_analyst_system()   # 常量，循环外算一次
    segment_on = bool(config.get("semantic", {}).get("segment_long_rounds", False))
    segment_overlap = max(0, int(config.get("semantic", {}).get("segment_overlap_frames", 2)))
    window_max_sec = float(config.get("semantic", {}).get("window_max_sec", 10.0))
    window_min_sec = float(config.get("semantic", {}).get("window_min_sec", 3.0))

    def _process_round(rnd) -> dict:
        payload = _semantic_payload(rnd)
        beats = payload.get("keyframes", [])      # 全帧 → compute_hype（不可降采样）
        hypes = compute_hype(beats, demo_rounds, tick_rate)

        # 取峰值而非均值定回合情绪：一局的情绪身份由它的最高光时刻决定，
        # 均值会被大量平淡 beat 稀释掉 ace/残局这种峰值（如纯走位局里一个1v3）。
        peak_hype = max(hypes) if hypes else 0.0
        avg_hype = round(sum(hypes) / len(hypes), 3) if hypes else 0.0
        round_emotion = dominant_round_emotion(peak_hype)
        speech_rate = _speech_rate_config()

        # ── 确定性切窗：由 demo 事件锚点决定，不依赖 LLM ──
        scene_wins = build_scene_windows(
            beats,
            rnd.start_sec,
            rnd.end_sec,
            window_max_sec=window_max_sec,
            window_min_sec=window_min_sec,
        )

        # 长回合决策树：三分支都产出与 _parse_scenes_response 同形的 parsed_scenes（或 None）。
        if dry_run:
            parsed_scenes = [{
                "t_start": rnd.start_sec, "t_end": rnd.end_sec,
                "scene": "full", "neutral": f"[dry-run] 第{rnd.round_no}局中性稿占位。",
            }]
        else:
            slim_full = _slim_payload_for_prompt(payload, downsample=False)   # 仅瘦字段不降帧
            est_tok = len(_dumps_compact(slim_full)) / _CHARS_PER_TOKEN
            if est_tok <= _ANALYST_PROMPT_TOKEN_BUDGET:
                # 绝大多数局：单次调用（全帧已落预算内）
                raw = _call_analyst(
                    _build_analyst_prompt(slim_full, windows=scene_wins), llm_cfg,
                    system_prompt=analyst_system, round_no=rnd.round_no, debug=debug_enabled)
                parsed_scenes = None if raw == _ANALYST_FAILED else _parse_scenes_response(raw)
            elif segment_on:
                # 超预算 + 开关 ON：切段无损，逐段分析后合并（附传确定性子窗口）
                parsed_scenes = _analyst_scenes_segmented(
                    slim_full, llm_cfg, analyst_system, rnd.round_no, debug_enabled,
                    segment_overlap, scene_windows=scene_wins)
            else:
                # 超预算 + 开关 OFF（默认）：二次压缩降采样，单次调用
                slim_ds = _slim_payload_for_prompt(payload)
                raw = _call_analyst(
                    _build_analyst_prompt(slim_ds, windows=scene_wins), llm_cfg,
                    system_prompt=analyst_system, round_no=rnd.round_no, debug=debug_enabled)
                parsed_scenes = None if raw == _ANALYST_FAILED else _parse_scenes_response(raw)

        # FIX-2+3：空/过短/预呓文/不可解析/全段失败 一律判失败，绝不把原始响应当 neutral。
        analyst_failed = parsed_scenes is None

        if parsed_scenes is not None:
            # ── 将 LLM 输出按确定性窗口对齐 ──
            # 建立 LLM 输出的 neutral 索引（按顺序，窗口数对齐时直接映射）
            llm_neutral: dict[int, str] = {}    # win_idx → neutral
            llm_scene: dict[int, str] = {}      # win_idx → scene name
            if len(parsed_scenes) == len(scene_wins):
                # LLM 按序输出，1-to-1 映射
                for wi, sc in enumerate(parsed_scenes):
                    llm_neutral[wi] = sc.get("neutral", "")
                    llm_scene[wi] = sc.get("scene", "")
            else:
                # 数量不匹配：按 t_start 最近窗匹配；同窗多 scene neutral 拼接
                win_starts = [w[0] for w in scene_wins]
                for sc in parsed_scenes:
                    t0 = float(sc.get("t_start", -1))
                    if not scene_wins:
                        break
                    wi = min(range(len(win_starts)), key=lambda i: abs(win_starts[i] - t0))
                    neu = sc.get("neutral", "")
                    if wi not in llm_neutral:
                        llm_neutral[wi] = neu
                        llm_scene[wi] = sc.get("scene", "")
                    elif neu.strip():
                        llm_neutral[wi] = llm_neutral[wi] + "。" + neu  # 同窗拼接

            scenes_out = []
            for wi, (t_start, t_end) in enumerate(scene_wins):
                duration = max(1.0, t_end - t_start)
                sc_hype = _scene_hype(beats, hypes, t_start, t_end)
                sc_emotion = dominant_round_emotion(sc_hype)
                char_budget = _compute_char_budget(duration, sc_emotion, speech_rate)
                neutral = llm_neutral.get(wi, "")

                # 事件窗兜底：LLM 漏写时自动生成 neutral
                if not neutral.strip():
                    fallback_parts: list[str] = []
                    for beat in beats:
                        t = float((beat.get("when") or {}).get("video_time", 0))
                        if not (t_start <= t < t_end):
                            continue
                        ev = beat.get("events") or {}
                        for k in (ev.get("kills") or []):
                            if k.get("is_corpse_shoot"):
                                continue
                            attacker = k.get("attacker", "?")
                            callout = k.get("callout") or ""
                            weapon = k.get("weapon", "?")
                            victim = k.get("victim", "?")
                            loc = f"在{callout}" if callout else ""
                            fallback_parts.append(f"{attacker}{loc}用{weapon}击杀{victim}")
                        c4 = ev.get("c4") or {}
                        if c4.get("planted"):
                            site = ""
                            for p in (beat.get("where", {}) or {}).get("players", []):
                                co = p.get("callout") or ""
                                if co:
                                    site = co
                                    break
                            fallback_parts.append(f"T方在{site}完成下包" if site else "T方完成下包")
                    if fallback_parts:
                        neutral = "【事件自动生成】" + "；".join(fallback_parts)

                scenes_out.append({
                    "t_start":    t_start,
                    "t_end":      t_end,
                    "scene":      llm_scene.get(wi, ""),
                    "neutral":    neutral,
                    "hype":       sc_hype,
                    "char_budget": char_budget,
                })
        else:
            # fallback: single scene covering full round
            duration = max(1.0, rnd.end_sec - rnd.start_sec)
            char_budget = _compute_char_budget(duration, round_emotion, speech_rate)
            scenes_out = [{
                "t_start":    rnd.start_sec,
                "t_end":      rnd.end_sec,
                "scene":      "full",
                "neutral":    "",   # 失败 → 空稿，phase3b 走占位分支；绝不落预呓文垃圾
                "hype":       peak_hype,
                "char_budget": char_budget,
            }]

        return {
            "round_no":        rnd.round_no,
            "start_sec":       rnd.start_sec,
            "end_sec":         rnd.end_sec,
            "demo_round_hint": rnd.demo_round_hint,
            "round_emotion":   round_emotion,
            "peak_hype":       peak_hype,
            "avg_hype":        avg_hype,
            "analyst_failed":  analyst_failed,
            "scenes":          scenes_out,
        }

    result_rounds = []
    with ThreadPoolExecutor(max_workers=concurrent_rounds) as pool:
        futures = {pool.submit(_process_round, rnd): rnd.round_no for rnd in match.rounds}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Phase3a analyst", unit="round"):
            result_rounds.append(fut.result())
    result_rounds.sort(key=lambda r: r["round_no"])

    if result_rounds and not dry_run:
        total = len(result_rounds)
        failed = sum(1 for r in result_rounds if r.get("analyst_failed"))
        print(f"[phase3a] analyst success {total - failed}/{total} rounds", file=sys.stderr)
        if failed == total or failed / total > 0.5:
            print(
                f"[phase3a] FATAL: {failed}/{total} rounds failed (>{50 if failed < total else 0}% threshold). "
                "Check LLM endpoint / API key. Exiting non-zero to prevent empty-shell output.",
                file=sys.stderr,
            )
            sys.exit(1)

    manifest = {
        "video_path": match.video_path,
        "map_name":   match.map_name,
        "model":      llm_cfg.get("model", ""),
        "rounds":     result_rounds,
    }
    write_json(output_path, manifest)
    return manifest
