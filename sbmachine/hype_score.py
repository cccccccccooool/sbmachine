"""硬事实强度与情绪档位计算：由击杀/炸弹/残血等硬事实算出每个 beat 的 hype 分并映射情绪。"""

from __future__ import annotations

import bisect
import math

from sbmachine.common import load_cs_game_rules, load_hype_rules


def _kill_id(kill: dict) -> tuple:
    return (
        kill.get("tick"),
        str(kill.get("attacker") or ""),
        str(kill.get("victim") or ""),
    )


def compute_hype(beats: list[dict]) -> list[float]:
    """逐 beat 计算硬事实强度，同一击杀去重、连杀计数不跨玩家膨胀。"""
    rules = load_hype_rules()
    tau = float(rules["decay_tau_sec"])
    base = rules["base_scores"]
    bonuses = rules["kill_flag_bonuses"]
    long_dist = float(rules.get("long_distance_threshold", 1000))

    def decay(dt: float) -> float:
        return math.exp(-dt / tau)

    events: list[tuple[float, float]] = []
    attacker_kill_count: dict[tuple[int, str], int] = {}
    seen_kills: set[tuple] = set()
    seen_objectives: set[tuple] = set()
    seen_low_health: set[tuple[int, str]] = set()
    exchange_kills: dict[int, list[tuple[float, str, tuple]]] = {}
    exchange_cfg = load_cs_game_rules().get("exchange", {})
    exchange_gap = float(exchange_cfg.get("max_gap_sec", 3.0))
    exchange_min = int(exchange_cfg.get("min_kills", 2))

    for beat in beats:
        when = beat.get("when", {}) or {}
        video_time = float(when.get("video_time", 0))
        round_no = int(when.get("round_no", 0))
        beat_events = beat.get("events") or {}

        for kill in beat_events.get("kills", []):
            if kill.get("is_corpse_shoot"):
                continue
            event_id = (round_no, *_kill_id(kill))
            if event_id in seen_kills:  # 同一击杀只计一次，防重复膨胀
                continue
            seen_kills.add(event_id)

            attacker = str(kill.get("attacker") or "")
            if attacker:
                counter_key = (round_no, attacker)
                attacker_kill_count[counter_key] = (
                    attacker_kill_count.get(counter_key, 0) + 1
                )
                count = attacker_kill_count[counter_key]
            else:
                count = 1
            score_key = f"kill_{min(count, 5)}k" if count >= 3 else "kill_single"
            score = float(base.get(score_key, base["kill_single"]))
            if kill.get("through_smoke"):
                score += float(bonuses.get("through_smoke", 0))
            if kill.get("no_scope"):
                score += float(bonuses.get("no_scope", 0))
            if kill.get("is_wallbang"):
                score += float(bonuses.get("is_wallbang", 0))
            if kill.get("attacker_blind"):
                score += float(bonuses.get("attacker_blind", 0))
            if float(kill.get("distance", 0) or 0) > long_dist:
                score += float(bonuses.get("long_distance", 0))
            events.append((video_time, score))
            exchange_kills.setdefault(round_no, []).append((video_time, attacker, event_id))

        c4 = beat_events.get("c4", {}) or {}
        if c4.get("planted"):
            event_id = ("bomb_plant", round_no, c4.get("plant_tick"))
            if event_id not in seen_objectives:
                seen_objectives.add(event_id)
                events.append((video_time, float(base["bomb_plant"])))
        if c4.get("begin_defuse_tick") and c4.get("defuser_has_kit") is False:
            event_id = ("no_kit_defuse", round_no, c4.get("begin_defuse_tick"))
            if event_id not in seen_objectives:
                seen_objectives.add(event_id)
                events.append((video_time, float(base["no_kit_defuse"])))
        for terminal_type, keys in (
            ("bomb_exploded", ("bomb_exploded_tick", "explode_tick", "exploded_tick")),
            ("bomb_defused", ("bomb_defused_tick", "defuse_tick", "defused_tick")),
        ):
            terminal_tick = next(
                (c4.get(key) for key in keys if c4.get(key) is not None), None
            )
            event_id = (terminal_type, round_no, terminal_tick)
            if terminal_tick is not None and event_id not in seen_objectives:
                seen_objectives.add(event_id)
                events.append((video_time, float(base.get(terminal_type, 1.0))))
        if str(when.get("phase") or "") == "post_round":
            event_id = ("round_end", round_no)
            if event_id not in seen_objectives:
                seen_objectives.add(event_id)
                events.append((video_time, float(base.get("round_end", 0.95))))

        for damage in beat_events.get("damages", []):
            victim = str(damage.get("victim") or "")
            low_health_id = (round_no, victim)
            if int(damage.get("health_after", 100)) <= 15 and (
                not victim or low_health_id not in seen_low_health
            ):
                if victim:
                    seen_low_health.add(low_health_id)
                events.append((video_time, float(base["low_blood"])))

        if c4.get("planted"):
            alive = {"T": 0, "CT": 0}
            for player in (beat.get("where") or {}).get("players") or []:
                side = str(player.get("side") or "").upper()
                hp = player.get("hp")
                if side in alive and isinstance(hp, (int, float)) and hp > 0:
                    alive[side] += 1
            clutch_id = ("clutch", round_no, c4.get("plant_tick"), tuple(sorted(alive.items())))
            t_alive, ct_alive = alive["T"], alive["CT"]
            man_disadvantage = (t_alive <= 2 and ct_alive >= 2) or (ct_alive <= 2 and t_alive >= 2)
            if man_disadvantage and clutch_id not in seen_objectives:
                seen_objectives.add(clutch_id)
                events.append((video_time, float(base.get("clutch", 0.6))))

    for round_no, round_events in exchange_kills.items():
        recent: list[tuple[float, str, tuple]] = []
        for video_time, attacker, event_id in sorted(round_events):
            recent.append((video_time, attacker, event_id))
            recent[:] = [item for item in recent if video_time - item[0] <= exchange_gap]
            if len(recent) >= exchange_min and len({name for _, name, _ in recent if name}) >= 2:
                exchange_id = ("exchange", round_no, tuple(item[2] for item in recent))
                if exchange_id not in seen_objectives:
                    seen_objectives.add(exchange_id)
                    score = exchange_cfg.get("critical_priority", 0.9) if len(recent) >= int(exchange_cfg.get("critical_min_kills", 3)) else exchange_cfg.get("key_priority", 0.72)
                    events.append((video_time, float(score)))

    # 按事件时刻排序后，用 bisect 定位 event_time <= video_time 的前缀切片，
    # 避免每个 beat 全表扫描（O(beats×events) -> O((beats+events)·log)）。
    events.sort(key=lambda e: e[0])
    event_times = [event_time for event_time, _ in events]

    scores = []
    for beat in beats:
        video_time = float((beat.get("when", {}) or {}).get("video_time", 0))
        # 只累计已发生（不晚于当前时刻）的事件，时间衰减后取峰值并夹到 [0, 1]
        hi = bisect.bisect_right(event_times, video_time)
        active = [
            score * decay(video_time - event_time) for event_time, score in events[:hi]
        ]
        scores.append(round(min(max(active, default=0.0), 1.0), 3))
    return scores


def dominant_round_emotion(avg_hype: float) -> str:
    """由回合平均硬事实强度得出主导情绪档位。"""
    emotions = load_hype_rules()["emotions"]
    if avg_hype >= float(emotions["尖叫"]["threshold"]):
        return "尖叫"
    if avg_hype >= float(emotions["激动"]["threshold"]):
        return "激动"
    return "平淡"


def _scene_hype(
    beats: list[dict], hypes: list[float], t_start: float, t_end: float
) -> float:
    """取 video_time 落在场景时间窗内的 beat 的硬事实强度峰值。"""
    scene_hypes = [
        hype
        for beat, hype in zip(beats, hypes)
        if t_start <= float((beat.get("when", {}) or {}).get("video_time", 0)) < t_end
    ]
    return round(max(scene_hypes), 3) if scene_hypes else 0.0


def _scene_scream_eligible(
    beats: list[dict],
    t_start: float,
    t_end: float,
    *,
    rapid_window_sec: float = 8.0,
) -> bool:
    """惊叹资格门：三杀连爆直通；两杀/单杀需凑够劣势/精彩加分项才放行。

    加分路径由 hype_rules.json 的 ``scream_gate`` 块驱动；配置缺失时 fail-closed，
    完全退回旧行为（仅三杀直通）。帧或字段缺失的加分项一律按"不成立"计，不猜。
    """
    gate = load_hype_rules().get("scream_gate")
    if not isinstance(gate, dict):
        gate = None
    pass_kill_count = int(gate.get("pass_kill_count", 3)) if gate else 3

    kills_by_attacker: dict[str, list[tuple[dict, float]]] = {}
    seen_kills: set[tuple] = set()
    window_beats: list[dict] = []
    window_start = t_start - rapid_window_sec
    for beat in beats:
        video_time = float((beat.get("when", {}) or {}).get("video_time", 0))
        if not (window_start <= video_time < t_end):
            continue
        window_beats.append(beat)
        for kill in (beat.get("events") or {}).get("kills") or []:
            if kill.get("is_corpse_shoot"):
                continue
            event_id = _kill_id(kill)
            if event_id in seen_kills:
                continue
            seen_kills.add(event_id)
            attacker = str(kill.get("attacker", "")).strip()
            if not attacker:
                continue
            kills_by_attacker.setdefault(attacker, []).append((kill, video_time))
            if len(kills_by_attacker[attacker]) >= pass_kill_count:
                return True  # 三杀直通：现行为不变

    if gate is None or not kills_by_attacker:
        return False

    # 加分放行路径：只看窗口内最大连杀数的攻击者（n==2 或 n==1）
    max_n = max(len(kills) for kills in kills_by_attacker.values())
    required = (gate.get("min_bonus") or {}).get(str(min(max_n, 2)))
    if required is None:
        return False
    factors = gate.get("bonus_factors") or {}
    for attacker, attacker_kills in kills_by_attacker.items():
        if len(attacker_kills) != max_n:
            continue
        if _scream_bonus_count(attacker, attacker_kills, window_beats, factors) >= int(
            required
        ):
            return True
    return False


def _nearest_players(window_beats: list[dict], video_time: float) -> list[dict]:
    """取窗口内 video_time 最接近且带选手列表的帧的 where.players；找不到返回空。"""
    best: list[dict] = []
    best_dt: float | None = None
    for beat in window_beats:
        players = (beat.get("where") or {}).get("players") or []
        if not players:
            continue
        dt = abs(float((beat.get("when", {}) or {}).get("video_time", 0)) - video_time)
        if best_dt is None or dt < best_dt:
            best, best_dt = players, dt
    return best


def _scream_bonus_count(
    attacker: str,
    attacker_kills: list[tuple[dict, float]],
    window_beats: list[dict],
    factors: dict,
) -> int:
    """统计该攻击者的加分项数，每个因子最多计 1 分；字段缺失一律按不成立计。"""
    # 每次击杀取最近 planning 帧的选手状态：(攻击者条目, 全体选手)
    states: list[tuple[dict | None, list[dict]]] = []
    for _kill, video_time in attacker_kills:
        players = _nearest_players(window_beats, video_time)
        me = next(
            (p for p in players if str(p.get("name") or "").strip() == attacker), None
        )
        states.append((me, players))

    count = 0

    # 无甲：任一击杀时刻攻击者 armor == 0
    if factors.get("no_armor") and any(
        me is not None and me.get("armor") == 0 for me, _ in states
    ):
        count += 1

    # 低血：任一击杀时刻攻击者 0 < hp <= low_hp_max
    low_hp_max = factors.get("low_hp_max")
    if isinstance(low_hp_max, (int, float)) and any(
        me is not None
        and isinstance(me.get("hp"), (int, float))
        and 0 < float(me["hp"]) <= float(low_hp_max)
        for me, _ in states
    ):
        count += 1

    # 仅副武器：本窗口该攻击者所有击杀的 weapon 都在名单内（大小写无关）
    sidearms = {str(w).strip().casefold() for w in factors.get("sidearm_only") or []}
    weapons = [
        str(kill.get("weapon") or "").strip().casefold() for kill, _ in attacker_kills
    ]
    if sidearms and weapons and all(w and w in sidearms for w in weapons):
        count += 1

    # 精彩flag：任一击杀带任一 highlight flag
    flags = factors.get("highlight_flags") or []
    if flags and any(
        any(kill.get(flag) for flag in flags) for kill, _ in attacker_kills
    ):
        count += 1

    # 人数劣势：同帧按实时 side 统计存活（hp>0），攻击者一方少 man_disadvantage_min 人以上。
    # 只用帧内实时 side 字段，不读 roster 首次观测值（换边后即错）。
    md_min = factors.get("man_disadvantage_min")
    if isinstance(md_min, (int, float)):
        for me, players in states:
            if me is None:
                continue
            side = str(me.get("side") or "").strip()
            if not side:
                continue
            alive = [
                p
                for p in players
                if isinstance(p.get("hp"), (int, float))
                and float(p["hp"]) > 0
                and str(p.get("side") or "").strip()
            ]
            own = sum(1 for p in alive if str(p["side"]).strip() == side)
            opp = len(alive) - own
            if opp - own >= float(md_min):
                count += 1
                break

    # 近身多敌：同帧敌方存活里与攻击者 x/y 平面距离 <= radius 的 >= near_enemies_min 人
    ne_min = factors.get("near_enemies_min")
    radius = factors.get("near_enemies_radius")
    if isinstance(ne_min, (int, float)) and isinstance(radius, (int, float)):
        for me, players in states:
            if me is None:
                continue
            side = str(me.get("side") or "").strip()
            if (
                not side
                or not isinstance(me.get("x"), (int, float))
                or not isinstance(me.get("y"), (int, float))
            ):
                continue
            near = 0
            for p in players:
                if (
                    str(p.get("side") or "").strip() == side
                    or not str(p.get("side") or "").strip()
                ):
                    continue
                if not (isinstance(p.get("hp"), (int, float)) and float(p["hp"]) > 0):
                    continue
                if not (
                    isinstance(p.get("x"), (int, float))
                    and isinstance(p.get("y"), (int, float))
                ):
                    continue
                if math.hypot(
                    float(p["x"]) - float(me["x"]), float(p["y"]) - float(me["y"])
                ) <= float(radius):
                    near += 1
            if near >= float(ne_min):
                count += 1
                break

    return count


def _speech_rate_config(config: dict | None = None) -> dict:
    rules = load_hype_rules()
    base = dict(rules.get(
        "speech_rate", {"base_char_per_sec": 5.0, "char_budget_factor": {}}
    ))
    semantic = config.get("semantic", {}) if isinstance(config, dict) else {}
    cfg = semantic.get("speech_rate") or {}
    if isinstance(cfg, dict):
        if "base_char_per_sec" in cfg:
            base["base_char_per_sec"] = float(cfg["base_char_per_sec"])
        factors = cfg.get("char_budget_factor")
        if isinstance(factors, dict):
            base["char_budget_factor"] = {**base.get("char_budget_factor", {}), **{k: float(v) for k, v in factors.items()}}
    return base


_HYPE_EMOTION_TO_TTS = {"平淡": "平述", "激动": "激动", "尖叫": "惊叹"}


def _compute_char_budget(duration: float, hype_emotion: str, speech_rate: dict) -> int:
    """字数预算 = 时长 × 基础语速 × 情绪字数系数（下限 8 字）。"""
    base = float(speech_rate.get("base_char_per_sec", 5.0))
    tts_emotion = _HYPE_EMOTION_TO_TTS.get(hype_emotion, "平述")
    factor = float(speech_rate.get("char_budget_factor", {}).get(tts_emotion, 1.0))
    return max(8, int(duration * base * factor))
