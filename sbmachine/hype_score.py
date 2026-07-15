"""硬事实强度与情绪档位计算：由击杀/炸弹/残血等硬事实算出每个 beat 的 hype 分并映射情绪。"""
from __future__ import annotations

import math

from sbmachine.common import load_hype_rules


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

    for beat in beats:
        when = beat.get("when", {}) or {}
        video_time = float(when.get("video_time", 0))
        round_no = int(when.get("round_no", 0))

        for kill in (beat.get("events", {}) or {}).get("kills", []):
            if kill.get("is_corpse_shoot"):
                continue
            event_id = (round_no, *_kill_id(kill))
            if event_id in seen_kills:  # 同一击杀只计一次，防重复膨胀
                continue
            seen_kills.add(event_id)

            attacker = str(kill.get("attacker") or "")
            if attacker:
                counter_key = (round_no, attacker)
                attacker_kill_count[counter_key] = attacker_kill_count.get(counter_key, 0) + 1
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

        c4 = (beat.get("events", {}) or {}).get("c4", {}) or {}
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

        for damage in (beat.get("events", {}) or {}).get("damages", []):
            victim = str(damage.get("victim") or "")
            low_health_id = (round_no, victim)
            if int(damage.get("health_after", 100)) <= 15 and (not victim or low_health_id not in seen_low_health):
                if victim:
                    seen_low_health.add(low_health_id)
                events.append((video_time, float(base["low_blood"])))

    scores = []
    for beat in beats:
        video_time = float((beat.get("when", {}) or {}).get("video_time", 0))
        # 只累计已发生（不晚于当前时刻）的事件，时间衰减后取峰值并夹到 [0, 1]
        active = [score * decay(video_time - event_time) for event_time, score in events if event_time <= video_time]
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


def _scene_hype(beats: list[dict], hypes: list[float], t_start: float, t_end: float) -> float:
    """取 video_time 落在场景时间窗内的 beat 的硬事实强度峰值。"""
    scene_hypes = [
        hype for beat, hype in zip(beats, hypes)
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
    """仅当单个玩家去重后达成三杀连爆，才允许该场景进入最高档（尖叫）。"""
    kills_by_attacker: dict[str, int] = {}
    seen_kills: set[tuple] = set()
    window_start = t_start - rapid_window_sec
    for beat in beats:
        video_time = float((beat.get("when", {}) or {}).get("video_time", 0))
        if not (window_start <= video_time < t_end):
            continue
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
            kills_by_attacker[attacker] = kills_by_attacker.get(attacker, 0) + 1
            if kills_by_attacker[attacker] >= 3:
                return True
    return False


def _speech_rate_config() -> dict:
    rules = load_hype_rules()
    return rules.get("speech_rate", {"base_char_per_sec": 5.0, "emotion_speed_factor": {}})


_HYPE_EMOTION_TO_TTS = {"平淡": "平述", "激动": "激动", "尖叫": "惊叹"}


def _compute_char_budget(duration: float, hype_emotion: str, speech_rate: dict) -> int:
    """字数预算 = 时长 × 基础语速 × 情绪语速系数（下限 20 字）。"""
    base = float(speech_rate.get("base_char_per_sec", 5.0))
    tts_emotion = _HYPE_EMOTION_TO_TTS.get(hype_emotion, "平述")
    factor = float(speech_rate.get("emotion_speed_factor", {}).get(tts_emotion, 1.0))
    return max(20, int(duration * base * factor))
