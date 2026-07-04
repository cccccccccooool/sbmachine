"""Hype score and emotion helpers."""
from __future__ import annotations

import math

from sbmachine.common import load_hype_rules


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

# 同构函数保留；后续可合并。
def _dominant_scene_emotion(hype: float) -> str:
    """Map scene hype score to emotion label (平淡/激动/尖叫). Same logic as phase3a."""
    em = load_hype_rules()["emotions"]
    if hype >= float(em["尖叫"]["threshold"]):
        return "尖叫"
    if hype >= float(em["激动"]["threshold"]):
        return "激动"
    return "平淡"


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

