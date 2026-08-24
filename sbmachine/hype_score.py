"""硬事实强度与情绪档位计算：由击杀、炸弹、残血等硬事实计算每个时间片的热度分并映射情绪。"""

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
    """逐个时间片计算硬事实强度，同一击杀去重，连杀计数不跨玩家膨胀。"""
    rules = load_hype_rules()
    decay_tau_sec = float(rules["decay_tau_sec"])
    base_scores = rules["base_scores"]
    kill_flag_bonuses = rules["kill_flag_bonuses"]
    long_distance_threshold = float(rules.get("long_distance_threshold", 1000))

    def decay(elapsed_seconds: float) -> float:
        return math.exp(-elapsed_seconds / decay_tau_sec)

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
        timing = beat.get("when", {}) or {}
        video_time = float(timing.get("video_time", 0))
        round_number = int(timing.get("round_no", 0))
        beat_events = beat.get("events") or {}

        for kill in beat_events.get("kills", []):
            if kill.get("is_corpse_shoot"):
                continue
            event_id = (round_number, *_kill_id(kill))
            if event_id in seen_kills:  # 同一击杀只计一次，防重复膨胀
                continue
            seen_kills.add(event_id)

            attacker = str(kill.get("attacker") or "")
            if attacker:
                counter_key = (round_number, attacker)
                attacker_kill_count[counter_key] = (
                    attacker_kill_count.get(counter_key, 0) + 1
                )
                kill_count = attacker_kill_count[counter_key]
            else:
                kill_count = 1
            score_key = f"kill_{min(kill_count, 5)}k" if kill_count >= 3 else "kill_single"
            score = float(base_scores.get(score_key, base_scores["kill_single"]))
            if kill.get("through_smoke"):
                score += float(kill_flag_bonuses.get("through_smoke", 0))
            if kill.get("no_scope"):
                score += float(kill_flag_bonuses.get("no_scope", 0))
            if kill.get("is_wallbang"):
                score += float(kill_flag_bonuses.get("is_wallbang", 0))
            if kill.get("attacker_blind"):
                score += float(kill_flag_bonuses.get("attacker_blind", 0))
            if float(kill.get("distance", 0) or 0) > long_distance_threshold:
                score += float(kill_flag_bonuses.get("long_distance", 0))
            events.append((video_time, score))
            exchange_kills.setdefault(round_number, []).append((video_time, attacker, event_id))

        bomb_state = beat_events.get("c4", {}) or {}
        if bomb_state.get("planted"):
            event_id = ("bomb_plant", round_number, bomb_state.get("plant_tick"))
            if event_id not in seen_objectives:
                seen_objectives.add(event_id)
                events.append((video_time, float(base_scores["bomb_plant"])))
        if bomb_state.get("begin_defuse_tick") and bomb_state.get("defuser_has_kit") is False:
            event_id = ("no_kit_defuse", round_number, bomb_state.get("begin_defuse_tick"))
            if event_id not in seen_objectives:
                seen_objectives.add(event_id)
                events.append((video_time, float(base_scores["no_kit_defuse"])))
        for terminal_type, terminal_keys in (
            ("bomb_exploded", ("bomb_exploded_tick", "explode_tick", "exploded_tick")),
            ("bomb_defused", ("bomb_defused_tick", "defuse_tick", "defused_tick")),
        ):
            terminal_tick = next(
                (bomb_state.get(key) for key in terminal_keys if bomb_state.get(key) is not None), None
            )
            event_id = (terminal_type, round_number, terminal_tick)
            if terminal_tick is not None and event_id not in seen_objectives:
                seen_objectives.add(event_id)
                events.append((video_time, float(base_scores.get(terminal_type, 1.0))))
        if str(timing.get("phase") or "") == "post_round":
            event_id = ("round_end", round_number)
            if event_id not in seen_objectives:
                seen_objectives.add(event_id)
                events.append((video_time, float(base_scores.get("round_end", 0.95))))

        for damage in beat_events.get("damages", []):
            victim = str(damage.get("victim") or "")
            low_health_id = (round_number, victim)
            if int(damage.get("health_after", 100)) <= 15 and (
                not victim or low_health_id not in seen_low_health
            ):
                if victim:
                    seen_low_health.add(low_health_id)
                events.append((video_time, float(base_scores["low_blood"])))

        if bomb_state.get("planted"):
            alive_counts = {"T": 0, "CT": 0}
            for player in (beat.get("where") or {}).get("players") or []:
                side = str(player.get("side") or "").upper()
                health = player.get("hp")
                if side in alive_counts and isinstance(health, (int, float)) and health > 0:
                    alive_counts[side] += 1
            clutch_id = (
                "clutch",
                round_number,
                bomb_state.get("plant_tick"),
                tuple(sorted(alive_counts.items())),
            )
            terrorists_alive, counter_terrorists_alive = alive_counts["T"], alive_counts["CT"]
            has_man_disadvantage = (terrorists_alive <= 2 and counter_terrorists_alive >= 2) or (
                counter_terrorists_alive <= 2 and terrorists_alive >= 2
            )
            if has_man_disadvantage and clutch_id not in seen_objectives:
                seen_objectives.add(clutch_id)
                events.append((video_time, float(base_scores.get("clutch", 0.6))))

    for round_number, round_events in exchange_kills.items():
        recent_kills: list[tuple[float, str, tuple]] = []
        for video_time, attacker, event_id in sorted(round_events):
            recent_kills.append((video_time, attacker, event_id))
            recent_kills[:] = [
                item for item in recent_kills if video_time - item[0] <= exchange_gap
            ]
            if (
                len(recent_kills) >= exchange_min
                and len({attacker_name for _, attacker_name, _ in recent_kills if attacker_name}) >= 2
            ):
                exchange_id = (
                    "exchange",
                    round_number,
                    tuple(item[2] for item in recent_kills),
                )
                if exchange_id not in seen_objectives:
                    seen_objectives.add(exchange_id)
                    score = (
                        exchange_cfg.get("critical_priority", 0.9)
                        if len(recent_kills) >= int(exchange_cfg.get("critical_min_kills", 3))
                        else exchange_cfg.get("key_priority", 0.72)
                    )
                    events.append((video_time, float(score)))

    # 按事件时刻排序后，用 bisect 定位 event_time <= video_time 的前缀切片，
    # 避免每个 beat 全表扫描（O(beats×events) -> O((beats+events)·log)）。
    events.sort(key=lambda event: event[0])
    event_times = [event_time for event_time, _ in events]

    scores = []
    for beat in beats:
        video_time = float((beat.get("when", {}) or {}).get("video_time", 0))
        # 只累计已发生（不晚于当前时刻）的事件，时间衰减后取峰值并夹到 [0, 1]
        event_end_index = bisect.bisect_right(event_times, video_time)
        decayed_scores = [
            score * decay(video_time - event_time)
            for event_time, score in events[:event_end_index]
        ]
        scores.append(round(min(max(decayed_scores, default=0.0), 1.0), 3))
    return scores


def dominant_round_emotion(avg_hype: float) -> str:
    """由传入的回合硬事实强度得出主导情绪档位。"""
    emotions = load_hype_rules()["emotions"]
    if avg_hype >= float(emotions["尖叫"]["threshold"]):
        return "尖叫"
    if avg_hype >= float(emotions["激动"]["threshold"]):
        return "激动"
    return "平淡"


def _scene_hype(
    beats: list[dict], hypes: list[float], t_start: float, t_end: float
) -> float:
    """取场景时间窗内各时间片的硬事实强度峰值。"""
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

    加分路径由 hype_rules.json 的 ``scream_gate`` 配置驱动；配置缺失时默认拒绝，
    仅保留三杀直通。帧或字段缺失的加分项一律按"不成立"计，不作猜测。
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
                # 达到直通击杀数即可通过。
                return True

    if gate is None or not kills_by_attacker:
        return False

    # 加分放行路径：只看窗口内连杀数最多的攻击者（仅考虑两杀或单杀）。
    max_kill_count = max(len(kills) for kills in kills_by_attacker.values())
    required_bonus_count = (gate.get("min_bonus") or {}).get(
        str(min(max_kill_count, 2))
    )
    if required_bonus_count is None:
        return False
    bonus_factors = gate.get("bonus_factors") or {}
    for attacker_name, attacker_kills in kills_by_attacker.items():
        if len(attacker_kills) != max_kill_count:
            continue
        if _compute_scream_bonus_count(
            attacker_name, attacker_kills, window_beats, bonus_factors
        ) >= int(
            required_bonus_count
        ):
            return True
    return False


def _find_nearest_players(window_beats: list[dict], video_time: float) -> list[dict]:
    """取窗口内与目标时刻最接近且带有选手列表的帧；找不到返回空列表。"""
    nearest_players: list[dict] = []
    nearest_delta: float | None = None
    for beat in window_beats:
        players = (beat.get("where") or {}).get("players") or []
        if not players:
            continue
        time_delta = abs(float((beat.get("when", {}) or {}).get("video_time", 0)) - video_time)
        if nearest_delta is None or time_delta < nearest_delta:
            nearest_players, nearest_delta = players, time_delta
    return nearest_players


def _compute_scream_bonus_count(
    attacker_name: str,
    attacker_kills: list[tuple[dict, float]],
    window_beats: list[dict],
    bonus_factors: dict,
) -> int:
    """统计该攻击者的加分项数，每个因子最多计 1 分；字段缺失一律按不成立计。"""
    # 每次击杀取最近的规划帧选手状态：(攻击者条目、全体选手)。
    attacker_states: list[tuple[dict | None, list[dict]]] = []
    for _kill, video_time in attacker_kills:
        players = _find_nearest_players(window_beats, video_time)
        attacker_player = next(
            (
                player
                for player in players
                if str(player.get("name") or "").strip() == attacker_name
            ),
            None,
        )
        attacker_states.append((attacker_player, players))

    bonus_count = 0

    # 无甲：任一击杀时刻攻击者的护甲值为 0。
    if bonus_factors.get("no_armor") and any(
        attacker_player is not None and attacker_player.get("armor") == 0
        for attacker_player, _ in attacker_states
    ):
        bonus_count += 1

    # 低血：任一击杀时刻攻击者的生命值处于配置的低血上限内。
    low_health_max = bonus_factors.get("low_hp_max")
    if isinstance(low_health_max, (int, float)) and any(
        attacker_player is not None
        and isinstance(attacker_player.get("hp"), (int, float))
        and 0 < float(attacker_player["hp"]) <= float(low_health_max)
        for attacker_player, _ in attacker_states
    ):
        bonus_count += 1

    # 仅副武器：本窗口该攻击者的所有击杀都使用名单中的武器（不区分大小写）。
    sidearms = {
        str(weapon).strip().casefold()
        for weapon in bonus_factors.get("sidearm_only") or []
    }
    weapons = [
        str(kill.get("weapon") or "").strip().casefold() for kill, _ in attacker_kills
    ]
    if sidearms and weapons and all(
        weapon_name and weapon_name in sidearms for weapon_name in weapons
    ):
        bonus_count += 1

    # 精彩标记：任一击杀带有任一高光标记。
    highlight_flags = bonus_factors.get("highlight_flags") or []
    if highlight_flags and any(
        any(kill.get(flag) for flag in highlight_flags) for kill, _ in attacker_kills
    ):
        bonus_count += 1

    # 人数劣势：按同帧实时阵营统计存活人数，攻击者一方至少少指定人数。
    # 仅使用帧内实时阵营字段，不读首次观测的选手列表，避免换边后失真。
    man_disadvantage_min = bonus_factors.get("man_disadvantage_min")
    if isinstance(man_disadvantage_min, (int, float)):
        for attacker_player, players in attacker_states:
            if attacker_player is None:
                continue
            side = str(attacker_player.get("side") or "").strip()
            if not side:
                continue
            alive_players = [
                player
                for player in players
                if isinstance(player.get("hp"), (int, float))
                and float(player["hp"]) > 0
                and str(player.get("side") or "").strip()
            ]
            own_alive_count = sum(
                1 for player in alive_players if str(player["side"]).strip() == side
            )
            opponent_alive_count = len(alive_players) - own_alive_count
            if opponent_alive_count - own_alive_count >= float(man_disadvantage_min):
                bonus_count += 1
                break

    # 近身多敌：同帧敌方存活选手中，距离攻击者不超过半径的至少达到指定人数。
    near_enemies_min = bonus_factors.get("near_enemies_min")
    near_enemies_radius = bonus_factors.get("near_enemies_radius")
    if isinstance(near_enemies_min, (int, float)) and isinstance(near_enemies_radius, (int, float)):
        for attacker_player, players in attacker_states:
            if attacker_player is None:
                continue
            side = str(attacker_player.get("side") or "").strip()
            if (
                not side
                or not isinstance(attacker_player.get("x"), (int, float))
                or not isinstance(attacker_player.get("y"), (int, float))
            ):
                continue
            near_enemy_count = 0
            for player in players:
                if (
                    str(player.get("side") or "").strip() == side
                    or not str(player.get("side") or "").strip()
                ):
                    continue
                if not (
                    isinstance(player.get("hp"), (int, float))
                    and float(player["hp"]) > 0
                ):
                    continue
                if not (
                    isinstance(player.get("x"), (int, float))
                    and isinstance(player.get("y"), (int, float))
                ):
                    continue
                if math.hypot(
                    float(player["x"]) - float(attacker_player["x"]),
                    float(player["y"]) - float(attacker_player["y"]),
                ) <= float(near_enemies_radius):
                    near_enemy_count += 1
            if near_enemy_count >= float(near_enemies_min):
                bonus_count += 1
                break

    return bonus_count


def _speech_rate_config(config: dict | None = None) -> dict:
    rules = load_hype_rules()
    speech_rate = dict(
        rules.get(
            "speech_rate", {"base_char_per_sec": 5.0, "char_budget_factor": {}}
        )
    )
    semantic_config = config.get("semantic", {}) if isinstance(config, dict) else {}
    speech_rate_override = semantic_config.get("speech_rate") or {}
    if isinstance(speech_rate_override, dict):
        if "base_char_per_sec" in speech_rate_override:
            speech_rate["base_char_per_sec"] = float(
                speech_rate_override["base_char_per_sec"]
            )
        factor_overrides = speech_rate_override.get("char_budget_factor")
        if isinstance(factor_overrides, dict):
            speech_rate["char_budget_factor"] = {
                **speech_rate.get("char_budget_factor", {}),
                **{key: float(value) for key, value in factor_overrides.items()},
            }
    return speech_rate


_HYPE_EMOTION_TO_TTS = {"平淡": "平述", "激动": "激动", "尖叫": "惊叹"}


def _compute_char_budget(duration: float, hype_emotion: str, speech_rate: dict) -> int:
    """字数预算 = 时长 × 基础语速 × 情绪字数系数（下限 8 字）。"""
    base_char_per_sec = float(speech_rate.get("base_char_per_sec", 5.0))
    tts_emotion = _HYPE_EMOTION_TO_TTS.get(hype_emotion, "平述")
    char_budget_factor = float(
        speech_rate.get("char_budget_factor", {}).get(tts_emotion, 1.0)
    )
    return max(8, int(duration * base_char_per_sec * char_budget_factor))
