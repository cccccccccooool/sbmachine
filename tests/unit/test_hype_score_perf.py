"""对拍测试：优化后的 compute_hype（bisect）必须与朴素线性扫描实现逐值等价。"""
import math
import random

from sbmachine.common import load_cs_game_rules, load_hype_rules
from sbmachine.hype_score import _kill_id, compute_hype


def _compute_hype_naive(beats: list[dict]) -> list[float]:
    """优化前的朴素参考实现：内层对 events 全表过滤 event_time <= video_time。

    事件构建逻辑与生产实现逐行一致，仅峰值扫描保持 O(beats×events) 线性过滤，
    作为对拍 oracle。
    """
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

        for kill in (beat.get("events", {}) or {}).get("kills", []):
            if kill.get("is_corpse_shoot"):
                continue
            event_id = (round_no, *_kill_id(kill))
            if event_id in seen_kills:
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
            exchange_kills.setdefault(round_no, []).append(
                (video_time, attacker, event_id)
            )

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

        for damage in (beat.get("events", {}) or {}).get("damages", []):
            victim = str(damage.get("victim") or "")
            low_health_id = (round_no, victim)
            if int(damage.get("health_after", 100)) <= 15 and (not victim or low_health_id not in seen_low_health):
                if victim:
                    seen_low_health.add(low_health_id)
                events.append((video_time, float(base["low_blood"])))

    for round_no, round_events in exchange_kills.items():
        recent: list[tuple[float, str, tuple]] = []
        for video_time, attacker, event_id in sorted(round_events):
            recent.append((video_time, attacker, event_id))
            recent[:] = [item for item in recent if video_time - item[0] <= exchange_gap]
            if (
                len(recent) >= exchange_min
                and len({name for _, name, _ in recent if name}) >= 2
            ):
                exchange_id = (
                    "exchange",
                    round_no,
                    tuple(item[2] for item in recent),
                )
                if exchange_id not in seen_objectives:
                    seen_objectives.add(exchange_id)
                    score = (
                        exchange_cfg.get("critical_priority", 0.9)
                        if len(recent)
                        >= int(exchange_cfg.get("critical_min_kills", 3))
                        else exchange_cfg.get("key_priority", 0.72)
                    )
                    events.append((video_time, float(score)))

    scores = []
    for beat in beats:
        video_time = float((beat.get("when", {}) or {}).get("video_time", 0))
        active = [score * decay(video_time - event_time) for event_time, score in events if event_time <= video_time]
        scores.append(round(min(max(active, default=0.0), 1.0), 3))
    return scores


def _random_beats(rng: random.Random, n_beats: int) -> list[dict]:
    players = ["A", "B", "C", "D", "E"]
    beats = []
    for i in range(n_beats):
        # video_time 故意非单调：随机时刻，允许重复/乱序
        video_time = round(rng.uniform(0.0, 60.0), 3)
        round_no = rng.randint(1, 3)
        events: dict = {}
        if rng.random() < 0.6:
            kills = []
            for _ in range(rng.randint(1, 3)):
                kills.append({
                    "tick": rng.randint(1, 50),
                    "attacker": rng.choice(players),
                    "victim": rng.choice(players),
                    "through_smoke": rng.random() < 0.2,
                    "no_scope": rng.random() < 0.2,
                    "is_wallbang": rng.random() < 0.2,
                    "attacker_blind": rng.random() < 0.2,
                    "is_corpse_shoot": rng.random() < 0.15,
                    "distance": round(rng.uniform(0, 2500), 1),
                })
            events["kills"] = kills
        if rng.random() < 0.3:
            events["c4"] = {
                "planted": rng.random() < 0.7,
                "plant_tick": rng.randint(1, 50),
                "begin_defuse_tick": rng.randint(0, 50),
                "defuser_has_kit": rng.choice([True, False]),
            }
        if rng.random() < 0.4:
            damages = []
            for _ in range(rng.randint(1, 3)):
                damages.append({
                    "victim": rng.choice(players + [""]),
                    "health_after": rng.randint(0, 100),
                })
            events["damages"] = damages
        beats.append({"when": {"video_time": video_time, "round_no": round_no}, "events": events})
    return beats


def test_compute_hype_matches_naive_oracle_random():
    rng = random.Random(20260715)
    for trial in range(200):
        beats = _random_beats(rng, rng.randint(0, 40))
        expected = _compute_hype_naive(beats)
        actual = compute_hype(beats)
        assert actual == expected, f"trial={trial} mismatch\nexpected={expected}\nactual={actual}"


def test_compute_hype_matches_naive_oracle_ties_and_nonmonotonic():
    # 大量相同 video_time / 事件时刻打平，验证 bisect_right 对等值的右界包含语义
    beats = [
        {"when": {"video_time": 5.0, "round_no": 1}, "events": {"kills": [{"tick": 1, "attacker": "A", "victim": "B"}]}},
        {"when": {"video_time": 5.0, "round_no": 1}, "events": {"kills": [{"tick": 2, "attacker": "A", "victim": "C"}]}},
        {"when": {"video_time": 5.0, "round_no": 1}, "events": {}},
        {"when": {"video_time": 2.0, "round_no": 1}, "events": {}},
        {"when": {"video_time": 5.0, "round_no": 1}, "events": {}},
    ]
    assert compute_hype(beats) == _compute_hype_naive(beats)


def test_compute_hype_empty():
    assert compute_hype([]) == _compute_hype_naive([]) == []
