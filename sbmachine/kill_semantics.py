"""保守的确定性击杀模式识别。

从击杀动作里识别"穿杀 / 扫射转火 / 武器压制 / 多杀 / 特殊击杀"等语义，
只在有几何或弹药证据支撑时才下结论（fail-closed），供解说话术选题使用。
"""

from __future__ import annotations

import math

from sbmachine.common import load_cs_game_rules
from sbmachine.rule_compare import rule_label, rule_priority
from sbmachine.scene_context import frame_time


def _normalize_weapon(value: object) -> str:
    """把武器名归一化：去掉 weapon_ 前缀、下划线转空格、大小写无关，便于比较。"""
    return str(value or "").replace("weapon_", "").replace("_", " ").strip().casefold()


def _normalize_weapons(values: list[object]) -> set[str]:
    return {_normalize_weapon(value) for value in values}


def _planar_vector(origin: list[float], target: list[float]) -> tuple[float, float]:
    """origin→target 的平面向量，只取 x/y（穿杀判定刻意忽略 z，不拿 z 当楼层代理）。"""
    return target[0] - origin[0], target[1] - origin[1]


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """两个平面向量的夹角（度）。"""
    na, nb = math.hypot(*a), math.hypot(*b)
    # 零长度向量没有方向，夹角无意义，直接当 0 度。
    if na <= 1e-6 or nb <= 1e-6:
        return 0.0
    # 点积除以模长得 cos；浮点误差可能让结果略微超出 [-1, 1]，夹紧后再取 acos 防止 domain error。
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cosine))


def _collateral(kills: list[dict], cfg: dict) -> bool:
    """判定是否为一枪穿杀：同一武器、几乎同一 tick、两名受害者近似在同一条射线上。"""
    if len(kills) != 2:
        return False
    first_kill, second_kill = kills[0], kills[1]
    if _normalize_weapon(first_kill.get("weapon")) != _normalize_weapon(second_kill.get("weapon")):
        return False
    try:
        # 两次击杀必须近乎同时发生，否则不可能是同一颗子弹穿过。
        if abs(int(first_kill["event_tick"]) - int(second_kill["event_tick"])) > int(
            cfg["collateral_tick_tolerance"]
        ):
            return False
        origin, first_victim, second_victim = (
            first_kill["attacker_pos"],
            first_kill["victim_pos"],
            second_kill["victim_pos"],
        )
        if not origin or not first_victim or not second_victim:
            return False
        ray, point = _planar_vector(origin, first_victim), _planar_vector(
            origin, second_victim
        )
        ray_len = math.hypot(*ray)
        # ray 是射手指向第一名受害者的方向；点积 <= 0 说明第二名受害者在身后，不可能被同一发子弹穿到。
        if ray_len <= 1e-6 or ray[0] * point[0] + ray[1] * point[1] <= 0:
            return False
        # |叉积| / ray_len 是第二名受害者到这条射线的垂直距离；在一个身位宽度内才算穿杀。
        return abs(ray[0] * point[1] - ray[1] * point[0]) / ray_len <= float(
            cfg["one_body_width_units"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ammo_trace(
    frames: list[dict], name: str, lo: float, hi: float, weapon: str
) -> list[int]:
    """在 [lo, hi] 时间窗内，按时间顺序收集某玩家该武器的弹药变化序列（去掉连续重复值）。"""
    ammo_trace: list[int] = []
    for frame in sorted(frames, key=frame_time):
        if not lo <= frame_time(frame) <= hi:
            continue
        for player in (frame.get("where") or {}).get("players") or []:
            if str(player.get("name")) != name or _normalize_weapon(
                player.get("weapon")
            ) != _normalize_weapon(weapon):
                continue
            try:
                ammo = int(player.get("ammo"))
            except (TypeError, ValueError):
                continue
            # 只记录数值发生变化的采样点，避免同一弹药数被重复写入。
            if not ammo_trace or ammo_trace[-1] != ammo:
                ammo_trace.append(ammo)
    return ammo_trace


def _spray_transfer(
    kills: list[dict], frames: list[dict], cfg: dict, automatic: set[str]
) -> bool:
    """判定是否为一次扫射转火：同一把自动武器连续击杀，中途大幅甩枪转向且弹药持续下降。"""
    if len(kills) < int(cfg["spray_transfer_min_kills"]):
        return False
    kill_times = [float(kill.get("event_time", 0.0)) for kill in kills]
    # 相邻击杀间隔过大就不是一梭子打完的连续压制。
    if any(
        kill_times[index] - kill_times[index - 1] > float(cfg["spray_transfer_max_kill_gap_sec"])
        for index in range(1, len(kill_times))
    ):
        return False
    weapon = _normalize_weapon(kills[0].get("weapon"))
    # 必须全程同一把、且属于可扫射转火的自动武器。
    if weapon not in automatic or any(
        _normalize_weapon(k.get("weapon")) != weapon for k in kills
    ):
        return False
    origin = kills[0].get("attacker_pos")
    positions = [kill.get("victim_pos") for kill in kills]
    if not origin or any(not pos for pos in positions):
        return False
    # 相邻受害者相对射手的夹角，代表这一梭子里的甩枪幅度。
    angle_turns = [
        _angle_between(
            _planar_vector(origin, positions[i - 1]),
            _planar_vector(origin, positions[i]),
        )
        for i in range(1, len(positions))
    ]
    if not angle_turns or max(angle_turns) < float(cfg["spray_transfer_min_angle_deg"]):
        return False
    ammo_trace = _ammo_trace(
        frames,
        str(kills[0].get("attacker", "")),
        float(kills[0].get("event_time", 0)) - 1.0,
        float(kills[-1].get("event_time", 0)) + 1.0,
        str(kills[0].get("weapon", "")),
    )
    # 弹药必须单调不增（没换弹/没停火），且总消耗达到阈值，才能证明是一梭子连续扫过来的。
    if len(ammo_trace) < 2 or any(ammo_trace[i] > ammo_trace[i - 1] for i in range(1, len(ammo_trace))):
        return False
    return ammo_trace[0] - ammo_trace[-1] >= int(cfg["spray_transfer_min_ammo_drop"])


def build_kill_topics(
    actions: list[dict], frames: list[dict], *, round_kills: list[dict] | None = None
) -> list[dict]:
    """按射手把相邻击杀聚成簇，只给有证据支撑的语义打标签。"""
    rules = load_cs_game_rules()
    cfg, classes = rules["kill_semantics"], rules["weapon_classes"]
    rule_cfg = rules.get("rule_compare", {})
    automatic = _normalize_weapons(classes["automatic_for_spray_transfer"])
    long_guns, pistols = _normalize_weapons(
        classes["long_guns_for_mismatch"]
    ), _normalize_weapons(classes["pistols"])
    gap = float(rules["scene"]["kill_chain_gap_sec"])
    kills = sorted(
        (a for a in actions if a.get("type") == "kill"),
        key=lambda k: float(k.get("event_time", 0)),
    )
    clusters: list[list[dict]] = []
    # 先按射手分组，再在每个射手内部按时间间隔切簇：间隔超过 gap 就断开成新的一串连杀。
    by_attacker: dict[str, list[dict]] = {}
    for kill in kills:
        by_attacker.setdefault(str(kill.get("attacker") or ""), []).append(kill)
    for attacker_kills in by_attacker.values():
        current: list[dict] = []
        for kill in attacker_kills:
            if (
                current
                and float(kill.get("event_time", 0))
                - float(current[-1].get("event_time", 0))
                > gap
            ):
                clusters.append(current)
                current = []
            current.append(kill)
        if current:
            clusters.append(current)
    # round_kills 提供整局视角，用来判断本簇是否包含本回合的首杀 / 残局收尾。
    all_kills = sorted(
        round_kills or kills, key=lambda k: float(k.get("event_time", 0))
    )
    first_tick = all_kills[0].get("event_tick") if all_kills else None
    last_tick = all_kills[-1].get("event_tick") if all_kills else None
    topics = []
    for cluster in clusters:
        collateral = _collateral(cluster, cfg)
        # 穿杀与扫射转火互斥：已判定为穿杀就不再做扫射转火的重活。
        spray = (
            False if collateral else _spray_transfer(cluster, frames, cfg, automatic)
        )
        mismatch = any(
            _normalize_weapon(k.get("weapon")) in long_guns
            and _normalize_weapon(k.get("victim_weapon")) in pistols
            for k in cluster
        )
        rule_tags = sorted(
            {
                str(tag)
                for kill in cluster
                for tag in (kill.get("rule_tags") or [])
                if tag
            },
            key=lambda tag: (-rule_priority(tag, rule_cfg), tag),
        )
        primary_rule = rule_tags[0] if rule_tags else None
        rule_strength = rule_priority(primary_rule, rule_cfg)
        rule_confidence = max(
            (
                float(kill.get("rule_confidence") or 0.0)
                for kill in cluster
                if kill.get("primary_rule") == primary_rule
            ),
            default=0.0,
        )
        observed_roles = {
            str(kill.get("pov_role") or "unavailable") for kill in cluster
        }
        pov_role = next(
            (
                role
                for role in ("killer", "victim", "observer", "unavailable")
                if role in observed_roles
            ),
            "unavailable",
        )
        pov_player = next(
            (
                str(kill.get("pov_player"))
                for kill in cluster
                if kill.get("pov_player")
            ),
            "",
        )
        round_tags = sorted(
            {
                str(tag)
                for kill in cluster
                for tag in (kill.get("round_tags") or [])
                if tag
            }
        )
        round_context = next(
            (
                kill.get("round_context")
                for kill in cluster
                if isinstance(kill.get("round_context"), dict)
            ),
            {},
        )
        special = any(
            k.get(key)
            for k in cluster
            for key in ("through_smoke", "no_scope", "is_wallbang", "attacker_blind")
        ) or bool(rule_tags)
        # 语义优先级从高到低：穿杀 > 扫射转火 > 单杀规则亮点 > 武器压制 > 多杀 > 普通击杀。
        if collateral:
            semantic, phrase, priority, confidence = "collateral", "串完了", 1.0, 0.95
        elif spray:
            semantic, phrase, priority, confidence = (
                "spray_transfer",
                "一个扫转",
                0.95,
                0.9,
            )
        elif (
            len(cluster) == 1
            and primary_rule is not None
            and rule_strength >= float(rule_cfg.get("rule_highlight_min_priority", 0.7))
        ):
            semantic, phrase, priority, confidence = (
                "rule_highlight",
                rule_label(primary_rule),
                rule_strength,
                rule_confidence,
            )
        elif mismatch:
            semantic, phrase, priority, confidence = "weapon_mismatch", "特", 0.78, 0.95
        elif len(cluster) >= 2:
            semantic, phrase, priority, confidence = "multi_kill", None, 0.82, 1.0
        elif special:
            semantic, phrase, priority, confidence = "special_kill", None, 0.75, 1.0
        else:
            semantic, phrase, priority, confidence = "plain_kill", None, 0.4, 1.0
        opening = first_tick is not None and cluster[0].get("event_tick") == first_tick
        final = last_tick is not None and cluster[-1].get("event_tick") == last_tick
        # 首杀 / 残局收尾自带看点，给一个优先级下限。
        if opening or final:
            priority = max(priority, 0.68)
        context_priorities = rule_cfg.get("round_context_priorities", {})
        if isinstance(context_priorities, dict):
            for tag in round_tags:
                try:
                    priority = max(priority, float(context_priorities.get(tag, 0.0)))
                except (TypeError, ValueError):
                    continue
        bonuses = rule_cfg.get("pov_priority_bonus", {})
        try:
            pov_bonus = float(bonuses.get(pov_role, 0.0)) if isinstance(bonuses, dict) else 0.0
        except (TypeError, ValueError):
            pov_bonus = 0.0
        priority = max(0.0, min(1.0, priority + pov_bonus))
        topics.append(
            {
                "type": "kill_topic",
                "semantic": semantic,
                "suggested_phrase": phrase,
                "priority": priority,
                "confidence": confidence,
                "narrate": priority >= float(cfg["narrate_priority_threshold"]),
                "opening_kill": opening,
                "final_kill": final,
                "attacker": cluster[0].get("attacker"),
                "victims": [k.get("victim") for k in cluster],
                "weapon": cluster[0].get("weapon"),
                "rule_tags": rule_tags,
                "primary_rule": primary_rule,
                "pov_role": pov_role,
                "pov_player": pov_player,
                "round_tags": round_tags,
                "round_context": round_context,
                "kills": cluster,
            }
        )
    # 高优先级在前；同优先级按最早击杀时间排，保证输出稳定。
    return sorted(
        topics,
        key=lambda topic: (
            -float(topic["priority"]),
            float(topic["kills"][0].get("event_time", 0)),
        ),
    )


def build_exchange_topics(
    actions: list[dict],
    frames: list[dict],
    *,
    result_state: dict | None = None,
) -> list[dict]:
    """聚合跨攻击者的连续击杀；只陈述事件序列与结果，不推断补枪等因果。"""
    rules = load_cs_game_rules()
    cfg = rules.get("exchange", {})
    gap = float(cfg.get("max_gap_sec", 3.0))
    minimum = int(cfg.get("min_kills", 2))
    kills = sorted(
        (action for action in actions if action.get("type") == "kill"),
        key=lambda action: float(action.get("event_time", 0.0)),
    )
    clusters: list[list[dict]] = []
    current: list[dict] = []
    for kill in kills:
        if (
            current
            and float(kill.get("event_time", 0.0))
            - float(current[-1].get("event_time", 0.0))
            > gap
        ):
            clusters.append(current)
            current = []
        current.append(kill)
    if current:
        clusters.append(current)

    topics: list[dict] = []
    for cluster in clusters:
        attackers = {
            str(kill.get("attacker") or "") for kill in cluster if kill.get("attacker")
        }
        if len(cluster) < minimum or len(attackers) < 2:
            continue
        participants: list[str] = []
        for kill in cluster:
            for key in ("attacker", "victim"):
                name = str(kill.get(key) or "")
                if name and name not in participants:
                    participants.append(name)
        event_ids = [
            str(
                kill.get("event_id")
                or f"kill:{kill.get('event_tick')}:{kill.get('attacker') or ''}:{kill.get('victim') or ''}"
            )
            for kill in cluster
        ]
        kill_count = len(cluster)
        topics.append(
            {
                "type": "exchange_topic",
                "event_ids": event_ids,
                "kill_count": kill_count,
                "participants": participants,
                "result_state": result_state or {},
                "priority_class": (
                    "critical_exchange"
                    if kill_count >= int(cfg.get("critical_min_kills", 3))
                    else "key_exchange"
                ),
                "priority": float(
                    cfg.get("critical_priority", 0.9)
                    if kill_count >= int(cfg.get("critical_min_kills", 3))
                    else cfg.get("key_priority", 0.72)
                ),
                "kills": cluster,
            }
        )
    return sorted(
        topics,
        key=lambda topic: (
            -float(topic["priority"]),
            -float(topic["kills"][-1].get("event_time", 0.0)),
        ),
    )
