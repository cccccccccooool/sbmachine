"""纯代码解说规划层：在 demo 事实与 LLM 措辞之间做话题选择与约束。"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

from sbmachine.common import count_spoken_chars, load_cs_game_rules
from sbmachine.kill_semantics import build_exchange_topics, build_kill_topics
from sbmachine.rule_compare import rule_label
from sbmachine.scene_context import (
    SceneWindow,
    _tick_event_time,
    extract_actions,
    frame_time,
    owns_time,
)
from sbmachine.spatial_context import resolve_spatial_context


@dataclass
class PlannerState:
    """跨窗口累计状态：限制平淡单杀数量、记录已解说过的击杀，避免重复。"""

    plain_kills_used: int = 0
    narrated_kill_ids: set[tuple] = field(default_factory=set)
    narrated_event_ids: set[str] = field(default_factory=set)
    seen_transition_ids: set[str] = field(default_factory=set)
    position_anchors_seen: set[str] = field(default_factory=set)
    active_tactic_rule_ids: set[str] = field(default_factory=set)
    utility_windows_used: int = 0
    attacker_round_kills: dict[str, int] = field(default_factory=dict)


def _kill_id(kill: dict) -> tuple:
    """用（tick, 攻击者, 受害者）唯一标识一次击杀，用于跨窗口去重。"""
    return (kill.get("event_tick"), kill.get("attacker"), kill.get("victim"))


# 同一回合内所有窗口都传入同一份 all_round_frames，逐窗口重算整回合击杀表纯属浪费。
# 缓存持有该 list 的对象引用并用 is 比较：既保证身份判定正确（避免 id 被 GC 复用后
# 误命中陈旧数据），又让被缓存对象保持存活；回合切换换新 frames 时用 is 判失效重算。
_ROUND_KILLS_CACHE: tuple[list[dict], list[dict]] | None = None


def _round_kills(all_round_frames: list[dict]) -> list[dict]:
    """返回整回合的击杀动作，按 all_round_frames 对象身份缓存，避免逐窗口重算。"""
    global _ROUND_KILLS_CACHE
    if _ROUND_KILLS_CACHE is not None and _ROUND_KILLS_CACHE[0] is all_round_frames:
        return _ROUND_KILLS_CACHE[1]
    round_actions = extract_actions(all_round_frames, float("-inf"), float("inf"))
    round_kills = [action for action in round_actions if action.get("type") == "kill"]
    _ROUND_KILLS_CACHE = (all_round_frames, round_kills)
    return round_kills


def _kill_summary(topic: dict) -> str:
    """把击杀话题转成一句中文摘要，按语义区分串杀/扫转/枪械压制。"""
    attacker = str(topic.get("attacker") or "进攻方")
    victims = (
        "、".join(str(value) for value in topic.get("victims") or [] if value) or "对手"
    )
    semantic = str(topic.get("semantic") or "plain_kill")
    if semantic == "collateral":
        return f"{attacker}一次击杀{victims}，形成串杀"
    if semantic == "spray_transfer":
        return f"{attacker}连续击杀{victims}，完成扫转"
    if semantic == "weapon_mismatch":
        return f"{attacker}在长枪对手枪的交火中击杀{victims}"
    if len(topic.get("victims") or []) >= 2:
        return f"{attacker}连续击杀{victims}"
    victim = str((topic.get("victims") or ["对手"])[0] or "对手")
    primary_rule = str(topic.get("primary_rule") or "")
    pov_is_victim = (
        topic.get("pov_role") == "victim"
        and str(topic.get("pov_player") or "") == victim
    )
    if semantic == "rule_highlight" and primary_rule:
        if pov_is_victim:
            victim_templates = {
                "air_noscope": f"{victim}被{attacker}空中盲狙击杀",
                "jump_kill": f"{victim}被{attacker}腾空击杀",
                "victim_airborne": f"{victim}在空中被{attacker}击落",
                "backstab": f"{victim}背身被{attacker}抓住并击杀",
                "unaware_kill": f"{victim}尚未发现{attacker}就被击杀",
                "blind_kill": f"{victim}被致盲状态下的{attacker}击杀",
                "through_smoke": f"{victim}被{attacker}隔烟击杀",
                "wallbang": f"{victim}被{attacker}穿墙击杀",
                "no_scope": f"{victim}被{attacker}盲狙击杀",
                "flick_shot": f"{victim}被{attacker}快速转向击杀",
                "one_tap": f"{victim}被{attacker}单发击杀",
                "caught_switching": f"{victim}切换装备时被{attacker}抓住",
                "point_blank": f"{victim}被{attacker}近距离击杀",
                "long_range": f"{victim}被{attacker}远距离击杀",
            }
            return victim_templates.get(
                primary_rule, f"{victim}被{attacker}击杀（{rule_label(primary_rule)}）"
            )
        attacker_templates = {
            "air_noscope": f"{attacker}空中盲狙击杀{victim}",
            "jump_kill": f"{attacker}腾空击杀{victim}",
            "victim_airborne": f"{attacker}击落空中的{victim}",
            "backstab": f"{attacker}抓住{victim}背身完成击杀",
            "unaware_kill": f"{attacker}在{victim}尚未发现自己时完成击杀",
            "blind_kill": f"{attacker}在致盲状态下击杀{victim}",
            "through_smoke": f"{attacker}隔烟击杀{victim}",
            "wallbang": f"{attacker}穿墙击杀{victim}",
            "no_scope": f"{attacker}盲狙击杀{victim}",
            "flick_shot": f"{attacker}快速转向击杀{victim}",
            "one_tap": f"{attacker}单发击杀{victim}",
            "caught_switching": f"{attacker}抓住{victim}切换装备的时机完成击杀",
            "point_blank": f"{attacker}近距离击杀{victim}",
            "long_range": f"{attacker}远距离击杀{victim}",
        }
        return attacker_templates.get(
            primary_rule, f"{attacker}击杀{victim}（{rule_label(primary_rule)}）"
        )
    if pov_is_victim:
        return f"{victim}被{attacker}击杀"
    return f"{attacker}击杀{victims}"


def _public_action(action: dict) -> dict:
    """裁剪动作字典为对外可见字段，剔除坐标等内部定位信息。"""
    if action.get("type") == "kill_topic":
        return {
            key: action.get(key)
            for key in (
                "type",
                "semantic",
                "suggested_phrase",
                "priority",
                "confidence",
                "opening_kill",
                "final_kill",
                "attacker",
                "victims",
                "weapon",
                "rule_tags",
                "primary_rule",
                "pov_role",
                "pov_player",
                "round_tags",
                "round_context",
            )
            if action.get(key) is not None
        }
    if action.get("type") == "exchange_topic":
        return {
            key: action.get(key)
            for key in (
                "type",
                "event_ids",
                "kill_count",
                "participants",
                "result_state",
                "priority_class",
            )
            if action.get(key) is not None
        }
    return {
        key: value
        for key, value in action.items()
        if key not in {"attacker_pos", "victim_pos", "throw_position", "destination"}
    }


_HARD_FACT_TYPES = frozenset(
    {
        "kill",
        "bomb_planted",
        "defuse_started",
        "bomb_exploded",
        "bomb_defused",
        "round_end",
        "team_eliminated",
        "utility_throw",
        "utility_effect",
        "effective_flash",
    }
)
_TERMINAL_TYPES = frozenset(
    {"bomb_exploded", "bomb_defused", "round_end", "team_eliminated"}
)


def _event_id(action: dict) -> str:
    if action.get("event_id"):
        return str(action["event_id"])
    kind = str(action.get("type") or "unknown")
    if kind == "kill":
        return f"kill:{action.get('event_tick')}:{action.get('attacker') or ''}:{action.get('victim') or ''}"
    return f"{kind}:{action.get('event_tick')}"


def _result_state(frames: list[dict]) -> dict:
    representative = next(
        (
            frame
            for frame in reversed(frames)
            if (frame.get("where") or {}).get("players")
        ),
        None,
    )
    result = {
        "T": {"alive_count": 0, "hp_total": 0, "players": []},
        "CT": {"alive_count": 0, "hp_total": 0, "players": []},
    }
    if representative is None:
        return result
    for player in (representative.get("where") or {}).get("players") or []:
        side = str(player.get("side") or "").upper()
        if side not in result:
            continue
        try:
            hp = max(0, int(player.get("hp") or 0))
        except (TypeError, ValueError):
            hp = 0
        result[side]["hp_total"] += hp
        result[side]["alive_count"] += int(hp > 0)
        if hp > 0:
            result[side]["players"].append(
                {"name": str(player.get("name") or ""), "hp": hp}
            )
    return result


def _state_text(result_state: dict, *, per_player: bool = False) -> str:
    t, ct = result_state.get("T") or {}, result_state.get("CT") or {}
    if per_player:
        parts = []
        for label, team in (("T", t), ("CT", ct)):
            players = team.get("players") or []
            if not players:
                continue
            names = "、".join(
                f"{p.get('name', '')}{p.get('hp', 0)}血"
                for p in players
                if p.get("name")
            )
            parts.append(f"{label}方{names}")
        return "，".join(parts)
    return f"T方{t.get('alive_count', 0)}人、CT方{ct.get('alive_count', 0)}人"


def _alive_state(frames: list[dict]) -> dict:
    """从帧内 players 重建某时刻的 T/CT 存活快照（与 _result_state 同构）。"""
    representative = next(
        (
            frame
            for frame in reversed(frames)
            if (frame.get("where") or {}).get("players")
        ),
        None,
    )
    result = {
        "T": {"alive_count": 0, "hp_total": 0, "players": []},
        "CT": {"alive_count": 0, "hp_total": 0, "players": []},
    }
    if representative is None:
        return result
    for player in (representative.get("where") or {}).get("players") or []:
        side = str(player.get("side") or "").upper()
        if side not in result:
            continue
        try:
            hp = max(0, int(player.get("hp") or 0))
        except (TypeError, ValueError):
            hp = 0
        result[side]["hp_total"] += hp
        result[side]["alive_count"] += int(hp > 0)
        if hp > 0:
            result[side]["players"].append(
                {"name": str(player.get("name") or ""), "hp": hp}
            )
    return result


def _state_summary(now: dict, before: dict) -> str | None:
    """当窗口内 T/CT 存活数发生实质变化时，确定性重建局面摘要；否则返回 None。"""
    t_now, t_before = now.get("T") or {}, before.get("T") or {}
    ct_now, ct_before = now.get("CT") or {}, before.get("CT") or {}
    t_alive, ct_alive = t_now.get("alive_count", 0), ct_now.get("alive_count", 0)
    if t_alive == (t_before.get("alive_count", 0)) and ct_alive == (ct_before.get("alive_count", 0)):
        return None
    if t_alive == 0 or ct_alive == 0:
        return None
    return f"T方{t_alive}人、CT方{ct_alive}人"


def _exchange_summary(topic: dict, char_budget: int | None) -> str:
    count = int(topic.get("kill_count") or 0)
    compact = f"交换，{count}人阵亡"
    result = topic.get("result_state") or {}
    t, ct = result.get("T") or {}, result.get("CT") or {}
    total_alive = int(t.get("alive_count", 0)) + int(ct.get("alive_count", 0))
    if char_budget is not None and char_budget <= 13:
        return compact
    if char_budget is not None and char_budget <= 25:
        return f"双方交换，T{(t.get('alive_count', 0))}打CT{(ct.get('alive_count', 0))}"
    # 残局：双方存活总数少且都有人，报 per-player 血量
    if 0 < total_alive <= 4:
        return f"双方连续交换，窗口结束时{_state_text(result, per_player=True)}"
    return f"双方连续交换，窗口结束时{_state_text(result)}"


def _kill_summary_budgeted(topic: dict, char_budget: int | None) -> str:
    summary = _kill_summary(topic)
    if char_budget is None or char_budget >= 26:
        return summary
    attacker = str(topic.get("attacker") or "进攻方")
    count = len(topic.get("victims") or [])
    if count >= 2:
        return f"{attacker}{count}杀"
    victim = str((topic.get("victims") or ["对手"])[0] or "对手")
    if topic.get("pov_role") == "victim" and topic.get("pov_player") == victim:
        return f"{victim}被{attacker}击杀"
    return f"{attacker}击杀{victim}"


def _kill_summary_with_streak(
    topic: dict, char_budget: int | None, prior_kills: int
) -> str:
    """在预算允许时，给本回合累计击杀数 >=2 的击杀话题加连杀弧线前缀。

    只陈述已确认事实（该攻击者本回合击杀总数），不新增强度、因果或战术。
    """
    summary = _kill_summary_budgeted(topic, char_budget)
    streak = prior_kills + len(topic.get("victims") or [])
    if streak < 2:
        return summary
    suffix = f"，本回合已连杀{streak}杀"
    if char_budget is not None and count_spoken_chars(summary) + count_spoken_chars(suffix) > char_budget:
        return summary
    return f"{summary}{suffix}"


def _terminal_summary(action: dict) -> str:
    kind = action.get("type")
    if kind == "bomb_exploded":
        return "C4爆炸，回合结束"
    if kind == "bomb_defused":
        return "C4已拆除，回合结束"
    if kind == "team_eliminated":
        return f"{action.get('side') or '一方'}被清零，回合结束"
    winner = action.get("winner")
    return f"{winner}方赢下回合" if winner else "回合结束"


def _required_facts(
    selected_candidates: list[dict], spatial: dict | None = None
) -> list[dict]:
    """将正式选中的一至两项候选编译为必保事实。"""
    result: list[dict] = []
    kind_counts: dict[str, int] = {}
    for candidate in selected_candidates:
        selected_action = candidate.get("action")
        canonical = str(candidate.get("summary") or "")
        if candidate.get("kind") == "silence" or not canonical:
            continue
        kind = str(candidate.get("kind") or "topic")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        suffix = "" if kind_counts[kind] == 1 else f":{kind_counts[kind]}"
        anchors: dict[str, list] = {
            key: []
            for key in ("players", "teams", "numbers", "events", "results", "locations", "weapons")
        }

        def add(anchor_kind: str, value: object) -> None:
            if value is not None and value != "" and value not in anchors[anchor_kind]:
                anchors[anchor_kind].append(value)

        if re.search(r"(?<![A-Za-z])CT(?:方)?(?![A-Za-z])", canonical, re.I):
            add("teams", "CT")
        if re.search(r"(?<![A-Za-z])T(?:方)?(?![A-Za-z])", canonical, re.I):
            add("teams", "T")
        for value in re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?", canonical):
            add("numbers", float(value) if "." in value else int(value))

        action_type = str((selected_action or {}).get("type") or "")
        if action_type == "kill_topic":
            add("events", "kill")
            for name in [(selected_action or {}).get("attacker"), *((selected_action or {}).get("victims") or [])]:
                if name and str(name) in canonical:
                    add("players", str(name))
        elif action_type == "exchange_topic":
            add("events", "kill_exchange")
        elif action_type in {"bomb_planted", "defuse_started", "bomb_exploded", "bomb_defused"}:
            add("events", action_type)
            if action_type in {"bomb_exploded", "bomb_defused"}:
                add("events", "round_end")
            if action_type == "defuse_started" and "CT" in canonical:
                add("teams", "CT")
        elif action_type == "team_eliminated":
            add("events", "team_eliminated")
            add("events", "round_end")
            add("results", "team_eliminated")
        elif action_type == "round_end":
            add("events", "round_end")
            if "赢" in canonical or "获胜" in canonical:
                add("results", "round_won")
        elif action_type in {"utility_throw", "utility_effect", "effective_flash"}:
            add("events", action_type)
            for key in ("attacker", "thrower", "victim"):
                name = (selected_action or {}).get(key)
                if name and str(name) in canonical:
                    add("players", str(name))
        elif kind == "position" and spatial:
            anchor = spatial.get("anchor") or {}
            name = str(anchor.get("name") or "")
            side = str(anchor.get("side") or "").upper()
            callout = anchor.get("callout_zh") or anchor.get("callout") or ""
            if name and name in canonical:
                add("players", name)
            if side in {"T", "CT"}:
                add("teams", side)
            if callout and str(callout) in canonical:
                add("locations", str(callout))

        result.append({
            "fact_id": f"topic:{kind}{suffix}",
            "type": kind,
            "canonical_text": canonical,
            "required": True,
            "anchors": anchors,
        })
    return result


_UTILITY_NAMES = {
    "smoke grenade": ("烟雾弹", "烟"),
    "smoke": ("烟雾弹", "烟"),
    "incendiary grenade": ("燃烧弹", "火"),
    "molotov": ("燃烧弹", "火"),
    "flashbang": ("闪光弹", "闪"),
    "flash": ("闪光弹", "闪"),
    "he grenade": ("手雷", "雷"),
    "high explosive grenade": ("手雷", "雷"),
    "decoy grenade": ("诱饵弹", "诱饵"),
}


def _budgeted_text(candidates: list[str], char_budget: int | None) -> str:
    """优先返回自然长句；预算不足时逐级降级，仍超限则返回最短候选供门禁拒绝。"""
    unique = list(dict.fromkeys(candidates))
    if char_budget is None:
        return unique[0]
    return next(
        (text for text in unique if count_spoken_chars(text) <= char_budget),
        min(unique, key=count_spoken_chars),
    )


def _utility_summary(action: dict, char_budget: int | None = None) -> str:
    """将道具类动作转成按窗口预算降级的自然语言描述。"""
    actor = action.get("attacker") or action.get("thrower") or "一名选手"
    if action.get("type") == "effective_flash":
        return f"{actor}的闪光让{action.get('victim') or '对手'}有效致盲{action.get('duration_s')}秒"
    utility = str(action.get("utility") or "道具")
    full_name, short_name = _UTILITY_NAMES.get(utility.casefold(), (utility, utility))
    if action.get("type") == "utility_effect":
        return _budgeted_text([
            f"{actor}的{full_name}已生效",
            f"{actor}的{short_name}生效",
        ], char_budget)
    return _budgeted_text([
        f"{actor}投出{full_name}",
        f"{actor}投{full_name}",
        f"{actor}投{short_name}",
    ], char_budget)


def _spatial_summary(spatial: dict) -> str:
    """把空间上下文转成一句摘要：锚点选手所在点位及附近队友/敌人。

    无人工审核地图模板时只报单人原始 callout（不推断附近关系）；
    缺 callout 或锚点不可信时返回空串（不输出“某人当前是关注对象”这类空话）。
    """
    anchor = spatial.get("anchor") or {}
    if not anchor:
        return ""
    name = str(anchor.get("name") or "一名选手")
    side = str(anchor.get("side") or "")
    subject = f"{name}（{side}）" if side else name
    callout = anchor.get("callout_zh") or anchor.get("callout") or ""
    if spatial.get("map_precision") != "reviewed_graph":
        if spatial.get("anchor_source") not in {"pov", "isolated_opposite"}:
            return ""
        if not callout:
            return ""
        return f"{subject}位于{callout}"

    summary = f"{subject}位于{callout}" if callout else ""
    nearby = spatial.get("nearby") or {}
    teammates = [
        str(item.get("name"))
        for item in nearby.get("teammates", [])
        if item.get("name")
    ]
    enemies = [
        str(item.get("name")) for item in nearby.get("enemies", []) if item.get("name")
    ]
    if teammates:
        summary += "，附近有队友" + "、".join(teammates)
    if enemies:
        summary += "，附近有敌人" + "、".join(enemies)
    return summary


_SOFT_CANDIDATE_KINDS = frozenset({"utility", "position", "state", "tactic"})
_PHASE_FOCUS = {
    "战术": ("utility", "position"),
    "下包前": ("position",),
    "下包后": ("position",),
    "结束": (),
}


def _narrative_phase(
    owned_frames: list[dict], owned_actions: list[dict], terminal_actions: list[dict]
) -> str | None:
    """根据当前窗口已确认事实确定内部叙事阶段；缺相对时间时保持旧软排序。"""
    if terminal_actions:
        return "结束"
    latest = (owned_frames or [{}])[-1]
    c4 = (latest.get("events") or {}).get("c4") or {}
    if c4.get("planted") or any(action.get("type") == "bomb_planted" for action in owned_actions):
        return "下包后"
    if any(action.get("type") == "kill" for action in owned_actions):
        return "下包前"
    return "战术"


def _candidate_sort_key(candidate: dict, narrative_phase: str | None) -> tuple:
    """硬事实保持原 rank；阶段只重排软候选。"""
    kind = str(candidate.get("kind") or "")
    rank = float(candidate.get("rank") or 0)
    if kind not in _SOFT_CANDIDATE_KINDS or narrative_phase is None:
        return (0, rank, -float(candidate.get("strength") or 0), -float(candidate.get("time") or 0))
    focus = _PHASE_FOCUS[narrative_phase]
    focus_rank = focus.index(kind) if kind in focus else len(focus)
    return (1, focus_rank, rank, -float(candidate.get("strength") or 0), -float(candidate.get("time") or 0))


def _utility_relation_matches(primary: dict, supporter: dict) -> bool:
    first, second = primary.get("action") or {}, supporter.get("action") or {}
    first_id, second_id = first.get("stable_event_id"), second.get("stable_event_id")
    if first_id and second_id:
        return first_id == second_id
    first_id, second_id = first.get("entity_id"), second.get("entity_id")
    return first_id is not None and second_id is not None and first_id == second_id


def _is_related_supporter(primary: dict, supporter: dict, narrative_phase: str | None) -> bool:
    """仅放行计划列出的关联组合，拒绝排序相邻的机械拼接。"""
    primary_kind, supporter_kind = primary.get("kind"), supporter.get("kind")
    primary_action = primary.get("action") or {}
    supporter_action = supporter.get("action") or {}
    if primary.get("event_ids", set()) & supporter.get("event_ids", set()):
        return False
    if primary_kind == supporter_kind == "utility" and narrative_phase == "战术":
        return True
    if primary_kind == supporter_kind == "utility":
        return _utility_relation_matches(primary, supporter)
    hard_types = {"bomb_planted", "defuse_started", "bomb_exploded", "bomb_defused", "round_end", "team_eliminated"}
    primary_hard = primary_action.get("type") in hard_types
    supporter_hard = supporter_action.get("type") in hard_types
    combat = {"kill", "exchange"}
    terminal_types = {"bomb_exploded", "bomb_defused", "round_end", "team_eliminated"}
    if primary_hard and supporter_kind in combat:
        return primary_action.get("type") not in terminal_types or supporter_kind == "kill"
    if supporter_hard and primary_kind in combat:
        return supporter_action.get("type") not in terminal_types or primary_kind == "kill"
    position_side = str(supporter.get("side") or "").upper()
    if supporter_kind == "position":
        if narrative_phase == "下包前" and primary_kind in combat:
            return position_side == "T"
        if narrative_phase == "下包后" and primary_hard:
            return position_side == "CT"
    if primary_kind == "position":
        position_side = str(primary.get("side") or "").upper()
        if narrative_phase == "下包前" and supporter_kind in combat:
            return position_side == "T"
        if narrative_phase == "下包后" and supporter_hard:
            return position_side == "CT"
    return False


def _select_candidates(
    candidates: list[dict], narrative_phase: str | None, char_budget: int | None
) -> list[dict]:
    """选择一个主事实和最多一个符合阶段且未超预算的相关事实。"""
    ordered = sorted(candidates, key=lambda item: _candidate_sort_key(item, narrative_phase))
    if not ordered:
        return []
    primary = ordered[0]
    for supporter in ordered[1:]:
        if not _is_related_supporter(primary, supporter, narrative_phase):
            continue
        if "position" in {primary.get("kind"), supporter.get("kind")} and char_budget is None:
            continue
        joined_chars = count_spoken_chars(primary["summary"]) + 1 + count_spoken_chars(supporter["summary"])
        if char_budget is not None and joined_chars > char_budget:
            continue
        return [primary, supporter]
    return [primary]


def plan_window(
    map_name: str | None,
    window: SceneWindow,
    ownership_frames: list[dict],
    context_frames: list[dict],
    all_round_frames: list[dict],
    state: PlannerState,
    *,
    tactic_hint: dict | None = None,
    is_last_window: bool = False,
    char_budget: int | None = None,
) -> dict:
    """生成 CommentaryPlan v2：先记全硬事实，再按动态优先级选择唯一主题。"""
    rules = load_cs_game_rules()
    priority_cfg = rules.get("topic_priority", {})

    def priority_rank(name: str, fallback: int) -> int:
        return int(priority_cfg.get(name, fallback))

    round_actions = extract_actions(
        all_round_frames, float("-inf"), float("inf"), include_end=True
    )
    actions = [
        action
        for action in round_actions
        if owns_time(
            float(action.get("event_time", 0)),
            window.context_start,
            window.context_end,
            include_end=is_last_window,
        )
    ]
    owned_actions = [
        action
        for action in round_actions
        if owns_time(
            float(action.get("event_time", 0)),
            window.t_start,
            window.t_end,
            include_end=is_last_window,
        )
    ]
    previous_transition_ids = set(state.seen_transition_ids)
    owned_frames = sorted(
        [
            frame
            for frame in all_round_frames
            if owns_time(
                frame_time(frame),
                window.t_start,
                window.t_end,
                include_end=is_last_window,
            )
        ],
        key=frame_time,
    ) or sorted(ownership_frames, key=frame_time)
    spatial = resolve_spatial_context(
        map_name, window.scene, owned_frames or context_frames, owned_actions
    )
    local_actions = spatial.pop("local_actions", [])
    local_ids = {_event_id(action) for action in local_actions}
    result_state = _result_state(owned_frames or context_frames)

    round_kills = _round_kills(all_round_frames)
    owned_kills = [action for action in owned_actions if action.get("type") == "kill"]
    owned_kill_ids = {_kill_id(action) for action in owned_kills}
    kill_topics = [
        topic
        for topic in build_kill_topics(actions, context_frames, round_kills=round_kills)
        if any(_kill_id(kill) in owned_kill_ids for kill in topic.get("kills") or [])
    ]
    exchange_topics = build_exchange_topics(
        owned_kills, owned_frames or context_frames, result_state=result_state
    )

    terminal_actions = [
        action for action in owned_actions if action.get("type") in _TERMINAL_TYPES
    ]
    if not any(
        action.get("type") in {"bomb_exploded", "bomb_defused"}
        for action in terminal_actions
    ):
        environment_kills = [
            kill for kill in owned_kills if not str(kill.get("attacker") or "").strip()
        ]
        if environment_kills and any(
            bool((((frame.get("events") or {}).get("c4") or {}).get("planted")))
            for frame in all_round_frames
            if frame_time(frame) <= float(environment_kills[-1].get("event_time", 0))
        ):
            kill = environment_kills[-1]
            terminal_actions.append(
                {
                    "type": "bomb_exploded",
                    "event_tick": kill.get("event_tick"),
                    "event_time": kill.get("event_time"),
                    "event_id": f"bomb_exploded:{kill.get('event_tick')}",
                }
            )
    observed_sides = {
        str(player.get("side") or "").upper()
        for frame in owned_frames[-1:]
        for player in ((frame.get("where") or {}).get("players") or [])
    }
    if owned_kills and {"T", "CT"}.issubset(observed_sides):
        for side in ("T", "CT"):
            other = "CT" if side == "T" else "T"
            if (
                result_state[side]["alive_count"] == 0
                and result_state[other]["alive_count"] > 0
            ):
                last_kill = owned_kills[-1]
                terminal_actions.append(
                    {
                        "type": "team_eliminated",
                        "side": side,
                        "event_tick": last_kill.get("event_tick"),
                        "event_time": last_kill.get("event_time"),
                        "event_id": f"team_eliminated:{last_kill.get('event_tick')}:{side}",
                    }
                )
                break

    narrative_phase = _narrative_phase(owned_frames, owned_actions, terminal_actions)

    candidates: list[dict] = []
    for action in terminal_actions:
        if _event_id(action) in previous_transition_ids:
            continue
        candidates.append(
            {
                "rank": priority_rank("terminal", 0),
                "strength": (
                    1.1
                    if action.get("type") in {"bomb_exploded", "bomb_defused"}
                    else 1.0
                ),
                "time": float(action.get("event_time", 0)),
                "kind": "round_end",
                "summary": _terminal_summary(action),
                "action": action,
                "event_ids": {_event_id(action)},
                "priority_class": "terminal",
            }
        )
    for topic in exchange_topics:
        candidates.append(
            {
                "rank": (
                    priority_rank("critical_exchange", 1)
                    if int(topic.get("kill_count") or 0) >= 3
                    else priority_rank("key_exchange", 2)
                ),
                "strength": float(topic.get("priority") or 0),
                "time": float((topic.get("kills") or [{}])[-1].get("event_time", 0)),
                "kind": "exchange",
                "summary": _exchange_summary(topic, char_budget),
                "action": topic,
                "event_ids": set(topic.get("event_ids") or []),
                "priority_class": topic.get("priority_class"),
            }
        )
    max_plain = int(rules["narration"]["max_plain_single_kills_per_round"])
    for topic in kill_topics:
        topic_ids = {_event_id(kill) for kill in topic.get("kills") or []}
        if topic_ids & state.narrated_event_ids:
            continue
        count = len(topic.get("kills") or [])
        semantic = topic.get("semantic")
        if semantic == "plain_kill" and state.plain_kills_used >= max_plain:
            continue
        rank = (
            priority_rank("critical_exchange", 1)
            if count >= 3
            else (
                priority_rank("key_exchange", 2)
                if count >= 2
                else priority_rank("ordinary_kill", 3)
            )
        )
        candidates.append(
            {
                "rank": rank,
                "strength": float(topic.get("priority") or 0),
                "time": float((topic.get("kills") or [{}])[-1].get("event_time", 0)),
                "kind": "kill",
                "summary": _kill_summary_with_streak(
                    topic, char_budget, state.attacker_round_kills.get(str(topic.get("attacker") or ""), 0)
                ),
                "action": topic,
                "event_ids": topic_ids,
                "priority_class": "multi_kill" if rank <= 2 else "ordinary_kill",
            }
        )
    for action in owned_actions:
        if action.get("type") not in {"bomb_planted", "defuse_started"}:
            continue
        if _event_id(action) in previous_transition_ids:
            continue
        summary = "C4已安装" if action.get("type") == "bomb_planted" else "CT开始拆弹"
        candidates.append(
            {
                "rank": priority_rank("key_exchange", 2),
                "strength": 0.7,
                "time": float(action.get("event_time", 0)),
                "kind": "retake",
                "summary": summary,
                "action": action,
                "event_ids": {_event_id(action)},
                "priority_class": "bomb_transition",
            }
        )
    # 4.2 已在更早窗口实际发生、当时被更高优先级话题压过而从未播报的
    # C4 transition，允许从当前已确认状态补报一次。判断看 narrated_event_ids
    # （真正播报过才算数），不看“是否曾在输入里见过”。terminal 已结束则不补。
    latest_frame = (owned_frames or context_frames or [{}])[-1]
    c4_state = (latest_frame.get("events") or {}).get("c4") or {}
    if not any(action.get("type") in _TERMINAL_TYPES for action in terminal_actions):
        current_tick = (latest_frame.get("when") or {}).get("tick")
        plant_tick = c4_state.get("plant_tick")
        if (
            c4_state.get("planted")
            and isinstance(plant_tick, (int, float))
            and int(plant_tick) > 0
            and (not isinstance(current_tick, (int, float)) or current_tick >= int(plant_tick))
        ):
            eid = f"bomb_planted:{int(plant_tick)}"
            if eid not in state.narrated_event_ids:
                canned_time = _tick_event_time(all_round_frames, int(plant_tick))
                transition_time = (
                    canned_time
                    if isinstance(canned_time, float) and math.isfinite(canned_time)
                    else window.t_end
                )
                candidates.append(
                    {
                        "rank": priority_rank("key_exchange", 2),
                        "strength": 0.7,
                        "time": transition_time,
                        "kind": "retake",
                        "summary": "C4已安装",
                        "action": {
                            "type": "bomb_planted",
                            "event_id": eid,
                            "event_tick": int(plant_tick),
                            "event_time": transition_time,
                        },
                        "event_ids": {eid},
                        "priority_class": "bomb_transition",
                    }
                )
        begin_defuse_tick = c4_state.get("begin_defuse_tick")
        if (
            isinstance(begin_defuse_tick, (int, float))
            and int(begin_defuse_tick) > 0
            and isinstance(current_tick, (int, float))
            and current_tick >= int(begin_defuse_tick)
        ):
            eid = f"defuse_started:{int(begin_defuse_tick)}"
            if eid not in state.narrated_event_ids:
                canned_time = _tick_event_time(all_round_frames, int(begin_defuse_tick))
                transition_time = (
                    canned_time
                    if isinstance(canned_time, float) and math.isfinite(canned_time)
                    else window.t_end
                )
                candidates.append(
                    {
                        "rank": priority_rank("key_exchange", 2),
                        "strength": 0.7,
                        "time": transition_time,
                        "kind": "retake",
                        "summary": "CT开始拆弹",
                        "action": {
                            "type": "defuse_started",
                            "event_id": eid,
                            "event_tick": int(begin_defuse_tick),
                            "event_time": transition_time,
                        },
                        "event_ids": {eid},
                        "priority_class": "bomb_transition",
                    }
                )
    utilities = [
        action
        for action in owned_actions
        if action.get("type") in {"utility_throw", "utility_effect", "effective_flash"}
        and not action.get("is_teammate")
    ]
    max_utility = int(
        rules.get("narration", {}).get("max_utility_windows_per_round", 2)
    )
    has_fight_context = bool(
        owned_kills
        or any(
            action.get("type") in _TERMINAL_TYPES
            for action in owned_actions
        )
        or any(
            action.get("type") in {"bomb_planted", "defuse_started"}
            for action in owned_actions
        )
    )
    for action in utilities:
        action_type = action.get("type")
        if _event_id(action) in state.narrated_event_ids:
            continue
        if action_type in {"utility_throw", "utility_effect"}:
            over_quota = state.utility_windows_used >= max_utility
            rank = priority_rank("utility", 4) + (1 if over_quota else 0)
            strength = 0.5 + (0.25 if has_fight_context else 0.0)
        else:
            rank = priority_rank("utility", 4)
            strength = 0.5
        candidates.append(
            {
                "rank": rank,
                "strength": strength,
                "time": float(action.get("event_time", 0)),
                "kind": "utility",
                "summary": _utility_summary(action, char_budget),
                "action": action,
                "event_ids": {_event_id(action)},
                "priority_class": "utility",
            }
        )

    public_tactic = None
    if isinstance(tactic_hint, dict):
        rule_id, label, hint, matched_at = (
            tactic_hint.get(key) for key in ("rule_id", "label", "hint", "matched_at")
        )
        if (
            all(isinstance(value, str) and value for value in (rule_id, label, hint))
            and isinstance(matched_at, (int, float))
            and not isinstance(matched_at, bool)
        ):
            public_tactic = {
                "rule_id": rule_id,
                "label": label,
                "hint": hint,
                "matched_at": float(matched_at),
            }
            candidates.append(
                {
                    "rank": priority_rank("verified_spatial", 5),
                    "strength": 0.5,
                    "time": float(matched_at),
                    "kind": "tactic",
                    "summary": label if hint == label else f"{label}：{hint}",
                    "action": None,
                    "event_ids": set(),
                    "priority_class": "verified_tactic",
                }
            )
    spatial_summary = _spatial_summary(spatial)
    before_frames = sorted(
        [
            frame
            for frame in all_round_frames
            if owns_time(
                frame_time(frame),
                window.context_start,
                window.t_start,
            )
        ],
        key=frame_time,
    ) or (owned_frames[:1] if owned_frames else [])
    state_summary = _state_summary(result_state, _alive_state(before_frames))
    if state_summary:
        candidates.append(
            {
                "rank": 3.5,
                "strength": 0.35,
                "time": window.t_end,
                "kind": "state",
                "summary": state_summary,
                "action": None,
                "event_ids": set(),
                "priority_class": "state",
            }
        )
    anchor_info = spatial.get("anchor") or {}
    anchor_name = str(anchor_info.get("name") or "")
    anchor_callout = str(
        anchor_info.get("callout_zh") or anchor_info.get("callout") or ""
    )
    position_allowed_source = spatial.get("anchor_source") in {
        "pov",
        "isolated_opposite",
    }
    if (
        spatial_summary
        and position_allowed_source
        and anchor_name
        and anchor_callout
        and (
            narrative_phase not in {"下包前", "下包后"}
            or str(anchor_info.get("side") or "").upper()
            == ("T" if narrative_phase == "下包前" else "CT")
        )
        and f"{anchor_name}|{anchor_callout.casefold()}"
        not in state.position_anchors_seen
    ):
        candidates.append(
            {
                "rank": priority_rank("verified_spatial", 5),
                "strength": 0.4,
                "time": window.t_end,
                "kind": "position",
                "summary": spatial_summary,
                "action": None,
                "event_ids": set(),
                "priority_class": "verified_position",
                "side": str(anchor_info.get("side") or "").upper(),
            }
        )
    selected_candidates = _select_candidates(candidates, narrative_phase, char_budget)
    selected = selected_candidates[0] if selected_candidates else None
    if selected:
        main_topic = {
            "kind": selected["kind"],
            "summary": selected["summary"],
            "priority_class": selected["priority_class"],
        }
        selected_actions = []
        seen_selected_action_ids: set[str] = set()
        for candidate in selected_candidates:
            action = candidate.get("action")
            if not action:
                continue
            action_id = _event_id(action)
            if action_id in seen_selected_action_ids:
                continue
            selected_actions.append(_public_action(action))
            seen_selected_action_ids.add(action_id)
        selected_event_ids = set().union(*(candidate["event_ids"] for candidate in selected_candidates))
    else:
        main_topic = {"kind": "silence", "summary": "", "priority_class": "silence"}
        selected_actions, selected_event_ids = [], set()

    selected_action = selected.get("action") if selected else None
    counted_kill_ids: set[str] = set()
    plain_kill_topics = 0
    for candidate in selected_candidates:
        action = candidate.get("action") or {}
        if action.get("type") != "kill_topic":
            continue
        kills = [kill for kill in action.get("kills") or [] if _event_id(kill) not in counted_kill_ids]
        counted_kill_ids.update(_event_id(kill) for kill in kills)
        state.narrated_kill_ids.update(_kill_id(kill) for kill in kills)
        if action.get("semantic") == "plain_kill" and kills:
            plain_kill_topics += 1
        attacker = str(action.get("attacker") or "")
        if attacker:
            state.attacker_round_kills[attacker] = state.attacker_round_kills.get(attacker, 0) + len(kills)
    state.plain_kills_used += plain_kill_topics
    if any(candidate["kind"] == "utility" for candidate in selected_candidates):
        state.utility_windows_used += 1
    if any(candidate["kind"] == "position" for candidate in selected_candidates) and anchor_name and anchor_callout:
        state.position_anchors_seen.add(f"{anchor_name}|{anchor_callout.casefold()}")
    state.narrated_event_ids.update(selected_event_ids)
    state.seen_transition_ids.update(
        _event_id(action)
        for action in owned_actions + terminal_actions
        if action.get("type") in {"bomb_planted", "defuse_started", "bomb_exploded", "bomb_defused", "round_end", "team_eliminated"}
    )

    event_ledger = []
    ledger_actions = list(owned_actions)
    ledger_ids = {_event_id(action) for action in ledger_actions}
    ledger_actions.extend(
        action for action in terminal_actions if _event_id(action) not in ledger_ids
    )
    for action in ledger_actions:
        if action.get("type") not in _HARD_FACT_TYPES:
            continue
        event_id = _event_id(action)
        action_pov_role = str(action.get("pov_role") or "unavailable")
        pov_relation = (
            "global"
            if action.get("type") != "kill"
            else (
                "on_pov"
                if action_pov_role in {"killer", "victim"}
                else "off_pov"
            )
        )
        row = {
            "event_id": event_id,
            "type": action.get("type"),
            "hard_fact": True,
            "owned_by_window": True,
            "pov_relation": pov_relation,
            "locality_verified": spatial.get("map_precision") == "reviewed_graph"
            and event_id in local_ids,
        }
        for key in (
            "event_tick",
            "event_time",
            "attacker",
            "victim",
            "weapon",
            "side",
            "winner",
            "primary_rule",
            "rule_tags",
            "rule_confidence",
            "pov_role",
            "pov_player",
            "round_tags",
            "round_context",
            "thrower",
            "utility",
            "entity_id",
            "stable_event_id",
            "duration_s",
        ):
            if action.get(key) is not None:
                row[key] = action.get(key)
        if event_id not in selected_event_ids:
            if event_id in previous_transition_ids:
                reason = "duplicate_event"
            elif event_id in state.narrated_event_ids:
                reason = "already_narrated"
            elif action.get("type") == "kill" and state.plain_kills_used >= max_plain:
                reason = "ordinary_kill_quota"
            elif pov_relation == "off_pov":
                reason = "off_pov_lower_rank"
            else:
                reason = "lower_priority"
            row["suppressed_reason"] = reason
        event_ledger.append(row)

    # position 没有公开 action；为规则模板路径提供同一窗口内的派生事实来源。
    for candidate in selected_candidates:
        if candidate.get("action") or candidate.get("kind") != "position":
            continue
        derived_type = str(candidate["kind"])
        derived_time = float(candidate.get("time") or window.t_end)
        row = {
            "event_id": f"derived:{derived_type}:{window.t_start:.3f}:{candidate.get('summary')}",
            "type": derived_type,
            "hard_fact": False,
            "owned_by_window": True,
            "event_time": derived_time,
            "event_tick": _sec_to_tick(derived_time),
            "summary": str(candidate.get("summary") or ""),
        }
        if derived_type == "position":
            row.update({
                "attacker": anchor_name,
                "side": str(candidate.get("side") or "").upper(),
                "callout": anchor_callout,
            })
        event_ledger.append(row)

    latest_frame = (owned_frames or context_frames or [{}])[-1]
    c4 = (latest_frame.get("events") or {}).get("c4") or {}
    bomb_state = {
        "state": "planted" if c4.get("planted") else "not_planted",
        "plant_tick": c4.get("plant_tick"),
        "defuse_started": bool(c4.get("begin_defuse_tick")),
    }
    bomb_candidates = [
        action
        for action in owned_actions + terminal_actions
        if action.get("type")
        in {"bomb_planted", "defuse_started", "bomb_exploded", "bomb_defused"}
        and _event_id(action) not in previous_transition_ids
    ]
    bomb_owned = max(
        bomb_candidates,
        key=lambda action: (
            action.get("type") in {"bomb_exploded", "bomb_defused"},
            float(action.get("event_time", 0)),
        ),
        default=None,
    )
    bomb_transition = {
        "type": bomb_owned.get("type") if bomb_owned else "none",
        "event_tick": bomb_owned.get("event_tick") if bomb_owned else None,
        "owned_by_window": bomb_owned is not None,
    }
    result = {
        "version": 2,
        "scene": window.scene,
        "ownership": {
            "t_start": window.t_start,
            "t_end": window.t_end,
            "include_end": is_last_window,
        },
        "read_only_context": {
            "t_start": window.context_start,
            "t_end": window.context_end,
        },
        "event_ledger": event_ledger,
        "bomb_state": bomb_state,
        "bomb_transition": bomb_transition,
        "main_topic": main_topic,
        "selected_actions": selected_actions,
        "required_facts": _required_facts(selected_candidates, spatial),
        "spatial": spatial,
        "constraints": {
            "max_kill_topics": 1,
            "do_not_narrate_suppressed": True,
            "avoid_weapon_names": True,
            "one_natural_sentence": True,
            "allowed_action_types": [
                "kill_topic",
                "exchange_topic",
                "bomb_planted",
                "defuse_started",
                "bomb_exploded",
                "bomb_defused",
                "round_end",
                "team_eliminated",
                "utility_throw",
                "utility_effect",
                "effective_flash",
            ],
        },
    }
    if selected_action and selected_action.get("type") in _TERMINAL_TYPES:
        result["scene_override"] = {
            "scene": "收尾",
            "reason": selected_action.get("type"),
        }
    if public_tactic is not None:
        result["tactic_hint"] = public_tactic
    if (
        char_budget is not None
        and sum(count_spoken_chars(str(candidate.get("summary") or "")) for candidate in selected_candidates) > char_budget
    ):
        result["projection_budget_error"] = True
    return result


def fallback_neutral(plan: dict) -> str:
    """确定性兜底覆盖全部必保事实，且不截断半条事实。"""
    facts = [
        fact for fact in (plan.get("required_facts") or [])
        if isinstance(fact, dict) and fact.get("required", True) and fact.get("canonical_text")
    ]
    if len(facts) <= 1:
        return str((plan.get("main_topic") or {}).get("summary") or "")[:100]
    return "，".join(str(fact["canonical_text"]) for fact in facts)


# ── Phase3a v4：原子事实模型（§7.3）──────────────────────────────────────────

_ANCHOR_CATEGORIES = ("players", "teams", "numbers", "events", "results", "locations", "weapons")
_TICK_PER_SEC = 30

# 事实族优先级：确定性数值越大越优先；排序用 (priority desc, anchor_tick asc, fact_id asc)
_FACT_PRIORITY = {
    "kill": 100,
    "bomb_planted": 120,
    "defuse_started": 110,
    "bomb_exploded": 130,
    "bomb_defused": 130,
    "round_result": 90,
    "team_eliminated": 95,
    "utility_throw": 60,
    "utility_effect": 60,
    "effective_flash": 65,
    "position": 45,
    "setup": 40,
}
# 视为"回合结果"类派生事实的 ledger 类型
_RESULT_LEDGER_TYPES = frozenset({"round_end", "team_eliminated", "bomb_exploded", "bomb_defused"})


def _sec_to_tick(sec: float) -> int:
    return int(round(float(sec) * _TICK_PER_SEC))


def _norm_player(value: object) -> str:
    return str(value or "").strip()


def _clause_for_ledger(row: dict) -> tuple[str, str, str]:
    """返回 (canonical_clause, capsule_clause, kind)。武器名不入句（constraints.avoid_weapon_names）。"""
    row_type = str(row.get("type") or "")
    attacker = _norm_player(row.get("attacker")) or "进攻方"
    victim = _norm_player(row.get("victim")) or "对手"
    winner = _norm_player(row.get("winner"))
    side = _norm_player(row.get("side")) or "一方"
    if row_type == "kill":
        return f"{attacker}击杀{victim}", f"{attacker}击杀{victim}", "kill"
    if row_type == "bomb_planted":
        return f"{attacker}安放C4", "C4安放", "bomb_planted"
    if row_type == "defuse_started":
        return f"{attacker}开始拆弹", "拆弹开始", "defuse_started"
    if row_type == "bomb_exploded":
        return "C4爆炸，回合结束", "C4爆炸", "bomb_exploded"
    if row_type == "bomb_defused":
        return "C4已拆除，回合结束", "C4拆除", "bomb_defused"
    if row_type == "round_end":
        return f"{winner}方赢下回合" if winner else "回合结束", f"{winner}胜" if winner else "回合结束", "round_result"
    if row_type == "team_eliminated":
        return f"{side}被清零，回合结束", f"{side}清零", "team_eliminated"
    if row_type in {"utility_throw", "utility_effect", "effective_flash"}:
        summary = _utility_summary(row)
        return summary, summary, row_type
    if row_type in {"position", "setup"}:
        summary = str(row.get("summary") or "")
        return summary, summary, row_type
    return "", "", ""


def _fact_fingerprint(kind: str, anchor_tick: int, row: dict) -> tuple[str, str]:
    """返回 (fingerprint_8, canonical_payload)。只哈希规范化的事实类型/tick/主体/客体/
    对象/结果（不含 canonical_clause 与数组序号）；canonical_payload 用于同 ID 冲突检测。"""
    parts = [
        kind,
        str(anchor_tick),
        _norm_player(row.get("attacker")),
        _norm_player(row.get("victim")),
        _norm_player(row.get("winner")),
        _norm_player(row.get("side")),
        str(row.get("entity_id") or row.get("stable_event_id") or ""),
        _norm_player(row.get("thrower")),
        str(row.get("summary") or ""),
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8], canonical


def _fact_anchors_for(row: dict, kind: str) -> dict[str, list]:
    """从结构化字段投影七类锚点（不从文本反推）。"""
    anchors: dict[str, list] = {key: [] for key in _ANCHOR_CATEGORIES}

    def add(cat: str, value: object) -> None:
        text = _norm_player(value)
        if text and text not in anchors[cat]:
            anchors[cat].append(text)

    for key in ("attacker", "victim"):
        add("players", row.get(key))
    add("players", row.get("thrower"))
    add("teams", row.get("winner"))
    if kind in {"bomb_planted", "defuse_started"}:
        add("events", "bomb")
    if kind == "round_result":
        add("events", "round_result")
        add("results", "round_won" if row.get("winner") else "round_end")
    if kind == "kill":
        add("events", "kill")
    if kind in {"utility_throw", "utility_effect", "effective_flash"}:
        add("events", kind)
    if kind == "position":
        add("teams", row.get("side"))
        add("locations", row.get("callout"))
    return anchors


def build_atomic_fact_units(window_id: str, plan: dict) -> dict:
    """从结构化选中事件生成原子 fact units（§7.3/§8.1）。

    只消费 plan 的 event_ledger / selected_actions / main_topic / ownership；
    禁止从 summary 字符串反向解析事实。
    """
    plan_map = plan if isinstance(plan, dict) else {}
    ownership = plan_map.get("ownership") or {}
    t_start = float(ownership.get("t_start") or 0.0)
    t_end = float(ownership.get("t_end") or 0.0)
    start_tick = _sec_to_tick(t_start)
    end_tick = _sec_to_tick(t_end)
    main_topic = plan_map.get("main_topic") or {}
    topic_kind = str(main_topic.get("kind") or "silence")

    if topic_kind == "silence":
        return {
            "window_id": window_id,
            "fact_units": [],
            "required_fact_ids": [],
            "fact_anchors": {key: [] for key in _ANCHOR_CATEGORIES},
            "target_units": 0,
            "hard_units": 0,
        }

    ledger = plan_map.get("event_ledger") or []
    if not isinstance(ledger, list):
        ledger = []

    units: list[dict] = []
    seen_ids: dict[str, dict] = {}
    for row in ledger:
        if not isinstance(row, dict):
            continue
        if row.get("suppressed_reason"):
            continue
        canonical_clause, capsule_clause, kind = _clause_for_ledger(row)
        if not canonical_clause:
            continue
        raw_tick = row.get("event_tick")
        anchor_tick = int(raw_tick) if isinstance(raw_tick, (int, float)) else end_tick
        fingerprint, payload_str = _fact_fingerprint(kind, anchor_tick, row)
        fact_id = f"fact:v1:{window_id}:{kind}:{anchor_tick:05d}:{fingerprint}"
        if fact_id in seen_ids:
            if seen_ids[fact_id].get("_payload") != payload_str:
                raise ValueError(f"fact ID collision with different payload: {fact_id}")
            continue
        origin = "derived" if row.get("type") in _RESULT_LEDGER_TYPES else "event"
        source_range = [start_tick, anchor_tick] if origin == "derived" else [anchor_tick, anchor_tick]
        unit = {
            "fact_id": fact_id,
            "kind": kind,
            "origin": origin,
            "anchor_tick": anchor_tick,
            "source_tick_range": source_range,
            "canonical_clause": canonical_clause,
            "capsule_clause": capsule_clause,
            "required": True,
            "priority": _FACT_PRIORITY.get(kind, 50),
            "anchors": _fact_anchors_for(row, kind),
            "_payload": payload_str,
        }
        units.append(unit)
        seen_ids[fact_id] = unit

    units.sort(key=lambda u: (-int(u["priority"]), int(u["anchor_tick"]), str(u["fact_id"])))
    required_fact_ids = [u["fact_id"] for u in units]

    merged_anchors: dict[str, list] = {key: [] for key in _ANCHOR_CATEGORIES}
    for unit in units:
        for key, values in unit["anchors"].items():
            for value in values:
                if value not in merged_anchors[key]:
                    merged_anchors[key].append(value)

    target_units = sum(count_spoken_chars(u["canonical_clause"]) for u in units)
    hard_units = max(target_units, int(target_units * 1.5))

    return {
        "window_id": window_id,
        "fact_units": units,
        "required_fact_ids": required_fact_ids,
        "fact_anchors": merged_anchors,
        "target_units": target_units,
        "hard_units": hard_units,
    }
