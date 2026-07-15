"""纯代码解说规划层：在 demo 事实与 LLM 措辞之间做话题选择与约束。"""
from __future__ import annotations

from dataclasses import dataclass, field

from sbmachine.common import load_cs_game_rules
from sbmachine.kill_semantics import build_kill_topics
from sbmachine.scene_context import SceneWindow, extract_actions
from sbmachine.spatial_context import resolve_spatial_context


@dataclass
class PlannerState:
    """跨窗口累计状态：限制平淡单杀数量、记录已解说过的击杀，避免重复。"""

    plain_kills_used: int = 0
    narrated_kill_ids: set[tuple] = field(default_factory=set)


def _kill_id(kill: dict) -> tuple:
    """用（tick, 攻击者, 受害者）唯一标识一次击杀，用于跨窗口去重。"""
    return (kill.get("event_tick"), kill.get("attacker"), kill.get("victim"))


def _kill_summary(topic: dict) -> str:
    """把击杀话题转成一句中文摘要，按语义区分串杀/扫转/枪械压制。"""
    attacker = str(topic.get("attacker") or "进攻方")
    victims = "、".join(str(value) for value in topic.get("victims") or [] if value) or "对手"
    semantic = str(topic.get("semantic") or "plain_kill")
    if semantic == "collateral":
        return f"{attacker}一次击杀{victims}，形成串杀"
    if semantic == "spray_transfer":
        return f"{attacker}连续击杀{victims}，完成扫转"
    if semantic == "weapon_mismatch":
        return f"{attacker}在长枪对手枪的交火中击杀{victims}"
    if len(topic.get("victims") or []) >= 2:
        return f"{attacker}连续击杀{victims}"
    return f"{attacker}击杀{victims}"


def _public_action(action: dict) -> dict:
    """裁剪动作字典为对外可见字段，剔除坐标等内部定位信息。"""
    if action.get("type") == "kill_topic":
        return {key: action.get(key) for key in (
            "type", "semantic", "suggested_phrase", "priority", "confidence",
            "opening_kill", "final_kill", "attacker", "victims", "weapon",
        ) if action.get(key) is not None}
    return {key: value for key, value in action.items() if key not in {"attacker_pos", "victim_pos", "throw_position", "destination"}}


def _utility_summary(action: dict) -> str:
    """将道具类动作转成一句自然语言描述。"""
    actor = action.get("attacker") or action.get("thrower") or "一名选手"
    if action.get("type") == "effective_flash":
        return f"{actor}的闪光让{action.get('victim') or '对手'}有效致盲{action.get('duration_s')}秒"
    utility = action.get("utility") or "道具"
    if action.get("type") == "utility_effect":
        return f"{actor}的{utility}已生效"
    return f"{actor}投出{utility}"


def _spatial_summary(spatial: dict) -> str:
    """把空间上下文转成一句摘要：锚点选手所在点位及附近队友/敌人。"""
    anchor = spatial.get("anchor") or {}
    if not anchor:
        return ""
    name = str(anchor.get("name") or "一名选手")
    side = str(anchor.get("side") or "")
    subject = f"{name}（{side}）" if side else name
    if spatial.get("map_precision") != "reviewed_graph":
        return f"{subject}是当前关注对象"

    callout = anchor.get("callout_zh") or anchor.get("callout")
    summary = f"{subject}位于{callout}" if callout else f"{subject}是当前关注对象"
    nearby = spatial.get("nearby") or {}
    teammates = [str(item.get("name")) for item in nearby.get("teammates", []) if item.get("name")]
    enemies = [str(item.get("name")) for item in nearby.get("enemies", []) if item.get("name")]
    if teammates:
        summary += "，附近有队友" + "、".join(teammates)
    if enemies:
        summary += "，附近有敌人" + "、".join(enemies)
    return summary


def plan_window(
    map_name: str | None,
    window: SceneWindow,
    ownership_frames: list[dict],
    context_frames: list[dict],
    all_round_frames: list[dict],
    state: PlannerState,
) -> dict:
    """为单个场景窗口选出主话题与配套动作，产出带约束的解说规划字典。"""
    rules = load_cs_game_rules()
    actions = extract_actions(context_frames, window.context_start, window.context_end)
    owned_actions = [action for action in actions if window.t_start <= float(action.get("event_time", 0)) < window.t_end]
    spatial = resolve_spatial_context(map_name, window.scene, ownership_frames or context_frames, owned_actions)
    local_actions = spatial.pop("local_actions", [])
    # 锚点缺失时空间归属未知，保留窗口内的 DEM 硬事实动作；只有锚点可信才做局部过滤。
    if spatial.get("anchor") is not None:
        owned_actions = [action for action in owned_actions if action in local_actions]
    round_actions = extract_actions(all_round_frames, float("-inf"), float("inf"))
    round_kills = [action for action in round_actions if action.get("type") == "kill"]
    owned_kill_ids = {_kill_id(action) for action in owned_actions if action.get("type") == "kill"}
    topics = [
        topic for topic in build_kill_topics(actions, context_frames, round_kills=round_kills)
        if any(_kill_id(kill) in owned_kill_ids for kill in topic.get("kills") or [])
    ]
    unseen_topics = [
        topic for topic in topics
        if not any(_kill_id(kill) in state.narrated_kill_ids for kill in topic.get("kills") or [])
    ]
    max_plain = int(rules["narration"]["max_plain_single_kills_per_round"])
    selected_kill = next((topic for topic in unseen_topics if topic.get("narrate") and topic.get("semantic") != "plain_kill"), None)
    if selected_kill is None and state.plain_kills_used < max_plain:
        candidates = []
        for topic in unseen_topics:
            if not (topic.get("opening_kill") or topic.get("final_kill")):
                continue
            last_time = float(topic["kills"][-1].get("event_time", 0))
            has_immediate_followup = any(
                action.get("attacker") == topic.get("attacker")
                and last_time < float(action.get("event_time", 0)) <= last_time + float(rules["scene"]["kill_chain_gap_sec"])
                for action in round_kills
            )
            if not has_immediate_followup:
                candidates.append(topic)
        selected_kill = candidates[0] if candidates else None
        if selected_kill is not None:
            state.plain_kills_used += 1
    if selected_kill is not None:
        state.narrated_kill_ids.update(_kill_id(kill) for kill in selected_kill.get("kills") or [])
    utilities = [
        action for action in owned_actions
        if action.get("type") in {"utility_throw", "utility_effect", "effective_flash"}
        and not action.get("is_teammate")
    ]
    utilities = utilities[:int(rules["narration"]["max_supporting_actions"])]

    scene = window.scene
    main_topic: dict
    if scene == "准备":
        main_topic = {"kind": "setup", "summary": _spatial_summary(spatial)}
    elif scene == "未下包" and utilities:
        main_topic = {"kind": "utility", "summary": _utility_summary(utilities[0])}
    elif scene == "炸弹":
        focus = _spatial_summary(spatial)
        summary = "C4已安装" + (f"，{focus}" if focus else "")
        main_topic = {"kind": "retake", "summary": summary}
    elif scene == "收尾" and selected_kill:
        main_topic = {"kind": "round_end", "summary": _kill_summary(selected_kill)}
    elif selected_kill:
        main_topic = {"kind": "kill", "summary": _kill_summary(selected_kill)}
    else:
        summary = _spatial_summary(spatial)
        main_topic = {"kind": "position" if summary else "silence", "summary": summary}

    selected_actions = [_public_action(action) for action in (([selected_kill] if selected_kill else []) + utilities)]
    action_counts: dict[str, int] = {}
    for action in owned_actions:
        kind = str(action.get("type") or "unknown")
        action_counts[kind] = action_counts.get(kind, 0) + 1
    return {
        "version": 1, "scene": scene,
        "ownership": {"t_start": window.t_start, "t_end": window.t_end},
        "read_only_context": {"t_start": window.context_start, "t_end": window.context_end},
        "main_topic": main_topic,
        "selected_actions": selected_actions,
        "suppressed_kill_topics": [topic.get("semantic") for topic in topics if topic is not selected_kill],
        "spatial": spatial,
        "constraints": {
            "max_kill_topics": 1, "do_not_narrate_suppressed": True,
            "avoid_weapon_names": True,
            "one_natural_sentence": True,
        },
    }


def fallback_neutral(plan: dict) -> str:
    """Safe fallback uses the selected topic only, never dumps every kill."""
    return str((plan.get("main_topic") or {}).get("summary") or "")[:100]
