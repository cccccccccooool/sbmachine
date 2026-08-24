"""Strict projections from rule-layer plans to LLM-visible window facts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

from sbmachine.common import count_spoken_chars


_ACTION_STRING_FIELDS = (
    "type",
    "semantic",
    "attacker",
    "weapon",
    "utility",
    "thrower",
    "victim",
)
_ACTION_NUMBER_FIELDS = ("priority", "confidence", "duration_s")
_ACTION_BOOL_FIELDS = ("opening_kill", "final_kill", "is_teammate")
_TOPIC_KINDS = frozenset({
    "setup", "utility", "retake", "round_end", "kill", "exchange",
    "position", "state", "tactic", "silence",
})
_TEAMS = ("T", "CT")
PROJECTION_VERSION = 2

_UTILITY_NAMES_ZH = {
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


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_mapping(value: object) -> Mapping:
    """Accept dict plans and the parallel CommentaryPlan v2 dataclass/object form."""
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(value)
        return converted if isinstance(converted, Mapping) else {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, Mapping) else {}
    return {}


def _safe_string(value: object, *, limit: int = 240) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and len(value) <= limit else None


def _project_actions(rows: object) -> list[dict]:
    if not isinstance(rows, (list, tuple)):
        return []
    result: list[dict] = []
    for row in rows:
        row = _as_mapping(row)
        if not row:
            continue
        item: dict = {}
        for key in _ACTION_STRING_FIELDS:
            value = _safe_string(row.get(key))
            if value is not None:
                item[key] = value
        for key in _ACTION_NUMBER_FIELDS:
            value = row.get(key)
            if _is_number(value):
                item[key] = value
        for key in _ACTION_BOOL_FIELDS:
            value = row.get(key)
            if isinstance(value, bool):
                item[key] = value
        victims = row.get("victims")
        if isinstance(victims, (list, tuple)):
            public_victims = [value for value in (_safe_string(item) for item in victims) if value is not None]
            if public_victims:
                item["victims"] = public_victims
        for key in ("event_ids", "participants"):
            values = row.get(key)
            if isinstance(values, (list, tuple)):
                public_values = [value for value in (_safe_string(entry) for entry in values) if value is not None]
                if public_values:
                    item[key] = public_values
        kill_count = row.get("kill_count")
        if isinstance(kill_count, int) and not isinstance(kill_count, bool) and kill_count >= 0:
            item["kill_count"] = kill_count
        result_state = _project_teams(row.get("result_state"))
        if result_state:
            item["result_state"] = result_state
        if item.get("type"):
            result.append(item)
    return result


def _project_tactic_hint(raw: object) -> dict | None:
    if not isinstance(raw, Mapping):
        return None
    rule_id = _safe_string(raw.get("rule_id"), limit=120)
    label = _safe_string(raw.get("label"))
    hint = _safe_string(raw.get("hint"))
    if not (rule_id and label and hint):
        return None
    return {"rule_id": rule_id, "label": label, "hint": hint}


def _project_main_topic(raw: object, tactic_hint: dict | None, selected_actions: list[dict] | None = None) -> dict:
    if not isinstance(raw, Mapping):
        return {"kind": "silence", "summary": ""}
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in _TOPIC_KINDS:
        return {"kind": "silence", "summary": ""}
    if kind == "tactic":
        if tactic_hint is None:
            return {"kind": "silence", "summary": ""}
        summary = tactic_hint["label"]
        if tactic_hint["hint"] != summary:
            summary = f"{summary}：{tactic_hint['hint']}"
        return {"kind": kind, "summary": summary}
    if kind == "retake":
        return {"kind": kind, "summary": "C4已安装"}
    if kind == "silence":
        return {"kind": "silence", "summary": ""}
    if kind == "state":
        return {"kind": kind, "summary": _safe_string(raw.get("summary")) or ""}
    # A2: 从 typed actions 确定性重建 summary（仅 kill/round_end/utility）。
    # position/setup/silence 及其他 kind 保持旧安全行为 summary=""，
    # 因为规则层已处理 reviewed map 证据区分，投影层不直接信任 free-form summary。
    if kind in ("kill", "round_end", "exchange"):
        actions = selected_actions if isinstance(selected_actions, list) else []
        exchange_action = next((a for a in actions if a.get("type") == "exchange_topic"), None)
        if isinstance(exchange_action, Mapping):
            count = exchange_action.get("kill_count")
            parts = [f"双方连续交换{count}次击杀" if isinstance(count, int) and count > 0 else "双方连续交换"]
            state = exchange_action.get("result_state")
            if isinstance(state, Mapping):
                state_text = []
                for side in _TEAMS:
                    team = state.get(side)
                    if isinstance(team, Mapping):
                        state_text.append(f"{side}方存活{team.get('alive_count')}人")
                total_alive = sum(
                    int((state.get(side) or {}).get("alive_count", 0))
                    for side in _TEAMS
                )
                if 0 < total_alive <= 4:
                    per_player = []
                    for side in _TEAMS:
                        team = state.get(side)
                        players = team.get("players") if isinstance(team, Mapping) else None
                        if isinstance(players, list):
                            names = "、".join(
                                f"{p.get('name', '')}{p.get('hp', 0)}血"
                                for p in players
                                if p.get("name")
                            )
                            if names:
                                per_player.append(f"{side}方{names}")
                    if per_player:
                        parts.extend(per_player)
                elif state_text:
                    parts.append("，".join(state_text))
            return {"kind": kind, "summary": "，".join(parts)}
        kill_action = next((a for a in actions if isinstance(a, Mapping) and a.get("type") == "kill_topic"), None)
        if isinstance(kill_action, Mapping):
            attacker = str(kill_action.get("attacker") or "进攻方")
            victims_list = kill_action.get("victims")
            victims = "、".join(str(v) for v in victims_list) if isinstance(victims_list, list) and victims_list else "对手"
            semantic = str(kill_action.get("semantic") or "plain_kill")
            if semantic == "collateral":
                summary = f"{attacker}一次击杀{victims}，形成串杀"
            elif semantic == "spray_transfer":
                summary = f"{attacker}连续击杀{victims}，完成扫转"
            elif semantic == "weapon_mismatch":
                summary = f"{attacker}在长枪对手枪的交火中击杀{victims}"
            elif isinstance(victims_list, list) and len(victims_list) >= 2:
                summary = f"{attacker}连续击杀{victims}"
            else:
                summary = f"{attacker}击杀{victims}"
        else:
            summary = _safe_string(raw.get("summary")) or ""
        return {"kind": kind, "summary": summary}
    if kind == "utility":
        actions = selected_actions if isinstance(selected_actions, list) else []
        utility_action = next((a for a in actions if isinstance(a, Mapping) and a.get("type") in {"utility_throw", "utility_effect", "effective_flash"}), None)
        if isinstance(utility_action, Mapping):
            actor = utility_action.get("attacker") or utility_action.get("thrower") or "一名选手"
            if utility_action.get("type") == "effective_flash":
                victim = utility_action.get("victim")
                duration = utility_action.get("duration_s")
                if victim and isinstance(duration, (int, float)) and not isinstance(duration, bool):
                    summary = f"{actor}的闪光让{victim}有效致盲{duration}秒"
                elif victim:
                    summary = f"{actor}的闪光让{victim}有效致盲"
                else:
                    summary = f"{actor}的闪光有效致盲了对手"
            else:
                utility_raw = str(utility_action.get("utility") or "道具")
                utility_zh, _ = _UTILITY_NAMES_ZH.get(utility_raw.casefold(), (utility_raw, utility_raw))
                if utility_action.get("type") == "utility_effect":
                    summary = f"{actor}的{utility_zh}已生效"
                else:
                    summary = f"{actor}投出{utility_zh}"
        else:
            summary = _safe_string(raw.get("summary")) or ""
        return {"kind": kind, "summary": summary}
    # position/setup/silence 及其他未列出的 kind：
    # 规则层已处理 reviewed map 证据与无空间推断通用摘要的区分；
    # 投影层不直接信任 free-form summary，保持旧安全行为。
    return {"kind": kind, "summary": ""}


def _project_teams(raw: object) -> dict[str, dict]:
    if not isinstance(raw, Mapping):
        return {}
    public_teams: dict[str, dict] = {}
    for side in _TEAMS:
        team = raw.get(side)
        if not isinstance(team, Mapping):
            continue
        alive_count = team.get("alive_count")
        # 队伍公开投影只保留存活人数；旧输入的 hp_total 一律在公开边界丢弃，
        # 避免 LLM 把整队总血量误说成"损失 XX 血"。
        if _is_number(alive_count) and alive_count >= 0:
            row: dict = {"alive_count": alive_count}
            players = team.get("players")
            if isinstance(players, list):
                public_players = []
                for p in players:
                    if not isinstance(p, Mapping):
                        continue
                    name = _safe_string(p.get("name"), limit=60)
                    hp = p.get("hp")
                    if name and _is_number(hp) and hp >= 0:
                        public_players.append({"name": name, "hp": hp})
                if public_players:
                    row["players"] = public_players
            public_teams[side] = row
    return public_teams


def _safe_rule_state(raw: object) -> dict | None:
    if not isinstance(raw, Mapping):
        return None
    kind = raw.get("kind")
    teams = raw.get("teams")
    if kind != "snapshot" or not isinstance(teams, Mapping):
        return None
    public_teams = _project_teams(teams)
    if set(public_teams) != set(_TEAMS):
        return None
    changed = raw.get("changed_teams")
    changed_teams = [side for side in _TEAMS if isinstance(changed, (list, tuple)) and side in changed]
    return {"kind": "snapshot", "teams": public_teams, "changed_teams": changed_teams}


def build_rule_state_delta(frames: object, reported: dict[str, dict[str, int | float]]) -> dict | None:
    """Return the complete end-of-window snapshot and changed side names.

    Raw player names, callouts, weapons, ammo and coordinates never leave this
    function. Unknown sides or invalid HP are ignored rather than guessed.
    """
    if not isinstance(frames, list):
        return None
    representative = next(
        (frame for frame in reversed(frames)
         if isinstance(frame, Mapping)
         and isinstance(frame.get("where"), Mapping)
         and isinstance(frame["where"].get("players"), list)),
        None,
    )
    if representative is None:
        return None

    snapshot: dict[str, dict] = {}
    players = representative["where"].get("players") or []
    for player in players:
        if not isinstance(player, Mapping):
            continue
        side = player.get("side")
        hp = player.get("hp")
        if side not in _TEAMS or not _is_number(hp) or hp < 0:
            continue
        team = snapshot.setdefault(
            side, {"alive_count": 0}
        )
        if hp > 0:
            team["alive_count"] += 1
    # 注意：rule_state 不向 LLM-A 暴露存活玩家名明细。
    # 若暴露，LLM-A 会在 retake 等非选手主体话题里顺手带上"谁存活"，
    # 导致 neutral 出现未被 required_facts 授权的玩家名，Phase3b 校验必挂。
    # 残局 per-player 血量由 exchange 的 result_state（_project_teams）单独提供。
    for side in snapshot:
        snapshot[side].pop("players", None)
    if set(snapshot) != set(_TEAMS):
        return None
    changed: list[str] = []
    for side in _TEAMS:
        team = snapshot[side]
        # 只按存活人数建立/比较阵营快照：单纯掉血不把阵营标为变化。
        aggregate = {
            "alive_count": team.get("alive_count", 0),
        }
        if reported.get(side) != aggregate:
            changed.append(side)
        reported[side] = dict(aggregate)
    return {"kind": "snapshot", "teams": snapshot, "changed_teams": changed}


def _project_anchors(raw: object) -> dict[str, list]:
    source = raw if isinstance(raw, Mapping) else {}
    result: dict[str, list] = {}
    for key in ("players", "events", "results", "locations", "weapons"):
        values = source.get(key)
        result[key] = [value for value in (_safe_string(item, limit=120) for item in values) if value is not None] if isinstance(values, (list, tuple)) else []
    teams = source.get("teams")
    result["teams"] = [side for side in _TEAMS if isinstance(teams, (list, tuple)) and side in teams]
    numbers = source.get("numbers")
    result["numbers"] = [item for item in numbers if _is_number(item)] if isinstance(numbers, (list, tuple)) else []
    return result


def _infer_anchors(text: str, actions: list[dict]) -> dict[str, list]:
    import re
    players: list[str] = []
    events: list[str] = []
    weapons: list[str] = []
    for action in actions:
        for key in ("attacker", "thrower", "victim"):
            value = _safe_string(action.get(key), limit=120)
            if value and value not in players:
                players.append(value)
        for value in action.get("victims", []):
            if value not in players:
                players.append(value)
        raw_action_type = _safe_string(action.get("type"), limit=120)
        action_type = {
            "kill_topic": "kill",
            "exchange_topic": "kill_exchange",
        }.get(raw_action_type, raw_action_type)
        if action_type and action_type not in events:
            events.append(action_type)
        weapon = _safe_string(action.get("weapon"), limit=120)
        if weapon and weapon not in weapons:
            weapons.append(weapon)
    numbers = [float(item) if "." in item else int(item) for item in re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?", text)]
    return {
        "players": players,
        "teams": [
            side
            for side in _TEAMS
            if re.search(rf"(?<![A-Za-z]){side}(?:方)?(?![A-Za-z])", text, re.I)
        ],
        "numbers": numbers,
        "events": events,
        "results": [],
        "locations": [],
        "weapons": weapons,
    }


def _project_required_facts(raw: object, topic: dict, actions: list[dict]) -> list[dict]:
    result: list[dict] = []
    if isinstance(raw, (list, tuple)):
        for index, row in enumerate(raw):
            if not isinstance(row, Mapping) or row.get("required", True) is not True:
                continue
            canonical = _safe_string(row.get("canonical_text"))
            if canonical is None:
                continue
            result.append({
                "fact_id": _safe_string(row.get("fact_id"), limit=160) or f"fact:{index + 1}",
                "type": _safe_string(row.get("type"), limit=120) or "topic",
                "canonical_text": canonical,
                "required": True,
                "anchors": _project_anchors(row.get("anchors")),
            })
    if result or topic.get("kind") == "silence":
        return result
    summary = _safe_string(topic.get("summary"))
    if summary:
        result.append({
            "fact_id": f"topic:{topic.get('kind')}",
            "type": str(topic.get("kind")),
            "canonical_text": summary,
            "required": True,
            "anchors": _infer_anchors(summary, actions),
        })
    return result


def merge_required_fact_anchors(required_facts: object) -> dict[str, list]:
    """Merge only projection-whitelisted required-fact anchors for downstream style input."""
    merged: dict[str, list] = {
        key: [] for key in ("players", "teams", "numbers", "events", "results", "locations", "weapons")
    }
    if not isinstance(required_facts, (list, tuple)):
        return merged
    for fact in required_facts:
        fact_map = _as_mapping(fact)
        anchors = fact_map.get("anchors")
        if not isinstance(anchors, Mapping):
            continue
        safe = _project_anchors(anchors)
        for key, values in safe.items():
            for value in values:
                if value not in merged[key]:
                    merged[key].append(value)
    return merged


_PLAYER_STATE_MAX_CHARS = 1000


def _safe_player_state(raw: object) -> str | None:
    """接受非空且不超过 1000 字符的选手状态字符串；超限整项省略，不截断半条。"""
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value or len(value) > _PLAYER_STATE_MAX_CHARS:
        return None
    return value


def build_llm_window_projection(plan: object, *, rule_state: object = None, player_state: object = None) -> dict:
    """Whitelist a rule plan for local and cloud LLMs.

    The returned object is the complete window data contract.  Anything absent
    here (spatial audit data, context frames, evidence and raw state) must not
    be serialized into a prompt or its input archive.
    """
    plan_map = _as_mapping(plan)
    tactic_hint = _project_tactic_hint(plan_map.get("tactic_hint"))
    selected_actions = _project_actions(plan_map.get("selected_actions"))
    topic = _project_main_topic(plan_map.get("main_topic"), tactic_hint, selected_actions)
    safe_state = _safe_rule_state(rule_state or plan_map.get("rule_state"))
    if topic.get("kind") in {"state", "position", "setup"} and not topic.get("summary") and safe_state:
        sides = safe_state["changed_teams"] or list(_TEAMS)
        topic["summary"] = "，".join(
            f"{side}方存活{safe_state['teams'][side]['alive_count']}人"
            for side in sides
        )
    required_facts = _project_required_facts(plan_map.get("required_facts"), topic, selected_actions)
    if required_facts and plan_map.get("required_facts") and topic.get("kind") != "silence":
        # Plan v2 的 canonical_text 已按字符预算确定；投影主题必须与同一事实基线一致，
        # 不能再用长摘要覆盖预算压缩结果。
        topic["summary"] = required_facts[0]["canonical_text"]
    result = {
        "projection_version": PROJECTION_VERSION,
        "main_topic": topic,
        "selected_actions": selected_actions,
        "required_facts": required_facts,
        "required_chars": sum(count_spoken_chars(fact["canonical_text"]) for fact in required_facts),
    }
    if safe_state is not None:
        result["rule_state"] = safe_state
    if tactic_hint is not None:
        result["tactic_hint"] = tactic_hint
    safe_player_state = _safe_player_state(player_state)
    if safe_player_state is not None:
        result["player_state"] = safe_player_state
    return result
