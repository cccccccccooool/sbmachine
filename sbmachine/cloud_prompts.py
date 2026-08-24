"""云端特化 prompt 拼装与窗口类型判定（仅 backend == "api" 路径使用）。

全部数据来源为规则层现有产物（phase3a 的 scene 结构），不新增共用 schema 字段：
- `scene.commentary_plan.selected_actions`（kill_topic：pov_role / final_kill / semantic /
  priority / round_tags / rule_tags / attacker / victims）
- `scene.commentary_plan.rule_state`（T/CT 存活数；不暴露队伍总血量）
- `scene.commentary_plan.player_state`（可选个人状态，首次基线/更新差量）
- `scene.hype` / `scene.fact_anchors`

共用文件（analyst_system.txt / style_system.txt / cs_rules.txt / style_skill.md / persona.txt）
一律只读不改。
"""
from __future__ import annotations

from core.prompt_loader import load_prompt
from sbmachine.common import load_cs_game_rules
from sbmachine.phase3b_prompt import _load_persona

# §4.1.2 meme 基调判定（仿 CS2 Insight meme_series_badges_for_kd，硬编码阈值）
# 判定顺序必须自上而下：211 在 kills==2 之前、i18 在 kills==1 之前检查。
def _meme_label(kills: int, deaths: int) -> str:
    if kills == 2 and deaths == 11:
        return "211（高材生）"
    if kills == 0:
        return "o系（研发）"
    if kills == 1 and deaths == 18:
        return "i18（典中典）"
    if kills == 1:
        return "i系"
    if kills == 2:
        return "z系（坐牢）"
    return ""

# §4.1 highlight：语义白名单 + 叙事优先级阈值（cs_game_rules.json kill_semantics 段）
_HIGHLIGHT_SEMANTICS = frozenset({
    "collateral", "spray_transfer", "rule_highlight",
    "weapon_mismatch", "multi_kill", "special_kill",
})
_MATCH_POINT_TAGS = frozenset({"ct_match_point", "t_match_point"})
# §4.1 fail：下饭类 rule_tags
_FAIL_RULE_TAGS = frozenset({"caught_switching"})

_WINDOW_TYPE_LABELS = {
    "highlight": "highlight（高光时刻，值得吹）",
    "fail": "fail（可惜/下饭，替观众叹气）",
    "clutch": "clutch（残局，以少打多）",
    "meme_death": "meme_death（玩梗死亡，带梗语气）",
    "flat": "平淡（过渡窗口，收着说）",
}

# 模块级整场 meme 基调：由 build_cloud_style_system(config, meme_profile) 设置，
# derive_window_type 在 fail 命中时将其升级为 meme_death。
_match_meme_profile: str = ""


def build_cloud_analyst_system() -> str:
    """云端 Phase3a system：规则 + JSON 契约全部内嵌在 Prompt 文件，不引用共用常量。"""
    return load_prompt("analyst_system_cloud")


def compute_match_meme_profile(neutral_data: dict) -> str:
    """统计目标玩家（POV）整场 K/D，返回 §4.1.2 的基调标签；无基调返回空串。

    目标玩家 = 首个非空的 kill_topic.pov_player；缺失时回退 scene.fact_anchors.players[0]。
    """
    global _match_meme_profile
    kills = deaths = 0
    target = ""
    for round_data in neutral_data.get("rounds", []) if isinstance(neutral_data, dict) else []:
        scenes = round_data.get("scenes", []) if isinstance(round_data, dict) else []
        if not isinstance(scenes, list):
            continue
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            plan = scene.get("commentary_plan") if isinstance(scene.get("commentary_plan"), dict) else {}
            actions = plan.get("selected_actions") if isinstance(plan.get("selected_actions"), list) else []
            for action in actions:
                if not isinstance(action, dict) or action.get("type") != "kill_topic":
                    continue
                if not target:
                    target = str(action.get("pov_player") or "")
                    if not target:
                        anchors = scene.get("fact_anchors")
                        if isinstance(anchors, dict) and isinstance(anchors.get("players"), list) and anchors["players"]:
                            target = str(anchors["players"][0])
                if not target:
                    continue
                if str(action.get("attacker") or "") == target:
                    kills += 1
                for victim in action.get("victims", []) if isinstance(action.get("victims"), list) else []:
                    if str(victim or "") == target:
                        deaths += 1
    _match_meme_profile = _meme_label(kills, deaths)
    return _match_meme_profile


def build_cloud_style_system(config: dict, meme_profile: str = "") -> str:
    """云端 Phase3b system：五段拼装；persona 仅作表达态度参考；meme 基调注入分档段。

    云端版不拼 style_skill.md（人格已内嵌 system），cs_rules 用精简内嵌版。
    """
    global _match_meme_profile
    system = load_prompt("style_system_cloud")
    persona = _load_persona()
    system = system.replace("{persona_hint}", persona)
    if meme_profile:
        _match_meme_profile = meme_profile
        meme_line = f"本场基调：{meme_profile}（死亡窗口允许玩「研发/坐牢」向梗，事实不变）"
    else:
        _match_meme_profile = ""
        meme_line = ""
    return system.replace("{meme_profile_line}", meme_line)


def resolve_target_player(scene: dict) -> str:
    """目标玩家（POV）：优先 kill_topic.pov_player，否则 scene.fact_anchors.players[0]。"""
    if not isinstance(scene, dict):
        return ""
    plan = scene.get("commentary_plan") if isinstance(scene.get("commentary_plan"), dict) else {}
    actions = plan.get("selected_actions") if isinstance(plan.get("selected_actions"), list) else []
    for action in actions:
        if isinstance(action, dict) and action.get("type") == "kill_topic":
            pov_player = str(action.get("pov_player") or "")
            if pov_player:
                return pov_player
    anchors = scene.get("fact_anchors")
    if isinstance(anchors, dict) and isinstance(anchors.get("players"), list) and anchors["players"]:
        return str(anchors["players"][0])
    return ""


def _action_pov_role(action: dict, target_player: str) -> str:
    """kill_topic 的视角角色：优先使用规则层 pov_role，缺失时用目标玩家兜底派生。"""
    role = action.get("pov_role")
    if role in {"killer", "victim", "observer"}:
        return str(role)
    if target_player:
        if str(action.get("attacker") or "") == target_player:
            return "killer"
        victims = action.get("victims")
        if isinstance(victims, list) and target_player in {str(v) for v in victims}:
            return "victim"
    return "unavailable"


def derive_window_type(scene: dict, target_player: str = "") -> str:
    """按 §4.1 判定规格派生窗口类型：clutch / highlight / fail / meme_death / flat。

    判定优先级从高到低：clutch > highlight > fail（meme 基调存在时升级 meme_death）> flat。
    缺字段即不触发（fail-closed）。
    """
    if not isinstance(scene, dict):
        return "flat"
    plan = scene.get("commentary_plan") if isinstance(scene.get("commentary_plan"), dict) else {}
    actions = plan.get("selected_actions") if isinstance(plan.get("selected_actions"), list) else []
    kill_actions = [a for a in actions if isinstance(a, dict) and a.get("type") == "kill_topic"]

    has_killer_kill = has_victim_kill = any_kill = False
    final_kill = False
    multi_kill = False
    killer_victims = 0
    highlight_semantic = False
    killer_priority = 0.0
    round_tags: set[str] = set()
    rule_tags: set[str] = set()
    for action in kill_actions:
        role = _action_pov_role(action, target_player)
        victims = action.get("victims") if isinstance(action.get("victims"), list) else []
        any_kill = True
        if role == "killer":
            has_killer_kill = True
            killer_victims = max(killer_victims, len(victims))
            killer_priority = max(killer_priority, float(action.get("priority") or 0.0))
            if str(action.get("semantic") or "") in _HIGHLIGHT_SEMANTICS:
                highlight_semantic = True
        elif role == "victim":
            has_victim_kill = True
        if action.get("final_kill") is True:
            final_kill = True
        if len(victims) >= 2:
            multi_kill = True
        for tag in action.get("round_tags", []) if isinstance(action.get("round_tags"), list) else []:
            if isinstance(tag, str) and tag:
                round_tags.add(tag)
        for tag in action.get("rule_tags", []) if isinstance(action.get("rule_tags"), list) else []:
            if isinstance(tag, str) and tag:
                rule_tags.add(tag)

    # clutch ①：规则层残局末态（某方存活 1 且敌方 ≥2）+ POV 参与的击杀
    rule_state = plan.get("rule_state") if isinstance(plan.get("rule_state"), dict) else {}
    teams = rule_state.get("teams") if isinstance(rule_state.get("teams"), dict) else {}
    alive_counts = [row.get("alive_count") for row in teams.values() if isinstance(row, dict) and isinstance(row.get("alive_count"), int)]
    clutch_by_rule_state = False
    if len(alive_counts) == 2 and (has_killer_kill or has_victim_kill):
        a, b = alive_counts
        if a >= 0 and b >= 0 and ((a == 1 and b >= 2) or (b == 1 and a >= 2)):
            clutch_by_rule_state = True
    if clutch_by_rule_state or final_kill:
        return "clutch"

    # highlight：目标视角击杀，且（多杀 / 亮点语义过叙事阈值 / 赛点回合有击杀）
    if has_killer_kill:
        narrate_threshold = 0.65
        rules = load_cs_game_rules()
        try:
            narrate_threshold = float(rules["kill_semantics"]["narrate_priority_threshold"])
        except (TypeError, ValueError, KeyError):
            pass
        if multi_kill or killer_victims >= 2:
            return "highlight"
        if highlight_semantic and killer_priority >= narrate_threshold:
            return "highlight"
        if round_tags & _MATCH_POINT_TAGS:
            return "highlight"

    # fail：目标被击杀（且本窗无目标击杀 或 round_won 明确为 False）；或下饭类 rule_tags
    if has_victim_kill and (not has_killer_kill or scene.get("round_won") is False):
        return "meme_death" if _match_meme_profile else "fail"
    if rule_tags & _FAIL_RULE_TAGS:
        return "meme_death" if _match_meme_profile else "fail"

    return "flat"


def inject_window_type(user_prompt: str, scene: dict, target_player: str = "") -> str:
    """在云端 Phase3b user prompt（JSON）后追加窗口类型行，供 system 的事件类型分档使用。"""
    if not target_player:
        target_player = resolve_target_player(scene)
    window_type = derive_window_type(scene, target_player)
    label = _WINDOW_TYPE_LABELS.get(window_type, _WINDOW_TYPE_LABELS["flat"])
    return f"{user_prompt}\n\n【窗口类型】{label}"
