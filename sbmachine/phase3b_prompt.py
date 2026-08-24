"""Phase 3b — 风格模型的提示词组装辅助（persona、绰号注入、JSON/尾句提取）。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from sbmachine.common import count_spoken_chars

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── persona ──

def _load_persona() -> str:
    path = _PROJECT_ROOT / "Prompt" / "persona.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


# ── prompt assembly ──

def _load_player_aliases() -> dict:
    path = _PROJECT_ROOT / "database" / "player_aliases.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _aliases_hint(neutral_text: str, aliases: dict) -> str:
    """仅为中性稿里真正出现过的选手注入绰号，避免无关绰号污染提示词。"""
    if not aliases:
        return ""
    lines = []
    nt_lower = neutral_text.lower()
    for std_name, info in aliases.items():
        if not isinstance(info, dict):
            continue
        if std_name.lower() not in nt_lower:
            continue
        alts = info.get("aliases", [])
        if not alts:
            continue
        alt_str = "、".join(f'"{a}"' for a in alts)
        desc = info.get("desc", "")
        line = f"  · {std_name} → 可叫 {alt_str}"
        if desc:
            line += f"（{desc}）"
        lines.append(line)
    if not lines:
        return ""
    return "选手绰号参考（可自然替换，不强制）：\n" + "\n".join(lines)


def _aliases_for_prompt(neutral_text: str, aliases: dict) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    lowered = neutral_text.lower()
    for name, info in aliases.items():
        if isinstance(name, str) and isinstance(info, dict) and name.lower() in lowered:
            result[name] = [str(v) for v in info.get("aliases", []) if str(v).strip()]
    return result


_EVENT_TERMS: dict[str, tuple[str, ...]] = {
    "kill": ("击杀", "击倒", "杀掉", "杀死", "收掉", "带走", "秒掉", "双杀", "三杀", "连杀", "清掉", "打掉", "拿下", "倒了"),
    "kill_exchange": ("交换", "交火", "互换", "互换了", "连换", "连续换", "对换", "换了一波", "换血"),
    "utility_throw": ("投出", "扔出", "丢出", "投了", "扔了", "丢了", "掷出", "投掷"),
    "utility_effect": ("生效", "起效", "烧起来", "烟起", "烧了", "爆了", "触发了", "散了"),
    "effective_flash": ("闪光", "致盲", "白了", "被闪了"),
    "bomb_planted": ("下包", "安包", "植弹", "炸弹已安装", "C4已安装", "放了包", "下了包", "下了", "放包", "安放"),
    "defuse_started": ("开始拆", "拆包", "拆弹", "动拆"),
    "bomb_defused": ("拆掉", "拆完", "拆弹成功", "拆包成功"),
    "bomb_exploded": ("爆炸", "炸了", "炸开"),
    "round_end": ("回合结束", "赢下回合", "结束这一回合", "这回合结束"),
    "smoke_effect": ("烟雾", "封烟", "烟起", "冒烟", "烟雾弹生效"),
    "smoke_dissipated": ("烟散", "烟雾消散", "烟消了"),
    "flash_effect": ("闪光", "致盲", "白了", "被闪"),
    "molotov_effect": ("燃烧瓶", "燃烧弹", "烧起来", "火起", "烧了", "火"),
    "team_eliminated": ("全灭", "清零", "全部淘汰", "全倒", "一个不剩"),
    "round_won": ("赢下", "获胜", "拿下这局", "赢下这局", "拿下"),
    "round_lost": ("输掉", "失利", "丢掉这局"),
}
_RESULT_TERMS = {"effect_active": ("生效", "已经烧", "烧起来", "起效"), "team_eliminated": _EVENT_TERMS["team_eliminated"], "round_won": _EVENT_TERMS["round_won"], "round_lost": _EVENT_TERMS["round_lost"]}
_WEAPON_TERMS = ("狙", "步枪", "手枪", "喷子", "AK", "M4", "AWP", "USP", "格洛克")
_WEAPON_FORMS = {"AK": ("AK", "步枪"), "M4": ("M4", "步枪"), "AWP": ("AWP", "狙"), "USP": ("USP", "手枪"), "GLOCK": ("格洛克", "Glock", "手枪")}
_LOCATION_TERMS = ("A点", "B点", "A区", "B区", "A大", "A小", "B通", "中路", "香蕉道", "长箱", "短箱", "警家", "匪家")
_TEAM_PATTERNS = {"T": ("T", "T方", "进攻方"), "CT": ("CT", "CT方", "防守方")}


def build_fact_anchors(scene: dict, aliases: dict) -> dict[str, list[Any]]:
    """读取 scene.fact_anchors，并仅从已验收中性稿做保守补充。

    数字按语义分三类：
    - numbers（人数，X人）：风格稿必须保留具体数字；
    - soft_numbers（血量，Y血）：允许"满血/残血"等语义表达，不强制数字出现；
    - all_numbers（neutral 全量）：作为 unexpected 基准，防捏造/说错数字。
    使用「X人/ Y血」模式提取，天然排除 gr1ks/C4 等内嵌数字。
    """
    neutral = str(scene.get("neutral") or "")
    result: dict[str, list[Any]] = {key: [] for key in ("players", "teams", "numbers", "locations", "weapons", "events", "results", "soft_numbers", "all_numbers")}

    def add(kind: str, value: Any) -> None:
        if value is not None and not isinstance(value, (dict, list, tuple, set)) and value not in result[kind]:
            result[kind].append(value)

    def _num(text: str) -> Any:
        return float(text) if "." in text else int(text)

    raw = scene.get("fact_anchors", {})
    if isinstance(raw, dict):
        for kind in result:
            if kind in ("numbers", "soft_numbers", "all_numbers"):
                continue
            values = raw.get(kind, [])
            for value in values if isinstance(values, list) else []:
                add(kind, value)
        for value in raw.get("numbers", []):
            if not isinstance(value, (dict, list, tuple, set)):
                add("all_numbers", value)
    lowered = neutral.lower()
    # 玩家名补提仅在话题确实以选手为主体时启用（kill / exchange / utility 由 raw anchors 授权或语义上需要），
    # 否则 LLM-A 可能从 rule_state 顺手带上存活玩家名（如 retake 的"Tauson与REZ存活"），
    # 但那不是主话题必需事实——把它们当硬锚点会导致风格稿必挂。
    cp = scene.get("commentary_plan") or {}
    mt = cp.get("main_topic") or {}
    topic_kind = str(mt.get("kind") or "")
    authorized_players = {str(v) for v in (raw.get("players") or []) if v is not None}
    player_sensitive = topic_kind in {"kill", "exchange"} or bool(authorized_players)
    if player_sensitive:
        for player, info in aliases.items():
            known = [str(player)]
            if isinstance(info, dict):
                known.extend(str(v) for v in info.get("aliases", []))
            # 别名匹配要求整词或长度>=3：s1 这类 2 字符短别名极易在
            # gr1ks100 之类字符串里被子串误匹配，导致无关选手混入锚点。
            matched = any(
                value
                and len(str(value)) >= 3
                and str(value).lower() in lowered
                for value in known
            )
            if matched:
                add("players", str(player))
    for team, forms in _TEAM_PATTERNS.items():
        if any(re.search(rf"(?<![A-Za-z]){re.escape(form)}(?![A-Za-z])", neutral, re.I) for form in forms):
            add("teams", team)
    hard_numbers = [_num(m) for m in re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)[人名]", neutral)]
    soft_numbers = [_num(m) for m in re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)血", neutral)]
    all_numbers = [float(n) if "." in n else int(n) for n in re.findall(r"\d+(?:\.\d+)?", neutral)]
    for value in hard_numbers:
        add("numbers", value)
    for value in soft_numbers:
        add("soft_numbers", value)
    for value in all_numbers:
        add("all_numbers", value)
    for location in _LOCATION_TERMS:
        if location.lower() in lowered:
            add("locations", location)
    for weapon in _WEAPON_TERMS:
        if weapon.lower() in lowered:
            add("weapons", weapon)
    for code, terms in _EVENT_TERMS.items():
        if any(term.lower() in lowered for term in terms):
            add("events", code)
    for code, terms in _RESULT_TERMS.items():
        if any(term.lower() in lowered for term in terms):
            add("results", code)
    return result


def build_delivery(scene: dict, *, variant_kind: str = "primary", slot_duration_sec: float | None = None, max_speed_factor: float = 1.5) -> dict[str, Any]:
    """构造模型调用输入的 delivery 块（§9.1）。

    - v4 neutral scene 提供 speech_budget 时，用其 target_units/hard_units 作为
      spoken-unit 预算默认（§9.1）；v3 neutral 保持现有字符预算路径（向后兼容）。
    - 始终携带 variant_kind / slot_duration_sec / max_speed_factor。
    """
    hype = float(scene.get("hype", 0.0) or 0.0)
    hard_limit = max(1, int(scene.get("char_budget", 100)))
    mode, pace, ceiling = (("high_energy", "fast", "惊叹") if hype >= 0.72 else ("live_reaction", "medium", "激动") if hype >= 0.35 else ("short_reaction", "slow", "平述"))
    delivery: dict[str, Any] = {
        "mode": mode,
        "pace": pace,
        "emotion_ceiling": ceiling,
        "min_chars": math.floor(hard_limit * 0.8),
        "max_chars": math.floor(hard_limit * 1.2),
        "target_chars": math.floor(hard_limit * 0.8),
        "hard_char_limit": hard_limit,
        "max_evaluative_clauses": 1,
        "variant_kind": variant_kind,
        "slot_duration_sec": None if slot_duration_sec is None else round(float(slot_duration_sec), 3),
        "max_speed_factor": round(float(max_speed_factor), 3),
    }
    budget = scene.get("speech_budget")
    if isinstance(budget, dict):
        target_units = budget.get("target_units")
        hard_units = budget.get("hard_units")
        if isinstance(target_units, int) and not isinstance(target_units, bool):
            delivery["target_units"] = target_units
        if isinstance(hard_units, int) and not isinstance(hard_units, bool):
            delivery["hard_units"] = hard_units
    return delivery


def _event_term_hint(anchors: dict) -> dict[str, list[str]]:
    allowed: dict[str, list[str]] = {}
    for code in (*anchors.get("events", []), *anchors.get("results", [])):
        terms = (*_EVENT_TERMS.get(str(code), ()), *_RESULT_TERMS.get(str(code), ()))
        if terms:
            allowed[str(code)] = sorted(set(terms), key=lambda v: (len(v), v))
    return allowed


def build_style_prompt(scene: dict, aliases: dict, recent_style_phrases: list[str], *, retry_feedback: dict[str, Any] | None = None, variant_kind: str = "primary", slot_duration_sec: float | None = None, max_speed_factor: float = 1.5) -> tuple[str, dict[str, list[Any]], dict[str, Any]]:
    """组装模型调用输入（§9.1）：neutral + fact_anchors + required_fact_ids + delivery + aliases。

    variant_kind="compact" 时 delivery 携带 compact 压缩约束；主稿 variant_kind="primary"。
    """
    neutral = str(scene.get("neutral") or "")
    anchors, delivery = build_fact_anchors(scene, aliases), build_delivery(
        scene, variant_kind=variant_kind, slot_duration_sec=slot_duration_sec,
        max_speed_factor=max_speed_factor,
    )
    prompt_aliases = _aliases_for_prompt(neutral, aliases)
    for player in anchors.get("players", []):
        info = aliases.get(str(player), {})
        if isinstance(info, dict):
            prompt_aliases[str(player)] = [str(v) for v in info.get("aliases", []) if str(v).strip()]
    payload: dict[str, Any] = {"neutral": neutral, "fact_anchors": anchors, "delivery": delivery, "recent_style_phrases": recent_style_phrases, "aliases": prompt_aliases, "allowed_event_terms": _event_term_hint(anchors)}
    required_fact_ids = scene.get("required_fact_ids")
    if isinstance(required_fact_ids, list) and required_fact_ids:
        payload["required_fact_ids"] = [str(fid) for fid in required_fact_ids]
    if variant_kind == "compact":
        payload["compact_constraint"] = _COMPACT_CONSTRAINT
    if retry_feedback:
        payload["retry_feedback"] = retry_feedback
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), anchors, delivery


_LEAK_MARKERS = ("```json", '"scenes"', '"t_start"', "【中性稿】", "【场景信息】", "【当前对局状态】")
_CONTAMINATION_MARKERS = ("任务", "注：", "字数预算约", "字数为", "根据以上信息", "【任务", "【核心")

# ── 数值门禁硬线（docs/plan/phase3c-phase4-numeric-gate-stacking-audit.md 收敛合同 §6.5）──
# B/C 的最终硬线都是固定最终值：任何配置（style_budget_hard_tolerance 等）
# 都不得在最终硬线之后再叠加 tolerance 乘子。
LLMB_HARD_CAP_FACTOR = 1.5  # LLM-B 草稿最终硬线：B 文本安全时长上界 / slot <= 1.5
LLMC_HARD_CAP_FACTOR = 1.25  # LLM-C 最终稿硬线：C 文本安全时长上界 / slot <= 1.25（不可解释为 1.25×(1+tolerance)）

# compact 压缩约束（§9.1）：复用同一响应结构，只改 variant_kind 并强制保留 required facts。
_COMPACT_CONSTRAINT = "请压缩为更短的完整口语句，但不得删除required facts中的任何事实，全部必需事实必须完整保留。"


def _strip_tags(s: str) -> str:
    return re.sub(r"\[[^\]]{1,4}\]", "", s).strip()


def _contains_team(text: str, team: str) -> bool:
    return any(re.search(rf"(?<![A-Za-z]){re.escape(form)}(?![A-Za-z])", text, re.I) for form in _TEAM_PATTERNS.get(team, (team,)))


def validate_style_commentary(commentary: str, neutral: str, anchors: dict, aliases: dict, recent: list[str], *, hard_char_limit: int, hard_cap_factor: float | None = None, phrase_max_reuse: int = 2, char_tolerance: float = 0.0, strong_fact_mode: bool = True, enforce_min_budget: bool = True) -> dict[str, Any]:
    """执行事实、预算的确定性验收（防错不防漏）。

    只拦截「说错/编造/过短/超预算/格式非法」，不拦截「漏报个别锚点」：
    - primary 下限=0.6×hard_char_limit，低于下限时以 under_budget 阻断并重试。
      compact 路径传 enforce_min_budget=False，保持原有压缩语义。
    - 双层预算：软上限=正常容量 hard_char_limit（超了标记 overage，不阻断），
      硬上限=hard_char_limit×1.5（超了阻断）。
    - B 最终硬线固定 1.5×hard_char_limit（LLMB_HARD_CAP_FACTOR），不再叠加
      char_tolerance：审计收敛合同要求任意配置都不能把 B 线扩大为 1.5×(1+tolerance)。
      hard_cap_factor 提供时覆盖该因子（仍受 1.5 上限约束，见读取处 clamp）。
    - budget_overage 口径：max(1.0, output_chars / hard_char_limit)，基准是正常容量
      而非硬线；仅供诊断/TTS 容速参考，不再与硬线叠加。
    - missing_anchor（漏说 players/teams/numbers/…）已于 2026-08-16 移除：
      LLM-B 保留表达自由度，允许语义化省略；unexpected_fact（新增未授权
      事实/说错数字）仍严格拦截。
    - strong_fact_mode：强事实依据模式总开关（semantic.strong_fact_mode）。
      True（默认）= 执行 unexpected_fact 事实越界拦截；False = 全面相信 LLM，
      跳过 unexpected_fact 段（空稿/情绪标签/预算硬线三项仍强制保留，属于
      运行基础，不得让步）。
    """
    plain = _strip_tags(commentary)
    output_chars = count_spoken_chars(plain)
    cap_factor = LLMB_HARD_CAP_FACTOR if hard_cap_factor is None else hard_cap_factor
    effective_hard = int(hard_char_limit * cap_factor)
    budget_overage = max(1.0, output_chars / hard_char_limit) if hard_char_limit > 0 and output_chars > hard_char_limit else 1.0
    tags = re.findall(r"\[([^\]]+)\]", commentary)
    if any(tag not in {"平述", "激动", "惊叹"} for tag in tags):
        return {"ok": False, "reason": "invalid_emotion", "details": tags, "output_chars": output_chars, "signature": "", "budget_overage": 1.0}
    if not plain:
        return {"ok": False, "reason": "empty_commentary", "details": [], "output_chars": output_chars, "signature": "", "budget_overage": 1.0}
    hard_limit = effective_hard
    if output_chars > hard_limit:
        return {"ok": False, "reason": "over_budget", "details": [output_chars, effective_hard, hard_limit, plain], "output_chars": output_chars, "signature": "", "budget_overage": budget_overage}
    if enforce_min_budget and output_chars < hard_char_limit * 0.6:
        return {"ok": False, "reason": "under_budget", "details": [output_chars, hard_char_limit, plain], "output_chars": output_chars, "signature": "", "budget_overage": budget_overage}

    if strong_fact_mode:
        unexpected: list[str] = []
        output_numbers = re.findall(r"\d+(?:\.\d+)?", plain)
        allowed_players = {str(v).lower() for v in anchors.get("players", [])}
        for player, info in aliases.items():
            forms = [str(player)] + ([str(v) for v in info.get("aliases", [])] if isinstance(info, dict) else [])
            # 与 build_fact_anchors 一致：别名匹配要求整词或长度>=3，防 s1 误匹配 gr1ks100。
            matched = any(
                form and len(str(form)) >= 3 and str(form).lower() in plain.lower()
                for form in forms
            )
            if str(player).lower() not in allowed_players and matched:
                unexpected.append(f"player:{player}")
        allowed_teams = {str(v) for v in anchors.get("teams", [])}
        for team in _TEAM_PATTERNS:
            if team not in allowed_teams and _contains_team(plain, team) and not _contains_team(neutral, team):
                unexpected.append(f"team:{team}")
        allowed_numbers = {str(v) for v in anchors.get("all_numbers", anchors.get("numbers", []))} | {str(int(v)) for v in anchors.get("all_numbers", anchors.get("numbers", [])) if isinstance(v, float)}
        unexpected.extend(f"number:{v}" for v in output_numbers if v not in allowed_numbers)
        allowed_locations = {str(v).lower() for v in anchors.get("locations", [])}
        for location in _LOCATION_TERMS:
            if location.lower() in plain.lower() and location.lower() not in allowed_locations and location.lower() not in neutral.lower():
                unexpected.append(f"location:{location}")
        allowed_weapon_forms = {form.lower() for weapon in anchors.get("weapons", []) for form in _WEAPON_FORMS.get(str(weapon).upper(), (str(weapon),))}
        for term in _WEAPON_TERMS:
            if term.lower() in plain.lower() and term.lower() not in allowed_weapon_forms and term.lower() not in neutral.lower():
                unexpected.append(f"weapon:{term}")
        if unexpected:
            return {"ok": False, "reason": "unexpected_fact", "details": sorted(set(unexpected)), "output_chars": output_chars, "signature": "", "budget_overage": budget_overage}

    return {"ok": True, "reason": "", "details": [], "output_chars": output_chars, "signature": "", "budget_overage": budget_overage}


def _extract_json_obj(raw: str) -> dict | None:
    """只解析恰好一个 JSON 对象；旧版代码围栏与前后散文一律拒绝。"""
    try:
        value = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _extract_tail(commentary: str, max_chars: int = 80) -> str:
    stripped = _strip_tags(commentary)
    parts = [p for p in re.split(r"(?<=[。！？])", stripped) if p.strip()]
    if not parts:
        return stripped[-40:] if len(stripped) > 40 else stripped
    tail = parts[-1]
    if len(parts) >= 2:
        candidate = parts[-2] + tail
        if len(candidate) <= max_chars:
            tail = candidate
    if len(tail) > max_chars:
        tail = tail[-40:]
    return tail

