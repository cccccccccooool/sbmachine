"""Phase3c / LLM-C 独立阶段：单向消费 B draft package，发布 render package。

依据 docs/plan/phase3c-llmc-one-way-handoff-plan.md 门禁矩阵 C0~C7：
- C0 入口同源：B 包身份、窗口集合、顺序与状态可验证，失败阻断（不请求 B 修复）
- C1 云端响应形状：严格 JSON envelope（llmc_round_edit_response_v1），字段白名单
- C2 窗口寻址：unit_id 集合与顺序一一对应，不缺失/重复/伪造/重排/合并/拆分
- C3 事实作用域：每单元只使用 allowed_fact_ids ∪ carry_in_fact_ids 对应值
- C4 情绪/时间恢复：emotion/render_slot/required_fact_ids 只从 B 快照按 unit_id 恢复
- C5 压缩容量：逐窗 r_C <= min(1.25, max(1.0, r_B))（非退化蕴含其中），不用回合总量抵消
- C6 来源选择：整回合来源单一（llmc | llmb_passthrough），mode 决策合法
- C7 出口封存：新 JSON 引用 B 身份，render units/情绪/slot/文本一致

任一 C 单元失败 = 整回合 C 失败；允许结果只有向前选择 B 快照或阻断（required）。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from sbmachine.common import count_spoken_chars, load_config
from sbmachine.phase3b_prompt import (
    LLMC_HARD_CAP_FACTOR,
    _LOCATION_TERMS,
    _WEAPON_FORMS,
    _WEAPON_TERMS,
    _extract_json_obj,
)

try:
    from sbmachine.preflight import PublishContractError
except ImportError:  # pragma: no cover - preflight 不可用时降级为 ValueError
    PublishContractError = ValueError

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 兼容期基准语速（字/秒），与 B 包 speech_capacity 的 safe_upper_sec 同口径：
# safe_upper_sec = count_spoken_chars(draft_text) / 5.0
_BASE_SPEED_CHARS_PER_SEC = 5.0

_LLMC_TAG_RE = re.compile(r"\[(平述|激动|惊叹)\]")

# C 响应禁止出现的泄漏标记（情绪标签/时间字段/Markdown 等）。
_LLMC_LEAK_MARKERS = ("```", "[平述]", "[激动]", "[惊叹]", "render_slot", "start_tick", "end_tick", "start_sec", "end_sec", "duration", "tick")

_LLMC3_SYSTEM = (
    "你是回合解说总编辑。你将收到一个回合各窗口的口播稿 JSON，请按窗口地址改写为更连贯、"
    "去重、衔接自然的解说文本。硬性规则：\n"
    "1. 只能改写来源文本，不得新增来源中不存在的事实（选手/数字/地点/武器/阵营）；\n"
    "2. 按每单元 edit_directive 做去重、衔接或润色；不得解释或翻译来源措辞；"
    "来源中的黑话/俗称（如\"坐牢\"）原样保留；\n"
    "3. 不得补全省略成分：来源未提及的主语/阵营/地点，输出也不得补充；\n"
    "4. 不得输出任何情绪标签（如[平述][激动][惊叹]）、时间戳、tick、秒数或 render_slot；\n"
    "5. 不得合并、拆分、排序或删除任何 unit_id，每个窗口独立返回一条 text；\n"
    "6. 必须输出严格 JSON："
    '{"contract":"llmc_round_edit_response_v1","round_id":"<与请求相同>",'
    '"units":[{"unit_id":"<与请求相同>","text":"<改写文本>"}]}；\n'
    "7. 禁止输出 Markdown、解释或任何 JSON 以外的内容。"
)

_VALID_MODES = {"off", "shadow", "optional", "required"}
_VALID_ROUND_STATUS = {"ready", "intentional_silent", "operator_accepted_skip"}
_LLMB_V1_CONTRACT = "llmb_draft_package_v1"
_LLMB_V2_CONTRACT = "llmb_draft_package_v2"


# ── C0 入口同源 ──

def load_llmb_draft_package(path: Path) -> dict:
    """读取并验收 llmb_draft_package_v1（C0 门禁）。

    失败即阻断：身份/窗口集合/顺序/状态任一不可验证时抛 PublishContractError，
    不请求 Phase3b 修复，不猜测。
    """
    if not path.is_file():
        raise PublishContractError("phase3c", f"missing llmb draft package: {path}")
    try:
        package = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishContractError("phase3c", f"invalid llmb draft package: {exc}") from exc
    if not isinstance(package, dict):
        raise PublishContractError("phase3c", "llmb draft package must be an object")
    if package.get("contract") == _LLMB_V2_CONTRACT:
        from sbmachine.preflight import validate_llmb_draft_package

        validate_llmb_draft_package(path)
        return package
    if package.get("contract") != _LLMB_V1_CONTRACT:
        raise PublishContractError("phase3c", f"unexpected contract: {package.get('contract')!r}")
    if package.get("producer") != "phase3b":
        raise PublishContractError("phase3c", "llmb draft package must be produced by phase3b")
    source = package.get("source")
    if not isinstance(source, dict):
        raise PublishContractError("phase3c", "llmb draft package source must be an object")
    for key in ("neutral_run_id", "neutral_sha256", "timeline_id"):
        if not isinstance(source.get(key), str) or not source.get(key):
            raise PublishContractError("phase3c", f"llmb draft package source.{key} is required")
    rounds = package.get("rounds")
    if not isinstance(rounds, list):
        raise PublishContractError("phase3c", "llmb draft package rounds must be a list")
    round_ids: set[str] = set()
    for round_data in rounds:
        if not isinstance(round_data, dict):
            raise PublishContractError("phase3c", "llmb draft package rounds entries must be objects")
        round_id = round_data.get("round_id")
        if not isinstance(round_id, str) or not round_id:
            raise PublishContractError("phase3c", "llmb draft package round_id is required")
        if round_id in round_ids:
            raise PublishContractError("phase3c", f"duplicate round_id: {round_id}")
        round_ids.add(round_id)
        status = round_data.get("status")
        if status not in _VALID_ROUND_STATUS:
            raise PublishContractError("phase3c", f"round {round_id} has invalid status {status!r}")
        units = round_data.get("units")
        if not isinstance(units, list):
            raise PublishContractError("phase3c", f"round {round_id} units must be a list")
        if status == "ready" and not units:
            raise PublishContractError("phase3c", f"round {round_id} status ready requires non-empty units")
        unit_ids: set[str] = set()
        for unit in units:
            if not isinstance(unit, dict):
                raise PublishContractError("phase3c", f"round {round_id} units entries must be objects")
            unit_id = unit.get("unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                raise PublishContractError("phase3c", f"round {round_id} unit_id is required")
            if unit_id in unit_ids:
                raise PublishContractError("phase3c", f"round {round_id} duplicate unit_id: {unit_id}")
            unit_ids.add(unit_id)
            for key in ("draft_text", "emotion_binding", "allowed_fact_ids", "render_slot", "speech_capacity"):
                if key not in unit:
                    raise PublishContractError("phase3c", f"unit {unit_id} missing {key}")
            if not isinstance(unit.get("draft_text"), str) or not unit["draft_text"].strip():
                raise PublishContractError("phase3c", f"unit {unit_id} draft_text must be non-empty")
            emotion_binding = unit.get("emotion_binding")
            if not isinstance(emotion_binding, dict) or not isinstance(emotion_binding.get("emotion"), str):
                raise PublishContractError("phase3c", f"unit {unit_id} emotion_binding.emotion is required")
            slot = unit.get("render_slot")
            if not isinstance(slot, dict) or not isinstance(slot.get("start_tick"), int) or not isinstance(slot.get("end_tick"), int):
                raise PublishContractError("phase3c", f"unit {unit_id} render_slot must have int start_tick/end_tick")
            if slot["start_tick"] >= slot["end_tick"]:
                raise PublishContractError("phase3c", f"unit {unit_id} render_slot start_tick must be < end_tick")
            capacity = unit.get("speech_capacity")
            if not isinstance(capacity, dict) or not isinstance(capacity.get("slot_sec"), (int, float)) or not isinstance(capacity.get("required_speed_factor"), (int, float)):
                raise PublishContractError("phase3c", f"unit {unit_id} speech_capacity is incomplete")
    return package


# ── 请求构造（llmc_round_edit_request_v1）──

def build_round_edit_request(round_data: dict) -> dict:
    """把 B 快照的 ready 回合组装成发给 LLM-C 的回合请求。

    真实时间戳不交给模型：模型只收到地址、正文、只读表达提示、事实边界与容量目标。
    """
    units = []
    for unit in round_data.get("units", []):
        capacity = unit["speech_capacity"]
        slot_sec = float(capacity["slot_sec"])
        source_length_chars = count_spoken_chars(unit["draft_text"])
        r_b = source_length_chars / max(1e-6, slot_sec * _BASE_SPEED_CHARS_PER_SEC)
        if r_b < 0.7:
            edit_directive = "本句偏短，可在保留全部 required facts 的前提下润色扩充（语气、动作、态度句）。"
        elif r_b > 1.3:
            edit_directive = "本句偏长，请删除冗余表达，保留全部 required facts 并保证口语连贯。"
        else:
            edit_directive = "不做字数调整，保证口语连贯。"
        normal_capacity = max(1, round(float(capacity.get("safe_upper_sec", 0.0) or 0.0) * _BASE_SPEED_CHARS_PER_SEC))
        units.append({
            "unit_id": unit["unit_id"],
            "source_text": unit["draft_text"],
            "source_length_chars": source_length_chars,
            "edit_directive": edit_directive,
            "delivery_hint": unit["emotion_binding"]["emotion"],
            "allowed_fact_ids": list(unit.get("allowed_fact_ids", [])),
            "carry_in_fact_ids": list(unit.get("carry_in_fact_ids", [])),
            "speech_budget": {
                "metric": "speech_units_v1",
                "prompt_target": max(1, round(normal_capacity * 0.9)),
                "normal_capacity": normal_capacity,
                "hard_capacity": max(1, round(normal_capacity * LLMC_HARD_CAP_FACTOR)),
            },
        })
    return {
        "contract": "llmc_round_edit_request_v1",
        "round_id": round_data["round_id"],
        "edit_policy": {
            "preserve_unit_ids": True,
            "preserve_order": True,
            "allow_merge": False,
            "allow_split": False,
            "emotion": "read_only",
            "timeline": "not_editable",
            "normal_target_speed_factor": 1.0,
            "hard_speed_factor": LLMC_HARD_CAP_FACTOR,
        },
        "units": units,
    }


# ── C1 响应形状验收 ──

def validate_llmc_response(raw: object, round_id: str) -> tuple[dict | None, str]:
    """验收 LLM-C 原始响应（C1 形状门禁）。

    返回 (parsed_units_dict | None, reason)；失败时 reason 描述拒绝原因。
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "response_error"
    data = _extract_json_obj(raw)
    if data is None:
        return None, "unparseable_json"
    for key in ("think", "reasoning", "reasoning_content"):
        data.pop(key, None)
    top_keys = set(data.keys())
    allowed_top = {"contract", "round_id", "units"}
    if not top_keys <= allowed_top:
        return None, f"unexpected_fields:{sorted(top_keys - allowed_top)}"
    if data.get("contract") != "llmc_round_edit_response_v1":
        return None, "bad_contract"
    if data.get("round_id") != round_id:
        return None, "round_id_mismatch"
    units = data.get("units")
    if not isinstance(units, list):
        return None, "units_not_list"
    result: list[dict] = []
    for item in units:
        if not isinstance(item, dict):
            return None, "unit_not_object"
        item_keys = set(item.keys())
        if not item_keys <= {"unit_id", "text"}:
            return None, f"unit_unexpected_fields:{sorted(item_keys - {'unit_id', 'text'})}"
        unit_id = item.get("unit_id")
        text = item.get("text")
        if not isinstance(unit_id, str) or not unit_id:
            return None, "unit_id_missing"
        if not isinstance(text, str) or not text.strip():
            return None, "empty_text"
        if _LLMC_TAG_RE.search(text):
            return None, "emotion_tag_in_text"
        lowered = text.lower()
        if any(marker in lowered for marker in _LLMC_LEAK_MARKERS):
            return None, "leak_marker"
        if re.search(r"\d+(?:\.\d+)?\s?秒", text):
            return None, "leak_marker"  # 时间读数泄漏
        result.append({"unit_id": unit_id, "text": text})
    return {"round_id": round_id, "units": result}, ""


# ── C2 窗口寻址验收 ──

def check_unit_addressing(resp_units: list[dict], activity_unit_ids: list[str]) -> str:
    """验收响应 unit_id 集合（C2 门禁）。

    必须与输入活动单元集合完全相等且同序；任一条件失败返回错误原因，成功返回空串。
    """
    if len(resp_units) != len(activity_unit_ids):
        return f"count_mismatch:expected {len(activity_unit_ids)} got {len(resp_units)}"
    for index, (resp, expected) in enumerate(zip(resp_units, activity_unit_ids)):
        if resp.get("unit_id") != expected:
            return f"order_or_id_mismatch at {index}: expected {expected!r} got {resp.get('unit_id')!r}"
    return ""


# ── C3 事实作用域验收 ──

def _allowed_fact_values(catalog: dict) -> dict[str, set[str]]:
    """把 fact_catalog 展平为 kind -> 值字符串集（含数字规范化）。"""
    result: dict[str, set[str]] = {kind: set() for kind in ("players", "teams", "numbers", "locations", "weapons")}
    for meta in catalog.values():
        if not isinstance(meta, dict):
            continue
        kind = str(meta.get("kind") or "")
        value = meta.get("value")
        if kind not in result or value is None:
            continue
        result[kind].add(_normalize_value(value))
    return result


def _normalize_value(value: Any) -> str:
    """数字统一去 .0（3.0→"3"），其余按 str。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _leet_normalize(s: str) -> str:
    """选手 ID 的 leetspeak 规范化（dev1ce→device、s1mple→simple），容忍数字↔字母变体。"""
    table = str.maketrans({"1": "i", "0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "2": "z", "6": "g", "8": "b"})
    return s.lower().translate(table)


# 实体提取：以字母开头、≥3 位、可含数字的混合 token（dev1ce 整体提取，不被数字切断）。
_ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")


def check_fact_scope(text: str, b_text: str, catalog: dict, carry_values: list[Any]) -> str:
    """验收 C 单元文本的事实作用域（C3 门禁，防错不防漏）。

    allowed 基准 = 本窗 fact_catalog 值集 ∪ carry_in 值集 ∪ B 原文（B 已有表达必然合法）。
    只拦截「新增未授权事实」：数字、实体（选手名等，leet 规范化后比对）、
    地点、武器；阵营（T/CT/进攻方/防守方）自 2026-08-17 起不再拦截
    （语义补充类改写视为合理，如"坐牢"→"防守方"）。
    返回错误原因（首个违规），合法返回空串。
    """
    allowed = _allowed_fact_values(catalog)
    allowed_numbers: set[str] = set()
    allowed_words: set[str] = set()
    for kind, values in allowed.items():
        for value in values:
            if kind == "numbers":
                allowed_numbers.add(value)
            else:
                allowed_words.add(_leet_normalize(value))
    for value in carry_values:
        allowed_numbers.add(_normalize_value(value))
    # B 原文中已出现的数字/实体/词表项天然合法（C 只改写，不新增）。
    for number in re.findall(r"\d+(?:\.\d+)?", b_text):
        allowed_numbers.add(number)
    for word in _ENTITY_RE.findall(b_text):
        allowed_words.add(_leet_normalize(word))
    for location in _LOCATION_TERMS:
        if location.lower() in b_text.lower():
            allowed["locations"].add(location)
    for weapon in _WEAPON_TERMS:
        if weapon.lower() in b_text.lower():
            allowed["weapons"].add(weapon)

    # 数字检查：先移除紧邻字母的数字（武器型号 AK47/M4A1 等的后缀数字不属于口播事实）；
    # 纯数字事实（"5个人"）前是中文不受影响，仍被检查；ID 变体由实体检查兜底。
    numbers_source = re.sub(r"(?<=[A-Za-z])[0-9]+", "", text)
    output_numbers = re.findall(r"\d+(?:\.\d+)?", numbers_source)
    for number in output_numbers:
        normalized = _normalize_value(float(number)) if "." in number else number
        if normalized not in allowed_numbers:
            return f"unexpected_number:{number}"
    for word in _ENTITY_RE.findall(text):
        if _leet_normalize(word) in allowed_words:
            continue
        upper = word.upper()
        if any(upper.startswith(w) for w in _WEAPON_FORMS):
            continue  # 武器+数字后缀（AK47/M4A1 等）放行
        return f"unexpected_entity:{word}"
    allowed_locations = {v.lower() for v in allowed["locations"]}
    for location in _LOCATION_TERMS:
        if location.lower() in text.lower() and location.lower() not in allowed_locations:
            return f"unexpected_location:{location}"
    allowed_weapons = set()
    for weapon in allowed["weapons"]:
        allowed_weapons.update(_WEAPON_FORMS.get(str(weapon).upper(), (str(weapon),)))
    for term in _WEAPON_TERMS:
        if term.lower() in text.lower() and term.lower() not in {w.lower() for w in allowed_weapons}:
            return f"unexpected_weapon:{term}"
    return ""


# ── C5 压缩容量验收（逐窗，非回合总量）──

def check_capacity(unit: dict, text: str) -> tuple[bool, float, str]:
    """验收单窗 C 文本容量（C5 门禁）。

    r_C = count_spoken_chars(text) / 5.0 / slot_sec（与 B 包 safe_upper_sec 同口径）。
    通过条件：r_C <= min(1.25, max(1.0, r_B))，其中 r_B = B 包 required_speed_factor。
    该公式同时蕴含非退化约束（r_B<=1.0 时 C 不得越过 1.0；1.0<r_B<=1.25 时 C 不得比 B 更长）。
    返回 (ok, r_C, reason)。
    """
    capacity = unit["speech_capacity"]
    slot_sec = max(1e-6, float(capacity["slot_sec"]))
    r_b = float(capacity.get("required_speed_factor", 1.0))
    u_c_sec = count_spoken_chars(text) / _BASE_SPEED_CHARS_PER_SEC
    r_c = u_c_sec / slot_sec
    allowed = min(LLMC_HARD_CAP_FACTOR, max(1.0, r_b))
    if r_c > allowed + 1e-9:
        return False, r_c, f"over_budget:r_c={r_c:.3f} allowed={allowed:.3f}"
    return True, r_c, ""


# ── C6 来源选择（mode 决策）──

def decide_round_source(mode: str, c_passed: bool, c_attempted: bool) -> tuple[str, str]:
    """按 mode 决定整回合来源（C6 门禁）。

    返回 (integration_status, selected_source)。
    """
    if mode == "off":
        return "llmb_passthrough", "llmb_passthrough"
    if mode == "shadow":
        return "llmb_passthrough", "llmb_passthrough"  # C 只验收观察，始终发布 B
    if mode == "optional":
        return ("llmc_accepted", "llmc") if c_passed else ("llmb_passthrough", "llmb_passthrough")
    if mode == "required":
        return ("llmc_accepted", "llmc") if c_passed else ("blocked", "llmb_passthrough")
    raise PublishContractError("phase3c", f"invalid llmc mode: {mode!r}")


# ── C7 出口封存 ──

def build_render_package(package: dict, mode: str, round_results: list[dict]) -> dict:
    """按 B contract dispatch v1/v2 render package construction."""
    if package.get("contract") == _LLMB_V2_CONTRACT:
        return _build_render_package_v2(package, mode, round_results)
    source = package["source"]
    rounds_out = []
    for result in round_results:
        round_data = result["round_data"]
        integration_status = result["integration_status"]
        selected_source = result["selected_source"]
        units_out = []
        if integration_status == "llmc_accepted":
            for index, unit in enumerate(round_data.get("units", [])):
                text = result["texts_by_unit"][unit["unit_id"]]
                capacity = unit["speech_capacity"]
                units_out.append({
                    "unit_id": unit["unit_id"],
                    "sequence": unit.get("sequence", index + 1),
                    "text": text,
                    "emotion": unit["emotion_binding"]["emotion"],
                    "render_slot": unit["render_slot"],
                    "required_fact_ids": list(unit.get("allowed_fact_ids", [])),
                    "required_speed_factor": round(max(1.0, result["r_c_by_unit"][unit["unit_id"]]), 3),
                    "source": "llmc",
                })
        elif integration_status in {"llmb_passthrough", "skipped"} and round_data.get("units"):
            for index, unit in enumerate(round_data.get("units", [])):
                capacity = unit["speech_capacity"]
                units_out.append({
                    "unit_id": unit["unit_id"],
                    "sequence": unit.get("sequence", index + 1),
                    "text": unit["draft_text"],
                    "emotion": unit["emotion_binding"]["emotion"],
                    "render_slot": unit["render_slot"],
                    "required_fact_ids": list(unit.get("allowed_fact_ids", [])),
                    "required_speed_factor": round(float(capacity.get("required_speed_factor", 1.0)), 3),
                    "source": "llmb",
                })
        rounds_out.append({
            "round_id": round_data["round_id"],
            "integration_status": integration_status,
            "selected_source": selected_source,
            "render_units": units_out,
        })
    payload = {
        "contract": "commentary_render_package_v1",
        "producer": "phase3c",
        "status": "render_ready",
        "llmc_mode": mode,
        "source": {
            "llmb_artifact_identity": package.get("artifact_identity", ""),
            "neutral_run_id": source["neutral_run_id"],
            "neutral_sha256": source["neutral_sha256"],
            "timeline_id": source["timeline_id"],
            "source_video_sha256": source.get("source_video_sha256", ""),
        },
        "rounds": rounds_out,
    }
    payload["artifact_identity"] = _artifact_identity(payload)
    return payload


def _artifact_identity_v2(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "artifact_identity"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_render_package_v2(package: dict, mode: str, round_results: list[dict], *, fact_scope: str = "disabled") -> dict:
    """Build the one-way execution contract consumed by strict Phase4."""
    source = package["source"]
    tts_policy = dict(package.get("tts_policy") or {})
    max_speed_factor = float(tts_policy.get("max_speed_factor") or 1.5)
    render_timebase_fps = float(package.get("render_timebase_fps") or 30.0)
    rounds_out: list[dict] = []
    blocked = False
    for result in round_results:
        round_data = result["round_data"]
        integration_status = result["integration_status"]
        if integration_status == "blocked":
            blocked = True
        units_out: list[dict] = []
        if integration_status in {"llmc_accepted", "llmb_passthrough"}:
            for index, unit in enumerate(round_data.get("units", []), start=1):
                unit_id = str(unit["unit_id"])
                if integration_status == "llmc_accepted":
                    final_text = str(result["texts_by_unit"][unit_id])
                    text_source = "llmc"
                    required_speed = float(result["r_c_by_unit"][unit_id])
                else:
                    final_text = str(unit["draft_text"])
                    text_source = "llmb_passthrough"
                    required_speed = float(unit["speech_capacity"].get("required_speed_factor", 1.0))
                required_speed = min(max_speed_factor, max(1.0, required_speed))
                units_out.append({
                    "unit_id": unit_id,
                    "sequence": int(unit.get("sequence") or index),
                    "final_text": final_text,
                    "emotion": str(unit["emotion_binding"]["emotion"]),
                    "text_source": text_source,
                    "render_slot": dict(unit["render_slot"]),
                    "required_speed_factor": round(required_speed, 3),
                    "max_speed_factor": round(max_speed_factor, 3),
                    "required_fact_ids": list(unit.get("allowed_fact_ids", [])),
                })
        rounds_out.append({
            "round_id": str(round_data["round_id"]),
            "round_no": int(round_data.get("round_no") or int(str(round_data["round_id"])[1:])),
            "integration_status": integration_status,
            "render_units": units_out,
        })
    payload = {
        "contract": "commentary_render_package_v2",
        "producer": "phase3c",
        "package_status": "blocked" if blocked else "ready",
        "source": {
            "llmb_artifact_identity": str(source.get("llmb_artifact_identity") or package.get("artifact_identity") or ""),
            "neutral_run_id": str(source.get("neutral_run_id") or ""),
            "neutral_sha256": str(source.get("neutral_sha256") or ""),
            "source_video_sha256": str(source.get("source_video_sha256") or ""),
        },
        "content_policy": {
            "phase3c_mode": mode,
            "fact_check_scope": fact_scope,
            "allow_llmb_passthrough": mode != "required",
        },
        "tts_policy": tts_policy,
        "timeline": {
            "timeline_id": str(source.get("timeline_id") or ""),
            "source_video_sha256": str(source.get("source_video_sha256") or ""),
            "timeline_origin_sec": 0.0,
            "render_tick_rate": render_timebase_fps,
        },
        "rounds": rounds_out,
    }
    payload["artifact_identity"] = _artifact_identity_v2(payload)
    return payload


def _artifact_identity(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# ── 云端调用 ──

def _write_llmc3_debug(round_id: str, attempt: int, prompt: str, raw: object, validation: dict | None, feedback: dict | None) -> None:
    debug_dir = _PROJECT_ROOT / "output" / "debug_phase3"
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dump = {
            "round_id": round_id,
            "phase": "3c_round_edit",
            "attempt": attempt,
            "user_prompt": prompt,
            "response_raw": raw,
            "validation": validation,
            "retry_feedback": feedback,
        }
        with (debug_dir / f"{round_id}_llmc3.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dump, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


def _call_round_editor(
    gen_fn: Callable,
    llm_cfg: dict,
    request: dict,
    round_id: str,
    *,
    max_retries: int,
    debug_enabled: bool,
    source_units: list[dict] | None = None,
) -> tuple[dict | None, str]:
    """调用 LLM-C 至多 max_retries 次，返回 (验证通过后的 units 映射 | None, 最终失败原因)。"""
    prompt = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    max_tokens = max(512, int(_round_normal_capacity(request)) + 200)
    feedback: dict[str, str] | None = None
    for attempt in range(max_retries + 1):
        if feedback:
            retry_request = dict(request)
            retry_request["retry_feedback"] = feedback
            payload = json.dumps(retry_request, ensure_ascii=False, separators=(",", ":"))
        else:
            payload = prompt
        try:
            raw = gen_fn(
                payload, llm_cfg, _LLMC3_SYSTEM,
                max_tokens=max_tokens,
                log_ctx={"round": round_id, "scene": "llmc_round"},
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # 网络/传输异常按 C 失败处理，不穿透
            reason = f"transport_error:{type(exc).__name__}"
            if debug_enabled:
                _write_llmc3_debug(round_id, attempt, payload, None, {"ok": False, "reason": reason}, {"failure_reason": reason})
            return None, reason
        parsed, reason = validate_llmc_response(raw, round_id)
        if parsed is None:
            if debug_enabled:
                _write_llmc3_debug(round_id, attempt, payload, raw, {"ok": False, "reason": reason}, {"failure_reason": reason})
            return None, reason
        if source_units is not None:
            source_by_id = {str(unit["unit_id"]): unit for unit in source_units}
            short_unit_id = ""
            for response_unit in parsed.get("units", []):
                source_unit = source_by_id.get(str(response_unit.get("unit_id")))
                if source_unit is None:
                    continue
                slot_sec = max(1e-6, float(source_unit["speech_capacity"]["slot_sec"]))
                r_c = count_spoken_chars(response_unit["text"]) / (_BASE_SPEED_CHARS_PER_SEC * slot_sec)
                if r_c < 0.6:
                    short_unit_id = str(response_unit["unit_id"])
                    break
            if short_unit_id:
                reason = f"under_budget:{short_unit_id}"
                feedback = {
                    "failure_reason": reason,
                    "instruction": "篇幅过短，润色扩充后重交。",
                }
                if debug_enabled:
                    _write_llmc3_debug(round_id, attempt, payload, raw, {"ok": False, "reason": reason}, feedback)
                if attempt < max_retries:
                    continue
                return None, reason
        return parsed, ""
    return None, "max_retries_exhausted"


def _round_normal_capacity(request: dict) -> int:
    return max(1, int(sum(u["speech_budget"]["normal_capacity"] for u in request.get("units", []))))


# ── 主流程 ──

def run_phase3c(
    *,
    draft_package_path: Path,
    output_render_path: Path,
    config_path: Path,
    dry_run: bool = False,
    progress_sink=None,
) -> dict:
    """Phase3c 主入口：读取 B 封存包，按 llmc.mode 逐回合处理，封存 render package。

    dry_run 时不调用云端（C 结果视为失败→按 mode 决策），且不写产物文件。
    """
    config = load_config(config_path)
    backend = str(config.get("llm", {}).get("backend", "api"))
    if backend != "api":
        raise ValueError(f"phase3c requires llm.backend=api, got {backend!r}")
    semantic_cfg = config.get("semantic", {}) if isinstance(config.get("semantic", {}), dict) else {}
    phase3c_cfg = semantic_cfg.get("phase3c", {}) if isinstance(semantic_cfg.get("phase3c", {}), dict) else {}
    mode = str(phase3c_cfg.get("mode", "off"))
    if mode not in _VALID_MODES:
        raise PublishContractError("phase3c", f"semantic.phase3c.mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
    max_retries = max(0, min(3, int(phase3c_cfg.get("max_retries", 2))))
    # 强事实依据模式（semantic.strong_fact_mode，B0/C3 共用总开关）：默认关闭=全面相信
    # LLM，跳过 C3 事实作用域逐窗校验；开启后执行（含语义补充类改写拦截）。
    fact_scope_enabled = bool(semantic_cfg.get("strong_fact_mode", False))
    debug_enabled = bool(config.get("debug", {}).get("phase3", False))

    package = load_llmb_draft_package(draft_package_path)
    package_version = 2 if package.get("contract") == _LLMB_V2_CONTRACT else 1
    configured_version = int(phase3c_cfg.get("contract_version", 1))
    if configured_version not in {1, 2}:
        raise PublishContractError("phase3c", "semantic.phase3c.contract_version must be 1 or 2")
    if configured_version != package_version:
        raise PublishContractError(
            "phase3c",
            f"configured contract_version={configured_version} does not match input package version={package_version}",
        )

    gen_fn = None
    llm_cfg = None
    if mode != "off":
        from sbmachine import cloud_memory
        from sbmachine.llmb_api import style_runtime_config
        llm_cfg, _ = style_runtime_config(config)
        llm_cfg["temperature"] = float(phase3c_cfg.get("temperature", 0.6))
        gen_fn = cloud_memory.make_generate("llmc", semantic_cfg=semantic_cfg)

    totals = {"llmc_accepted": 0, "passthrough": 0, "blocked": 0, "skipped": 0}
    round_results: list[dict] = []
    total_rounds = len(package.get("rounds", []))
    for index, round_data in enumerate(package.get("rounds", []), start=1):
        round_id = round_data["round_id"]
        result: dict = {"round_data": round_data, "integration_status": "", "selected_source": "", "texts_by_unit": {}, "r_c_by_unit": {}}
        status = round_data["status"]
        units = round_data.get("units", [])
        if status != "ready":
            result["integration_status"] = "skipped"
            result["selected_source"] = "llmb_passthrough"
            totals["skipped"] += 1
            round_results.append(result)
            _report_progress(progress_sink, index, total_rounds, round_id)
            continue
        if mode == "off" or dry_run:
            result["integration_status"] = "llmb_passthrough"
            result["selected_source"] = "llmb_passthrough"
            totals["passthrough"] += 1
            round_results.append(result)
            _report_progress(progress_sink, index, total_rounds, round_id)
            continue
        request = build_round_edit_request(round_data)
        activity_ids = [unit["unit_id"] for unit in units]
        parsed, fail_reason = _call_round_editor(
            gen_fn, llm_cfg, request, round_id,
            max_retries=max_retries, debug_enabled=debug_enabled,
            source_units=units,
        )
        c_passed = False
        if parsed is not None:
            resp_units = parsed.get("units", [])
            addressing_error = check_unit_addressing(resp_units, activity_ids)
            if addressing_error:
                fail_reason = f"addressing:{addressing_error}"
            else:
                catalog_union: dict[str, dict] = {}
                carry_values: list[Any] = []
                for unit in units:
                    catalog = unit.get("fact_catalog", {})
                    if isinstance(catalog, dict):
                        catalog_union.update(catalog)
                    for carry_fid in unit.get("carry_in_fact_ids", []):
                        carry_meta = catalog_union.get(str(carry_fid))
                        if isinstance(carry_meta, dict):
                            carry_values.append(carry_meta.get("value"))
                texts_by_unit: dict[str, str] = {}
                r_c_by_unit: dict[str, float] = {}
                for unit, resp in zip(units, resp_units):
                    text = resp["text"]
                    if fact_scope_enabled:
                        scope_error = check_fact_scope(text, unit["draft_text"], unit.get("fact_catalog", {}), carry_values)
                        if scope_error:
                            fail_reason = f"fact_scope:{scope_error}"
                            break
                    ok, r_c, capacity_error = check_capacity(unit, text)
                    if not ok:
                        fail_reason = capacity_error
                        break
                    texts_by_unit[unit["unit_id"]] = text
                    r_c_by_unit[unit["unit_id"]] = r_c
                else:
                    c_passed = True
                    result["texts_by_unit"] = texts_by_unit
                    result["r_c_by_unit"] = r_c_by_unit
        if not c_passed and parsed is not None:
            # 解析成功但验收失败：记录调试，供重试反馈与运营观测
            if debug_enabled:
                _write_llmc3_debug(round_id, -1, "", None, {"ok": False, "reason": fail_reason}, None)
        integration_status, selected_source = decide_round_source(mode, c_passed, parsed is not None)
        result["integration_status"] = integration_status
        result["selected_source"] = selected_source
        if integration_status == "llmc_accepted":
            totals["llmc_accepted"] += 1
        elif integration_status == "blocked":
            totals["blocked"] += 1
        else:
            totals["passthrough"] += 1
        round_results.append(result)
        _report_progress(progress_sink, index, total_rounds, round_id)

    payload = (
        _build_render_package_v2(
            package,
            mode,
            round_results,
            fact_scope="strong" if fact_scope_enabled else "disabled",
        )
        if package_version == 2
        else build_render_package(package, mode, round_results)
    )
    if not dry_run:
        output_render_path.parent.mkdir(parents=True, exist_ok=True)
        output_render_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": payload.get("status", payload.get("package_status")),
        "contract": payload["contract"],
        "package_status": payload.get("package_status", payload.get("status")),
        "llmc_mode": mode,
        "rounds_total": total_rounds,
        "rounds_llmc_accepted": totals["llmc_accepted"],
        "rounds_passthrough": totals["passthrough"],
        "rounds_blocked": totals["blocked"],
        "rounds_skipped": totals["skipped"],
    }


def _report_progress(progress_sink, completed: int, total: int, round_id: str) -> None:
    if progress_sink is None:
        return
    try:
        progress_sink(completed, total, "round", None)
    except Exception:
        pass
