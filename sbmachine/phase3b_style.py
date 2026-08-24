"""Phase 3b — 风格模型：中性稿 + hype hint → 6657 口播 + [情绪] 标签。"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from core.prompt_loader import load_prompt
from audio_service.emotion import parse_emotional_text
from sbmachine.common import count_spoken_chars, load_config, load_hype_rules, resolve_backend, write_json
from sbmachine.neutral_contract import CLOUD_PHASE3A_MODE, validate_neutral_manifest, validate_neutral_v4
from sbmachine.preflight import validate_llmb_draft_package, validate_neutral_publishable
from sbmachine.emotion_policy import EmotionPolicy, capsule_emotion, normalize_commentary_emotion, weighted_intensity
from sbmachine.phase3b_prompt import (
    _CONTAMINATION_MARKERS, _LEAK_MARKERS, _extract_json_obj, _load_persona,
    _load_player_aliases, _strip_tags, build_style_prompt,
    validate_style_commentary,
    LLMB_HARD_CAP_FACTOR,
)
from sbmachine.media_clock import round_half_even
from sbmachine.llm_shim import _request_error_category, accept_api_response
from sbmachine.schemas import EmotionSegment, SemanticData, load_match, save_match
from sbmachine import speech_measure
from sbmachine.voice_task_contract import validate_commentary_v3

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── llmb 诊断落盘（非 debug 也写摘要；每次尝试一条，供重试根因观测）──
_DIAGNOSTICS_LOCK = threading.Lock()
_DIAGNOSTICS_DIR: Path | None = None
_DIAGNOSTICS_RUN_ID: str = ""


def _init_style_diagnostics(output_dir: Path, run_id: str) -> None:
    global _DIAGNOSTICS_DIR, _DIAGNOSTICS_RUN_ID
    _DIAGNOSTICS_DIR = output_dir / "diagnostics" / "phase3b"
    _DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    _DIAGNOSTICS_RUN_ID = run_id


def _write_style_diagnostic(
    round_no: int,
    window_id: str,
    scene_idx: int,
    attempt: int,
    max_tokens: int | None,
    validation_ok: bool,
    validation_reason: str | None,
    meta: dict | None,
    dispatch_order: int | None = None,
    completion_order: int | None = None,
) -> None:
    """写一条 llmb 窗口诊断（脱敏，不含推理正文/提示词）。

    dispatch_order/completion_order 是阶段 2 有界并发新增的可选诊断字段，
    仅写入诊断 JSONL，不进发布契约。
    """
    if _DIAGNOSTICS_DIR is None:
        return
    entry: dict[str, object] = {
        "run_id": _DIAGNOSTICS_RUN_ID,
        "round_no": round_no,
        "window_id": window_id,
        "scene_index": scene_idx,
        "attempt": attempt,
        "max_tokens": max_tokens,
        "validation_ok": validation_ok,
        "validation_reason": validation_reason,
    }
    if dispatch_order is not None:
        entry["dispatch_order"] = dispatch_order
    if completion_order is not None:
        entry["completion_order"] = completion_order
    meta = meta or {}
    if meta.get("finish_reason") is not None:
        entry["finish_reason"] = meta["finish_reason"]
    if meta.get("http_status") is not None:
        entry["http_status"] = meta["http_status"]
    if meta.get("reasoning_chars") is not None:
        entry["reasoning_chars"] = meta["reasoning_chars"]
    if meta.get("error") is not None:
        entry["error"] = meta["error"]
    usage = meta.get("usage")
    if isinstance(usage, dict):
        entry["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        }
    with _DIAGNOSTICS_LOCK:
        diag_path = _DIAGNOSTICS_DIR / f"{_DIAGNOSTICS_RUN_ID}_diagnostics.jsonl"
        with diag_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _pace_label(scene_hype: float) -> str:
    """与 Prompt/json/hype_rules.json emotions 阈值保持一致：>=0.72 高、>=0.35 中、其余低。"""
    if scene_hype >= 0.72:
        return "高"
    if scene_hype >= 0.35:
        return "中"
    return "低"


class _ValidatedStyleCommentary(str):
    """校验通过的口播文本，附带原始响应，供最终 accept 反馈给后端。"""

    def __new__(cls, value: str, raw_response: object) -> "_ValidatedStyleCommentary":
        result = super().__new__(cls, value)
        result.raw_response = raw_response
        return result

    def accept(self, output: str) -> None:
        accept_api_response(self.raw_response, output=output)


# 无状态生成：不累积任何对话历史。
def _style_retry_instruction(validation: dict, delivery: dict, neutral: str) -> str:
    reason = str(validation.get("reason") or "")
    details = validation.get("details") or []
    hard = delivery.get("hard_char_limit", 100)
    out = validation.get("output_chars")
    if reason == "over_budget":
        out = details[0] if len(details) > 0 else validation.get("output_chars")
        soft = details[1] if len(details) > 1 else hard
        prior = details[3] if len(details) > 3 else ""
        if prior:
            return f"你上一次输出「{prior}」（{out}字），超过本窗口上限{soft}字。保留同样的事实，压缩到{soft}字以内的完整短句。"
        return f"你上一次输出{out}字，超过本窗口上限{soft}字。压缩到{soft}字以内的完整短句。"
    if reason == "under_budget":
        return "篇幅过短（不足预算六成），请在0.8~1.2区间内补足语气/细节，保留全部 required facts。"
    if reason == "missing_anchor":
        missing_str = "、".join(str(d) for d in details[:6])
        return f"你上一次输出缺少了事实: {missing_str}。必须保留中性稿里的全部事实，从allowed_event_terms中选词。"
    if reason == "unexpected_fact":
        extra_str = "、".join(str(d) for d in details[:6])
        return f"你上一次输出新增了未经授权的事实: {extra_str}。只改写原中性事实，不得增加任何新内容。"
    if reason == "high_repetition":
        phrase = details[0] if details else "某短语"
        return f"你上一次输出的风格短语\"{phrase}\"与近期窗口重复。换一种说法，但不能删改任何事实。"
    if reason == "incomplete_sentence":
        return f"你上一次输出是残句（不以完整标点结尾或括号不成对）。输出完整口语句。"
    if reason == "invalid_emotion":
        return f"你上一次输出的情绪标签不合法。只使用[平述]、[激动]、[惊叹]三种标签。"
    if reason == "empty_commentary":
        return f"你上一次输出为空。必须输出非空的口播文本。"
    if reason == "response_error":
        return f"你上一次输出不是合法的JSON对象。只输出{{\"commentary\":\"...\",\"felt_intensity\":0.5}}。"
    return "只改写原中性事实，输出预算内的完整短句；不得增加任何新事实。"


def _call_style(system: str, user_prompt: str, llm_cfg: dict, gen_fn, round_no: int = 0, scene_idx: int = 0, debug: bool = False,
                max_tokens: int | None = None, log_ctx: dict | None = None) -> tuple[str, float, dict]:
    """返回 (口播文本, felt_intensity, meta)；走 /api/generate，无状态。"""
    meta: dict[str, object] = {}
    try:
        raw = gen_fn(user_prompt, llm_cfg, system, max_tokens=max_tokens, log_ctx=log_ctx,
                     response_format={"type": "json_object"})
    except Exception as exc:
        meta["error"] = type(exc).__name__
        # 阶段 2：记录基础设施错误分类，供回合末兜底重试的跳过判断。
        response = getattr(exc, "response", None)
        http_status = getattr(response, "status_code", None)
        if isinstance(http_status, int):
            meta["http_status"] = http_status
        meta["retry_category"] = _request_error_category(exc)
        return f"[style error: {exc}]", 0.0, meta
    # 诊断元数据：usage / finish_reason / reasoning 摘要（api 后端 _ApiChatResult 携带）
    if getattr(raw, "finish_reason", None) is not None:
        meta["finish_reason"] = raw.finish_reason
    if getattr(raw, "http_status", None) is not None:
        meta["http_status"] = raw.http_status
    raw_usage = getattr(raw, "usage", None)
    if isinstance(raw_usage, dict):
        meta["usage"] = raw_usage
    reasoning = getattr(raw, "reasoning_content", "")
    if isinstance(reasoning, str) and reasoning:
        meta["reasoning_chars"] = len(reasoning)

    data = _extract_json_obj(raw)
    # Qwen3 可能输出 JSON 级 think/reasoning 字段；严格校验前剥离。
    if data is not None:
        for _key in ("think", "reasoning", "reasoning_content"):
            data.pop(_key, None)
    if data is None or set(data) != {"commentary", "felt_intensity"}:
        commentary, felt = "[style error: unparseable]", 0.0
    else:
        raw_commentary = data.get("commentary")
        commentary = raw_commentary if isinstance(raw_commentary, str) else ""
        if (not isinstance(raw_commentary, str)
                or not commentary.strip()
                or any(marker in commentary for marker in _LEAK_MARKERS)
                or commentary.lstrip().startswith("{")):
            error = "contract-leak" if isinstance(raw_commentary, str) else "unparseable"
            commentary, felt = f"[style error: {error}]", 0.0
        else:
            try:
                raw_felt = data["felt_intensity"]
                if isinstance(raw_felt, bool):
                    raise ValueError("boolean intensity")
                felt = float(raw_felt)
                if not math.isfinite(felt) or not 0.0 <= felt <= 1.0:
                    raise ValueError("intensity outside [0, 1]")
                commentary = _ValidatedStyleCommentary(commentary, raw)
            except (TypeError, ValueError):
                commentary, felt = "[style error: unparseable]", 0.0

    # debug 模式落盘：记录本次无状态请求与响应原文。
    if debug:
        debug_dir = _PROJECT_ROOT / "output" / "debug_phase3"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dump = {
            "round_no":      round_no,
            "model":         llm_cfg.get("model", ""),
            "phase":         "3b_style",
            "system_prompt": system,
            "user_prompt":   user_prompt,
            "response_raw":  raw,
            "commentary":    commentary,
            "felt_intensity": felt,
        }
        out = debug_dir / f"r{round_no:03d}_s{scene_idx:02d}_3b_style.json"
        out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")

    return commentary, felt, meta


# ── 阶段 2：LLM-B 有界并发（确定性准备 → 并发请求 → 顺序验收/提交）──

# 可恢复基础设施错误类别：transport/http/rate_limit 等；业务层失败
# （response_error/预算/事实类）不算，不享受回合末重复补偿。
_RECOVERABLE_INFRA_CATEGORIES = frozenset({
    "transport_error", "http_error", "rate_limit",
    "timeout", "connection_error", "server_error",
})


def _is_recoverable_infra_failure(meta: dict) -> bool:
    """判定一次失败是否属于可恢复基础设施错误（transport/http/rate_limit 等）。

    供回合末兜底重试的跳过判断：仅当主循环失败类别明确属于可恢复基础设施
    错误时，才允许再补偿一次；HTTP 200 形态/业务验收类失败不再重复补偿。
    """
    category = meta.get("retry_category")
    if isinstance(category, str) and category in _RECOVERABLE_INFRA_CATEGORIES:
        return True
    http_status = meta.get("http_status")
    if isinstance(http_status, int):
        return http_status in (408, 429) or http_status >= 500
    error = str(meta.get("error") or "")
    return any(token in error for token in (
        "ConnectionError", "Timeout", "SSLError",
        "ChunkedEncodingError", "RetryError",
    ))


# ── 阶段 3：commentary v3 稀疏候选（voice task）辅助 ──

_VOICE_TASK_MAX_SPEED_FACTOR = 1.5


def _classify_risk(safe_upper_sec: float | None, slot_sec: float, max_speed_factor: float) -> str:
    """§5.1 U/S/M 风险公式：green: U<=S；amber: S<U<=S*M；red: U>S*M。

    无安全上界（profile 不可用）时按 unknown 处理，禁止正式分级。
    """
    if safe_upper_sec is None:
        return "unknown"
    if safe_upper_sec <= slot_sec:
        return "green"
    if safe_upper_sec <= slot_sec * max_speed_factor:
        return "amber"
    return "red"


def _profile_readiness(profile_id: str, voice_task_cfg: dict) -> tuple[dict | None, str]:
    """profile 就绪检查：存在且 validated 且引擎/声线/预处理指纹与配置完全一致。

    返回 (profile, status)：status ∈ {"ok", "profile_id_missing", "profile_missing",
    "profile_not_validated", "fingerprint_mismatch"}。任一失败都不允许正式风险分级
    （§1.3/§5.1：profile 缺失或指纹不匹配只写 risk_class=unknown 影子诊断）。
    """
    if not profile_id:
        return None, "profile_id_missing"
    profile = speech_measure.load_profile(profile_id)
    if profile is None:
        return None, "profile_missing"
    status = speech_measure.validate_profile_status(profile)
    if status != "validated":
        return profile, "profile_not_validated"
    engine = str(voice_task_cfg.get("engine_fingerprint") or "")
    voice = str(voice_task_cfg.get("voice_fingerprint") or "")
    preprocess = str(voice_task_cfg.get("preprocess_fingerprint") or "")
    if not (engine and voice and preprocess):
        return profile, "fingerprint_mismatch"
    if not speech_measure.check_profile_match(
        profile,
        engine_fingerprint=engine,
        voice_fingerprint=voice,
        preprocess_fingerprint=preprocess,
    ):
        return profile, "fingerprint_mismatch"
    return profile, "ok"


def _write_voice_task_diagnostic(entry: dict) -> None:
    """写一条 voice task 诊断（risk/候选成败），不落正文候选。"""
    if _DIAGNOSTICS_DIR is None:
        return
    record = {"run_id": _DIAGNOSTICS_RUN_ID, **entry}
    with _DIAGNOSTICS_LOCK:
        diag_path = _DIAGNOSTICS_DIR / f"{_DIAGNOSTICS_RUN_ID}_voice_task.jsonl"
        with diag_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_unknown_shadow_diagnostics(neutral_data: dict, reason: str) -> None:
    """profile 不可用时的影子诊断：每个非静默窗口写 risk_class=unknown（§5.1）。"""
    for round_data in neutral_data.get("rounds", []):
        if not isinstance(round_data, dict):
            continue
        for scene in round_data.get("scenes", []):
            if not isinstance(scene, dict) or not str(scene.get("neutral") or "").strip():
                continue
            _write_voice_task_diagnostic({
                "window_id": str(scene.get("window_id") or ""),
                "variant_id": None,
                "risk_class": "unknown",
                "pre_risk_class": None,
                "semantic_state": "unknown",
                "reason": reason,
                "safe_duration_upper_bound_at_base_speed_sec": None,
            })


def _v4_preserved_facts(text: str, scene: dict) -> dict:
    """preserved_fact_ids 由原子级校验器计算（§9.2），不接受模型自报。"""
    from sbmachine.phase3a_prompt import compute_preserved_fact_ids
    return compute_preserved_fact_ids(
        _strip_tags(text),
        scene.get("fact_catalog") or [],
        scene.get("required_fact_ids") or [],
    )


def _v3_candidate(
    *,
    variant_id: str,
    source: str,
    text: str,
    felt_intensity: float,
    preserved_fact_ids: list[str],
    profile_id: str,
    profile: dict,
    slot_sec: float,
) -> dict:
    """组装单个候选（§9.4）：时长/单位由 speech_measure 计算，min_speed 由校验器建议。"""
    measure = speech_measure.measure_text(text, profile_id=profile_id)
    speed = speech_measure.estimate_required_speed_factor(text, profile, slot_sec)
    return {
        "variant_id": variant_id,
        "source": source,
        "text": text,
        "felt_intensity": round(float(felt_intensity), 3),
        "spoken_units": int(measure.get("spoken_units", 0)),
        "safe_duration_upper_bound_at_base_speed_sec": measure.get("safe_duration_upper_bound_at_base_speed_sec"),
        "minimum_required_speed_factor": None if speed is None else round(float(speed), 2),
        "preserved_fact_ids": list(preserved_fact_ids),
    }


def _v4_render_slot(scene: dict) -> dict:
    slot = scene.get("render_slot")
    if isinstance(slot, dict):
        return {
            "start_sec": float(slot.get("start_sec") if slot.get("start_sec") is not None else scene.get("t_start")),
            "end_sec": float(slot.get("end_sec") if slot.get("end_sec") is not None else scene.get("t_end")),
            "start_tick": int(slot.get("start_tick", 0)),
            "end_tick": int(slot.get("end_tick", 0)),
            "gap_policy": str(slot.get("gap_policy") or "independent_window"),
        }
    t_start = float(scene.get("t_start", 0.0))
    t_end = float(scene.get("t_end", t_start + 1.0))
    return {
        "start_sec": t_start,
        "end_sec": t_end,
        "start_tick": int(round(t_start * 30)),
        "end_tick": int(round(t_end * 30)),
        "gap_policy": "independent_window",
    }


class _StyleWindowPlan:
    """窗口的不可变请求计划：预计算阶段按时间顺序生成，供并发 worker 只读。

    worker 只读取 user_prompt/scene_max_tokens/log_ctx 等预计算字段，
    不修改任何全局结构；scene 仅由主线程验收时读取。
    """

    __slots__ = (
        "scene_idx", "window_id", "scene", "scene_neutral", "scene_hype",
        "duration", "user_prompt", "anchors", "delivery",
        "scene_max_tokens", "log_ctx",
    )

    def __init__(
        self,
        *,
        scene_idx: int,
        window_id: str,
        scene: dict,
        scene_neutral: str,
        scene_hype: float,
        duration: float,
        user_prompt: str,
        anchors: dict,
        delivery: dict,
        scene_max_tokens: int,
        log_ctx: dict,
    ) -> None:
        self.scene_idx = scene_idx
        self.window_id = window_id
        self.scene = scene
        self.scene_neutral = scene_neutral
        self.scene_hype = scene_hype
        self.duration = duration
        self.user_prompt = user_prompt
        self.anchors = anchors
        self.delivery = delivery
        self.scene_max_tokens = scene_max_tokens
        self.log_ctx = log_ctx


def _style_request_worker(
    plan: _StyleWindowPlan,
    round_no: int,
    debug: bool,
    system_content: str,
    llm_cfg: dict,
    gen_fn,
    prompt: str | None = None,
) -> tuple[str, float, dict]:
    """worker 只执行一次无状态生成并返回候选结果（不写任何共享结构）。

    prompt 为 None 时使用 plan 的预计算 user_prompt；重试路径由主线程
    构造带 retry_feedback 的新 prompt 传入。
    """
    return _call_style(
        system_content,
        plan.user_prompt if prompt is None else prompt,
        llm_cfg,
        gen_fn,
        round_no=round_no,
        scene_idx=plan.scene_idx,
        debug=debug,
        max_tokens=plan.scene_max_tokens,
        log_ctx=plan.log_ctx,
    )


# ── 阶段 3：commentary v3 稀疏候选生产（voice task，§5.1/§9.4）──

def _run_phase3b_v4(
    *,
    neutral_path: Path,
    neutral_data: dict,
    match,
    config: dict,
    llm_cfg: dict,
    style_cfg: dict,
    gen_fn,
    backend: str,
    system_content: str,
    emotion_policy: EmotionPolicy,
    aliases: dict,
    voice_task_cfg: dict,
    profile: dict,
    profile_id: str,
    debug_enabled: bool,
    output_rounds_path: Path,
    commentary_path: Path,
    draft_package_path: Path | None,
    draft_contract_version: int,
    render_timebase_fps: float,
    dry_run: bool,
    progress_sink=None,
) -> dict:
    """Phase3b v4 生产路径：按风险稀疏生成候选并写 commentary v3。

    进入本函数前已确认：neutral schema v4、voice_task 配置启用、profile
    validated 且指纹匹配。场景流程（§5.1 两次判定）：
    - 预判：以 neutral 的安全上界 U 与 slot S 判定 green/amber/red；red 直接 compact。
    - 终判：primary 生成后按实际文本重新测量；green/amber 维持，变 red 则 primary
      只进诊断并改生成 compact。
    - capsule 由规则层直接组装（不调 LLM-B）。
    """
    max_speed_factor = float(voice_task_cfg.get("max_speed_factor") or _VOICE_TASK_MAX_SPEED_FACTOR)
    if max_speed_factor < 1.0 or max_speed_factor > _VOICE_TASK_MAX_SPEED_FACTOR:
        # 合同上界：Phase4 全局物理边界 1.5x，任何来源都不可突破（审计 §4/§5.1）。
        max_speed_factor = _VOICE_TASK_MAX_SPEED_FACTOR
    if backend == "api" and style_cfg.get("cloud_style_output_max_tokens"):
        style_output_max_tokens = max(1, int(style_cfg["cloud_style_output_max_tokens"]))
    else:
        style_output_max_tokens = max(1, int(style_cfg["style_output_max_tokens"]))
    recent_limit = max(1, int(style_cfg["style_recent_window_count"]))
    max_retries = max(0, min(2, int(style_cfg["style_max_retries"])))
    char_tolerance = min(0.5, max(0.0, float(style_cfg.get("style_budget_hard_tolerance", 0.0))))
    # 强事实依据模式（semantic.strong_fact_mode）：False=全面相信 LLM，跳过 unexpected_fact。
    strong_fact_mode = bool((config.get("semantic") or {}).get("strong_fact_mode", False))
    recent_style_phrases: list[str] = []
    neutral_by_round: dict[int, dict] = {
        int(r["round_no"]): r
        for r in neutral_data.get("rounds", [])
    }
    source_neutral_sha256 = hashlib.sha256(neutral_path.read_bytes()).hexdigest()
    run_id = str(neutral_data["run_id"])

    voice_tasks: list[dict] = []
    errors: list[dict] = []
    accepted_run_samples: list[tuple[_ValidatedStyleCommentary, str]] = []

    for completed, rnd in enumerate(tqdm(match.rounds, desc="Phase3b voice task", unit="round"), start=1):
        round_data = neutral_by_round.get(rnd.round_no, {})
        scenes = round_data.get("scenes", [])
        avg_hype = float(round_data.get("avg_hype", 0.0))
        analyst_failed = bool(round_data.get("analyst_failed", False))
        scene_commentaries: list[str] = []
        round_scenes_v3: list[dict] = []
        round_scream_used = False

        for scene_idx, scene in enumerate(scenes):
            window_id = str(scene.get("window_id") or f"r{rnd.round_no:03d}_w{scene_idx + 1:02d}")
            scene_neutral = str(scene.get("neutral") or "")
            neutral_source = str(scene.get("neutral_source") or "")
            scene_hype = float(scene.get("hype", avg_hype))
            char_budget = max(1, int(scene.get("char_budget", 100)))
            t_start = float(scene.get("t_start", rnd.start_sec))
            t_end = float(scene.get("t_end", rnd.end_sec))

            if dry_run:
                continue
            if analyst_failed:
                continue
            if neutral_source == "unrecoverable":
                continue
            if not scene_neutral.strip():
                continue
            if neutral_source not in {"rule_template", "tiny_assembler"}:
                continue

            slot = _v4_render_slot(scene)
            slot_sec = max(1.0, slot["end_sec"] - slot["start_sec"])
            scream_eligible = bool(scene.get("scream_eligible", False)) and not round_scream_used
            log_ctx = {"run_id": run_id, "round": f"round{rnd.round_no}", "scene": window_id}
            scene_max_tokens = min(
                style_output_max_tokens,
                max(96, int(char_budget * 2.2) + 80),
            ) if backend != "api" else style_output_max_tokens

            # ── 预判（§5.1）：用 neutral 与 validated profile 的安全上界 U vs S ──
            pre_measure = speech_measure.measure_text(scene_neutral, profile_id=profile_id)
            pre_upper = pre_measure.get("safe_duration_upper_bound_at_base_speed_sec")
            pre_risk = _classify_risk(pre_upper, slot_sec, max_speed_factor)
            diag_entries: list[dict] = [{
                "window_id": window_id, "variant_id": None,
                "risk_class": pre_risk, "pre_risk_class": pre_risk,
                "semantic_state": "pre_judge", "reason": "",
                "safe_duration_upper_bound_at_base_speed_sec": pre_upper,
            }]
            production: list[dict] = []
            final_risk = "red" if pre_risk == "red" else pre_risk
            primary_measured_risk: str | None = None
            accepted_candidate = None  # (raw_candidate, normalized, decision, felt)

            def call_and_validate(variant_kind: str, max_tokens: int) -> tuple[str, float, dict]:
                retry_feedback = None
                attempts = max_retries + 1 if variant_kind == "primary" else 1
                for attempt in range(attempts):
                    user_prompt, anchors, delivery = build_style_prompt(
                        scene, aliases, recent_style_phrases[-recent_limit:],
                        retry_feedback=retry_feedback,
                        variant_kind=variant_kind, slot_duration_sec=slot_sec,
                        max_speed_factor=max_speed_factor,
                    )
                    if backend == "api":
                        from sbmachine import cloud_prompts
                        user_prompt = cloud_prompts.inject_window_type(user_prompt, scene)
                    candidate, felt, meta = _call_style(
                        system_content, user_prompt, llm_cfg, gen_fn,
                        round_no=rnd.round_no, scene_idx=scene_idx, debug=debug_enabled,
                        max_tokens=max_tokens, log_ctx=log_ctx,
                    )
                    if candidate.startswith("[style error:") or any(
                        marker in candidate for marker in _CONTAMINATION_MARKERS
                    ):
                        validation = {"ok": False, "reason": "response_error", "details": [candidate], "output_chars": None, "signature": ""}
                    else:
                        validation = validate_style_commentary(
                            candidate, scene_neutral, anchors, aliases,
                            recent_style_phrases[-recent_limit:],
                            hard_char_limit=delivery["hard_char_limit"],
                            char_tolerance=char_tolerance,
                            strong_fact_mode=strong_fact_mode,
                            enforce_min_budget=variant_kind == "primary",
                        )
                    _write_style_diagnostic(
                        rnd.round_no, window_id, scene_idx, attempt, max_tokens,
                        bool(validation["ok"]), str(validation["reason"]), meta,
                    )
                    if validation["ok"] or attempt + 1 >= attempts:
                        return candidate, felt, validation
                    retry_feedback = {
                        "failure_reason": validation["reason"],
                        "details": validation["details"],
                        "instruction": _style_retry_instruction(validation, delivery, scene_neutral),
                    }
                raise AssertionError("unreachable")

            def record_variant(variant_id: str, plain_text: str, semantic_state: str, reason: str, felt: float, measure: dict) -> None:
                speed = speech_measure.estimate_required_speed_factor(plain_text, profile, slot_sec)
                _write_voice_task_diagnostic({
                    "window_id": window_id, "variant_id": variant_id,
                    "risk_class": final_risk, "pre_risk_class": pre_risk,
                    "semantic_state": semantic_state, "reason": reason,
                    "felt_intensity": round(float(felt), 3),
                    "spoken_units": int(measure.get("spoken_units", 0)),
                    "safe_duration_upper_bound_at_base_speed_sec": measure.get("safe_duration_upper_bound_at_base_speed_sec"),
                    "minimum_required_speed_factor": None if speed is None else round(float(speed), 2),
                })

            # ── primary（预判 red 不生成）──
            if pre_risk != "red":
                candidate, felt, validation = call_and_validate("primary", scene_max_tokens)
                if validation["ok"]:
                    plain = _strip_tags(candidate)
                    primary_measure = speech_measure.measure_text(plain, profile_id=profile_id)
                    primary_upper = primary_measure.get("safe_duration_upper_bound_at_base_speed_sec")
                    primary_measured_risk = _classify_risk(primary_upper, slot_sec, max_speed_factor)
                    if primary_measured_risk == "red":
                        # 终判降级：primary 只进诊断，随后生成 compact（§5.1-2）
                        final_risk = "red"
                        record_variant("primary", plain, "timing_red", "primary_exceeds_max_speed", felt, primary_measure)
                    else:
                        final_risk = primary_measured_risk
                        decision = emotion_policy.decide(
                            hard_intensity=scene_hype, llmb_intensity=felt,
                            scream_eligible=scream_eligible,
                        )
                        if decision.label == "惊叹":
                            round_scream_used = True
                        normalized = normalize_commentary_emotion(candidate, decision.label)
                        preserved = _v4_preserved_facts(normalized, scene)
                        if preserved["missing_required"] or preserved["unexpected_fact_ids"]:
                            reason = f"missing_required={preserved['missing_required']};unexpected={preserved['unexpected_fact_ids']}"
                            record_variant("primary", plain, "rejected", reason, felt, primary_measure)
                        else:
                            production.append(_v3_candidate(
                                variant_id="primary", source="llmb", text=plain,
                                felt_intensity=decision.score,
                                preserved_fact_ids=preserved["preserved_fact_ids"],
                                profile_id=profile_id, profile=profile, slot_sec=slot_sec,
                            ))
                            record_variant("primary", plain, "ok", "", felt, primary_measure)
                            accepted_candidate = (candidate, normalized, decision, felt)
                else:
                    record_variant("primary", _strip_tags(candidate), "rejected", str(validation["reason"]), felt, speech_measure.measure_text(_strip_tags(candidate), profile_id=profile_id))

            # ── 终判：primary 实测 red → final red；green 不生成 compact ──
            if pre_risk == "red":
                final_risk = "red"
            elif primary_measured_risk is not None:
                final_risk = primary_measured_risk
            else:
                final_risk = pre_risk

            if final_risk in ("amber", "red"):
                candidate_c, felt_c, validation_c = call_and_validate("compact", scene_max_tokens)
                if validation_c["ok"]:
                    plain_c = _strip_tags(candidate_c)
                    compact_measure = speech_measure.measure_text(plain_c, profile_id=profile_id)
                    preserved_c = _v4_preserved_facts(candidate_c, scene)
                    if preserved_c["missing_required"] or preserved_c["unexpected_fact_ids"]:
                        reason_c = f"missing_required={preserved_c['missing_required']};unexpected={preserved_c['unexpected_fact_ids']}"
                        record_variant("compact", plain_c, "rejected", reason_c, felt_c, compact_measure)
                    else:
                        decision_c = emotion_policy.decide(
                            hard_intensity=scene_hype, llmb_intensity=felt_c, scream_eligible=False,
                        )
                        normalized_c = normalize_commentary_emotion(candidate_c, decision_c.label)
                        production.append(_v3_candidate(
                            variant_id="compact", source="llmb_compact", text=plain_c,
                            felt_intensity=decision_c.score,
                            preserved_fact_ids=preserved_c["preserved_fact_ids"],
                            profile_id=profile_id, profile=profile, slot_sec=slot_sec,
                        ))
                        record_variant("compact", plain_c, "ok", "", felt_c, compact_measure)
                else:
                    record_variant("compact", _strip_tags(candidate_c), "rejected", str(validation_c["reason"]), felt_c, speech_measure.measure_text(_strip_tags(candidate_c), profile_id=profile_id))

            # ── capsule：规则层直接组装，不调 LLM-B（§9.4 source=rule_capsule）──
            capsule_text = str(scene.get("rule_capsule") or "").strip()
            if capsule_text:
                capsule_plain = _strip_tags(capsule_text)
                preserved_capsule = _v4_preserved_facts(capsule_text, scene)
                capsule_measure = speech_measure.measure_text(capsule_plain, profile_id=profile_id)
                capsule_label, capsule_score = capsule_emotion(scene_hype)
                if preserved_capsule["missing_required"] or preserved_capsule["unexpected_fact_ids"]:
                    reason_capsule = f"missing_required={preserved_capsule['missing_required']};unexpected={preserved_capsule['unexpected_fact_ids']}"
                    record_variant("capsule", capsule_plain, "rejected", reason_capsule, capsule_score, capsule_measure)
                else:
                    production.append(_v3_candidate(
                        variant_id="capsule", source="rule_capsule", text=capsule_plain,
                        felt_intensity=capsule_score,
                        preserved_fact_ids=preserved_capsule["preserved_fact_ids"],
                        profile_id=profile_id, profile=profile, slot_sec=slot_sec,
                    ))
                    record_variant("capsule", capsule_plain, "ok", "", capsule_score, capsule_measure)
            else:
                record_variant("capsule", "", "rejected", "missing_rule_capsule", 0.0, speech_measure.measure_text("", profile_id=profile_id))

            for entry in diag_entries:
                _write_voice_task_diagnostic(entry)

            if not production:
                # §1.1：关键候选全部失败 → 显式失败，不伪装 silent。
                errors.append({"round": f"round{rnd.round_no}", "round_no": rnd.round_no, "scene": window_id, "error": "all_variants_failed", "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                continue

            task = {
                "voice_task_id": window_id,
                "window_id": window_id,
                "render_slot": slot,
                "required_fact_ids": [str(fid) for fid in (scene.get("required_fact_ids") or [])],
                "speech_profile_id": profile_id,
                "risk_class": final_risk,
                "selection_order": [c["variant_id"] for c in production],
                "max_speed_factor": max_speed_factor,
                "candidates": [dict(c) for c in production],
                "semantic_state": "ok",
            }
            voice_tasks.append(task)

            primary_candidate = next((c for c in production if c["variant_id"] == "primary"), None)
            if accepted_candidate is not None and primary_candidate is not None:
                _raw_candidate, primary_normalized, decision_primary, felt_primary = accepted_candidate
                primary_label = decision_primary.label
                primary_score = decision_primary.score
            else:
                primary_normalized = None
                primary_label = "平述"
                primary_score = 0.0
            if primary_candidate is not None and primary_normalized is None:
                primary_label = "平述"
                primary_score = primary_candidate["felt_intensity"]
                primary_normalized = f"[{primary_label}]{primary_candidate['text']}"
            scene_text = primary_normalized.removeprefix(f"[{primary_label}]") if primary_normalized else production[0]["text"]
            scene_label = primary_label if primary_normalized else capsule_label
            output_chars = count_spoken_chars(_strip_tags(primary_normalized if primary_normalized else scene_text))
            round_scenes_v3.append({
                "window_id": window_id,
                "t_start": t_start,
                "t_end": t_end,
                "text": scene_text,
                "emotion": scene_label,
                "voice_task_id": window_id,
                "primary_variant_id": primary_candidate["variant_id"] if primary_candidate else None,
                "emotion_score": round(primary_score if primary_normalized else capsule_score, 3),
                "hard_intensity": round(scene_hype, 3),
                "char_budget": char_budget,
                "output_chars": output_chars,
                "style_status": "ok",
            })
            if primary_normalized:
                scene_commentaries.append(primary_normalized)
                if accepted_candidate is not None and isinstance(accepted_candidate[0], _ValidatedStyleCommentary):
                    accepted_run_samples.append((accepted_candidate[0], json.dumps(
                        {"commentary": primary_normalized, "felt_intensity": felt_primary},
                        ensure_ascii=False, separators=(",", ":"),
                    )))

        commentary = "".join(scene_commentaries)
        parsed = parse_emotional_text(commentary)
        rnd.phase3_semantic = SemanticData(
            model_profile=str(config.get("profile", "style")),
            model_name=str(llm_cfg.get("model", "")),
            commentary_text=commentary,
            emotion_segments=[EmotionSegment(seg.emotion, seg.text, i) for i, seg in enumerate(parsed)],
        )
        rnd.scenes = round_scenes_v3
        if progress_sink is not None:
            try:
                progress_sink(completed, len(match.rounds), "round", None)
            except Exception:
                pass

    if errors:
        err_path = _PROJECT_ROOT / "logs" / "error.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if err_path.exists():
            try:
                existing = json.loads(err_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.extend(errors)
        err_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    for rnd in match.rounds:
        rnd.phase2_yolo = None
    # §9.7：commentary v3 必须先过权威结构契约（voice_task_contract.validate_commentary_v3）
    # 再落盘；缺字段拒绝，不隐式退化。全部场景均无生产候选时显式失败（§1.1）。
    v3_payload = {
        "commentary_schema_version": 3,
        "voice_task_contract_version": 1,
        "candidate_policy": str(voice_task_cfg.get("candidate_policy") or "sparse_v1"),
        "speech_metric_version": speech_measure.METRIC_VERSION,
        "source_neutral_run_id": run_id,
        "source_neutral_sha256": source_neutral_sha256,
        "voice_tasks": voice_tasks,
        "rounds": [],
    }
    contract_errors = [] if dry_run else validate_commentary_v3(v3_payload)
    if contract_errors:
        raise ValueError("commentary v3 does not pass voice_task_contract: " + "; ".join(contract_errors))
    save_match(output_rounds_path, match)
    write_json(commentary_path, v3_payload)
    # rounds_with_commentary v3 需要在顶层带 source_neutral_sha256（§10.2/§9.7）。
    rounds_v3_payload = json.loads(output_rounds_path.read_text(encoding="utf-8"))
    rounds_v3_payload["source_neutral_sha256"] = source_neutral_sha256
    output_rounds_path.write_text(json.dumps(rounds_v3_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if draft_package_path is not None and not dry_run:
        if draft_contract_version != 2:
            raise ValueError("schema v4 voice-task Phase3b requires llmb_draft_package_v2")
        _export_llmb_draft_package_v2(
            {
                "source_neutral_run_id": run_id,
                "source_neutral_sha256": source_neutral_sha256,
            },
            neutral_path,
            match.video_path,
            render_timebase_fps,
            draft_package_path,
            config=config,
            rounds_payload=rounds_v3_payload,
        )
    for response, output in accepted_run_samples:
        response.accept(output)
    return v3_payload


# ── B 封存包导出（llmb_draft_package_v1，docs/plan/phase3c-llmc-one-way-handoff-plan.md §3.1/§8 B1）──

_LLMB_DRAFT_PACKAGE_CONTRACT = "llmb_draft_package_v1"
_LLMB_DRAFT_PACKAGE_V2_CONTRACT = "llmb_draft_package_v2"
# v2 fact adapter 的 7 类固定顺序（players/teams/numbers/events/results/locations/weapons）
_FACT_KINDS = ("players", "teams", "numbers", "events", "results", "locations", "weapons")
# 兼容期基准语速（字/秒）：safe_upper_sec = 口播字数 / 5.0（§6.5 字符预算兼容期口径）
_BASE_SPEECH_RATE_CHAR_PER_SEC = 5.0
_B_DRAFT_HARD_SPEED_FACTOR = 1.5  # B 草稿硬线：B 文本安全时长上界 / slot <= 1.5（§6.2）


def _normalize_fact_value(value: object) -> str:
    """v2 fact adapter 值规范化：int/float 统一为字符串且去 .0（3→"3"、3.0→"3"）。

    事件/结果类为事件代码字符串，原样 str；bool 与 1/0 区分，防误归一。
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def _fact_catalog_from_anchors(unit_id: str, anchors: dict) -> dict:
    """从 fact_anchors 确定性重建 fact_catalog（fact_id 带值指纹，同类同值必同 ID）。

    fact_id = fact:v1:{unit_id}:{kind}:{seq:05d}:{sha256(kind|value)[:8]}；
    seq 为 kind 内 1 起递增，catalog 顺序即生成顺序。
    """
    catalog: dict[str, dict] = {}
    for kind in _FACT_KINDS:
        values = anchors.get(kind)
        if not isinstance(values, list):
            continue
        for seq, value in enumerate(values, start=1):
            if value is None:
                continue
            value_str = _normalize_fact_value(value)
            value_sha8 = hashlib.sha256(f"{kind}|{value_str}".encode("utf-8")).hexdigest()[:8]
            fact_id = f"fact:v1:{unit_id}:{kind}:{seq:05d}:{value_sha8}"
            catalog[fact_id] = {"kind": kind, "value": value_str}
    return catalog


def _build_unit_fact_scope(unit_id: str, neutral_scene: dict | None) -> tuple[list[str], dict]:
    """v2 fact adapter：为成功窗口建立确定性事实作用域（不猜 ID，不伪造）。

    fail-closed（与计划书 C0 入口同源一致）：成功窗口必须能建立完整事实作用域——
    neutral 对应 scene 缺失、fact_anchors 非 dict 或 7 类全部为空时抛 ValueError
    （窗口级 fact_anchors 全空但窗口有成功 B 稿时同样 fail-closed）。
    v4 路径 scene 已有 required_fact_ids 时直接使用并跳过 adapter：
    fact_catalog 仍从 fact_anchors 尽力重建，失败则 fact_catalog={}，
    但 allowed_fact_ids=required_fact_ids。
    """
    if neutral_scene is None:
        raise ValueError(f"unit {unit_id}: neutral scene missing, cannot build fact scope (fail-closed)")
    required_fact_ids = neutral_scene.get("required_fact_ids")
    if isinstance(required_fact_ids, list) and required_fact_ids:
        allowed = [str(fid) for fid in required_fact_ids]
        anchors = neutral_scene.get("fact_anchors")
        catalog: dict[str, dict] = {}
        if isinstance(anchors, dict):
            try:
                catalog = _fact_catalog_from_anchors(unit_id, anchors)
            except Exception:
                catalog = {}
        return allowed, catalog
    anchors = neutral_scene.get("fact_anchors")
    if not isinstance(anchors, dict):
        raise ValueError(f"unit {unit_id}: fact_anchors is not a dict, cannot build fact scope (fail-closed)")
    catalog = _fact_catalog_from_anchors(unit_id, anchors)
    if not catalog:
        raise ValueError(f"unit {unit_id}: fact_anchors empty for successful window, cannot build fact scope (fail-closed)")
    return list(catalog.keys()), catalog


def _unit_render_slot(unit_id: str, scene: dict, neutral_scene: dict | None, timeline_id: str, fps: float) -> dict:
    """render_slot：v4 neutral render_slot 透传（slot_id 固定 unit_id，timeline_id 同源）。

    无 render_slot（v2 路径）时由 scene 的 t_start/t_end × fps 构造 tick。
    """
    slot = neutral_scene.get("render_slot") if isinstance(neutral_scene, dict) else None
    t_start = float(scene.get("t_start", 0.0))
    t_end = float(scene.get("t_end", t_start + 1.0))
    if isinstance(slot, dict):
        start_tick, end_tick = slot.get("start_tick"), slot.get("end_tick")
        if not (isinstance(start_tick, int) and isinstance(end_tick, int)):
            start_tick = int(round(float(slot.get("start_sec", t_start)) * fps))
            end_tick = int(round(float(slot.get("end_sec", t_end)) * fps))
        return {
            "slot_id": unit_id,
            "timeline_id": timeline_id,
            "start_tick": start_tick,
            "end_tick": end_tick,
        }
    return {
        "slot_id": unit_id,
        "timeline_id": timeline_id,
        "start_tick": int(round(t_start * fps)),
        "end_tick": int(round(t_end * fps)),
    }


def _unit_render_slot_v2(unit_id: str, scene: dict, neutral_scene: dict | None, timeline_id: str, fps: float) -> dict:
    """Build an execution-ready slot while retaining authoritative seconds.

    Seconds come from the neutral render slot when present, otherwise from the
    scene window. Ticks are only a checked companion coordinate; they are never
    used to reconstruct the seconds.
    """
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError(f"unit {unit_id}: render_timebase_fps must be positive")
    try:
        scene_start = float(scene["t_start"])
        scene_end = float(scene["t_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unit {unit_id}: scene t_start/t_end are required") from exc
    if not math.isfinite(scene_start) or not math.isfinite(scene_end) or scene_start >= scene_end:
        raise ValueError(f"unit {unit_id}: scene t_start must be < t_end")

    source_slot = neutral_scene.get("render_slot") if isinstance(neutral_scene, dict) else None
    source_slot = source_slot if isinstance(source_slot, dict) else {}
    start_value = source_slot.get("start_sec", scene_start)
    end_value = source_slot.get("end_sec", scene_end)
    try:
        start_sec = float(start_value)
        end_sec = float(end_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unit {unit_id}: render_slot start_sec/end_sec are required") from exc
    if not math.isfinite(start_sec) or not math.isfinite(end_sec) or start_sec >= end_sec:
        raise ValueError(f"unit {unit_id}: render_slot start_sec must be < end_sec")
    if not math.isclose(start_sec, scene_start, abs_tol=1e-6) or not math.isclose(end_sec, scene_end, abs_tol=1e-6):
        raise ValueError(f"unit {unit_id}: authoritative render seconds do not match scene window")

    start_tick = source_slot.get("start_tick")
    end_tick = source_slot.get("end_tick")
    tick_rate = Fraction(str(float(fps)))
    if not isinstance(start_tick, int) or isinstance(start_tick, bool):
        start_tick = round_half_even(Fraction(str(start_sec)) * tick_rate)
    if not isinstance(end_tick, int) or isinstance(end_tick, bool):
        end_tick = round_half_even(Fraction(str(end_sec)) * tick_rate)
    if start_tick >= end_tick:
        raise ValueError(f"unit {unit_id}: render_slot start_tick must be < end_tick")
    if abs(float(start_tick) - start_sec * float(fps)) > 2.0 or abs(float(end_tick) - end_sec * float(fps)) > 2.0:
        raise ValueError(f"unit {unit_id}: render_slot seconds and ticks do not agree")
    return {
        "slot_id": unit_id,
        "timeline_id": timeline_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "start_tick": start_tick,
        "end_tick": end_tick,
    }


def _unit_speech_capacity(draft_text: str, t_start: float, t_end: float) -> dict:
    """speech_capacity：安全上界按基准语速 5 字/秒估算（兼容期口径，§6.5）。"""
    slot_sec = round(t_end - t_start, 3)
    safe_upper_sec = round(count_spoken_chars(draft_text) / _BASE_SPEECH_RATE_CHAR_PER_SEC, 3)
    required_speed_factor = round(max(1.0, safe_upper_sec / slot_sec if slot_sec > 0 else 1.0), 3)
    return {
        "slot_sec": slot_sec,
        "safe_upper_sec": safe_upper_sec,
        "required_speed_factor": required_speed_factor,
        "draft_hard_speed_factor": round(_B_DRAFT_HARD_SPEED_FACTOR, 3),
    }


def _file_sha256(path: str | Path | None) -> str:
    """文件 sha256（分块读，防大视频内存峰值）；路径缺失/文件不存在/不可读返回空串。"""
    if not path:
        return ""
    try:
        p = Path(path)
        if not p.is_file():
            return ""
        digest = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def _llmb_artifact_identity(package: dict) -> str:
    """artifact_identity：对排除自身后的全包 sort_keys 序列化求 sha256。"""
    body = {k: v for k, v in package.items() if k != "artifact_identity"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _llmb_tts_policy(config: dict | None, manifest: dict | None = None) -> dict:
    """Resolve the closed-set TTS policy recorded in B v2."""
    semantic = config.get("semantic", {}) if isinstance(config, dict) else {}
    semantic = semantic if isinstance(semantic, dict) else {}
    voice_cfg = semantic.get("voice_task", {})
    voice_cfg = voice_cfg if isinstance(voice_cfg, dict) else {}
    profile_id = str(voice_cfg.get("speech_profile_id") or "")
    if not profile_id and isinstance(manifest, dict):
        for task in manifest.get("voice_tasks", []):
            if isinstance(task, dict) and task.get("speech_profile_id"):
                profile_id = str(task["speech_profile_id"])
                break
    require_profile = bool(voice_cfg.get("require_validated_profile", True))
    max_speed = float(voice_cfg.get("max_speed_factor") or _VOICE_TASK_MAX_SPEED_FACTOR)
    max_speed = min(_VOICE_TASK_MAX_SPEED_FACTOR, max(1.0, max_speed))
    if not require_profile:
        profile_status = "not_required"
    else:
        profile = speech_measure.load_profile(profile_id) if profile_id else None
        if profile is None:
            profile_status = "missing"
        else:
            try:
                status = speech_measure.validate_profile_status(profile)
            except speech_measure.ProfileError:
                status = "stale"
            if status == "stale":
                profile_status = "stale"
            elif status != "validated":
                profile_status = "stale"
            else:
                configured = (
                    str(voice_cfg.get("engine_fingerprint") or ""),
                    str(voice_cfg.get("voice_fingerprint") or ""),
                    str(voice_cfg.get("preprocess_fingerprint") or ""),
                )
                if all(configured) and not speech_measure.check_profile_match(
                    profile,
                    engine_fingerprint=configured[0],
                    voice_fingerprint=configured[1],
                    preprocess_fingerprint=configured[2],
                ):
                    profile_status = "mismatch"
                else:
                    profile_status = "validated"
    return {
        "speech_profile_id": profile_id,
        "profile_status": profile_status,
        "require_validated_profile": require_profile,
        "max_speed_factor": round(max_speed, 3),
    }


def _llmb_timeline_id(neutral_payload: dict, manifest: dict, video_path: str | None, fps: float) -> tuple[str, str]:
    video_sha = _file_sha256(video_path)
    timeline_id = str(neutral_payload.get("timeline_id") or "").strip()
    if not timeline_id:
        scope = video_sha[:12] if video_sha else str(manifest.get("source_neutral_sha256") or "")[:12]
        timeline_id = f"tl:{scope}:{max(1, int(round(fps))):03d}"
    return timeline_id, video_sha


def _build_llmb_v2_units(
    round_entry: dict,
    neutral_round: dict,
    timeline_id: str,
    fps: float,
) -> tuple[str, list[dict]]:
    """Convert either legacy B manifest rounds or v3 output rounds to B v2."""
    status = str(round_entry.get("status") or "")
    if status in {"silent", "intentional_silent"}:
        return "intentional_silent", []
    if status in {"empty", "operator_accepted_skip"}:
        return "operator_accepted_skip", []

    scenes = [scene for scene in round_entry.get("scenes", []) if isinstance(scene, dict)]
    if not scenes:
        return "intentional_silent", []
    window_results = [item for item in round_entry.get("window_results", []) if isinstance(item, dict)]
    scene_by_id = {str(scene.get("window_id") or ""): scene for scene in scenes}
    if window_results:
        source_items = [
            item for item in window_results
            if item.get("style_status") in {"ok", "retry_success"}
        ]
    else:
        source_items = scenes
    if not source_items:
        raise ValueError(f"round {round_entry.get('round_no')}: no successful windows for B v2")

    neutral_scenes = [scene for scene in neutral_round.get("scenes", []) if isinstance(scene, dict)]
    neutral_by_id = {str(scene.get("window_id") or ""): scene for scene in neutral_scenes}
    units: list[dict] = []
    for sequence, item in enumerate(source_items, start=1):
        unit_id = str(item.get("window_id") or "")
        scene = scene_by_id.get(unit_id, item)
        if not unit_id:
            raise ValueError(f"round {round_entry.get('round_no')}: successful window is missing window_id")
        neutral_scene = neutral_by_id.get(unit_id)
        allowed_fact_ids, fact_catalog = _build_unit_fact_scope(unit_id, neutral_scene)
        draft_text = str(scene.get("text") or "").strip()
        if not draft_text:
            raise ValueError(f"unit {unit_id}: draft text is empty")
        start_sec = float(scene.get("t_start"))
        end_sec = float(scene.get("t_end"))
        units.append({
            "unit_id": unit_id,
            "sequence": int(item.get("sequence") or sequence),
            "draft_text": draft_text,
            "emotion_binding": {
                "emotion": str(scene.get("emotion") or "平述"),
                "authority": "emotion_policy",
            },
            "render_slot": _unit_render_slot_v2(unit_id, scene, neutral_scene, timeline_id, fps),
            "speech_capacity": _unit_speech_capacity(draft_text, start_sec, end_sec),
            "allowed_fact_ids": allowed_fact_ids,
            "carry_in_fact_ids": [],
            "fact_catalog": fact_catalog,
        })
    return "ready", units


def _export_llmb_draft_package_v2(
    manifest: dict,
    neutral_path: Path,
    video_path: str | None,
    fps: float,
    out_path: Path,
    *,
    config: dict | None = None,
    rounds_payload: dict | None = None,
) -> Path:
    """Export the execution-ready B v2 package without changing B v1."""
    neutral_payload = json.loads(neutral_path.read_text(encoding="utf-8"))
    neutral_rounds = {
        int(item["round_no"]): item
        for item in neutral_payload.get("rounds", [])
        if isinstance(item, dict) and item.get("round_no") is not None
    }
    timeline_id, video_sha = _llmb_timeline_id(neutral_payload, manifest, video_path, fps)
    source_payload = rounds_payload if isinstance(rounds_payload, dict) else manifest
    rounds_out: list[dict] = []
    for round_entry in source_payload.get("rounds", []):
        if not isinstance(round_entry, dict):
            raise ValueError("B v2 round entry must be an object")
        round_no = int(round_entry.get("round_no"))
        round_id = str(round_entry.get("round_id") or f"r{round_no:03d}")
        round_status, units = _build_llmb_v2_units(
            round_entry,
            neutral_rounds.get(round_no, {}),
            timeline_id,
            fps,
        ) if str(round_entry.get("status") or "") not in {"silent", "empty", "intentional_silent", "operator_accepted_skip"} else (
            "intentional_silent" if str(round_entry.get("status")) in {"silent", "intentional_silent"} else "operator_accepted_skip",
            [],
        )
        rounds_out.append({
            "round_id": round_id,
            "round_no": round_no,
            "status": round_status,
            "units": units,
        })
    if not rounds_out:
        raise ValueError("B v2 package must contain rounds")
    payload: dict = {
        "contract": _LLMB_DRAFT_PACKAGE_V2_CONTRACT,
        "producer": "phase3b",
        "run_id": str(manifest.get("source_neutral_run_id") or neutral_payload.get("run_id") or ""),
        "source": {
            "neutral_run_id": str(manifest.get("source_neutral_run_id") or neutral_payload.get("run_id") or ""),
            "neutral_sha256": str(manifest.get("source_neutral_sha256") or hashlib.sha256(neutral_path.read_bytes()).hexdigest()),
            "timeline_id": timeline_id,
            "source_video_sha256": video_sha,
        },
        "tts_policy": _llmb_tts_policy(config, manifest),
        "render_timebase_fps": float(fps),
        "rounds": rounds_out,
    }
    payload["artifact_identity"] = _llmb_artifact_identity(payload)
    write_json(out_path, payload)
    validate_llmb_draft_package(out_path)
    return out_path


def _self_check_llmb_draft_package(out_path: Path) -> None:
    """模块内 B1 合同字段完整性检查（preflight 权威校验不可用时兜底）。"""
    package = json.loads(out_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if package.get("contract") != _LLMB_DRAFT_PACKAGE_CONTRACT:
        errors.append("contract != llmb_draft_package_v1")
    if package.get("producer") != "phase3b":
        errors.append("producer != phase3b")
    if not str(package.get("run_id") or ""):
        errors.append("run_id is required")
    source = package.get("source")
    if not isinstance(source, dict):
        errors.append("source is required")
    else:
        for key in ("neutral_run_id", "neutral_sha256", "timeline_id"):
            if not str(source.get(key) or ""):
                errors.append(f"source.{key} is required")
    rounds = package.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        errors.append("rounds must be a non-empty list")
    else:
        for rnd in rounds:
            if not isinstance(rnd, dict):
                errors.append("round entry must be a dict")
                continue
            round_id = str(rnd.get("round_id") or "")
            if not round_id:
                errors.append("round.round_id is required")
            if rnd.get("status") not in {"ready", "intentional_silent", "operator_accepted_skip"}:
                errors.append(f"round {round_id} has unmappable status {rnd.get('status')!r}")
            units = rnd.get("units")
            if not isinstance(units, list):
                errors.append(f"round {round_id} units must be a list")
            else:
                for unit in units:
                    if not isinstance(unit, dict):
                        errors.append(f"round {round_id} unit must be a dict")
                        continue
                    for key in ("unit_id", "sequence", "draft_text", "emotion_binding",
                                "allowed_fact_ids", "carry_in_fact_ids", "render_slot",
                                "speech_capacity", "fact_catalog"):
                        if key not in unit:
                            errors.append(f"round {round_id} unit {unit.get('unit_id')} missing {key}")
    if package.get("artifact_identity") != _llmb_artifact_identity(package):
        errors.append("artifact_identity mismatch")
    if errors:
        raise ValueError("llmb_draft_package self-check failed: " + "; ".join(errors))


def _export_llmb_draft_package(manifest: dict, neutral_path: Path, video_path: str | None, fps: float, out_path: Path) -> Path:
    """B1 出口封存：把已定稿 manifest（commentary_schema_version=2）封装为不可变 B 草稿包。

    三态映射（§9.3/§3.1）：ok / partial（有成功窗口）→ ready；silent →
    intentional_silent；empty → operator_accepted_skip（现网 empty 经人工决策放行）；
    style_failed / analyst_failed / partial（零成功）→ ValueError fail-closed。
    每成功窗口一个 unit：unit_id=window_id、sequence=scenes 内序号（1 起）、
    emotion_binding 由确定性 EmotionPolicy 绑定、fact 作用域由 v2 fact adapter
    从 neutral fact_anchors 确定性生成、render_slot 由 neutral render_slot（v4）
    透传或 t_start/t_end×fps 构造。
    """
    neutral_payload = json.loads(neutral_path.read_text(encoding="utf-8"))
    neutral_rounds: dict[int, dict] = {}
    if isinstance(neutral_payload, dict):
        for round_data in neutral_payload.get("rounds", []):
            if isinstance(round_data, dict):
                neutral_rounds[int(round_data.get("round_no"))] = round_data
    neutral_timeline_id = ""
    if isinstance(neutral_payload, dict):
        candidate = neutral_payload.get("timeline_id")
        if isinstance(candidate, str) and candidate.strip():
            neutral_timeline_id = candidate.strip()

    fps_int = max(1, int(round(fps)))
    video_sha = _file_sha256(video_path)
    if not neutral_timeline_id:
        # timeline_id 不可用 run_id 伪造：v4 neutral 无 timeline_id 时用视频指纹构造
        # tl:{video_sha[:12]}:{fps:03d}；视频文件不存在时以 neutral 包 sha256 前 12 位兜底。
        scope = video_sha[:12] if video_sha else str(manifest.get("source_neutral_sha256") or "")[:12]
        neutral_timeline_id = f"tl:{scope}:{fps_int:03d}"
    timeline_id = neutral_timeline_id

    rounds_out: list[dict] = []
    for round_entry in manifest.get("rounds", []):
        if not isinstance(round_entry, dict):
            raise ValueError("manifest round entry is not a dict (fail-closed)")
        round_no = int(round_entry.get("round_no"))
        round_id = f"r{round_no:03d}"
        status = str(round_entry.get("status") or "")
        window_results = round_entry.get("window_results", [])
        scenes = round_entry.get("scenes", [])
        success_windows = [
            item for item in window_results
            if isinstance(item, dict) and item.get("style_status") in {"ok", "retry_success"}
        ]
        if status == "silent":
            units: list[dict] = []
            round_status = "intentional_silent"
        elif status == "empty":
            # 现网 empty 经人工决策放行（empty_round_decision），B 包标记 operator_accepted_skip，
            # 不导出 units；Phase3c 不将其解释为可发布文本。
            units = []
            round_status = "operator_accepted_skip"
        elif status in {"ok", "partial"}:
            if not success_windows:
                raise ValueError(f"round {round_id}: status={status!r} 无成功窗口，不可映射到 B 包（fail-closed）")
            round_status = "ready"
            neutral_round = neutral_rounds.get(round_no, {})
            neutral_by_window = {
                str(s.get("window_id") or ""): s
                for s in (neutral_round.get("scenes", []) if isinstance(neutral_round, dict) else [])
                if isinstance(s, dict)
            }
            scene_by_window = {
                str(s.get("window_id") or ""): s
                for s in scenes if isinstance(s, dict)
            }
            units = []
            for seq, item in enumerate(success_windows, start=1):
                window_id = str(item.get("window_id") or "")
                scene = scene_by_window.get(window_id)
                if scene is None:
                    raise ValueError(f"round {round_id} window {window_id}: manifest scene missing (fail-closed)")
                neutral_scene = neutral_by_window.get(window_id)
                allowed_fact_ids, fact_catalog = _build_unit_fact_scope(window_id, neutral_scene)
                t_start = float(scene.get("t_start", 0.0))
                t_end = float(scene.get("t_end", t_start + 1.0))
                draft_text = str(scene.get("text") or "")
                units.append({
                    "unit_id": window_id,
                    "sequence": seq,
                    "draft_text": draft_text,
                    "emotion_binding": {
                        "emotion": str(scene.get("emotion") or "平述"),
                        "authority": "emotion_policy",
                    },
                    "allowed_fact_ids": allowed_fact_ids,
                    "carry_in_fact_ids": [],
                    "render_slot": _unit_render_slot(window_id, scene, neutral_scene, timeline_id, fps),
                    "speech_capacity": _unit_speech_capacity(draft_text, t_start, t_end),
                    "fact_catalog": fact_catalog,
                })
        else:
            raise ValueError(
                f"round {round_id}: status={status!r} 不可映射到 B 包"
                "（style_failed/analyst_failed/partial 零成功均 fail-closed，过不了 B0/B1 门禁）"
            )
        rounds_out.append({"round_id": round_id, "round_no": round_no, "status": round_status, "units": units})

    payload: dict = {
        "contract": _LLMB_DRAFT_PACKAGE_CONTRACT,
        "producer": "phase3b",
        "run_id": str(manifest.get("source_neutral_run_id") or ""),
        "source": {
            "neutral_run_id": str(manifest.get("source_neutral_run_id") or ""),
            "neutral_sha256": str(manifest.get("source_neutral_sha256") or ""),
            "timeline_id": timeline_id,
            "source_video_sha256": video_sha or "",
        },
        "rounds": rounds_out,
    }
    payload["artifact_identity"] = _llmb_artifact_identity(payload)
    write_json(out_path, payload)
    # B1 出口自检：优先走 preflight.validate_llmb_draft_package（并行 agent 实现）；
    # 未合入（ImportError）或签名未定（TypeError）时退回模块内合同字段完整性检查。
    try:
        from sbmachine.preflight import validate_llmb_draft_package as _preflight_validate_package
    except ImportError:
        _preflight_validate_package = None
    if _preflight_validate_package is None:
        _self_check_llmb_draft_package(out_path)
    else:
        try:
            result = _preflight_validate_package(out_path)
        except TypeError:
            _self_check_llmb_draft_package(out_path)
        else:
            if result:
                raise ValueError("llmb_draft_package rejected by preflight: " + "; ".join(str(e) for e in result))
    return out_path


# ── main runner ──

def run_phase3b(
    *,
    neutral_path: Path,
    rounds_path: Path,
    output_rounds_path: Path,
    commentary_path: Path,
    config_path: Path,
    draft_package_path: Path | None = None,
    dry_run: bool = False,
    progress_sink=None,
) -> dict:
    import os

    config = load_config(config_path)
    semantic_cfg = config.get("semantic", {}) if isinstance(config.get("semantic"), dict) else {}
    debug_enabled = bool(config.get("debug", {}).get("phase3", False) or os.getenv("AI6657_DEBUG_PHASE3"))
    backend = resolve_backend(config, "style")
    if backend not in {"api", "vllm"}:
        raise ValueError(f"unsupported style backend: {backend}; use vllm or api")
    from sbmachine import llmb_api as _llmb_backend
    llm_cfg, style_cfg = _llmb_backend.style_runtime_config(config)
    if backend == "api":
        from sbmachine import cloud_memory
        # 阶段 2 第 5 点：LLM-B 云端会话默认关闭（cloud_conversation_max_rounds 缺省 0）。
        # cloud_memory.make_generate 的 llmb 会话分支保留，但不再作为默认路径；
        # 显式配置 >0 时仍启用会话。
        llmb_semantic = dict(semantic_cfg)
        llmb_semantic.setdefault("cloud_conversation_max_rounds", 0)
        llmb_semantic.setdefault("cloud_conversation_max_tokens", 0)
        gen_fn = cloud_memory.make_generate("llmb", semantic_cfg=llmb_semantic)
    else:
        gen_fn = _llmb_backend.generate
    # 阶段 2：LLM-B 有界并发度（缺省按后端：云端 4、本地 1=完全复现串行；显式配置可覆盖）。
    default_scenes = 4 if backend == "api" else 1
    style_concurrent_scenes = max(1, int(semantic_cfg.get("style_concurrent_scenes", default_scenes) or default_scenes))
    style_model = config.get("semantic", {}).get("style_model") or config.get("semantic", {}).get("model", "")
    if style_model:
        llm_cfg["model"] = style_model

    aliases = _load_player_aliases()
    persona = _load_persona()

    neutral_payload = json.loads(neutral_path.read_text(encoding="utf-8"))
    is_v4 = isinstance(neutral_payload, dict) and neutral_payload.get("schema_version") == 4
    if is_v4:
        # schema v4（rule_neutral_renderer）：走 voice task 生产路径；v3 发布门禁
        # （preflight.validate_neutral_publishable）不适用 v4 neutral。
        neutral_data = validate_neutral_v4(neutral_payload)
    else:
        neutral_data = validate_neutral_manifest(neutral_payload, rounds_path)
        validate_neutral_publishable(neutral_path)
    # llmb 诊断：非 debug 也写脱敏摘要（usage/reasoning/finish_reason/validation），
    # 供云端思考模型重试根因观测。
    _init_style_diagnostics(output_rounds_path.parent, str(neutral_data["run_id"]))
    phase3a_mode = neutral_data["phase3a_mode"]
    neutral_by_round: dict[int, dict] = {
        int(r["round_no"]): r
        for r in neutral_data.get("rounds", [])
    }

    match = load_match(rounds_path)
    profile = str(config.get("profile", "style"))

    manifest_rounds = []
    errors: list[dict] = []
    accepted_run_samples: list[tuple[_ValidatedStyleCommentary, str]] = []
    recent_style_phrases: list[str] = []
    recent_limit = max(1, int(style_cfg["style_recent_window_count"]))
    phrase_max_reuse = max(1, int(style_cfg["style_phrase_max_reuse"]))
    max_retries = max(0, min(2, int(style_cfg["style_max_retries"])))
    # 云端思考模型（reasoning 挤占预算导致验证失败重试，实测每窗 2-3 次）：
    # 云端用 cloud_style_output_max_tokens 放开；本地 vllm 保持 style_output_max_tokens 原值。
    if backend == "api" and style_cfg.get("cloud_style_output_max_tokens"):
        style_output_max_tokens = max(1, int(style_cfg["cloud_style_output_max_tokens"]))
    else:
        style_output_max_tokens = max(1, int(style_cfg["style_output_max_tokens"]))
    char_tolerance = min(0.5, max(0.0, float(style_cfg.get("style_budget_hard_tolerance", 0.0))))
    # LLM-B 硬限护栏：固定 1.5×（审计 §2.1/§4 收敛：B 最终硬线=1.5B，恒生效，
    # 不再依赖 llmc 开关；读取处钳制 [1.0, 1.5]，配置可调低但不得放大，
    # 任何配置（style_budget_hard_tolerance 等）都不得在最终硬线后再叠加 tolerance 乘子）。
    llmb_hard_cap_factor = min(LLMB_HARD_CAP_FACTOR, max(1.0, float(style_cfg.get("style_output_hard_cap_factor", LLMB_HARD_CAP_FACTOR))))
    # 强事实依据模式（semantic.strong_fact_mode）：False=全面相信 LLM，跳过 B0 unexpected_fact。
    strong_fact_mode = bool(semantic_cfg.get("strong_fact_mode", False))
    # B 封存包时间轴帧率（semantic.phase3c.render_timebase_fps；缺省 30）。
    phase3c_cfg = semantic_cfg.get("phase3c")
    render_timebase_fps = float(phase3c_cfg.get("render_timebase_fps", 30.0)) if isinstance(phase3c_cfg, dict) else 30.0
    draft_contract_version = int(phase3c_cfg.get("contract_version", 1)) if isinstance(phase3c_cfg, dict) else 1
    if draft_contract_version not in {1, 2}:
        raise ValueError("semantic.phase3c.contract_version must be 1 or 2")
    cs_rules_path = _PROJECT_ROOT / "Prompt" / "cs_rules.txt"
    cs_rules = cs_rules_path.read_text(encoding="utf-8").strip() if cs_rules_path.exists() else ""
    if backend == "api":
        from sbmachine import cloud_prompts
        meme_profile = cloud_prompts.compute_match_meme_profile(neutral_data)
        system_content = cloud_prompts.build_cloud_style_system(config, meme_profile)
    else:
        system_content = "\n\n".join(filter(None, [
            load_prompt("style_system").replace("{persona_hint}", persona),
            cs_rules,
            _llmb_backend.load_style_skill(config),
        ]))

    emotion_policy = EmotionPolicy.from_rules(load_hype_rules())

    # ── voice task（commentary v3）生产开关（§12.2）──
    # 仅 schema v4 neutral 且 voice_task.enabled=true 且 profile 就绪时走 v3；
    # 否则沿用 v2 单稿路径（profile 不可用只写 risk_class=unknown 影子诊断，§5.1）。
    voice_task_cfg = semantic_cfg.get("voice_task", {})
    if not isinstance(voice_task_cfg, dict):
        voice_task_cfg = {}
    use_voice_task_v3 = False
    if is_v4 and voice_task_cfg.get("enabled"):
        profile_id = str(voice_task_cfg.get("speech_profile_id") or "")
        if not profile_id:
            for round_data in neutral_data.get("rounds", []):
                for scene in (round_data.get("scenes") or []):
                    budget = scene.get("speech_budget") if isinstance(scene, dict) else None
                    if isinstance(budget, dict) and budget.get("profile_id"):
                        profile_id = str(budget["profile_id"])
                        break
                if profile_id:
                    break
        _voice_profile, profile_status = _profile_readiness(profile_id, voice_task_cfg)
        use_voice_task_v3 = profile_status == "ok"
        if not use_voice_task_v3:
            _write_unknown_shadow_diagnostics(neutral_data, profile_status)
    if use_voice_task_v3:
        return _run_phase3b_v4(
            neutral_path=neutral_path,
            neutral_data=neutral_data,
            match=match,
            config=config,
            llm_cfg=llm_cfg,
            style_cfg=style_cfg,
            gen_fn=gen_fn,
            backend=backend,
            system_content=system_content,
            emotion_policy=emotion_policy,
            aliases=aliases,
            voice_task_cfg=voice_task_cfg,
            profile=_voice_profile,
            profile_id=profile_id,
            debug_enabled=debug_enabled,
            output_rounds_path=output_rounds_path,
            commentary_path=commentary_path,
            draft_package_path=draft_package_path,
            draft_contract_version=draft_contract_version,
            render_timebase_fps=render_timebase_fps,
            dry_run=dry_run,
            progress_sink=progress_sink,
        )

    for completed, rnd in enumerate(tqdm(match.rounds, desc="Phase3b style", unit="round"), start=1):
        round_data = neutral_by_round.get(rnd.round_no, {})
        scenes = round_data.get("scenes", [])
        avg_hype = float(round_data.get("avg_hype", 0.0))
        analyst_failed = bool(round_data.get("analyst_failed", False))
        round_final_intensity = avg_hype
        round_status = "ok"
        last_tail = ""
        round_scream_used = False  # 每回合最多 1 次惊叹，逐回合重置。
        accepted_samples: list[tuple[_ValidatedStyleCommentary, str]] = []

        scene_commentaries: list[str] = []
        scenes_manifest: list[dict] = []
        window_results: list[dict] = []
        felt_samples: list[tuple[float, float]] = []
        final_intensity_samples: list[tuple[float, float]] = []

        # ── 阶段 1：确定性预计算（串行，主线程；按时间顺序生成不可变计划）──
        plans: list[object | None] = []
        for scene_idx, scene in enumerate(scenes):
            window_id = str(scene.get("window_id") or f"r{rnd.round_no:03d}_w{scene_idx + 1:02d}")
            scene_neutral = str(scene.get("neutral") or "")
            neutral_source = str(scene.get("neutral_source") or "")
            generation_status = str(scene.get("generation_status") or "")
            scene_hype = float(scene.get("hype", avg_hype))
            char_budget = max(1, int(scene.get("char_budget", 100)))
            scene_name = str(scene.get("scene") or "")
            t_start = float(scene.get("t_start", rnd.start_sec))
            t_end = float(scene.get("t_end", rnd.end_sec))
            duration = max(1.0, t_end - t_start)
            result = {"window_id": window_id, "t_start": t_start, "t_end": t_end, "neutral_source": neutral_source, "neutral_nonempty": bool(scene_neutral.strip()), "style_status": "upstream_failed", "retry_count": 0, "failure_reason": None, "char_budget": char_budget, "output_chars": None, "published_scene_index": None}
            window_results.append(result)

            if dry_run:
                result["failure_reason"] = "dry_run"
                plans.append(None)
                continue
            if analyst_failed:
                result["failure_reason"] = "analyst_failed"
                plans.append(None)
                continue
            if neutral_source == "unrecoverable":
                result["style_status"] = "skipped_unrecoverable"
                result["failure_reason"] = "unrecoverable"
                plans.append(None)
                continue
            if not scene_neutral.strip():
                if neutral_source == "intentional_empty" and generation_status == "success":
                    result["style_status"] = "skipped_intentional_empty"
                else:
                    result["failure_reason"] = "invalid_empty_neutral"
                plans.append(None)
                continue
            if is_v4:
                # v4 neutral（rule_neutral_renderer）没有 LLM-A 失败语义；fallback 路径
                # （voice_task 未启用或 profile 不可用）直接消费规则层 neutral。
                if neutral_source not in {"rule_template", "tiny_assembler"}:
                    result["failure_reason"] = neutral_source or "invalid_neutral_source"
                    plans.append(None)
                    continue
            elif generation_status != "success" or neutral_source not in {"llm", "llm_retry", "rule"}:
                result["failure_reason"] = generation_status or "invalid_neutral_source"
                plans.append(None)
                continue

            # 短语快照：主循环期间 recent_style_phrases 不变，所有窗口取同一快照，
            # 随计划不可变，供并发请求只读使用。
            user_prompt, anchors, delivery = build_style_prompt(scene, aliases, recent_style_phrases[-recent_limit:], retry_feedback=None)
            if backend == "api":
                from sbmachine import cloud_prompts
                user_prompt = cloud_prompts.inject_window_type(user_prompt, scene)
            # max_tokens：云端全额放开（按 cloud_style_output_max_tokens 发送），
            # 本地按口播字数公式 min() 计算。
            if backend == "api":
                scene_max_tokens = style_output_max_tokens
            else:
                scene_max_tokens = min(
                    style_output_max_tokens,
                    max(96, int(char_budget * 2.2) + 80),
                )
            log_ctx = {"run_id": neutral_data["run_id"], "round": f"round{rnd.round_no}", "scene": window_id}
            plans.append(_StyleWindowPlan(
                scene_idx=scene_idx, window_id=window_id, scene=scene,
                scene_neutral=scene_neutral, scene_hype=scene_hype, duration=duration,
                user_prompt=user_prompt, anchors=anchors, delivery=delivery,
                scene_max_tokens=scene_max_tokens, log_ctx=log_ctx,
            ))

        # ── 阶段 2 + 3：有界并发请求 + 按时间顺序验收/提交（主线程）──
        # 滑动窗口调度：主请求 in-flight 不超过 style_concurrent_scenes；
        # 重试作业由主线程按窗口顺序提交并等待，验收顺序与完成顺序无关。
        plan_failure_meta: dict[int, dict] = {}
        jobs: dict[int, list] = {}
        consumed: dict[int, int] = {}
        inflight_count = 0
        next_primary = 0
        dispatch_seq = 0
        completion_seq = 0
        executor = ThreadPoolExecutor(max_workers=style_concurrent_scenes, thread_name_prefix="phase3b-style")
        try:
            def submit_request(plan: _StyleWindowPlan, plan_idx: int, retry_feedback: dict | None) -> int:
                """主线程提交一次生成请求；返回 dispatch_order 诊断序号。"""
                nonlocal inflight_count, dispatch_seq
                prompt = plan.user_prompt
                if retry_feedback is not None:
                    user_prompt_retry, _, _ = build_style_prompt(
                        plan.scene, aliases, recent_style_phrases[-recent_limit:],
                        retry_feedback=retry_feedback,
                    )
                    if backend == "api":
                        from sbmachine import cloud_prompts
                        user_prompt_retry = cloud_prompts.inject_window_type(user_prompt_retry, plan.scene)
                    prompt = user_prompt_retry
                dispatch_seq += 1
                future = executor.submit(
                    _style_request_worker, plan, rnd.round_no, debug_enabled,
                    system_content, llm_cfg, gen_fn, prompt=prompt,
                )
                jobs.setdefault(plan_idx, []).append((future, dispatch_seq))
                inflight_count += 1
                return dispatch_seq

            def submit_primary() -> None:
                """按窗口顺序补充主请求，保持 in-flight 不超过并发上限。"""
                nonlocal next_primary
                while next_primary < len(plans) and inflight_count < style_concurrent_scenes:
                    plan_idx = next_primary
                    next_primary += 1
                    plan = plans[plan_idx]
                    if plan is None:
                        continue
                    submit_request(plan, plan_idx, None)

            submit_primary()
            for plan_idx, plan in enumerate(plans):
                if plan is None:
                    continue
                accepted_response = None
                scene_felt = 0.0
                validation = {"ok": False, "reason": "response_error", "details": [], "output_chars": None, "signature": ""}
                style_meta: dict = {}
                window_id = plan.window_id
                scene_idx = plan.scene_idx
                scene_hype = plan.scene_hype
                duration = plan.duration
                while jobs.get(plan_idx):
                    attempt = consumed.get(plan_idx, 0)
                    future, dispatch_order = jobs[plan_idx][attempt]
                    candidate, scene_felt, style_meta = future.result()
                    consumed[plan_idx] = attempt + 1
                    inflight_count -= 1
                    completion_seq += 1
                    result = window_results[plan_idx]
                    t_start = result["t_start"]
                    t_end = result["t_end"]
                    char_budget = result["char_budget"]
                    result["retry_count"] = attempt
                    if candidate.startswith("[style error:"):
                        validation = {"ok": False, "reason": "response_error", "details": [candidate], "output_chars": None, "signature": ""}
                    elif any(marker in candidate for marker in _CONTAMINATION_MARKERS):
                        validation = {"ok": False, "reason": "unexpected_fact", "details": ["prompt_contamination"], "output_chars": len(_strip_tags(candidate)), "signature": ""}
                    else:
                        validation = validate_style_commentary(candidate, plan.scene_neutral, plan.anchors, aliases, recent_style_phrases, hard_char_limit=plan.delivery["hard_char_limit"], phrase_max_reuse=phrase_max_reuse, char_tolerance=char_tolerance, hard_cap_factor=llmb_hard_cap_factor, strong_fact_mode=strong_fact_mode)
                    _write_style_diagnostic(
                        rnd.round_no, window_id, scene_idx, attempt, plan.scene_max_tokens,
                        bool(validation["ok"]), str(validation["reason"]), style_meta,
                        dispatch_order=dispatch_order, completion_order=completion_seq,
                    )
                    if validation["reason"] == "under_budget" and attempt >= max_retries:
                        # 最低字数只用于推动扩写重试；重试耗尽后保留最后一稿，
                        # 避免短但真实的内容被强制丢弃。
                        accepted_response = candidate
                        if backend == "api":
                            raw_response = getattr(accepted_response, "raw_response", None)
                            if raw_response is not None:
                                cloud_memory.commit_round(raw_response)
                        break
                    if validation["ok"]:
                        accepted_response = candidate
                        if backend == "api":
                            # 业务验收通过后才写入会话历史（失败轮/空稿不入，防空稿污染）
                            raw_response = getattr(accepted_response, "raw_response", None)
                            if raw_response is not None:
                                cloud_memory.commit_round(raw_response)
                        break
                    if attempt >= max_retries:
                        break
                    retry_feedback = {"failure_reason": validation["reason"], "details": validation["details"], "instruction": _style_retry_instruction(validation, plan.delivery, plan.scene_neutral)}
                    submit_request(plan, plan_idx, retry_feedback)

                result["output_chars"] = validation["output_chars"]
                if accepted_response is None:
                    result["style_status"] = "style_failed"
                    result["failure_reason"] = validation["reason"]
                    plan_failure_meta[plan_idx] = style_meta
                    errors.append({"round": f"round{rnd.round_no}", "round_no": rnd.round_no, "scene": window_id, "error": validation["reason"], "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                    submit_primary()
                    continue

                scream_eligible = bool(plan.scene.get("scream_eligible", False)) and not round_scream_used
                decision = emotion_policy.decide(hard_intensity=scene_hype, llmb_intensity=scene_felt, scream_eligible=scream_eligible)
                if decision.label == "惊叹":
                    round_scream_used = True
                normalized = normalize_commentary_emotion(accepted_response, decision.label)
                result["style_status"] = "retry_success" if result["retry_count"] else "ok"
                result["failure_reason"] = None
                result["published_scene_index"] = len(scenes_manifest)
                result["output_chars"] = count_spoken_chars(_strip_tags(normalized))
                scenes_manifest.append({"window_id": window_id, "t_start": t_start, "t_end": t_end, "emotion": decision.label, "emotion_score": decision.score, "hard_intensity": round(scene_hype, 3), "llmb_intensity": round(scene_felt, 3), "scream_eligible": scream_eligible, "text": normalized.removeprefix(f"[{decision.label}]"), "char_budget": char_budget, "output_chars": result["output_chars"], "style_status": result["style_status"], "budget_overage": validation.get("budget_overage", 1.0)})
                scene_commentaries.append(normalized)
                felt_samples.append((scene_felt, duration))
                final_intensity_samples.append((decision.score, duration))
                if isinstance(accepted_response, _ValidatedStyleCommentary):
                    accepted_samples.append((accepted_response, json.dumps({"commentary": normalized, "felt_intensity": scene_felt}, ensure_ascii=False, separators=(",", ":"))))
                submit_primary()
        finally:
            executor.shutdown(wait=True)

        # ── 回合末兜底重试：给本轮 style_failed 的窗口各补一次调用 ──
        for ri, result in enumerate(window_results):
            if result.get("style_status") != "style_failed":
                continue
            if result.get("retry_count", 0) >= 1 and not _is_recoverable_infra_failure(plan_failure_meta.get(ri, {})):
                # 阶段 2 第 4 点：主循环已做过一次有效重试 → 不再回合末重复补偿；
                # 仅当最后失败类别明确为可恢复基础设施错误（transport/http/rate_limit）时仍补偿一次。
                continue
            scene = scenes[ri]
            scene_neutral_retry = str(scene.get("neutral") or "")
            scene_hype_retry = float(scene.get("hype", avg_hype))
            char_budget_retry = max(1, int(scene.get("char_budget", 100)))
            window_id_retry = result["window_id"]
            t_start_retry = float(scene.get("t_start", rnd.start_sec))
            t_end_retry = float(scene.get("t_end", rnd.end_sec))
            duration_retry = max(1.0, t_end_retry - t_start_retry)

            retry_feedback_round = {
                "failure_reason": result.get("failure_reason", "unknown"),
                "details": [],
                "instruction": _style_retry_instruction(
                    {"reason": result.get("failure_reason", ""), "details": [], "output_chars": result.get("output_chars")},
                    {"hard_char_limit": char_budget_retry},
                    scene_neutral_retry,
                ),
            }
            user_prompt_retry, anchors_retry, delivery_retry = build_style_prompt(
                scene, aliases, recent_style_phrases[-recent_limit:], retry_feedback=retry_feedback_round,
            )
            if backend == "api":
                from sbmachine import cloud_prompts
                user_prompt_retry = cloud_prompts.inject_window_type(user_prompt_retry, scene)
            log_ctx_retry = {"run_id": neutral_data["run_id"], "round": f"round{rnd.round_no}", "scene": window_id_retry}
            if backend == "api":
                # 云端兜底重试同样全额放开（与主调用一致），否则思考占满小预算必败。
                scene_max_tokens_retry = style_output_max_tokens
            else:
                scene_max_tokens_retry = min(style_output_max_tokens, max(96, int(char_budget_retry * 2.2) + 80))
            candidate_retry, scene_felt_retry, style_meta_retry = _call_style(
                system_content, user_prompt_retry, llm_cfg, gen_fn,
                round_no=rnd.round_no, scene_idx=ri, debug=debug_enabled,
                max_tokens=scene_max_tokens_retry, log_ctx=log_ctx_retry,
            )

            if candidate_retry.startswith("[style error:") or any(
                marker in candidate_retry for marker in _CONTAMINATION_MARKERS
            ):
                _write_style_diagnostic(
                    rnd.round_no, window_id_retry, ri, max_retries + 1, scene_max_tokens_retry,
                    False, "response_error", style_meta_retry,
                )
                continue
            validation_retry = validate_style_commentary(
                candidate_retry, scene_neutral_retry, anchors_retry, aliases, recent_style_phrases,
                hard_char_limit=delivery_retry["hard_char_limit"], phrase_max_reuse=phrase_max_reuse,
                char_tolerance=char_tolerance, hard_cap_factor=llmb_hard_cap_factor,
                strong_fact_mode=strong_fact_mode,
            )
            _write_style_diagnostic(
                rnd.round_no, window_id_retry, ri, max_retries + 1, scene_max_tokens_retry,
                bool(validation_retry["ok"]), str(validation_retry["reason"]), style_meta_retry,
            )
            if not validation_retry["ok"]:
                continue

            if backend == "api":
                raw_response_retry = getattr(candidate_retry, "raw_response", None)
                if raw_response_retry is not None:
                    cloud_memory.commit_round(raw_response_retry)

            # ── 补救成功：更新该窗的全部状态 ──
            scream_eligible_retry = bool(scene.get("scream_eligible", False)) and not round_scream_used
            decision_retry = emotion_policy.decide(hard_intensity=scene_hype_retry, llmb_intensity=scene_felt_retry, scream_eligible=scream_eligible_retry)
            if decision_retry.label == "惊叹":
                round_scream_used = True
            normalized_retry = normalize_commentary_emotion(candidate_retry, decision_retry.label)
            result["style_status"] = "retry_success"
            result["failure_reason"] = None
            result["published_scene_index"] = len(scenes_manifest)
            result["output_chars"] = count_spoken_chars(_strip_tags(normalized_retry))
            scenes_manifest.append({
                "window_id": window_id_retry, "t_start": t_start_retry, "t_end": t_end_retry,
                "emotion": decision_retry.label, "emotion_score": decision_retry.score,
                "hard_intensity": round(scene_hype_retry, 3), "llmb_intensity": round(scene_felt_retry, 3),
                "scream_eligible": scream_eligible_retry,
                "text": normalized_retry.removeprefix(f"[{decision_retry.label}]"),
                "char_budget": char_budget_retry, "output_chars": result["output_chars"],
                "style_status": result["style_status"],
                "budget_overage": validation_retry.get("budget_overage", 1.0),
            })
            scene_commentaries.append(normalized_retry)
            felt_samples.append((scene_felt_retry, duration_retry))
            final_intensity_samples.append((decision_retry.score, duration_retry))
            if validation_retry["signature"]:
                recent_style_phrases.append(validation_retry["signature"])
            if isinstance(candidate_retry, _ValidatedStyleCommentary):
                accepted_samples.append((candidate_retry, json.dumps(
                    {"commentary": normalized_retry, "felt_intensity": scene_felt_retry},
                    ensure_ascii=False, separators=(",", ":"),
                )))

        commentary = "".join(scene_commentaries)
        required = [item for item in window_results if item["neutral_nonempty"] and item["neutral_source"] != "unrecoverable"]
        successes = [item for item in required if item["style_status"] in {"ok", "retry_success"}]
        if analyst_failed:
            round_status = "analyst_failed"
        elif not required and window_results and all(item["style_status"] == "skipped_intentional_empty" for item in window_results):
            round_status = "silent"
        elif required and len(successes) == len(required) and not any(item["style_status"] in {"upstream_failed", "skipped_unrecoverable"} for item in window_results):
            round_status = "ok"
        elif required and not successes and all(item["style_status"] == "style_failed" for item in required):
            round_status = "style_failed"
        else:
            round_status = "partial"
        # ── LLM-C 回合级整合已移除：LLM-C 独立为 Phase3c，B 仅通过
        # llmb_draft_package_v1 封存包单向交接（见 docs/plan/phase3c-llmc-one-way-handoff-plan.md）。

        felt_intensity = weighted_intensity(felt_samples)
        round_final_intensity = weighted_intensity(final_intensity_samples)


        parsed = parse_emotional_text(commentary)
        rnd.phase3_semantic = SemanticData(
            model_profile=profile,
            model_name=str(llm_cfg.get("model", "")),
            commentary_text=commentary,
            emotion_segments=[EmotionSegment(seg.emotion, seg.text, i) for i, seg in enumerate(parsed)],
        )
        # 方案 R（§6.4）：不可恢复窗的 commentary silent 带 silent_reason 区分于规则层静默。
        round_silent_reason = ""
        if round_status == "silent":
            if any(
                isinstance(s, dict) and str(s.get("neutral_source") or "") == "unrecoverable"
                for s in scenes
            ):
                round_silent_reason = "unrecoverable_failure"
            else:
                round_silent_reason = "intentional"
        manifest_rounds.append({
            "round_no":        rnd.round_no,
            "start_sec":       rnd.start_sec,
            "end_sec":         rnd.end_sec,
            "commentary_text": commentary,
            "status":          round_status,
            "hype_avg":        round(avg_hype, 3),
            "felt_intensity":  round(felt_intensity, 3),
            "final_intensity": round(round_final_intensity, 3),
            "emotion_segments": [seg.__dict__ for seg in rnd.phase3_semantic.emotion_segments],
            "window_results":  window_results,
            "scenes":          scenes_manifest,
        })
        if round_silent_reason:
            manifest_rounds[-1]["silent_reason"] = round_silent_reason
        if round_status == "ok":
            accepted_run_samples.extend(accepted_samples)
        if progress_sink is not None:
            try:
                progress_sink(completed, len(match.rounds), "round", None)
            except Exception:
                pass

    if errors:
        err_path = _PROJECT_ROOT / "logs" / "error.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if err_path.exists():
            try:
                existing = json.loads(err_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.extend(errors)
        err_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    # commentary 只承载解说产物；phase2 视觉时间线（YOLO 框、DEM 投影）的权威副本在
    # rounds_with_yolo.json，此处不透传，防止产物膨胀。phase4 与发布契约均不读 _phase2_yolo。
    for rnd in match.rounds:
        rnd.phase2_yolo = None
    save_match(output_rounds_path, match)
    manifest = {
        "commentary_schema_version": 2,
        "source_neutral_run_id": str(neutral_data["run_id"]),
        "source_neutral_sha256": hashlib.sha256(neutral_path.read_bytes()).hexdigest(),
        "source_window_count": sum(len(round_data.get("scenes", [])) for round_data in neutral_data.get("rounds", []) if isinstance(round_data, dict)),
        "video_path":    match.video_path,
        "map_name":      match.map_name,
        "model_profile": profile,
        "model_name":    str(llm_cfg.get("model", "")),
        "effective_style_config": style_cfg,
        "rounds":        manifest_rounds,
    }
    write_json(commentary_path, manifest)
    if draft_package_path is not None and not dry_run:
        # B1 出口封存（§3.1/§8 门禁矩阵）：manifest 定稿后导出不可变 B 草稿包并自检。
        if draft_contract_version == 2:
            _export_llmb_draft_package_v2(
                manifest, neutral_path, manifest.get("video_path"), render_timebase_fps,
                draft_package_path, config=config,
            )
        else:
            _export_llmb_draft_package(
                manifest, neutral_path, manifest.get("video_path"), render_timebase_fps,
                draft_package_path,
            )
    for response, output in accepted_run_samples:
        response.accept(output)
    return manifest
