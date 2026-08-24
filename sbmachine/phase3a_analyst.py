"""Phase 3a 分析模型：确定性时间窗 -> 每窗一条中性稿。"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from sbmachine.common import count_spoken_chars, debug_output_dir, load_config, resolve_backend, resolve_path, write_json
from sbmachine.hype_score import _compute_char_budget, _scene_hype, _scene_scream_eligible, _speech_rate_config, compute_hype, dominant_round_emotion
from sbmachine.llm_shim import accept_api_response, infra_backoff_delay, retry_after_seconds
from sbmachine.phase3a_payload import _semantic_payload, build_state_block, load_semantic_frames
from sbmachine.llm_projection import build_llm_window_projection, build_rule_state_delta, merge_required_fact_anchors
from sbmachine.phase3a_prompt import (
    _build_analyst_system, _build_window_prompt, _first_json_obj,
    _parse_window_neutral_response, validate_neutral_semantics,
)
from sbmachine.neutral_contract import new_manifest_metadata
from sbmachine.commentary_planner import PlannerState, build_atomic_fact_units, fallback_neutral, plan_window
from sbmachine.rule_neutral_renderer import (
    RendererUnfitError,
    render_capsule,
    render_neutral,
    validate_preserved_facts as renderer_validate,
)
from sbmachine.debug_phase3a import DebugRecorder, DebugWindowRecord
from sbmachine.scene_context import build_scene_contexts
from sbmachine.schemas import load_match
from sbmachine.tactic_book import load_tactic_book
from sbmachine.tactic_projection import build_window_rule_projection
from sbmachine.phase3a_audit import (
    build_audit_artifact,
    build_window_statistics,
    compute_source_hashes,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ANALYST_FAILED = "__ANALYST_FAILED__"
_NEUTRAL_HARD_LIMIT = 100  # 硬上限：任何窗口的 neutral 不得超过此值
_DEFAULT_EFFECTIVE_CHAR_LIMIT = 100  # 默认有效字符上限，被每窗 effective_char_limit 覆盖
_MAX_WINDOW_FRAMES = 5
_MIN_WINDOW_FRAMES = 2

# ── S2 诊断落盘 ──
_DIAGNOSTICS_LOCK = threading.Lock()
_DIAGNOSTICS_DIR: Path | None = None
_DIAGNOSTICS_RUN_ID: str = ""

_CONSOLE_STATUS_ORDER = (
    "success",
    "transport_error",
    "http_error",
    "response_error",
    "truncated",
    "parse_error",
    "contract_error",
    "semantic_contract_error",
    "required_fact_missing",
    "unexpected_fact",
    "side_mismatch",
    "projection_budget_error",
)


def _format_window_statistics(window_stats: dict) -> str:
    """用机器统计对象生成完整控制台摘要，并显示未来新增的非零状态。"""
    counts = window_stats.get("generation_status_counts") or {}
    ordered = [f"{status}={int(counts.get(status, 0) or 0)}" for status in _CONSOLE_STATUS_ORDER]
    known = set(_CONSOLE_STATUS_ORDER)
    ordered.extend(
        f"{status}={int(count)}"
        for status, count in sorted(counts.items())
        if status not in known and int(count or 0) != 0
    )
    return (
        f"[phase3a] windows: total={window_stats['windows_total']} "
        f"model_calls={window_stats['model_calls']} "
        f"intentional_silence={window_stats['intentional_silence']} "
        + " ".join(ordered)
        + f" unrecoverable={window_stats.get('unrecoverable_count', 0)} "
        f"fallback={window_stats['fallback_windows']} "
        f"publishable={window_stats['publishable']}"
    )


def _init_diagnostics(output_dir: Path, run_id: str) -> None:
    global _DIAGNOSTICS_DIR, _DIAGNOSTICS_RUN_ID
    _DIAGNOSTICS_DIR = output_dir / "diagnostics" / "phase3a"
    _DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    _DIAGNOSTICS_RUN_ID = run_id


def _write_window_diagnostic(
    round_no: int,
    window_idx: int,
    window_id: str,
    t_start: float,
    t_end: float,
    generation_status: str,
    *,
    neutral_source: str = "",
    error_stage: str | None = None,
    error_type: str | None = None,
    error_detail: str | None = None,
    http_status: int | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    content_present: bool = False,
    content_chars: int = 0,
    reasoning_present: bool = False,
    reasoning_chars: int = 0,
    raw_response_saved: bool = False,
    attempt: int = 0,
    final: bool = True,
    retry_count: int = 0,
    first_attempt_status: str | None = None,
    first_attempt_detail: str | None = None,
    silent_reason: str | None = None,
) -> None:
    """写入一条脱敏诊断摘要（不含 reasoning 正文），线程安全。

    方案 R（§8）：重试窗写多条——首次失败（attempt=0, final=false）、每次重试、
    最终成功（final=true, neutral_source=llm_retry）或不可恢复终态
    （final=true, neutral_source=unrecoverable, silent_reason=unrecoverable_failure）。
    单条旧记录按 attempt=0, final=true 读取，向后兼容。"""
    if _DIAGNOSTICS_DIR is None:
        return
    entry = {
        "run_id": _DIAGNOSTICS_RUN_ID,
        "round_no": round_no,
        "window_id": window_id,
        "window_index": window_idx,
        "t_start": t_start,
        "t_end": t_end,
        "generation_status": generation_status,
        "neutral_source": neutral_source,
        "attempt": attempt,
        "final": final,
    }
    if retry_count:
        entry["retry_count"] = retry_count
    if first_attempt_status is not None:
        entry["first_attempt_status"] = first_attempt_status
    if first_attempt_detail is not None:
        entry["first_attempt_detail"] = first_attempt_detail
    if silent_reason is not None:
        entry["silent_reason"] = silent_reason
    if error_stage is not None:
        entry["error_stage"] = error_stage
    if error_type is not None:
        entry["error_type"] = error_type
    if error_detail is not None:
        entry["error_detail"] = error_detail
    if http_status is not None:
        entry["http_status"] = http_status
    if finish_reason is not None:
        entry["finish_reason"] = finish_reason
    if usage is not None:
        entry["usage"] = usage
    entry["content_present"] = content_present
    entry["content_chars"] = content_chars
    entry["reasoning_present"] = reasoning_present
    entry["reasoning_chars"] = reasoning_chars
    entry["raw_response_saved"] = raw_response_saved
    with _DIAGNOSTICS_LOCK:
        diag_path = _DIAGNOSTICS_DIR / f"{_DIAGNOSTICS_RUN_ID}_diagnostics.jsonl"
        with diag_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


# ── LLM-A 语料收集（data/llma/，供小模型拼接层微调）──
_LLMA_CORPUS_LOCK = threading.Lock()
_LLMA_CORPUS_PATH: Path | None = None


def _init_llma_corpus(run_id: str) -> None:
    """按 run_id 初始化语料文件 data/llma/<run_id>.jsonl（追加写）。"""
    global _LLMA_CORPUS_PATH
    corpus_dir = _PROJECT_ROOT / "data" / "llma"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    _LLMA_CORPUS_PATH = corpus_dir / f"{run_id}.jsonl"


def _append_llma_corpus(entry: dict) -> None:
    """追加一条 (input 投影, output neutral) 样本；任何失败都不影响主链。"""
    if _LLMA_CORPUS_PATH is None:
        return
    try:
        with _LLMA_CORPUS_LOCK:
            with _LLMA_CORPUS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


@dataclass
class AnalystResult:
    """单窗口 LLM 调用的结构化结果。

    neutral_source 只管"文本从哪来"（三值互斥）：
      - "llm": LLM 成功返回合法 JSON，neutral 非空
      - "intentional_empty": LLM 返回 {"neutral":""} 或 main_topic 无可靠内容
      - "fallback": dry_run 或 plan 不存在（生产路径不应出现）

    generation_status 只管"处理过程出了什么错"：
      - "success": 一切正常
      - "http_error": HTTP 请求失败
      - "parse_error": 无法解析为 JSON
      - "contract_error": JSON 合法但字段不符契约（含超长）
      - "truncated": finish_reason == "length" 且输出不完整
    """
    content: str = ""
    neutral_source: str = ""
    generation_status: str = "success"
    error_stage: str | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    usage: dict | None = None
    error_type: str | None = None
    error_detail: str | None = None
    raw_response: str | None = None
    api_result: object | None = None  # 原始 _ApiChatResult 实例（调试用）
    # 方案 R 增量字段（缺省时等价于未经历恢复）
    retry_count: int = 0
    first_attempt_status: str | None = None
    first_attempt_detail: str | None = None
    retry_after_sec: float | None = None


def _call_analyst(
    prompt: str,
    llm_cfg: dict,
    gen_fn,
    system_prompt: str | None = None,
    round_no: int = 0,
    run_id: str | None = None,
    debug: bool = False,
    seg: int = 0,
    max_tokens: int | None = None,
    char_limit: int = _DEFAULT_EFFECTIVE_CHAR_LIMIT,
    projection: dict | None = None,
) -> AnalystResult:
    try:
        log_ctx = {"round": f"round{round_no}"}
        if run_id:
            log_ctx["run_id"] = run_id
        if seg > 0:
            log_ctx["scene"] = f"win{seg}"
        raw = gen_fn(prompt, llm_cfg, system_prompt=system_prompt, max_tokens=max_tokens, log_ctx=log_ctx,
                     response_format={"type": "json_object"})
    except Exception as exc:
        print(f"[phase3a] round {round_no} analyst error: {type(exc).__name__}: {exc}", file=sys.stderr)
        response = getattr(exc, "response", None)
        http_status = getattr(response, "status_code", None)
        if isinstance(http_status, int) and 400 <= http_status < 500 and http_status not in (408, 429):
            # 客户端 4xx（除 408/429）是参数/鉴权类错误，重试无意义：标记为不可恢复类别。
            status = "http_client_error"
        else:
            status = "http_error" if isinstance(http_status, int) else "transport_error"
        retry_after = retry_after_seconds(response) if status == "http_error" and response is not None else None
        return AnalystResult(
            generation_status=status,
            error_stage="http" if status in ("http_error", "http_client_error") else "transport",
            http_status=http_status if isinstance(http_status, int) else None,
            error_type=type(exc).__name__,
            error_detail=str(exc),
            retry_after_sec=retry_after,
        )

    finish_reason = getattr(raw, "finish_reason", None)
    api_result = raw  # 保留原始 _ApiChatResult 供调试
    raw_envelope = getattr(raw, "raw_response", None)
    raw_usage = getattr(raw, "usage", None)
    usage = raw_usage if isinstance(raw_usage, dict) else (
        raw_envelope.get("usage")
        if isinstance(raw_envelope, dict) and isinstance(raw_envelope.get("usage"), dict)
        else None
    )
    http_status = getattr(raw, "http_status", None)
    if isinstance(raw_envelope, dict):
        choices = raw_envelope.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            return AnalystResult(
                generation_status="response_error",
                error_stage="response_envelope",
                finish_reason=finish_reason,
                http_status=http_status if isinstance(http_status, int) else None,
                usage=usage,
                error_type="InvalidResponseEnvelope",
                error_detail="OpenAI-compatible response is missing a string choices[0].message.content",
                raw_response=str(raw),
                api_result=api_result,
            )
    if finish_reason == "length":
        return AnalystResult(
            generation_status="truncated",
            error_stage="response_contract",
            finish_reason=finish_reason,
            http_status=http_status if isinstance(http_status, int) else None,
            usage=usage,
            error_type="IncompleteJSON",
            error_detail="finish_reason=length",
            raw_response=str(raw),
            api_result=api_result,
        )

    if debug:
        debug_dir = _PROJECT_ROOT / "output" / "debug_phase3"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dump = {
            "round_no": round_no,
            "seg": seg,
            "model": llm_cfg.get("model", ""),
            "phase": "3a_analyst_window",
            "prompt": prompt,
            "response": str(raw),
        }
        name = f"r{round_no:03d}_w{seg:02d}_3a_analyst.json" if seg else f"r{round_no:03d}_3a_analyst.json"
        (debug_dir / name).write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")

    # 解析 JSON
    raw_text = str(raw)
    parsed = _parse_window_neutral_response(raw_text, debug=debug)
    if parsed is None:
        # 诊断失败原因
        json_obj = _first_json_obj(raw_text)
        if json_obj is None:
            return AnalystResult(
                generation_status="parse_error",
                error_stage="json_parse",
                finish_reason=finish_reason,
                http_status=http_status if isinstance(http_status, int) else None,
                usage=usage,
                error_type="InvalidJSON",
                error_detail="not a valid JSON object",
                raw_response=raw_text,
                api_result=api_result,
            )
        else:
            for _key in ("think", "reasoning", "reasoning_content"):
                json_obj.pop(_key, None)
            extra = set(json_obj) - {"neutral"}
            if extra:
                detail = f"unexpected fields: {sorted(extra)}"
            elif "neutral" not in json_obj:
                detail = "missing 'neutral' field"
            elif not isinstance(json_obj.get("neutral"), str):
                detail = "neutral is not a string"
            else:
                detail = "unknown contract error"
            # finish_reason == "length" 且契约失败 → 判定为截断
            status = "truncated" if finish_reason == "length" else "contract_error"
            return AnalystResult(
                generation_status=status,
                error_stage="response_contract",
                finish_reason=finish_reason,
                http_status=http_status if isinstance(http_status, int) else None,
                usage=usage,
                error_type="IncompleteJSON" if status == "truncated" else "NeutralContractError",
                error_detail=detail,
                raw_response=raw_text,
                api_result=api_result,
            )

    # 解析成功 → 长度校验 + 来源判定
    if count_spoken_chars(parsed) > char_limit:
        return AnalystResult(
            generation_status="contract_error",
            error_stage="response_contract",
            finish_reason=finish_reason,
            http_status=http_status if isinstance(http_status, int) else None,
            usage=usage,
            error_type="NeutralLengthExceeded",
            error_detail=f"neutral too long: {count_spoken_chars(parsed)} chars (limit {char_limit})",
            raw_response=raw_text,
            api_result=api_result,
        )

    semantic_error, semantic_detail = validate_neutral_semantics(parsed, projection) if projection is not None else (None, None)
    if semantic_error is not None:
        return AnalystResult(
            generation_status=semantic_error,
            error_stage="semantic_contract",
            finish_reason=finish_reason,
            http_status=http_status if isinstance(http_status, int) else None,
            usage=usage,
            error_type=semantic_error,
            error_detail=semantic_detail,
            raw_response=raw_text,
            api_result=api_result,
        )

    if parsed:
        return AnalystResult(
            content=parsed,
            neutral_source="llm",
            generation_status="success",
            finish_reason=finish_reason,
            http_status=http_status if isinstance(http_status, int) else None,
            usage=usage,
            api_result=api_result,
        )
    else:
        return AnalystResult(
            content="",
            neutral_source="intentional_empty",
            generation_status="success",
            finish_reason=finish_reason,
            http_status=http_status if isinstance(http_status, int) else None,
            usage=usage,
            api_result=api_result,
        )


# ── 方案 R：可恢复错误自纠重试层（§5 R1 / §6 R2 / §7 R3 / §8 R4）─────────────
# 本块为孤立新增：不修改 _call_analyst 的 S2 分类顺序，仅在分类后对可恢复错误
# 做"错误反馈重生成"或"原样重调用+退避"，耗尽则诚实标 unrecoverable 并留白。
# 不生成 fallback、不截字、不掩盖失败（每次尝试均留诊断，§8 R4）。
_RECOVERABLE_MODEL_ERRORS = (
    "contract_error", "parse_error", "truncated", "semantic_contract_error",
    "required_fact_missing", "unexpected_fact", "side_mismatch",
)
_INFRA_ERROR_CLASSES = ("transport_error", "http_error", "response_error")
_REASONING_STRIP_KEYS = ("think", "reasoning", "reasoning_content")
_RECOVERY_DEFAULT_MAX_RETRIES = 3
_RESPONSE_ERROR_MAX_RETRIES = 2  # §6.2：response_error 同形 2 次即止，不第 3 次
_DEFAULT_CLIENT_ERROR_THRESHOLD = 5  # 连续客户端 4xx 达到该值即熔断中止


class Phase3aCircuitBreak(RuntimeError):
    """连续客户端 4xx（鉴权/参数类）达到阈值，立即中止整阶段，避免无效重试白跑。"""


class _ClientErrorCircuitBreaker:
    """线程安全计数器：连续 http_client_error 达到阈值即触发熔断。"""

    def __init__(self, threshold: int) -> None:
        self.threshold = max(1, int(threshold))
        self._count = 0
        self._lock = threading.Lock()

    def record_failure(self) -> bool:
        with self._lock:
            self._count += 1
            return self._count >= self.threshold

    def record_success(self) -> None:
        with self._lock:
            self._count = 0


def _recovery_config(semantic_cfg: dict) -> dict:
    """从 semantic.recovery 读取恢复配置；缺省 enabled=False → 与现行零容忍一致。"""
    rec = semantic_cfg.get("recovery")
    if not isinstance(rec, dict):
        rec = {}
    return {
        "enabled": bool(rec.get("enabled", False)),
        "max_retries": int(rec.get("max_retries", _RECOVERY_DEFAULT_MAX_RETRIES)),
    }


def _diag_meta_from_result(result: AnalystResult) -> dict:
    """从 AnalystResult 提取诊断字段（content/reasoning 摘要），镜像 _process_round 旧逻辑。"""
    api_raw = result.api_result
    return dict(
        error_stage=result.error_stage,
        error_type=result.error_type,
        error_detail=result.error_detail,
        http_status=result.http_status,
        finish_reason=result.finish_reason,
        usage=result.usage,
        content_present=bool(api_raw),
        content_chars=len(str(api_raw)) if api_raw is not None else 0,
        reasoning_present=bool(getattr(api_raw, "reasoning_content", None) if api_raw is not None else None),
        reasoning_chars=len(getattr(api_raw, "reasoning_content", "") or "") if api_raw is not None else 0,
        raw_response_saved=bool(getattr(api_raw, "raw_response", None) if api_raw is not None else None),
    )


def _try_strip_reparse(result: AnalystResult, char_limit: int, projection: dict | None = None) -> AnalystResult | None:
    """R3（§7）：truncated 时防御性去除 reasoning/think 后重解析。

    仅当 raw 中确有 reasoning/think 字段（或 message 级 reasoning_content）且去除后
    得到合法 {"neutral":"..."} 且长度合规时，改判 success——"截断其实是被 think
    吃掉的预算"。无 reasoning 可去则返回 None（保持 truncated，走 R1 修正重试），
    以保留 S2 "finish_reason=length 即使 content 恰好形成合法 JSON 仍判 truncated"
    的不变量（§7.2 / §9 测试 4 "残留 reasoning"）。"""
    if result.generation_status != "truncated":
        return None
    raw_text = result.raw_response or ""
    if not raw_text:
        return None
    obj = _first_json_obj(raw_text)
    if obj is None:
        return None
    stripped_any = any(key in obj for key in _REASONING_STRIP_KEYS)
    if not stripped_any:
        api_raw = result.api_result
        if api_raw is not None and getattr(api_raw, "reasoning_content", None):
            stripped_any = True
    if not stripped_any:
        return None
    for key in _REASONING_STRIP_KEYS:
        obj.pop(key, None)
    if set(obj) != {"neutral"}:
        return None
    value = obj.get("neutral")
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or count_spoken_chars(value) > char_limit:
        return None
    if projection is not None and validate_neutral_semantics(value, projection)[0] is not None:
        return None
    return AnalystResult(
        content=value,
        neutral_source="llm",
        generation_status="success",
        finish_reason=result.finish_reason,
        http_status=result.http_status,
        usage=result.usage,
        api_result=result.api_result,
        retry_count=0,
        first_attempt_status="truncated",
        first_attempt_detail=result.error_detail,
    )


def _build_correction_prompt(category: str, prior: str, detail: str | None, char_limit: int, base_prompt: str) -> str:
    """R1（§5.2）：类别专属修正 prompt（文本约束），追加到原投影 prompt 之后。"""
    if category == "truncated":
        correction = (
            f"你上一次输出被 token 上限截断，非完整 JSON。"
            f"输出完整且≤{char_limit}字的 JSON {{\"neutral\":\"...\"}}。"
            f"（已移除思考内容，预算全用于正文。）"
        )
    elif category == "parse_error":
        correction = (
            "你上一次输出不是合法 JSON 对象。"
            "只输出一个 JSON 对象 {\"neutral\":\"...\"}，不要 markdown 围栏、不要解释文字。"
        )
    elif category == "contract_error":
        if detail and "too long" in detail:
            correction = (
                f"你上一次输出 {prior}（{detail}）超过本窗口上限 {char_limit} 字。"
                f"只输出 JSON {{\"neutral\":\"...\"}}，≤{char_limit}字，必须保留人名/队伍/事件/武器等事实，"
                f"只压缩措辞，不得新增或丢弃事实。"
            )
        else:
            correction = (
                f"你上一次输出 {prior} 字段不符：{detail or '不符'}。"
                f"只输出 JSON {{\"neutral\":\"...\"}}，仅含字符串字段 neutral，不要其他字段。"
            )
    elif category in {"semantic_contract_error", "required_fact_missing", "unexpected_fact", "side_mismatch"}:
        correction = (
            f"你上一次输出违反事实契约：{category}（{detail or '不符'}）。"
            "只根据原投影重写，不得添加任何新事实；逐字保留全部 required_facts[].canonical_text，"
            "T 是进攻方、CT 是防守方且不得互换。"
        )
    else:
        correction = ""
    if not correction:
        return base_prompt
    return base_prompt + "\n\n【修正要求】\n" + correction


def _is_budget_silence(result: AnalystResult) -> bool:
    """预算静默标记：成本护栏超限时的预期留白（非模型错误），不判错、不重试。"""
    return bool(getattr(result.api_result, "budget_silence", False))


def _reclassify_empty_neutral(result: AnalystResult) -> AnalystResult:
    """A3：非 silence 主题但 LLM 返回空稿 → contract_error（可恢复）。"""
    return AnalystResult(
        content="",
        neutral_source="",
        generation_status="contract_error",
        error_stage="response_contract",
        finish_reason=result.finish_reason,
        http_status=result.http_status,
        usage=result.usage,
        error_type="EmptyNeutral",
        error_detail="neutral is empty for non-silence topic",
        raw_response=result.raw_response,
        api_result=result.api_result,
    )


def _make_unrecoverable(result: AnalystResult, first_status: str, first_detail: str | None, retries_done: int) -> AnalystResult:
    """R2/R1 耗尽后构造不可恢复终态（§6.4 留白）：保留真实失败类，不填假内容。"""
    return AnalystResult(
        content="",
        neutral_source="unrecoverable",
        generation_status=result.generation_status,
        error_stage=result.error_stage,
        finish_reason=result.finish_reason,
        http_status=result.http_status,
        usage=result.usage,
        error_type=result.error_type,
        error_detail=result.error_detail,
        raw_response=result.raw_response,
        api_result=result.api_result,
        retry_count=retries_done,
        first_attempt_status=first_status,
        first_attempt_detail=first_detail,
    )


def _recover_analyst_window(
    prompt: str,
    llm_cfg: dict,
    gen_fn,
    *,
    system_prompt: str | None,
    round_no: int,
    run_id: str | None,
    debug: bool,
    seg: int,
    max_tokens: int | None,
    char_limit: int,
    window_id: str,
    t_start: float,
    t_end: float,
    recovery_cfg: dict,
    expect_content: bool,
    projection: dict | None = None,
) -> AnalystResult:
    """方案 R 主循环：R1 错误反馈重试 + R2 基建退避重调用 + R3 think-strip + R4 多记录。

    返回最终 AnalystResult（含 retry_count/first_attempt_status 等增量字段）。
    每次尝试写一条诊断（§8 R4，不掩盖）。恢复未启用时退化为单次调用 + 单条诊断，
    与现行零容忍行为完全一致。"""
    enabled = bool(recovery_cfg.get("enabled"))
    max_retries = int(recovery_cfg.get("max_retries", _RECOVERY_DEFAULT_MAX_RETRIES)) if enabled else 0
    diag_base = dict(round_no=round_no, window_idx=seg, window_id=window_id, t_start=t_start, t_end=t_end)

    def _diag(result: AnalystResult, attempt: int, final: bool, *, source: str | None = None,
              first_status: str | None = None, first_detail: str | None = None,
              silent_reason: str | None = None) -> None:
        meta = _diag_meta_from_result(result)
        kwargs = dict(diag_base)
        kwargs["generation_status"] = result.generation_status
        kwargs["neutral_source"] = result.neutral_source if source is None else source
        if final and source is None and result.generation_status == "success":
            # 成功终态的来源由调用方在 result 上设置（llm / llm_retry）
            kwargs["neutral_source"] = result.neutral_source
        kwargs["attempt"] = attempt
        kwargs["final"] = final
        kwargs["retry_count"] = result.retry_count if final else 0
        if first_status is not None:
            kwargs["first_attempt_status"] = first_status
        if first_detail is not None:
            kwargs["first_attempt_detail"] = first_detail
        if silent_reason is not None:
            kwargs["silent_reason"] = silent_reason
        kwargs.update(meta)
        _write_window_diagnostic(**kwargs)

    # ── 首次调用 ──
    first = _call_analyst(
        prompt, llm_cfg, gen_fn, system_prompt=system_prompt,
        round_no=round_no, run_id=run_id, debug=debug, seg=seg,
        max_tokens=max_tokens, char_limit=char_limit,
        projection=projection,
    )
    first_status = first.generation_status
    first_detail = first.error_detail

    # A3：非 silence 主题的空稿 → contract_error（可恢复）
    # 预算静默（budget_silence）是成本护栏的预期留白，非模型错误：不判错、不重试。
    if (first.generation_status == "success" and first.neutral_source == "intentional_empty"
            and expect_content and not _is_budget_silence(first)):
        first = _reclassify_empty_neutral(first)
        first_status = first.generation_status
        first_detail = first.error_detail

    # R3：truncated 先 strip+reparse（仅恢复启用时，§7 防御性兜底）
    if enabled and first.generation_status == "truncated":
        stripped = _try_strip_reparse(first, char_limit, projection)
        if stripped is not None:
            stripped.first_attempt_status = "truncated"
            stripped.first_attempt_detail = first_detail
            _diag(stripped, 0, True, first_status="truncated", first_detail=first_detail)
            return stripped

    # 首次即成功（含规则层 silence 的 intentional_empty）
    if first.generation_status == "success":
        first.retry_count = 0
        _diag(first, 0, True)
        return first

    # 恢复未启用：单次失败即终态（与现行零容忍一致，不标 unrecoverable）
    if not enabled or max_retries <= 0:
        _diag(first, 0, True)
        return first

    # ── 恢复启用：首次失败留记录（attempt=0, final=false）──
    _diag(first, 0, False, first_status=first_status, first_detail=first_detail)

    result = first
    response_error_retries = 0
    retries_done = 0
    for attempt in range(1, max_retries + 1):
        if result.generation_status == "success":
            break
        if result.generation_status not in (_RECOVERABLE_MODEL_ERRORS + _INFRA_ERROR_CLASSES):
            break  # 非可恢复类别，不再重试

        # §6.2：response_error 同形 2 次重调用即止，不第 3 次
        if (result.generation_status == "response_error"
                and response_error_retries >= _RESPONSE_ERROR_MAX_RETRIES):
            break  # 已做 2 次重调用仍同形，视为端点不兼容

        if result.generation_status in _INFRA_ERROR_CLASSES:
            # R2：原样重调用 + 退避（无修正 prompt）；http_error 尊重 Retry-After
            backoff = infra_backoff_delay(attempt - 1, result.retry_after_sec)
            time.sleep(backoff)
            retry_prompt = prompt
        else:
            # R1：错误反馈修正 prompt（原系统prompt + 原投影 + 修正）
            retry_prompt = _build_correction_prompt(
                result.generation_status, result.raw_response or "",
                result.error_detail, char_limit, prompt,
            )

        result = _call_analyst(
            retry_prompt, llm_cfg, gen_fn, system_prompt=system_prompt,
            round_no=round_no, run_id=run_id, debug=debug, seg=seg,
            max_tokens=max_tokens, char_limit=char_limit,
            projection=projection,
        )
        retries_done = attempt

        # R3：对重试输出也做 think-strip
        if result.generation_status == "truncated":
            stripped = _try_strip_reparse(result, char_limit, projection)
            if stripped is not None:
                result = stripped

        if result.generation_status == "response_error":
            response_error_retries += 1

        is_final = result.generation_status == "success"
        _diag(result, attempt, is_final, first_status=first_status, first_detail=first_detail)

    # ── 最终裁决 ──
    if result.generation_status == "success":
        result.neutral_source = "llm_retry"
        result.retry_count = retries_done
        result.first_attempt_status = first_status
        result.first_attempt_detail = first_detail
        return result

    # 耗尽 → 不可恢复终态（§6.4 留白）
    unrecoverable = _make_unrecoverable(result, first_status, first_detail, retries_done)
    _diag(unrecoverable, retries_done, True, source="unrecoverable",
          first_status=first_status, first_detail=first_detail,
          silent_reason="unrecoverable_failure")
    return unrecoverable


def _video_time(frame: dict) -> float:
    return float((frame.get("when") or {}).get("video_time", 0.0))


def _evenly_sample(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return list(items)
    if limit <= 0:
        return []
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def _is_priority_frame(frame: dict) -> bool:
    ev = frame.get("events") or {}
    c4 = ev.get("c4") or {}
    return bool(ev.get("kills") or c4.get("planted") or c4.get("begin_defuse_tick"))


def _is_context_frame(frame: dict) -> bool:
    ev = frame.get("events") or {}
    return bool(
        ev.get("damages")
        or ev.get("flashes")
        or ev.get("smokes_active")
        or ev.get("infernos_active")
    )


def _select_window_frames(frames: list[dict], lo: float, hi: float, *, is_last: bool) -> list[dict]:
    selected: list[dict] = []
    for frame in frames:
        t = _video_time(frame)
        upper_ok = t <= hi if is_last else t < hi
        if lo <= t and upper_ok:
            selected.append(frame)

    if len(selected) < _MIN_WINDOW_FRAMES and frames:
        mid = (lo + hi) / 2.0
        existing = {_video_time(frame) for frame in selected}
        for frame in sorted(frames, key=lambda item: abs(_video_time(item) - mid)):
            if _video_time(frame) in existing:
                continue
            selected.append(frame)
            existing.add(_video_time(frame))
            if len(selected) >= _MIN_WINDOW_FRAMES:
                break

    selected.sort(key=_video_time)
    if len(selected) <= _MAX_WINDOW_FRAMES:
        return selected

    priority = [frame for frame in selected if _is_priority_frame(frame)]
    priority_times = {_video_time(frame) for frame in priority}
    if len(priority) >= _MAX_WINDOW_FRAMES:
        return _evenly_sample(priority, _MAX_WINDOW_FRAMES)

    context = [
        frame
        for frame in selected
        if _video_time(frame) not in priority_times and _is_context_frame(frame)
    ]
    keep = priority + _evenly_sample(context, _MAX_WINDOW_FRAMES - len(priority))
    keep_times = {_video_time(frame) for frame in keep}
    if len(keep) < _MAX_WINDOW_FRAMES:
        rest = [frame for frame in selected if _video_time(frame) not in keep_times]
        keep.extend(_evenly_sample(rest, _MAX_WINDOW_FRAMES - len(keep)))
    return sorted(keep, key=_video_time)


def _build_window_payload(
    payload: dict,
    frames: list[dict],
    plan: dict,
    *,
    rule_state: dict | None = None,
) -> dict:
    """Keep raw rule inputs out of the local LLM prompt and its audit file."""
    del payload, frames
    return build_llm_window_projection(plan, rule_state=rule_state)


@dataclass
class _WindowRequest:
    """阶段 1 规则预计算产物：单个窗口的待请求描述（不携带任何共享状态）。

    主线程按窗口顺序构建；仅 prompt 非空的窗口进入阶段 2 网络请求。
    prebuilt 承载 required_chars/required_facts 检查失败的预置 AnalystResult
    （不访问网络）；future 是阶段 2 提交后的并发句柄（兼容模式下为 None，
    阶段 3 同步调用）。"""
    idx: int
    window_id: str
    t_start: float
    t_end: float
    context_start: float
    context_end: float
    scene_label: str
    plan: dict
    window_payload: dict
    char_budget: int
    effective_char_limit: int
    topic_kind: str
    sc_hype: float
    scream_eligible: bool
    prompt: str = ""
    prebuilt: AnalystResult | None = None
    future: object | None = None
    rule_v4: dict | None = None


def run_phase3a(
    *,
    rounds_path: Path,
    output_path: Path,
    config_path: Path,
    dry_run: bool = False,
    progress_sink=None,
) -> dict:
    import os

    config = load_config(config_path)
    debug_enabled = not dry_run and bool(config.get("debug", {}).get("phase3", False) or os.getenv("AI6657_DEBUG_PHASE3"))
    debug_run_id = uuid.uuid4().hex if debug_enabled else ""
    debug_recorder = DebugRecorder(debug_enabled, debug_output_dir(debug_run_id) / "phase3a")
    llm_cfg = dict(config.get("llm", {}))
    semantic_cfg = config.get("semantic", {}) if isinstance(config.get("semantic", {}), dict) else {}
    # Phase3a 生成器模式（§12.2）：rule_template=纯规则中性句；legacy_llma=旧 LLM-A 路径。
    generator_cfg = semantic_cfg.get("phase3a_generator") or {}
    generator_mode = str(generator_cfg.get("mode", "legacy_llma"))
    if generator_mode not in {"rule_template", "legacy_llma"}:
        raise ValueError(f"unsupported phase3a_generator.mode: {generator_mode}")
    recovery_cfg = _recovery_config(semantic_cfg)
    analyst_model = semantic_cfg.get("analyst_model") or semantic_cfg.get("model", "")
    if analyst_model:
        llm_cfg["model"] = analyst_model
    # 请求速率治理：拆分发送（每窗口一次）时压制请求频率，避免触发服务端限流/封禁。
    if semantic_cfg.get("analyst_request_interval_sec"):
        llm_cfg["request_interval_sec"] = float(semantic_cfg["analyst_request_interval_sec"])

    backend = resolve_backend(config, "analyst")
    if backend not in {"api", "vllm"}:
        raise ValueError(f"unsupported analyst backend: {backend}; use vllm or api")

    # S4: Phase3a 专属 max_tokens。
    # 新 key analyst_output_max_tokens 直接发送原值；旧 key analyst_max_tokens 保持旧有效行为（×10 兜底），
    # 因为 llm_shim 不再隐藏 ×10，兼容责任移到此处显式计算。
    # 云端用 cloud_analyst_output_max_tokens 放开；本地 vllm 保持 analyst_output_max_tokens 原值。
    if backend == "api" and "cloud_analyst_output_max_tokens" in semantic_cfg:
        llm_cfg["max_tokens"] = int(semantic_cfg["cloud_analyst_output_max_tokens"])
    elif "analyst_output_max_tokens" in semantic_cfg:
        llm_cfg["max_tokens"] = int(semantic_cfg["analyst_output_max_tokens"])
    else:
        if "analyst_max_tokens" in semantic_cfg:
            import warnings
            warnings.warn(
                "semantic.analyst_max_tokens is deprecated; use semantic.analyst_output_max_tokens",
                DeprecationWarning,
                stacklevel=2,
            )
        llm_cfg["max_tokens"] = int(semantic_cfg.get("analyst_max_tokens", 256)) * 10

    llm_cfg["temperature"] = float(semantic_cfg.get("analyst_temperature", 0.3))
    llm_cfg["repeat_penalty"] = float(semantic_cfg.get("analyst_repeat_penalty", 1.3))
    llm_cfg["top_p"] = float(semantic_cfg.get("analyst_top_p", 0.9))

    if backend == "api":
        from sbmachine import cloud_memory
        gen_fn = cloud_memory.make_generate("llma", semantic_cfg=semantic_cfg)
    else:
        from sbmachine import llma_api as _llma_backend
        gen_fn = _llma_backend.generate

    run_metadata = new_manifest_metadata(rounds_path)
    run_id = str(run_metadata["run_id"])
    # S2: 初始化诊断目录（与 debug 解耦，非 debug 模式也写摘要）
    _init_diagnostics(output_path.parent, run_id)
    _init_llma_corpus(run_id)
    match = load_match(rounds_path)
    # 物理帧来源是 phase2 产出的精简 DEM-fact sidecar；
    # rounds_with_yolo.json 仍通过 load_match 作为契约/身份锚点。
    configured_semantic_path = resolve_path(
        config.get("paths", {}).get("rounds_with_yolo_semantic_json")
    )
    semantic_path = configured_semantic_path or rounds_path.with_name("rounds_with_yolo_semantic.json")
    if configured_semantic_path is not None and not semantic_path.is_file():
        raise ValueError(f"Phase3a semantic input is missing: {semantic_path}")
    semantic_frames = load_semantic_frames(semantic_path)

    # 回合级并发：api 后端默认 5（回合内窗口默认串行，PlannerState/rule_state 顺序依赖不变）；
    # 本地 vllm 保持 1（显存/吞吐受限）。
    default_concurrent = 5 if backend == "api" else 1
    concurrent_rounds = max(1, int(semantic_cfg.get("analyst_concurrent_rounds", default_concurrent)))
    # 阶段 3：LLM-A 两阶段并发——窗口级有界并发（缺省按后端：云端 4、本地 1=串行；
    # 显式配置可覆盖）。
    default_window_concurrency = 4 if backend == "api" else 1
    window_concurrency = max(1, int(semantic_cfg.get("analyst_window_concurrency", default_window_concurrency)))
    client_error_threshold = int(semantic_cfg.get("analyst_client_error_threshold", _DEFAULT_CLIENT_ERROR_THRESHOLD))
    breaker = _ClientErrorCircuitBreaker(client_error_threshold)
    if backend == "api":
        from sbmachine import cloud_prompts
        analyst_system = cloud_prompts.build_cloud_analyst_system()
    else:
        analyst_system = _build_analyst_system()
    window_max_sec = float(semantic_cfg.get("window_max_sec", 10.0))
    window_min_sec = float(semantic_cfg.get("window_min_sec", 3.0))
    speech_rate = _speech_rate_config(config)
    tactic_book = load_tactic_book(match.map_name)

    def _process_round(rnd, window_pool=None) -> tuple[dict, list[tuple[object, str]]]:
        payload = _semantic_payload(rnd, external_frames=semantic_frames.get(rnd.round_no))
        beats = payload.get("keyframes", [])
        hypes = compute_hype(beats)
        peak_hype = max(hypes) if hypes else 0.0
        avg_hype = round(sum(hypes) / len(hypes), 3) if hypes else 0.0
        round_emotion = dominant_round_emotion(peak_hype)

        def _request_window(req: _WindowRequest) -> AnalystResult:
            """单窗口 LLM-A 请求（含方案 R 恢复与逐尝试诊断）。
            future 存在（并发模式）时取 worker 结果；否则同步调用（并发度 1 兼容）。"""
            if req.future is not None:
                return req.future.result()
            return _recover_analyst_window(
                req.prompt, llm_cfg, gen_fn, system_prompt=analyst_system,
                round_no=rnd.round_no, run_id=run_id, debug=debug_enabled, seg=req.idx,
                max_tokens=int(llm_cfg.get("max_tokens", 256)),
                char_limit=req.effective_char_limit,
                window_id=req.window_id, t_start=req.t_start, t_end=req.t_end,
                recovery_cfg=recovery_cfg, expect_content=True,
                projection=req.window_payload,
            )

        windows = build_scene_contexts(
            beats, rnd.start_sec, rnd.end_sec,
            window_max_sec=window_max_sec,
            window_min_sec=window_min_sec,
            runtime_config=config,
        )
        scenes_out: list[dict] = []
        failed_windows = 0
        unrecoverable_windows = 0
        fallback_windows = 0
        planner_state = PlannerState()
        reported_rule_state: dict[str, dict] = {}
        reported_player_state: dict[str, dict] = {}
        accepted_samples: list[tuple[object, str]] = []
        llma_inputs: list[dict] = []

        # ── 阶段 1：规则层预计算（按回合串行，不访问网络）──
        # PlannerState / reported_rule_state 仅在此阶段修改；产出"待请求窗口"列表，
        # 每个元素是自包含的 _WindowRequest，之后不再触碰共享规则状态。
        phase1: list[_WindowRequest] = []
        for idx, window in enumerate(windows, start=1):
            t_start, t_end, scene_label = window.t_start, window.t_end, window.scene
            window_id = f"r{rnd.round_no:03d}_w{idx:02d}"
            is_last = idx == len(windows)
            ownership_frames = [frame for frame in beats if t_start <= _video_time(frame) and (_video_time(frame) <= t_end if is_last else _video_time(frame) < t_end)]
            context_frames = [frame for frame in beats if window.context_start <= _video_time(frame) and (_video_time(frame) <= window.context_end if is_last else _video_time(frame) < window.context_end)]
            sc_hype = _scene_hype(beats, hypes, t_start, t_end)
            sc_emotion = dominant_round_emotion(sc_hype)
            char_budget = _compute_char_budget(max(1.0, t_end - t_start), sc_emotion, speech_rate)
            effective_char_limit = min(_NEUTRAL_HARD_LIMIT, max(1, char_budget))
            projection = build_window_rule_projection(
                match.map_name, window, ownership_frames, context_frames, beats, planner_state, tactic_book,
                is_last_window=is_last,
                char_budget=effective_char_limit,
            )
            if debug_enabled and projection.debug is not None:
                debug_dir = _PROJECT_ROOT / "output" / "debug_phase3"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"r{rnd.round_no:03d}_w{idx:02d}_tactic_match.json"
                debug_path.write_text(json.dumps(projection.debug, ensure_ascii=False, indent=2), encoding="utf-8")
            plan = projection.plan
            rule_state = build_rule_state_delta(ownership_frames, reported_rule_state)

            window_payload = _build_window_payload(payload, ownership_frames, plan, rule_state=rule_state)
            # A5: 将字符上限注入投影，供 Prompt builder 和审计产物使用。
            window_payload["character_limit"] = effective_char_limit
            # The archive is the same strict projection passed to LLM-A.
            llma_inputs.append(window_payload)

            topic_kind = (plan.get("main_topic") or {}).get("kind")
            req = _WindowRequest(
                idx=idx, window_id=window_id, t_start=t_start, t_end=t_end,
                context_start=window.context_start, context_end=window.context_end,
                scene_label=scene_label, plan=plan, window_payload=window_payload,
                char_budget=char_budget, effective_char_limit=effective_char_limit,
                topic_kind=topic_kind, sc_hype=sc_hype,
                scream_eligible=_scene_scream_eligible(beats, t_start, t_end),
            )
            if generator_mode == "rule_template" and topic_kind != "silence":
                # v4 规则中性句渲染：不调用 LLM-A；dry_run 也渲染（无网络副作用）。
                fact_task = build_atomic_fact_units(window_id, plan)
                try:
                    render_out = render_neutral(fact_task)
                    capsule = render_capsule(fact_task)
                except Exception as exc:
                    req.prebuilt = AnalystResult(
                        generation_status="semantic_contract_error",
                        error_stage="rule_renderer",
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )
                else:
                    validation = renderer_validate(
                        render_out["neutral"], fact_task["fact_units"], fact_task["required_fact_ids"]
                    )
                    if validation["missing_required"]:
                        req.prebuilt = AnalystResult(
                            generation_status="semantic_contract_error",
                            error_stage="rule_renderer",
                            error_type="fact_preservation_failed",
                            error_detail=f"missing required facts: {validation['missing_required']}",
                        )
                    else:
                        # rule_contract_unfit 保守检查（§7.4）：无 validated profile 期间用
                        # 5 字/秒 × 最高 1.5 倍速的最短 capsule 估算；仍超 slot 则写 JSON 前失败。
                        slot_sec = max(0.001, t_end - t_start)
                        capsule_sec_est = count_spoken_chars(capsule) / 5.0 / 1.5
                        if capsule_sec_est > slot_sec:
                            raise RendererUnfitError(
                                f"{window_id}: shortest capsule ({count_spoken_chars(capsule)} chars) "
                                f"estimated {capsule_sec_est:.2f}s > slot {slot_sec:.2f}s"
                            )
                        req.prebuilt = AnalystResult(
                            content=render_out["neutral"],
                            neutral_source="rule_template",
                            generation_status="success",
                        )
                        req.rule_v4 = {
                            "fact_catalog": fact_task["fact_units"],
                            "required_fact_ids": fact_task["required_fact_ids"],
                            "fact_anchors": fact_task["fact_anchors"],
                            "capsule": capsule,
                            "target_units": fact_task["target_units"],
                            "hard_units": fact_task["hard_units"],
                            "neutral_renderer": {
                                "selected": render_out["neutral_source"],
                                "policy": render_out["renderer_policy"],
                            },
                        }
            elif not dry_run and topic_kind != "silence":
                # A3: 规则层 silence 不调用 LLM-A，直接产出 intentional_empty/success。
                required_chars = window_payload.get("required_chars")
                required_facts = window_payload.get("required_facts")
                if not isinstance(required_chars, int) or required_chars > effective_char_limit:
                    req.prebuilt = AnalystResult(
                        generation_status="projection_budget_error",
                        error_stage="projection_contract",
                        error_type="projection_budget_error",
                        error_detail=f"required_chars={required_chars!r} exceeds character_limit={effective_char_limit}",
                    )
                elif not isinstance(required_facts, list) or not required_facts:
                    req.prebuilt = AnalystResult(
                        generation_status="semantic_contract_error",
                        error_stage="projection_contract",
                        error_type="semantic_contract_error",
                        error_detail="non-silence projection has no required facts",
                    )
                else:
                    # 个人状态：legacy_llma 非静默且预算/事实检查通过的窗口才注入，
                    # 首个真正发给 LLM-A 的窗口拿到完整基线；静默与预算失败窗口
                    # 不调用，避免提前消耗首窗基线。
                    player_state = build_state_block(ownership_frames, reported_player_state)
                    if player_state:
                        window_payload["player_state"] = player_state
                    req.prompt = _build_window_prompt(window_payload)
            phase1.append(req)

        # ── 阶段 2：无状态 LLM-A 请求（有界窗口并发）──
        # worker 只执行"请求+恢复重试"，返回 AnalystResult，不修改任何共享结构；
        # 请求带固定 window_id，跨窗口无会话状态。兼容模式（window_pool=None）
        # 不提交，由阶段 3 按序同步调用，行为与现状完全一致。
        if window_pool is not None:
            for req in phase1:
                if req.prompt:
                    req.future = window_pool.submit(
                        _recover_analyst_window,
                        req.prompt, llm_cfg, gen_fn,
                        system_prompt=analyst_system,
                        round_no=rnd.round_no, run_id=run_id, debug=debug_enabled, seg=req.idx,
                        max_tokens=int(llm_cfg.get("max_tokens", 256)),
                        char_limit=req.effective_char_limit,
                        window_id=req.window_id, t_start=req.t_start, t_end=req.t_end,
                        recovery_cfg=recovery_cfg, expect_content=True,
                        projection=req.window_payload,
                    )

        # ── 阶段 3：按 window_id 顺序归并（验收/兜底/诊断/组装，全部主线程）──
        try:
            for req in phase1:
                idx = req.idx
                t_start, t_end, scene_label = req.t_start, req.t_end, req.scene_label
                window_id = req.window_id
                plan = req.plan
                window_payload = req.window_payload
                effective_char_limit = req.effective_char_limit
                char_budget = req.char_budget
                topic_kind = req.topic_kind

                # neutral_source 只管"文本从哪来"，generation_status 只管"处理过程出了什么错"。
                # 两者严格分离：错误时不填假文本，neutral 保持 ""。
                neutral = ""
                neutral_source = ""
                generation_status = "success"
                analyst_result = None  # silence/dry_run 路径保持 None
                prompt = req.prompt  # silence/dry_run/prebuilt 路径无 Prompt

                if req.prompt:
                    analyst_result = _request_window(req)
                    neutral = analyst_result.content
                    neutral_source = analyst_result.neutral_source
                    generation_status = analyst_result.generation_status
                elif dry_run:
                    generation_status = "dry_run"
                elif topic_kind == "silence":
                    # A3: 规则层 silence 不调用 LLM-A，直接产出 intentional_empty/success。
                    neutral_source = "intentional_empty"
                else:
                    # projection 检查失败（阶段 1 已构造 AnalystResult，此处补诊断）
                    analyst_result = req.prebuilt
                    neutral = analyst_result.content
                    neutral_source = analyst_result.neutral_source
                    generation_status = analyst_result.generation_status
                    _write_window_diagnostic(
                        rnd.round_no, idx, window_id, t_start, t_end,
                        analyst_result.generation_status,
                        error_stage=analyst_result.error_stage,
                        error_type=analyst_result.error_type,
                        error_detail=analyst_result.error_detail,
                    )

                if generation_status == "http_client_error":
                    # 客户端 4xx（鉴权/参数类）：连续达到阈值即熔断中止整阶段
                    if breaker.record_failure():
                        raise Phase3aCircuitBreak(
                            f"round {rnd.round_no} window {window_id}: {analyst_result.error_detail}"
                        )
                elif generation_status == "success":
                    breaker.record_success()
                    # llm / llm_retry 成功：可作训练样本（延迟到整场门禁通过后提交）
                    if neutral_source in ("llm", "llm_retry") and neutral and analyst_result.api_result is not None:
                        accepted_response = analyst_result.api_result
                        if backend == "api":
                            # 业务验收通过后才写入会话历史（失败轮/空稿不入，
                            # 防空稿污染后续窗口上下文诱导连续空输出）
                            cloud_memory.commit_round(accepted_response)
                        accepted_output = json.dumps(
                            {"neutral": neutral}, ensure_ascii=False, separators=(",", ":"),
                        )
                        accepted_samples.append((accepted_response, accepted_output))
                    # intentional_empty 仅在 silence 主题（expect_content=False）时可达，
                    # 已由 _recover_analyst_window 直接返回 success，此处不再二次裁决
                else:
                    # 任何非 success（含 unrecoverable）：LLM-A 失败时用规则层 summary 兜底。
                    # neutral_source="rule" 表示文本来自规则层确定性投影（main_topic.summary），
                    # 不填假内容、不掩盖失败（first_attempt_status 保留原始失败类供诊断/配额）。
                    fallback_text = fallback_neutral(plan)
                    if fallback_text and not (generation_status == "success" and neutral):
                        neutral = fallback_text
                        neutral_source = "rule"
                        generation_status = "success"
                        if analyst_result is not None:
                            analyst_result.first_attempt_status = analyst_result.first_attempt_status or analyst_result.generation_status
                            analyst_result.first_attempt_detail = analyst_result.first_attempt_detail or analyst_result.error_detail
                            analyst_result.retry_count = 0
                    else:
                        failed_windows += 1
                    if neutral_source == "unrecoverable":
                        unrecoverable_windows += 1

                if neutral_source == "fallback":
                    fallback_windows += 1

                # ── LLM-A 语料采集（data/llma/）：无论成败都收，供小模型拼接层微调 ──
                # input = LLM 看到的窗口投影（window_payload）；output = LLM 实际输出
                # （优先取 LLM 原始响应；成功时取最终 neutral）。status 保留兜底前的
                # 原始分类（analyst_result.generation_status，兜底只改局部变量不影响它），
                # 失败样本也保留，训练时可筛选。
                # v4 规则模式不调用 LLM-A，无 LLM 输出可收，跳过采集。
                if not dry_run and generator_mode != "rule_template":
                    llm_output = ""
                    if analyst_result is not None:
                        llm_output = str(analyst_result.raw_response or "") or (
                            str(analyst_result.api_result) if analyst_result.api_result is not None else ""
                        )
                    if not llm_output and generation_status == "success":
                        llm_output = neutral
                    original_status = (
                        analyst_result.generation_status
                        if analyst_result is not None and analyst_result.generation_status
                        else generation_status
                    )
                    _append_llma_corpus({
                        "run_id": run_id,
                        "round_no": rnd.round_no,
                        "window_id": window_id,
                        "t_start": t_start,
                        "t_end": t_end,
                        "input": window_payload,
                        "output": llm_output,
                        "status": original_status,
                        "source": neutral_source if neutral_source in ("llm", "llm_retry", "rule") else "unknown",
                        "char_budget": char_budget,
                    })

                fact_anchors = (
                    merge_required_fact_anchors(window_payload.get("required_facts"))
                    if generation_status == "success" and bool(neutral)
                    else merge_required_fact_anchors([])
                )
                if req.rule_v4 is not None:
                    fact_anchors = req.rule_v4["fact_anchors"]

                scene = {
                    "t_start": t_start, "t_end": t_end,
                    "context_start": req.context_start, "context_end": req.context_end,
                    "scene": scene_label,
                    "window_id": window_id,
                    "actions": [action.get("type") for action in plan.get("selected_actions", [])],
                    "commentary_plan": plan,
                    "neutral": neutral,
                    "neutral_source": neutral_source,
                    "generation_status": generation_status,
                    "fact_anchors": fact_anchors,
                    "hype": req.sc_hype,
                    "scream_eligible": req.scream_eligible,
                    "char_budget": req.char_budget,
                    "retry_count": analyst_result.retry_count if analyst_result is not None else 0,
                    "first_attempt_status": analyst_result.first_attempt_status if analyst_result is not None else None,
                    "first_attempt_detail": analyst_result.first_attempt_detail if analyst_result is not None else None,
                }
                if req.rule_v4 is not None:
                    start_tick = int(round(t_start * 30))
                    end_tick = int(round(t_end * 30))
                    scene["neutral_renderer"] = req.rule_v4["neutral_renderer"]
                    scene["rule_capsule"] = req.rule_v4["capsule"]
                    scene["fact_catalog"] = [
                        {
                            "fact_id": unit["fact_id"],
                            "kind": unit["kind"],
                            "origin": unit["origin"],
                            "anchor_tick": unit["anchor_tick"],
                            "source_tick_range": unit["source_tick_range"],
                            "canonical_clause": unit["canonical_clause"],
                            "required": unit["required"],
                            "priority": unit["priority"],
                        }
                        for unit in req.rule_v4["fact_catalog"]
                    ]
                    scene["required_fact_ids"] = req.rule_v4["required_fact_ids"]
                    scene["render_slot"] = {
                        "start_sec": t_start,
                        "end_sec": t_end,
                        "start_tick": start_tick,
                        "end_tick": end_tick,
                        "continuity_group_id": None,
                        "gap_policy": "independent_window",
                    }
                    scene["speech_budget"] = {
                        "target_units": req.rule_v4["target_units"],
                        "hard_units": req.rule_v4["hard_units"],
                        "profile_id": "speech-profile-v1",
                    }
                scenes_out.append(scene)

                # ── S2 诊断落盘 ──
                # 方案 R：real-call 窗口的诊断由 _recover_analyst_window 按每次尝试多记录
                # （§8 R4，不掩盖）；此处仅补写 silence / dry_run 路径的单条摘要。
                if topic_kind == "silence" or dry_run:
                    ar = analyst_result if not dry_run and topic_kind != "silence" else None
                    api_raw = ar.api_result if ar is not None else None
                    _write_window_diagnostic(
                        round_no=rnd.round_no,
                        window_idx=idx,
                        window_id=window_id,
                        t_start=t_start,
                        t_end=t_end,
                        generation_status=generation_status if not dry_run else "dry_run",
                        neutral_source=neutral_source,
                        error_stage=ar.error_stage if ar is not None else None,
                        error_type=ar.error_type if ar is not None else None,
                        error_detail=ar.error_detail if ar is not None else None,
                        http_status=ar.http_status if ar is not None else None,
                        finish_reason=ar.finish_reason if ar is not None else None,
                        usage=ar.usage if ar is not None else None,
                        content_present=bool(api_raw),
                        content_chars=len(str(api_raw)) if api_raw is not None else 0,
                        reasoning_present=bool(getattr(api_raw, "reasoning_content", None) if api_raw is not None else None),
                        reasoning_chars=len(getattr(api_raw, "reasoning_content", "") or "") if api_raw is not None else 0,
                        raw_response_saved=bool(getattr(api_raw, "raw_response", None) if api_raw is not None else None),
                    )

                # ── 调试数据断点：每窗完成后落盘全链路快照 ──
                if debug_enabled:
                    fb_text = fallback_neutral(plan)
                    raw_http_body = None
                    raw_vllm = None
                    msg_content = None
                    reasoning = None
                    think = None
                    cleaned = None

                    ar = analyst_result  # 可能是 None（silence/dry_run）
                    api_raw = ar.api_result if ar is not None and not dry_run else None
                    if api_raw is not None:
                        raw_http_body = getattr(api_raw, "request_payload", None)
                        raw_vllm = getattr(api_raw, "raw_response", None)
                        msg_content = str(api_raw)
                        reasoning = getattr(api_raw, "reasoning_content", None)
                        if raw_vllm:
                            orig_content = (
                                (raw_vllm.get("choices", [{}])[0].get("message", {}) or {})
                                .get("content") or ""
                            )
                            if "</think>" in orig_content:
                                think = orig_content
                                cleaned = orig_content.rsplit("</think>", 1)[-1].strip()

                    # 解析诊断：复用 AnalystResult 的诊断信息
                    if ar is not None:
                        json_parsed = _first_json_obj(ar.raw_response or "") if ar.raw_response else None
                        parse_err = ar.error_detail if ar.generation_status == "parse_error" else None
                        contract_ok = ar.generation_status == "success"
                        contract_err = ar.error_detail if ar.generation_status in ("contract_error", "truncated") else None
                        parsed_neu = ar.content if contract_ok else None
                        error_stage = ar.error_stage
                        http_status = ar.http_status
                        finish_reason = ar.finish_reason
                        usage = ar.usage
                    else:
                        json_parsed = None
                        parse_err = None
                        contract_ok = (generation_status == "success")
                        contract_err = None
                        parsed_neu = None
                        error_stage = None
                        http_status = None
                        finish_reason = None
                        usage = None

                    record = DebugWindowRecord(
                        round_no=rnd.round_no,
                        window_idx=idx,
                        t_start=t_start,
                        t_end=t_end,
                        scene=scene_label,
                        run_id=run_id,
                        raw_plan=plan,
                        llm_projection=window_payload,
                        system_prompt=analyst_system if topic_kind != "silence" else "",
                        user_prompt=prompt if topic_kind != "silence" else "[silence — no LLM call]",
                        http_request_body=raw_http_body,
                        llm_config=dict(llm_cfg),
                        vllm_raw_response=raw_vllm,
                        message_content=msg_content,
                        reasoning_content=reasoning,
                        think_text=think,
                        cleaned_content=cleaned,
                        json_parse_result=json_parsed,
                        parse_error=parse_err,
                        contract_valid=contract_ok,
                        contract_error=contract_err,
                        parsed_neutral=parsed_neu,
                        final_neutral=neutral,
                        neutral_source=neutral_source,
                        generation_status=generation_status,
                        error_stage=error_stage,
                        http_status=http_status,
                        finish_reason=finish_reason,
                        usage=usage,
                        content_present=bool(msg_content),
                        content_chars=len(msg_content or ""),
                        reasoning_present=bool(reasoning),
                        reasoning_chars=len(reasoning or ""),
                        raw_response_saved=raw_vllm is not None,
                        equals_fallback=(neutral == fb_text),
                        fallback_text=fb_text,
                    )
                    debug_recorder.record_window(record)
        except Phase3aCircuitBreak:
            # 熔断中止整阶段：取消本回合尚未开始的窗口请求。
            for req in phase1:
                if req.future is not None:
                    req.future.cancel()
            raise

        # 方案 R：全失败的回合若窗口均已标 unrecoverable（恢复耗尽），不算
        # "analyst_failed"（已诚实留白，交 K 配额裁决）；仅未恢复的裸失败才判失败。
        analyst_failed = bool(scenes_out) and failed_windows == len(scenes_out) and not any(
            str(scene.get("neutral", "")).strip() for scene in scenes_out
        ) and unrecoverable_windows == 0
        # 窗口级状态分布
        status_counts: dict[str, int] = {}
        for scene in scenes_out:
            gs = str(scene.get("generation_status", "unknown"))
            status_counts[gs] = status_counts.get(gs, 0) + 1
        result = {
            "round_no": rnd.round_no,
            "start_sec": rnd.start_sec,
            "end_sec": rnd.end_sec,
            "demo_round_hint": rnd.demo_round_hint,
            "round_emotion": round_emotion,
            "peak_hype": peak_hype,
            "avg_hype": avg_hype,
            "analyst_failed": analyst_failed,
            "fallback_windows": fallback_windows,
            "failed_windows": failed_windows,
            "unrecoverable_windows": unrecoverable_windows,
            "generation_status_counts": status_counts,
            "scenes": scenes_out,
        }
        llma_input_round = {"round_no": rnd.round_no, "windows": llma_inputs}
        return result, (accepted_samples if not analyst_failed else []), llma_input_round

    # 窗口并发池：全局同一 pool、跨回合复用（阶段 2 派发）。
    # 缺省并发度 1 时不创建线程池，_process_round 在阶段 3 按序同步调用。
    window_pool = ThreadPoolExecutor(max_workers=window_concurrency) if window_concurrency > 1 else None
    result_rounds = []
    llma_input_rounds = []
    accepted_run_samples: list[tuple[object, str]] = []
    completed_rounds = 0
    try:
        with ThreadPoolExecutor(max_workers=concurrent_rounds) as pool:
            futures = {pool.submit(_process_round, rnd, window_pool): rnd.round_no for rnd in match.rounds}
            try:
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Phase3a llma_analyze", unit="round"):
                    result, accepted_samples_part, llma_input_round = fut.result()
                    result_rounds.append(result)
                    completed_rounds += 1
                    if progress_sink is not None:
                        try:
                            progress_sink(completed_rounds, len(futures), "round", None)
                        except Exception:
                            pass
                    accepted_run_samples.extend(accepted_samples_part)
                    llma_input_rounds.append(llma_input_round)
            except Phase3aCircuitBreak as exc:
                for fut in futures:
                    fut.cancel()
                raise
    finally:
        if window_pool is not None:
            window_pool.shutdown(wait=True)
    result_rounds.sort(key=lambda r: r["round_no"])
    llma_input_rounds.sort(key=lambda r: r["round_no"])

    # S5: llma_input.json 升级为审计产物（artifact_kind/contract_version/run_id/
    # source_hashes/每窗投影，§10.2）。落盘逻辑由 phase3a_audit 模块纯计算，此处只写盘。
    llma_input_path = resolve_path(
        config.get("paths", {}).get("llma_input_json")
    ) or output_path.with_name("llma_input.json")
    if not dry_run:
        prompt_dir = _PROJECT_ROOT / "Prompt"
        source_hashes = compute_source_hashes(
            rounds_path, semantic_path,
            prompt_dir / "analyst_system.txt",
            prompt_dir / "analyst_round.txt",
        )
        audit_artifact = build_audit_artifact(
            llma_input_rounds, result_rounds, run_id, source_hashes,
        )
        write_json(llma_input_path, audit_artifact)

    if result_rounds and not dry_run:
        window_stats = build_window_statistics(result_rounds)
        # S5: 机器可读窗口级统计写入 diagnostics（§10.2 控制台和机器可读汇总统一按窗口）
        if _DIAGNOSTICS_DIR is not None:
            stats_path = _DIAGNOSTICS_DIR / f"{_DIAGNOSTICS_RUN_ID}_window_stats.json"
            stats_path.write_text(
                json.dumps(window_stats, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        # S5: 控制台汇总统一按窗口（不再以 round 为分母，§10.1）
        print(_format_window_statistics(window_stats), file=sys.stderr)
        total = len(result_rounds)
        failed = sum(1 for r in result_rounds if r.get("analyst_failed"))
        if failed == total or failed / total > 0.5:
            print(
                f"[phase3a] FATAL: {failed}/{total} rounds failed. Check LLM endpoint / API key.",
                file=sys.stderr,
            )
            sys.exit(1)

    manifest = {
        **run_metadata,
        "video_path": match.video_path,
        "map_name": match.map_name,
        "model": llm_cfg.get("model", ""),
        "recovery": {
            "enabled": recovery_cfg["enabled"],
            "max_retries": recovery_cfg["max_retries"],
        },
        "rounds": result_rounds,
    }
    if generator_mode == "rule_template":
        # v4 契约（§7.4）：规则渲染器产出；schema 版本与 mode 随产物冻结。
        manifest["schema_version"] = 4
        manifest["phase3a_mode"] = "rule_neutral_renderer"
        manifest["speech_metric_version"] = "speech_units_v1"
        manifest["model"] = ""
    if dry_run:
        return {
            "mode": "phase3a_dry_run",
            "writes_performed": False,
            "publish_path": None,
            "rounds": len(result_rounds),
            "windows": sum(len(item.get("scenes", [])) for item in result_rounds),
            "fallback_windows": 0,
        }
    write_json(output_path, manifest)
    # S5: 训练样本延迟到整体验收后才提交（§10.2/§10.3）。
    # 单窗口解析成功但整体正式产物未通过 publishable 门禁时，不提交样本。
    from sbmachine.preflight import PublishContractError, validate_neutral_publishable
    try:
        validate_neutral_publishable(output_path)
    except PublishContractError:
        pass  # 产物未通过门禁，不提交训练样本；下游 checkpoint 会 fail closed
    else:
        for response, output in accepted_run_samples:
            accept_api_response(response, output=output)
    return manifest
