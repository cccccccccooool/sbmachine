"""Phase3a 审计产物与窗口级统计（S5）。

llma_input.json 升级为独立审计产物：包含 artifact_kind、contract_version、run_id、
source_hashes 和每窗投影（§10.2）。窗口级统计替代旧的 round 级成功日志。训练样本
延迟到整体验收后才提交（§10.2/§10.3）。

本模块不产生副作用：所有函数纯计算，落盘由调用方负责。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

AUDIT_ARTIFACT_KIND = "phase3a_llm_input"
AUDIT_CONTRACT_VERSION = 3

_KNOWN_GENERATION_STATUSES = (
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


def sha256_file(path: Path) -> str:
    """返回 'sha256:<hex>' 格式的文件哈希。"""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def compute_source_hashes(
    rounds_path: Path,
    semantic_path: Path,
    analyst_system_path: Path,
    analyst_round_path: Path,
) -> dict[str, str]:
    """计算审计产物所需的源文件哈希（§10.2 source_hashes）。

    缺失的文件跳过（不写入对应 key），调用方据此判断基线是否完整。
    """
    hashes: dict[str, str] = {}
    for key, path in (
        ("rounds_with_yolo.json", rounds_path),
        ("semantic_frames.json", semantic_path),
        ("analyst_system.txt", analyst_system_path),
        ("analyst_round.txt", analyst_round_path),
    ):
        if path.is_file():
            hashes[key] = sha256_file(path)
    return hashes


def build_audit_artifact(
    llma_input_rounds: list[dict],
    result_rounds: list[dict],
    run_id: str,
    source_hashes: dict[str, str],
) -> dict:
    """构建 llma_input.json 审计产物（contract version 3）。

    将每窗投影（llma_input_rounds）与窗口身份（result_rounds.scenes）按
    round_no + 窗口顺序对齐，输出 §10.2 规定的结构：

      {window_index, window_id, t_start, t_end, scene, character_limit, projection}

    projection 是 build_llm_window_projection 的白名单输出（main_topic、
    selected_actions、可选 rule_state / tactic_hint），character_limit 从
    window_payload 顶层提取到窗口层级。
    """
    result_by_round: dict[int, dict] = {
        int(r["round_no"]): r
        for r in result_rounds
        if isinstance(r, dict) and isinstance(r.get("round_no"), int)
    }
    audit_rounds: list[dict] = []
    for llma_round in llma_input_rounds:
        round_no = llma_round.get("round_no")
        result_round = (
            result_by_round.get(int(round_no))
            if isinstance(round_no, int)
            else None
        )
        scenes = (result_round or {}).get("scenes") or []
        llma_windows = llma_round.get("windows") or []
        audit_windows: list[dict] = []
        for idx, window_payload in enumerate(llma_windows):
            scene = scenes[idx] if idx < len(scenes) else {}
            projection = {
                key: value
                for key, value in dict(window_payload).items()
                if key != "character_limit"
            }
            fallback_id = (
                f"r{int(round_no):03d}_w{idx + 1:02d}"
                if isinstance(round_no, int)
                else f"window[{idx}]"
            )
            audit_window: dict[str, Any] = {
                "window_index": idx + 1,
                "window_id": scene.get("window_id") or fallback_id,
                "t_start": scene.get("t_start"),
                "t_end": scene.get("t_end"),
                "scene": scene.get("scene", ""),
                "character_limit": window_payload.get("character_limit"),
                "projection": projection,
            }
            audit_windows.append(audit_window)
        audit_rounds.append({"round_no": round_no, "windows": audit_windows})
    return {
        "artifact_kind": AUDIT_ARTIFACT_KIND,
        "contract_version": AUDIT_CONTRACT_VERSION,
        "run_id": run_id,
        "source_hashes": source_hashes,
        "rounds": audit_rounds,
    }


def build_window_statistics(result_rounds: list[dict]) -> dict:
    """构建窗口级统计（§10.2 控制台与机器可读汇总）。

    不变量（§10.5 Gate 5）：
    - windows_total = sum(generation_status_counts)
    - model_calls + intentional_silence = windows_total
    - fallback_windows = 0

    在非 dry-run 模式下，每个窗口要么是规则层静默（intentional_silence），
    要么调用了模型（model_calls，无论成功或失败），故两者之和等于总窗口数。
    intentional_silence = generation_status=="success" 且 neutral_source=="intentional_empty"
    的窗口（规则层 silence 跳过 LLM-A 调用）。
    """
    rounds_total = len(result_rounds)
    status_counts: dict[str, int] = {status: 0 for status in _KNOWN_GENERATION_STATUSES}
    windows_total = 0
    fallback_windows = 0
    intentional_silence = 0
    unrecoverable_count = 0
    infra_error_count = 0
    retried_windows = 0
    required_fact_count = 0
    required_fact_covered = 0

    _infra_statuses = ("transport_error", "http_error", "response_error")

    for rnd in result_rounds:
        scenes = rnd.get("scenes") or []
        windows_total += len(scenes)
        rnd_status = rnd.get("generation_status_counts") or {}
        for status, count in rnd_status.items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
        fallback_windows += int(rnd.get("fallback_windows", 0) or 0)
        for scene in scenes:
            source = scene.get("neutral_source")
            gstatus = scene.get("generation_status")
            if gstatus == "success" and source == "intentional_empty":
                intentional_silence += 1
            if source == "unrecoverable":
                unrecoverable_count += 1
            first_attempt = scene.get("first_attempt_status")
            if first_attempt in _infra_statuses:
                infra_error_count += 1
            retry_count = scene.get("retry_count")
            if isinstance(retry_count, int) and retry_count > 0:
                retried_windows += 1
            plan = scene.get("commentary_plan")
            facts = plan.get("required_facts") if isinstance(plan, dict) else None
            if isinstance(facts, list):
                required_fact_count += len(facts)
                neutral = scene.get("neutral")
                if isinstance(neutral, str):
                    required_fact_covered += sum(
                        1 for fact in facts
                        if isinstance(fact, dict)
                        and isinstance(fact.get("canonical_text"), str)
                        and fact["canonical_text"] in neutral
                    )

    model_calls = windows_total - intentional_silence
    non_success = sum(
        count for status, count in status_counts.items() if status != "success"
    )
    retry_rate = (retried_windows / model_calls) if model_calls > 0 else 0.0
    infra_ratio = (infra_error_count / windows_total) if windows_total > 0 else 0.0
    publishable = fallback_windows == 0 and non_success == 0

    return {
        "rounds_total": rounds_total,
        "windows_total": windows_total,
        "model_calls": model_calls,
        "intentional_silence": intentional_silence,
        "generation_status_counts": status_counts,
        "fallback_windows": fallback_windows,
        "publishable": publishable,
        "unrecoverable_count": unrecoverable_count,
        "infra_error_count": infra_error_count,
        "infra_ratio": round(infra_ratio, 4),
        "retry_rate": round(retry_rate, 4),
        "retried_windows": retried_windows,
        "required_fact_count": required_fact_count,
        "required_fact_coverage": (
            round(required_fact_covered / required_fact_count, 4)
            if required_fact_count else 1.0
        ),
        "semantic_contract_error_count": status_counts.get("semantic_contract_error", 0),
        "side_mismatch_count": status_counts.get("side_mismatch", 0),
        "unexpected_fact_count": status_counts.get("unexpected_fact", 0),
    }


def read_audit_artifact(path: Path) -> dict:
    """按 contract_version 读取审计产物（§10.4 向后兼容）。

    无 artifact_kind / contract_version 的旧 llma_input.json 按 version 1 解释，
    仅用于 replay，不作为新运行的发布证明或 A/B 样本。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("llma_input artifact must be a JSON object")
    if "artifact_kind" not in payload or "contract_version" not in payload:
        return {"contract_version": 1, "rounds": payload.get("rounds") or []}
    return payload
