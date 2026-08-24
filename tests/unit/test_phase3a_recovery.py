"""方案 R（可恢复错误自纠重试）的契约测试。

覆盖 phase3_plan_r_recovery_spec.md §9 的可测子集：
- R1 错误反馈重试（超长 1 次恢复）、重试上限（3 次后 unrecoverable）
- R2 基建退避重调用（response_error 同形 2 次即止）、10% 中止、K 配额、K=0 零容忍
- R3 think-strip 恢复
- R4 多记录（_make_unrecoverable 终态字段）
- S5 桥接：build_window_statistics 的恢复指标
- preflight：llm_retry 放行、unrecoverable 配额裁决、基建占比中止
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sbmachine.phase3a_analyst as analyst_mod
from sbmachine.phase3a_analyst import AnalystResult
from sbmachine.phase3a_audit import build_window_statistics
from sbmachine.preflight import PublishContractError, validate_neutral_publishable


# ── 工具 ──────────────────────────────────────────────────────────────


def _write_neutral_manifest(
    tmp_path: Path, rounds: list[dict], recovery: dict | None = None
) -> Path:
    manifest = {
        "schema_version": 3,
        "phase3a_mode": "llma_slicer_then_llma_analyze",
        "run_id": "r-test",
        "source_rounds_sha256": "0" * 64,
        "video_path": "/dev/null",
        "map_name": "de_test",
        "model": "test-model",
        "rounds": rounds,
    }
    if recovery is not None:
        manifest["recovery"] = recovery
    path = tmp_path / "rounds_with_neutral.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def _scene(
    status: str = "success",
    source: str = "llm",
    neutral: str = "",
    **kw,
) -> dict:
    return {
        "t_start": 0.0,
        "t_end": 3.0,
        "scene": "s",
        "neutral": neutral,
        "neutral_source": source,
        "generation_status": status,
        **kw,
    }


def _scripted_call(script: list[AnalystResult]):
    """返回 (mock, calls)：mock 复现 script 序列，calls 记录每次 prompt。"""
    it = iter(script)
    calls: list[str] = []

    def _mock(prompt, llm_cfg, gen_fn, *, system_prompt=None, round_no=0,
              run_id=None, debug=False, seg=0, max_tokens=None,
              char_limit=100, projection=None):
        calls.append(prompt)
        return next(it)

    return _mock, calls


def _recover(script, monkeypatch, *, char_limit=14, max_retries=3):
    """跑一次 _recover_analyst_window，屏蔽 sleep 与诊断落盘。"""
    mock, calls = _scripted_call(script)
    monkeypatch.setattr(analyst_mod, "_call_analyst", mock)
    monkeypatch.setattr(analyst_mod, "_write_window_diagnostic", lambda *a, **k: None)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    result = analyst_mod._recover_analyst_window(
        "prompt", {}, gen_fn=None, system_prompt="sys", round_no=1, run_id="r",
        debug=False, seg=1, max_tokens=256, char_limit=char_limit,
        window_id="r001_w01", t_start=0.0, t_end=3.0,
        recovery_cfg={"enabled": True, "max_retries": max_retries},
        expect_content=True,
    )
    return result, calls


# ── R3：think-strip 纯函数 ────────────────────────────────────────────


def test_try_strip_reparse_recovers_when_think_ate_budget():
    result = AnalystResult(
        generation_status="truncated",
        raw_response='{"think":"分析过程","neutral":"双杀"}',
        finish_reason="length",
    )
    recovered = analyst_mod._try_strip_reparse(result, char_limit=20)
    assert recovered is not None
    assert recovered.generation_status == "success"
    assert recovered.content == "双杀"
    assert recovered.first_attempt_status == "truncated"


def test_try_strip_reparse_returns_none_without_reasoning():
    result = AnalystResult(
        generation_status="truncated",
        raw_response='{"neutral":"不完整',
        finish_reason="length",
    )
    assert analyst_mod._try_strip_reparse(result, char_limit=20) is None


def test_try_strip_reparse_returns_none_when_too_long_after_strip():
    result = AnalystResult(
        generation_status="truncated",
        raw_response='{"think":"x","neutral":"这是一句非常长的中文中性稿内容超预算"}',
        finish_reason="length",
    )
    assert analyst_mod._try_strip_reparse(result, char_limit=5) is None


# ── R1：修正 prompt 纯函数 ────────────────────────────────────────────


def test_build_correction_prompt_length():
    p = analyst_mod._build_correction_prompt(
        "contract_error", '{"neutral":"x"}',
        "neutral too long: 21 chars (limit 14)", 14, "BASE",
    )
    assert "14" in p and "BASE" in p and "压缩" in p


def test_build_correction_prompt_parse_error():
    p = analyst_mod._build_correction_prompt("parse_error", "```json", None, 20, "BASE")
    assert "BASE" in p and ("markdown" in p or "围栏" in p)


def test_build_correction_prompt_truncated():
    p = analyst_mod._build_correction_prompt("truncated", "", None, 20, "BASE")
    assert "截断" in p


def test_build_correction_prompt_field_mismatch():
    p = analyst_mod._build_correction_prompt(
        "contract_error", '{"neutral":"x","extra":1}', "unexpected fields", 20, "BASE",
    )
    assert "字段" in p


# ── R4：_make_unrecoverable 终态 ───────────────────────────────────────


def test_make_unrecoverable_preserves_real_failure_and_blanks_content():
    src = AnalystResult(
        generation_status="http_error", error_type="HTTPError",
        error_detail="503", http_status=503, finish_reason=None,
    )
    u = analyst_mod._make_unrecoverable(src, "http_error", "503", 3)
    assert u.neutral_source == "unrecoverable"
    assert u.generation_status == "http_error"
    assert u.retry_count == 3
    assert u.first_attempt_status == "http_error"
    assert u.content == ""


# ── R1 重试循环 ───────────────────────────────────────────────────────


def test_length_overrun_recovers_on_first_retry(monkeypatch):
    script = [
        AnalystResult(
            content="", generation_status="contract_error",
            error_type="NeutralLengthExceeded",
            error_detail="neutral too long: 21 chars (limit 14)",
            raw_response='{"neutral":"Tauson连续击杀tabseN"}', finish_reason="stop",
        ),
        AnalystResult(
            content="Tauson双杀", neutral_source="llm",
            generation_status="success", finish_reason="stop",
        ),
    ]
    result, calls = _recover(script, monkeypatch, char_limit=14)
    assert result.generation_status == "success"
    assert result.neutral_source == "llm_retry"
    assert result.retry_count == 1
    assert result.content == "Tauson双杀"
    assert len(calls) == 2


def test_retry_cap_three_then_unrecoverable(monkeypatch):
    err = AnalystResult(
        content="", generation_status="contract_error",
        error_type="NeutralLengthExceeded",
        error_detail="neutral too long: 30 chars (limit 14)",
        raw_response='{"neutral":"..."}', finish_reason="stop",
    )
    result, calls = _recover([err] * 5, monkeypatch, char_limit=14)
    assert result.neutral_source == "unrecoverable"
    assert result.retry_count == 3
    assert len(calls) == 4


# ── R2：基建退避重调用 ───────────────────────────────────────────────


def test_response_error_two_strikes_unrecoverable(monkeypatch):
    err = AnalystResult(
        generation_status="response_error", error_type="EnvelopeMalformed",
        error_detail="missing choices", finish_reason=None,
    )
    result, calls = _recover([err] * 4, monkeypatch, char_limit=20)
    assert result.neutral_source == "unrecoverable"
    assert result.retry_count == 2
    assert len(calls) == 3


# ── S5 桥接：build_window_statistics 恢复指标 ─────────────────────────


def test_window_statistics_tracks_recovery_metrics():
    rounds = [{
        "round_no": 1,
        "scenes": [
            {"generation_status": "success", "neutral_source": "llm"},
            {"generation_status": "success", "neutral_source": "intentional_empty"},
            {"generation_status": "contract_error", "neutral_source": "unrecoverable",
             "first_attempt_status": "transport_error", "retry_count": 3},
            {"generation_status": "success", "neutral_source": "llm_retry",
             "retry_count": 1},
        ],
        "generation_status_counts": {"success": 3, "contract_error": 1},
        "fallback_windows": 0,
    }]
    stats = build_window_statistics(rounds)
    assert stats["windows_total"] == 4
    assert stats["intentional_silence"] == 1
    assert stats["model_calls"] == 3
    assert stats["unrecoverable_count"] == 1
    assert stats["infra_error_count"] == 1
    assert stats["retried_windows"] == 2
    assert stats["retry_rate"] == round(2 / 3, 4)
    assert stats["infra_ratio"] == round(1 / 4, 4)
    assert stats["publishable"] is False


# ── preflight：K 配额、10% 中止、llm_retry 放行、K=0 零容忍 ─────────


def test_llm_retry_source_is_publishable(tmp_path):
    path = _write_neutral_manifest(tmp_path, [{
        "round_no": 1, "analyst_failed": False,
        "scenes": [_scene("success", "llm_retry", "Tauson双杀", retry_count=1)],
    }], recovery={"enabled": True})
    validate_neutral_publishable(path)


def test_unrecoverable_within_K_passes(tmp_path):
    scenes = [_scene("success", "llm")] * 92 + [
        _scene("contract_error", "unrecoverable", first_attempt_status="contract_error"),
        _scene("contract_error", "unrecoverable", first_attempt_status="contract_error"),
    ]
    path = _write_neutral_manifest(tmp_path, [{
        "round_no": 1, "analyst_failed": False,
        "scenes": scenes,
        "generation_status_counts": {"success": 92, "contract_error": 2},
    }], recovery={"enabled": True})
    validate_neutral_publishable(path)


def test_unrecoverable_exceeding_K_rejected(tmp_path):
    scenes = [_scene("success", "llm")] * 91 + [
        _scene("contract_error", "unrecoverable") for _ in range(3)
    ]
    path = _write_neutral_manifest(tmp_path, [{
        "round_no": 1, "analyst_failed": False,
        "scenes": scenes,
        "generation_status_counts": {"success": 91, "contract_error": 3},
    }], recovery={"enabled": True})
    with pytest.raises(PublishContractError, match="exceed K quota"):
        validate_neutral_publishable(path)


def test_K_zero_default_when_recovery_disabled(tmp_path):
    path = _write_neutral_manifest(tmp_path, [{
        "round_no": 1, "analyst_failed": False,
        "scenes": [_scene("contract_error", "unrecoverable")],
    }])
    with pytest.raises(PublishContractError, match="exceed K quota"):
        validate_neutral_publishable(path)


def test_infra_ratio_over_10pct_no_longer_aborts(tmp_path):
    # 基建错误占比 >10% 的中止裁决已于 2026-08-16 移除：失败窗口交由
    # K 配额与下游空回合语义处理，不再因占比直接判整场失败。
    scenes = [_scene("success", "llm")] * 17 + [
        _scene("success", "llm_retry",
               first_attempt_status="transport_error", retry_count=1)
        for _ in range(3)
    ]
    path = _write_neutral_manifest(tmp_path, [{
        "round_no": 1, "analyst_failed": False,
        "scenes": scenes,
        "generation_status_counts": {"success": 20},
    }], recovery={"enabled": True})
    # 覆盖率不足 K 配额（unrecoverable=0 时），校验不再因基建占比拒绝。
    validate_neutral_publishable(path)
