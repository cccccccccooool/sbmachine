"""validate_neutral_publishable 的契约测试：neutral_source 显式字段区分 llm / fallback。"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from sbmachine.preflight import PublishContractError, validate_neutral_publishable


def _write_neutral_manifest(tmp_path: Path, rounds: list[dict]) -> Path:
    manifest = {
        "schema_version": 2,
        "phase3a_mode": "llma_slicer_then_llma_analyze",
        "run_id": uuid.uuid4().hex,
        "source_rounds_sha256": "0" * 64,
        "video_path": "/dev/null",
        "map_name": "de_test",
        "model": "test-model",
        "rounds": rounds,
    }
    path = tmp_path / "rounds_with_neutral.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def _scene(neutral: str, neutral_source: str = "llm", **kwargs) -> dict:
    return {
        "t_start": 10.0,
        "t_end": 15.0,
        "scene": "test_scene",
        "commentary_plan": {
            "main_topic": {"kind": "retake", "summary": "C4已安装"},
            "selected_actions": [],
        },
        "neutral": neutral,
        "neutral_source": neutral_source,
        "generation_status": "success",
        **kwargs,
    }


class TestValidateNeutralPublishable:
    def test_llm_source_never_rejected_even_when_text_matches_rule_summary(self, tmp_path):
        """模型输出恰好等于规则摘要但仍属于合法 LLM 输出，不应被拒绝。"""
        path = _write_neutral_manifest(tmp_path, [
            {
                "round_no": 1,
                "start_sec": 0.0,
                "end_sec": 30.0,
                "analyst_failed": False,
                "scenes": [
                    _scene("C4已安装", neutral_source="llm"),
                ],
            },
        ])
        # Must not raise — same text as fallback, but explicitly llm source.
        validate_neutral_publishable(path)

    def test_fallback_source_with_content_is_rejected(self, tmp_path):
        """neutral_source=fallback 且 neutral 非空 → 拒绝发布。"""
        path = _write_neutral_manifest(tmp_path, [
            {
                "round_no": 1,
                "start_sec": 0.0,
                "end_sec": 30.0,
                "analyst_failed": False,
                "scenes": [
                    _scene("兜底文本", neutral_source="fallback"),
                ],
            },
        ])
        with pytest.raises(PublishContractError, match="rule fallback neutral is not publishable"):
            validate_neutral_publishable(path)

    def test_fallback_source_with_empty_neutral_is_rejected(self, tmp_path):
        """空 fallback 也没有可验证来源，不能作为可发布 neutral。"""
        path = _write_neutral_manifest(tmp_path, [
            {
                "round_no": 1,
                "start_sec": 0.0,
                "end_sec": 30.0,
                "analyst_failed": False,
                "scenes": [
                    _scene("", neutral_source="fallback"),
                ],
            },
        ])
        with pytest.raises(PublishContractError, match="rule fallback neutral is not publishable"):
            validate_neutral_publishable(path)

    def test_mixed_llm_and_fallback_windows(self, tmp_path):
        """混合场景：llm 窗口通过，fallback 窗口拒绝。"""
        path = _write_neutral_manifest(tmp_path, [
            {
                "round_no": 1,
                "start_sec": 0.0,
                "end_sec": 30.0,
                "analyst_failed": False,
                "scenes": [
                    _scene("合法LLM输出", neutral_source="llm"),
                    _scene("兜底文本", neutral_source="fallback"),
                ],
            },
        ])
        with pytest.raises(PublishContractError, match="rule fallback neutral is not publishable"):
            validate_neutral_publishable(path)

    def test_tactic_kind_llm_matches_rule_summary(self, tmp_path):
        """tactic 话题下 LLM 复述了 tactic_hint 文本（投影层直接写入 summary）→ 不拒绝。"""
        path = _write_neutral_manifest(tmp_path, [
            {
                "round_no": 1,
                "start_sec": 0.0,
                "end_sec": 30.0,
                "analyst_failed": False,
                "scenes": [
                    {
                        "t_start": 10.0, "t_end": 15.0,
                        "scene": "tactic_scene",
                        "commentary_plan": {
                            "main_topic": {"kind": "tactic", "summary": "默认A区爆弹"},
                            "selected_actions": [{"type": "tactic_event"}],
                        },
                        "neutral": "默认A区爆弹",
                        "neutral_source": "llm",
                        "generation_status": "success",
                    },
                ],
            },
        ])
        # Must not raise — same text as plan.main_topic.summary, but llm source.
        validate_neutral_publishable(path)

    def test_missing_generation_status_or_source_is_rejected_as_legacy(self, tmp_path):
        """旧格式可供人工读取，但不能直接进入 Phase3b 或发布。"""
        path = _write_neutral_manifest(tmp_path, [
            {
                "round_no": 1,
                "start_sec": 0.0,
                "end_sec": 30.0,
                "analyst_failed": False,
                "scenes": [
                    {
                        "t_start": 10.0, "t_end": 15.0,
                        "scene": "legacy",
                        "commentary_plan": {"main_topic": {"kind": "retake", "summary": "C4已安装"}},
                        "neutral": "C4已安装",
                        # 缺少 generation_status 和 neutral_source。
                    },
                ],
            },
        ])
        with pytest.raises(PublishContractError, match="legacy neutral is not publishable"):
            validate_neutral_publishable(path)
    def test_unknown_neutral_source_is_not_publishable(self, tmp_path):
        """success 不能让未定义的来源绕过发布门禁。"""
        path = _write_neutral_manifest(tmp_path, [
            {
                "round_no": 1,
                "start_sec": 0.0,
                "end_sec": 30.0,
                "analyst_failed": False,
                "scenes": [_scene("未经证明的文本", neutral_source="unknown")],
            },
        ])

        with pytest.raises(PublishContractError, match="neutral source .* is not publishable"):
            validate_neutral_publishable(path)

    def test_contract_error_with_window_location(self, tmp_path):
        """contract_error 窗口应被拒绝，错误消息包含 window_id 定位。"""
        path = _write_neutral_manifest(tmp_path, [
            {
                "round_no": 2,
                "start_sec": 0.0,
                "end_sec": 60.0,
                "analyst_failed": False,
                "scenes": [
                    _scene("", neutral_source="", generation_status="contract_error"),
                ],
            },
        ])
        with pytest.raises(PublishContractError, match="r002_w01"):
            validate_neutral_publishable(path)
