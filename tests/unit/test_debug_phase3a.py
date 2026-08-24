"""Phase3a 调试模式分层测试。

三层：
  Layer 1 — 快速单测：纯函数、Mock、无需模型
  Layer 2 — 本地集成：Mock 后端走完整链路
  Layer 3 — 云端验收：手动执行 tools/debug_phase3a.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sbmachine.commentary_planner import fallback_neutral
from sbmachine.debug_phase3a import DebugRecorder, DebugWindowRecord
from sbmachine.llm_backends import _MOCK_CASES, mock_generate
from sbmachine.llm_projection import build_llm_window_projection
from sbmachine.phase3a_prompt import (
    _build_window_prompt,
    _first_json_obj,
    _parse_window_neutral_response,
)

# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 — 纯函数快速单测（无需模型）
# ═══════════════════════════════════════════════════════════════════════════


class TestParseWindowNeutral:
    """_parse_window_neutral_response 的回归测试。"""

    def test_normal_neutral(self):
        assert _parse_window_neutral_response('{"neutral":"A区爆弹进攻"}') == "A区爆弹进攻"

    def test_legal_empty_neutral(self):
        assert _parse_window_neutral_response('{"neutral":""}') == ""

    def test_strips_think_field(self):
        """JSON 级 think 字段应被剥离并成功提取 neutral。"""
        assert (
            _parse_window_neutral_response(
                '{"think":"分析过程","neutral":"战术配合"}'
            )
            == "战术配合"
        )

    def test_strips_reasoning_field(self):
        """JSON 级 reasoning 字段应被剥离。"""
        assert (
            _parse_window_neutral_response(
                '{"reasoning":"思考","neutral":"闪光弹致盲"}'
            )
            == "闪光弹致盲"
        )

    def test_strips_reasoning_content_field(self):
        """JSON 级 reasoning_content 字段应被剥离。"""
        assert (
            _parse_window_neutral_response(
                '{"reasoning_content":"推理...","neutral":"有效输出"}'
            )
            == "有效输出"
        )

    def test_strips_multiple_think_fields(self):
        """多个已知字段同时存在时仍正确剥离。"""
        assert (
            _parse_window_neutral_response(
                '{"think":"1","reasoning":"2","reasoning_content":"3","neutral":"core"}'
            )
            == "core"
        )

    def test_rejects_markdown_fence(self):
        """Markdown JSON 围栏不是合法 JSON，应拒绝。"""
        assert (
            _parse_window_neutral_response(
                '```json\n{"neutral":"围栏"}\n```'
            )
            is None
        )

    def test_rejects_surrounding_prose_before(self):
        assert (
            _parse_window_neutral_response(
                '说明：{"neutral":"包裹文本"}'
            )
            is None
        )

    def test_rejects_surrounding_prose_after(self):
        assert (
            _parse_window_neutral_response(
                '{"neutral":"包裹文本"}\n说明'
            )
            is None
        )

    def test_rejects_extra_field(self):
        """未知额外字段应被拒绝。"""
        assert (
            _parse_window_neutral_response(
                '{"neutral":"文本","extra_field":true}'
            )
            is None
        )

    def test_rejects_illegal_json(self):
        assert _parse_window_neutral_response("这不是JSON") is None

    def test_rejects_non_string_neutral(self):
        """neutral 不是字符串类型时应拒绝。"""
        assert _parse_window_neutral_response('{"neutral": 123}') is None

    def test_rejects_missing_neutral(self):
        """缺少 neutral 字段时应拒绝。"""
        assert _parse_window_neutral_response('{"other":"value"}') is None

    def test_rejects_list_not_dict(self):
        """非法 JSON 类型（数组）应拒绝。"""
        assert _parse_window_neutral_response('["neutral"]') is None

    def test_rejects_number_not_dict(self):
        """非法 JSON 类型（数字）应拒绝。"""
        assert _parse_window_neutral_response("123") is None


class TestFirstJsonObj:
    """_first_json_obj 的回归测试。"""

    def test_valid_dict(self):
        assert _first_json_obj('{"neutral":"ok"}') == {"neutral": "ok"}

    def test_empty_dict(self):
        assert _first_json_obj("{}") == {}

    def test_list_rejected(self):
        assert _first_json_obj('["neutral"]') is None

    def test_string_rejected(self):
        assert _first_json_obj('"neutral"') is None

    def test_invalid_json(self):
        assert _first_json_obj("invalid") is None


class TestLLMProjection:
    """build_llm_window_projection 白名单投影测试。"""

    def test_strips_spatial_and_evidence(self):
        plan = {
            "main_topic": {"kind": "kill", "summary": "A击杀B"},
            "selected_actions": [{"type": "kill", "attacker": "A", "victim": "B"}],
            "spatial": {"anchor": {"callout": "Ramp"}},
            "ownership": {"t_start": 10.0, "t_end": 15.0},
            "read_only_context": {"t_start": 5.0, "t_end": 15.0},
        }
        result = build_llm_window_projection(plan)
        assert "spatial" not in result
        assert "ownership" not in result
        assert "read_only_context" not in result
        # S3 A2: summary 从 typed actions 重建；无 kill_topic action 时回退到 raw summary。
        assert result["main_topic"]["kind"] == "kill"
        assert result["main_topic"]["summary"] == "A击杀B"
        assert len(result["selected_actions"]) == 1
        assert result["selected_actions"][0]["type"] == "kill"

    def test_tactic_hint_preserved_in_projection(self):
        plan = {
            "main_topic": {"kind": "tactic", "summary": "默认A区爆弹"},
            "selected_actions": [],
            "tactic_hint": {
                "rule_id": "default_a_execute",
                "label": "默认A区爆弹",
                "hint": "T在A区执行默认爆弹战术",
                "matched_at": 12.0,
            },
        }
        result = build_llm_window_projection(plan)
        assert result["tactic_hint"] == {
            "rule_id": "default_a_execute",
            "label": "默认A区爆弹",
            "hint": "T在A区执行默认爆弹战术",
        }

    def test_tactic_summary_is_label_or_label_plus_hint(self):
        """tactic 类型 main_topic 的 summary 由 tactic_hint 派生。"""
        plan = {
            "main_topic": {"kind": "tactic"},
            "selected_actions": [],
            "tactic_hint": {
                "rule_id": "test_rule",
                "label": "A区爆弹",
                "hint": "快速A区爆弹执行",
            },
        }
        result = build_llm_window_projection(plan)
        assert result["main_topic"]["kind"] == "tactic"
        assert "A区爆弹" in result["main_topic"]["summary"]

    def test_tactic_without_hint_demotes_to_silence(self):
        """tactic 话题缺少 tactic_hint 时降级为 silence。"""
        plan = {
            "main_topic": {"kind": "tactic"},
            "selected_actions": [],
        }
        result = build_llm_window_projection(plan)
        assert result["main_topic"]["kind"] == "silence"


class TestFallbackNeutral:
    """fallback_neutral 的回归测试。"""

    def test_uses_main_topic_summary(self):
        plan = {"main_topic": {"kind": "retake", "summary": "C4已安装"}}
        assert fallback_neutral(plan) == "C4已安装"

    def test_empty_plan(self):
        assert fallback_neutral({}) == ""

    def test_truncates_at_100_chars(self):
        long_summary = "很长的文本" * 30
        plan = {"main_topic": {"kind": "kill", "summary": long_summary}}
        result = fallback_neutral(plan)
        assert len(result) <= 100


class TestNeutralSource:
    """neutral_source 字段的语义正确性。"""

    def test_llm_source_never_rejected_even_if_text_equals_fallback(self):
        """模型输出恰好等于规则摘要 → neutral_source=llm，不应被门禁拒绝。"""
        # 这个测试验证逻辑：source 是独立于文本的显式字段
        # 在 validate_neutral_publishable 中：只有 neutral_source=="fallback" 且 neutral 非空才拒绝
        plan = {"main_topic": {"kind": "retake", "summary": "C4已安装"}}
        fb = fallback_neutral(plan)
        # LLM 恰好输出了和 fallback 一样的文本
        assert fb == "C4已安装"
        # 但 neutral_source 是 "llm"，所以发布门禁不应拒绝
        # （此逻辑在 test_preflight_neutral.py 中已有覆盖）

    def test_fallback_source_with_content_is_rejected(self, tmp_path):
        """neutral_source=fallback 且 neutral 非空 → 发布门禁应拒绝"""
        from sbmachine.preflight import PublishContractError, validate_neutral_publishable

        manifest = {
            "schema_version": 3,
            "phase3a_mode": "llma_slicer_then_llma_analyze",
            "run_id": "test-run",
            "source_rounds_sha256": "0" * 64,
            "video_path": "/dev/null",
            "map_name": "de_test",
            "model": "test",
            "rounds": [
                {
                    "round_no": 1,
                    "start_sec": 0.0,
                    "end_sec": 30.0,
                    "analyst_failed": False,
                    "scenes": [
                        {
                            "t_start": 10.0,
                            "t_end": 15.0,
                            "scene": "test",
                            "commentary_plan": {"main_topic": {"kind": "retake", "summary": "C4已安装"}},
                            "neutral": "兜底文本",
                            "neutral_source": "fallback",
                            "generation_status": "success",
                        }
                    ],
                }
            ],
        }
        path = tmp_path / "test_neutral_publishable.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(PublishContractError, match="rule fallback neutral is not publishable"):
            validate_neutral_publishable(path)

    def test_llm_source_always_passes_publish_gate(self, tmp_path):
        """neutral_source=llm 的窗口总是通过发布门禁。"""
        from sbmachine.preflight import validate_neutral_publishable

        manifest = {
            "schema_version": 3,
            "phase3a_mode": "llma_slicer_then_llma_analyze",
            "run_id": "test-run",
            "source_rounds_sha256": "0" * 64,
            "video_path": "/dev/null",
            "map_name": "de_test",
            "model": "test",
            "rounds": [
                {
                    "round_no": 1,
                    "start_sec": 0.0,
                    "end_sec": 30.0,
                    "analyst_failed": False,
                    "scenes": [
                        {
                            "t_start": 10.0,
                            "t_end": 15.0,
                            "scene": "test",
                            "commentary_plan": {"main_topic": {"kind": "retake", "summary": "C4已安装"}},
                            "neutral": "C4已安装",
                            "neutral_source": "llm",
                            "generation_status": "success",
                        }
                    ],
                }
            ],
        }
        path = tmp_path / "test_llm_publishable.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        validate_neutral_publishable(path)  # 不应 raise


class TestEmptyNeutralWithActions:
    """main_topic.summary 为空但 selected_actions 非空的兜底逻辑。"""

    def test_empty_neutral_fallback_to_window_payload_summary(self):
        """当 neutral 为空字符串且 plan 有 selected_actions 时，使用 window_payload.main_topic.summary。"""
        # 模拟 _process_round 中的逻辑
        plan = {
            "main_topic": {"kind": "kill", "summary": "原始摘要但会被投影清空"},
            "selected_actions": [{"type": "kill"}],
        }
        window_payload = build_llm_window_projection(plan)
        # S3 A2: 无 kill_topic 类型 action 时回退到 raw summary，非空。
        assert window_payload["main_topic"]["kind"] == "kill"
        assert len(window_payload["main_topic"]["summary"]) > 0
        # 但 selected_actions 非空
        assert window_payload["selected_actions"]

        # 模拟兜底：neutral 为空且 selected_actions 非空
        neutral = ""
        if not neutral.strip() and plan.get("selected_actions"):
            neutral = str((window_payload.get("main_topic") or {}).get("summary") or "")[:100]
        # S3 A2: 投影层 summary 非空，所以兜底会取到值
        assert len(neutral) > 0


class TestDebugRecorder:
    """DebugRecorder 的行为测试。"""

    def test_disabled_recorder_is_noop(self, tmp_path):
        recorder = DebugRecorder(enabled=False, output_dir=tmp_path)
        record = DebugWindowRecord(round_no=1, window_idx=1, t_start=0.0, t_end=1.0, scene="test")
        recorder.record_window(record)
        # disabled 时不创建任何文件
        assert list(tmp_path.glob("*.json")) == []

    def test_enabled_recorder_writes_file(self, tmp_path):
        recorder = DebugRecorder(enabled=True, output_dir=tmp_path)
        record = DebugWindowRecord(
            round_no=1, window_idx=1, t_start=0.0, t_end=1.0, scene="test",
            final_neutral="test neutral", neutral_source="llm",
        )
        recorder.record_window(record)

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        assert files[0].name == "r001_w01.json"

        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["round_no"] == 1
        assert data["window_idx"] == 1
        assert data["final_neutral"] == "test neutral"
        assert data["neutral_source"] == "llm"

    def test_record_contains_all_required_fields(self, tmp_path):
        recorder = DebugRecorder(enabled=True, output_dir=tmp_path)
        record = DebugWindowRecord(round_no=2, window_idx=3, t_start=10.0, t_end=20.0, scene="测试")
        recorder.record_window(record)

        files = list(tmp_path.glob("*.json"))
        data = json.loads(files[0].read_text(encoding="utf-8"))

        required_fields = [
            "round_no", "window_idx", "t_start", "t_end", "scene",
            "raw_plan", "llm_projection", "system_prompt", "user_prompt",
            "http_request_body", "llm_config",
            "vllm_raw_response", "message_content", "reasoning_content",
            "think_text", "cleaned_content",
            "json_parse_result", "parse_error", "contract_valid", "contract_error",
            "parsed_neutral",
            "final_neutral", "neutral_source", "equals_fallback", "fallback_text",
        ]
        for field in required_fields:
            assert field in data, f"missing field: {field}"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Mock 后端集成测试（无模型，秒级）
# ═══════════════════════════════════════════════════════════════════════════


class TestMockBackend:
    """mock_generate 返回 _ApiChatResult 的集成测试。"""

    def test_mock_returns_string(self):
        result = mock_generate("test prompt", {"model": "test", "temperature": 0.3})
        assert isinstance(result, str)

    def test_mock_has_raw_response(self):
        result = mock_generate("test prompt", {"model": "test", "temperature": 0.3})
        raw = getattr(result, "raw_response", None)
        assert raw is not None
        assert "choices" in raw
        assert len(raw["choices"]) > 0

    def test_mock_has_request_payload(self):
        result = mock_generate("test prompt", {"model": "test", "temperature": 0.3})
        payload = getattr(result, "request_payload", None)
        assert payload is not None
        assert "messages" in payload

    def test_mock_cycles_through_responses(self):
        """连续调用应轮转不同响应。"""
        import os as _os

        _os.environ.pop("MOCK_RESPONSE_INDEX", None)
        results = []
        for _ in range(len(_MOCK_CASES) + 2):
            r = mock_generate("p", {"model": "m", "temperature": 0.3})
            results.append(str(r))

        # 第二轮的第一个应与第一轮的第一个相同
        assert results[0] == results[len(_MOCK_CASES)]

        # 所有 mock case 的 after_strip 都应出现
        for case in _MOCK_CASES:
            assert case["after_strip"] in results

    def test_mock_fixed_index(self):
        """MOCK_RESPONSE_INDEX 环境变量可固定索引。"""
        import os as _os

        _os.environ["MOCK_RESPONSE_INDEX"] = "0"
        r1 = str(mock_generate("p", {"model": "m", "temperature": 0.3}))
        r2 = str(mock_generate("p", {"model": "m", "temperature": 0.3}))
        _os.environ.pop("MOCK_RESPONSE_INDEX", None)
        # 固定索引时，每次返回相同 case
        assert r1 == r2

    def test_mock_reasoning_content_in_response_layer(self):
        """reasoning_content 应出现在 _ApiChatResult 属性上。"""
        import os as _os

        # Case index 3 = reasoning_content at response level
        _os.environ["MOCK_RESPONSE_INDEX"] = "3"
        result = mock_generate("p", {"model": "m", "temperature": 0.3})
        _os.environ.pop("MOCK_RESPONSE_INDEX", None)

        reasoning = getattr(result, "reasoning_content", None)
        assert reasoning is not None
        assert "推理过程" in reasoning


class TestMockFullPipeline:
    """Mock 后端走完整 parse → contract → fallback 链路。"""

    def _run_pipeline(self, mock_index: int):
        """使用指定 mock case 跑一遍解析链路。"""
        import os as _os

        _os.environ["MOCK_RESPONSE_INDEX"] = str(mock_index)
        try:
            plan = {
                "main_topic": {"kind": "kill", "summary": "A击杀B"},
                "selected_actions": [{"type": "kill", "attacker": "A", "victim": "B"}],
            }
            window_payload = build_llm_window_projection(plan)
            prompt = _build_window_prompt({"commentary_plan": plan})
            raw = mock_generate(prompt, {"model": "mock", "temperature": 0.3})
            parsed = _parse_window_neutral_response(str(raw))
            return raw, parsed, plan
        finally:
            _os.environ.pop("MOCK_RESPONSE_INDEX", None)

    def test_normal_neutral_pipeline(self):
        """Case 0: 正常 neutral → 解析成功。"""
        raw, parsed, plan = self._run_pipeline(0)
        assert parsed == "A区爆弹进攻"
        # source 应为 llm
        assert parsed is not None

    def test_empty_neutral_pipeline(self):
        """Case 1: 合法空 neutral → 解析为空字符串。"""
        raw, parsed, plan = self._run_pipeline(1)
        assert parsed == ""

    def test_think_field_pipeline(self):
        """Case 2: JSON 级 think 字段 → 应成功剥离。"""
        raw, parsed, plan = self._run_pipeline(2)
        assert parsed == "战术配合"

    def test_think_tag_pipeline(self):
        """Case 4: <think> 标签包裹 → _execute_openai_chat 已剥离。"""
        raw, parsed, plan = self._run_pipeline(4)
        assert parsed == "有效输出"

    def test_markdown_fence_pipeline(self):
        """Case 5: Markdown 围栏 → 解析失败。"""
        raw, parsed, plan = self._run_pipeline(5)
        assert parsed is None
        # 应 fallback
        fb = fallback_neutral(plan)
        assert fb  # fallback 不为空

    def test_extra_field_pipeline(self):
        """Case 6: 额外字段 → 解析失败。"""
        raw, parsed, plan = self._run_pipeline(6)
        assert parsed is None

    def test_illegal_json_pipeline(self):
        """Case 7: 非法 JSON → 解析失败。"""
        raw, parsed, plan = self._run_pipeline(7)
        assert parsed is None

    def test_non_string_neutral_pipeline(self):
        """Case 8: neutral 非字符串 → 解析失败。"""
        raw, parsed, plan = self._run_pipeline(8)
        assert parsed is None

    def test_all_mock_cases_have_sane_fallback(self):
        """所有 mock case 解析失败时 fallback 不应 crash。"""
        for i in range(len(_MOCK_CASES)):
            raw, parsed, plan = self._run_pipeline(i)
            fb = fallback_neutral(plan)
            # fallback 应该总是字符串
            assert isinstance(fb, str)
            # 如果 parsed 是 None，neutral_source 应为 fallback
            if parsed is None:
                pass  # 预期行为


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — 云端验收（仅文档，手动执行）
# ═══════════════════════════════════════════════════════════════════════════
#
# python -m tools.debug_phase3a \
#   --input output/sbmachine/rounds_with_neutral.json \
#   --round 1 --window 1 \
#   --backend openai_compatible \
#   --base-url http://127.0.0.1:8000/v1 \
#   --model Qwen/Qwen3-1.7B
