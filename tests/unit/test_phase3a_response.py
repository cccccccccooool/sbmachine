import json

import pytest

from sbmachine import llm_shim
from sbmachine.phase3a_analyst import _call_analyst
from sbmachine.phase3a_prompt import _parse_window_neutral_response, validate_neutral_semantics


def test_analyst_response_accepts_only_one_exact_json_object():
    assert _parse_window_neutral_response('{"neutral":"有效中性稿"}') == "有效中性稿"


@pytest.mark.parametrize(
    "response",
    [
        '说明：{"neutral":"包裹文本"}',
        '{"neutral":"包裹文本"}\n说明',
        '```json\n{"neutral":"围栏"}\n```',
        '{"neutral":"文本","extra":true}',
    ],
)
def test_analyst_response_rejects_wrappers_fences_and_extra_fields(response):
    assert _parse_window_neutral_response(response) is None


def test_analyst_tolerates_qwen_think_field():
    """Qwen3 emits JSON-level think field; parser should strip it and still extract neutral."""
    assert _parse_window_neutral_response('{"think":"分析过程","neutral":"有效中性稿"}') == "有效中性稿"
    assert _parse_window_neutral_response(
        '{"reasoning":"思考","think":"分析","neutral":"多字段剥离"}'
    ) == "多字段剥离"
    # Unknown extra field should still be rejected
    assert _parse_window_neutral_response('{"neutral":"文本","unknown_field":true}') is None


def test_analyst_parser_does_not_persist_before_round_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_shim, "_LOG_DIR", tmp_path)
    response = llm_shim._ApiChatResult(
        '{"neutral":"有效中性稿"}',
        scope="llma",
        source_run_id="run-a",
        request_payload={"messages": [{"role": "user", "content": "分析输入"}]},
        log_ctx={"round": "round1", "scene": "win1"},
    )

    assert _parse_window_neutral_response(response) == "有效中性稿"
    assert list(tmp_path.glob("api_training_*.jsonl")) == []


def test_analyst_marks_length_finished_json_as_truncated():
    raw = llm_shim._ApiChatResult(
        '{"neutral":"看似完整但达到输出上限"}',
        scope="llma",
        source_run_id="length-case",
        request_payload={"messages": []},
        log_ctx=None,
        raw_response={
            "choices": [{
                "message": {"content": '{"neutral":"看似完整但达到输出上限"}'},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
        finish_reason="length",
    )

    result = _call_analyst("prompt", {}, lambda *args, **kwargs: raw, system_prompt="system")

    assert result.generation_status == "truncated"
    assert result.content == ""
    assert result.finish_reason == "length"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

def test_analyst_classifies_timeout_as_transport_error():
    import requests

    def timeout(*args, **kwargs):
        raise requests.Timeout("request timed out")

    result = _call_analyst("prompt", {}, timeout, system_prompt="system")

    assert result.generation_status == "transport_error"
    assert result.error_type == "Timeout"

def test_analyst_classifies_missing_message_content_as_response_error():
    raw = llm_shim._ApiChatResult(
        "",
        scope="llma",
        source_run_id="bad-envelope",
        request_payload={"messages": []},
        log_ctx=None,
        raw_response={"choices": [{"finish_reason": "stop"}]},
        finish_reason="stop",
    )

    result = _call_analyst("prompt", {}, lambda *args, **kwargs: raw, system_prompt="system")

    assert result.generation_status == "response_error"


# ── S2 Gate 2 补充测试 ──

def test_gate2_empty_content_with_stop_is_contract_error():
    """reasoning 存在、content 为空、finish_reason=stop → contract_error"""
    raw = llm_shim._ApiChatResult(
        "",
        scope="llma",
        source_run_id="empty-content",
        request_payload={"messages": []},
        log_ctx=None,
        raw_response={
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        },
        finish_reason="stop",
    )
    result = _call_analyst("prompt", {}, lambda *args, **kwargs: raw, system_prompt="system")
    # 空串解析：parser 返回 None（不是合法 JSON），classification 检查
    # 即使是 stop，空 content 也应按顺序归类为 parse_error（不是 JSON）
    assert result.generation_status in ("parse_error", "contract_error")


def test_gate2_length_with_empty_content_is_truncated():
    """thinking 未结束，finish_reason=length，content 为空 → truncated"""
    raw = llm_shim._ApiChatResult(
        "",
        scope="llma",
        source_run_id="think-unfinished",
        request_payload={"messages": []},
        log_ctx=None,
        raw_response={
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2560, "total_tokens": 2660},
        },
        finish_reason="length",
    )
    result = _call_analyst("prompt", {}, lambda *args, **kwargs: raw, system_prompt="system")
    # finish_reason=length 在分类顺序中排在 parse 之前，直接判为 truncated
    assert result.generation_status == "truncated"
    assert result.finish_reason == "length"
    assert result.error_type == "IncompleteJSON"


def test_gate2_length_with_truncated_json_is_truncated():
    """reasoning 完整，content 中 JSON 截断，finish_reason=length → truncated"""
    raw = llm_shim._ApiChatResult(
        '{"neutral":"未完成的句子',
        scope="llma",
        source_run_id="json-trunc",
        request_payload={"messages": []},
        log_ctx=None,
        raw_response={
            "choices": [{"message": {"content": '{"neutral":"未完成的句子'}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        },
        finish_reason="length",
    )
    result = _call_analyst("prompt", {}, lambda *args, **kwargs: raw, system_prompt="system")
    assert result.generation_status == "truncated"
    assert result.finish_reason == "length"


def test_gate2_neutral_too_long_is_contract_error():
    """neutral 超过字符上限 → contract_error"""
    long_neutral = "测试" * 60  # 120 chars, > 100 limit
    raw = llm_shim._ApiChatResult(
        json.dumps({"neutral": long_neutral}, ensure_ascii=False),
        scope="llma",
        source_run_id="too-long",
        request_payload={"messages": []},
        log_ctx=None,
        raw_response={
            "choices": [{"message": {"content": json.dumps({"neutral": long_neutral}, ensure_ascii=False)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
        },
        finish_reason="stop",
    )
    result = _call_analyst("prompt", {}, lambda *args, **kwargs: raw, system_prompt="system")
    assert result.generation_status == "contract_error"
    assert result.error_type == "NeutralLengthExceeded"


def test_gate2_http_503_is_http_error():
    """503 → http_error，保留状态码"""
    import requests

    class Fake503Response:
        status_code = 503

    def raise_503(*args, **kwargs):
        exc = requests.HTTPError("Service Unavailable")
        exc.response = Fake503Response()
        raise exc

    result = _call_analyst("prompt", {}, raise_503, system_prompt="system")
    assert result.generation_status == "http_error"
    assert result.http_status == 503


def _semantic_projection() -> dict:
    return {
        "projection_version": 2,
        "main_topic": {"kind": "state", "summary": "T方存活5人，总血量494"},
        "selected_actions": [],
        "required_facts": [{
            "fact_id": "state:end:T", "type": "team_state",
            "canonical_text": "T方存活5人，总血量494", "required": True,
            "anchors": {"players": [], "teams": ["T"], "numbers": [5, 494],
                        "events": [], "results": ["alive_count", "hp_total"]},
        }],
        "rule_state": {"kind": "snapshot", "teams": {
            "T": {"alive_count": 5, "hp_total": 494},
            "CT": {"alive_count": 5, "hp_total": 500},
        }, "changed_teams": ["T"]},
        "required_chars": 15,
    }


def test_semantic_contract_requires_canonical_text_verbatim():
    assert validate_neutral_semantics("T方还有5人。", _semantic_projection())[0] == "required_fact_missing"


def test_semantic_contract_classifies_side_swap():
    assert validate_neutral_semantics("CT方存活5人，总血量494。", _semantic_projection())[0] == "side_mismatch"


def test_semantic_contract_rejects_unexpected_number_and_location():
    assert validate_neutral_semantics("T方存活5人，总血量494，形成3打2。", _semantic_projection())[0] == "unexpected_fact"
    assert validate_neutral_semantics("T方存活5人，总血量494，转向A区。", _semantic_projection())[0] == "unexpected_fact"


def _bomb_projection_with_state() -> dict:
    return {
        "projection_version": 2,
        "main_topic": {"kind": "bomb", "summary": "C4已安装"},
        "selected_actions": [{"type": "bomb_planted"}],
        "required_facts": [{
            "fact_id": "topic:bomb", "type": "bomb",
            "canonical_text": "C4已安装", "required": True,
            "anchors": {"players": [], "teams": [], "numbers": [],
                        "events": ["bomb_planted"], "results": []},
        }],
        "rule_state": {"kind": "snapshot", "teams": {
            "T": {
                "alive_count": 3, "hp_total": 170,
                "players": [
                    {"name": "t1", "hp": 100},
                    {"name": "t2", "hp": 70},
                    {"name": "t3", "hp": 0},
                ],
            },
            "CT": {
                "alive_count": 2, "hp_total": 95,
                "players": [
                    {"name": "ct1", "hp": 95},
                    {"name": "ct2", "hp": 0},
                ],
            },
        }, "changed_teams": ["T", "CT"]},
        "required_chars": 6,
    }


def test_semantic_contract_allows_rule_state_chinese_terms():
    projection = _bomb_projection_with_state()
    assert validate_neutral_semantics("C4已安装，T方3人存活，CT方2人存活。", projection)[0] is None
    # per-player 血量：规则层残局会输出"X方t1 100血"这类明细，"血量"是可口播语义词
    assert validate_neutral_semantics("C4已安装，T方t1 100血、t2 70血，CT方ct1 95血。", projection)[0] is None
    # 总血量：规则层已不再输出整队求和血量，词不被授权
    assert validate_neutral_semantics("C4已安装，T方总血量170，CT方总血量95。", projection)[0] == "unexpected_fact"


def test_semantic_contract_rejects_state_terms_without_rule_state_fields():
    projection = _bomb_projection_with_state()
    projection.pop("rule_state")
    assert validate_neutral_semantics("C4已安装，仍有选手存活。", projection)[0] == "unexpected_fact"
    assert validate_neutral_semantics("C4已安装，双方人数不明。", projection)[0] == "unexpected_fact"


def test_semantic_contract_allows_player_state_facts():
    projection = _bomb_projection_with_state()
    projection["player_state"] = "首次快照：JDC（CT，100血，M4，中路）；REZ已阵亡"
    assert validate_neutral_semantics("C4已安装，JDC还剩100血。", projection)[0] is None
    assert validate_neutral_semantics("C4已安装，JDC拿着M4。", projection)[0] is None
    assert validate_neutral_semantics("C4已安装，REZ已阵亡。", projection)[0] is None


def test_semantic_contract_rejects_unprovided_player_state_facts():
    projection = _bomb_projection_with_state()
    projection["player_state"] = "首次快照：JDC（CT，100血，M4，中路）；REZ已阵亡"
    assert validate_neutral_semantics("C4已安装，ZONETIC只剩34血。", projection)[0] == "unexpected_fact"
    assert validate_neutral_semantics("C4已安装，JDC只剩34血。", projection)[0] == "unexpected_fact"
