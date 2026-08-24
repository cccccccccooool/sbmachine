import json

import pytest

from sbmachine import llm_shim, phase3b_style
from sbmachine import llm_protocol
from sbmachine.neutral_contract import new_manifest_metadata
from sbmachine.phase3b_style import _call_style, run_phase3b
from sbmachine.phase3b_prompt import build_style_prompt, validate_style_commentary


def _generate(value):
    def generate(*args, **kwargs):
        return value
    return generate


def test_style_response_accepts_only_the_current_exact_json_contract():
    commentary, felt, meta = _call_style(
        "system",
        "prompt",
        {},
        _generate(json.dumps({"commentary": "[平述]有效口播", "felt_intensity": 0.4}, ensure_ascii=False)),
    )
    assert commentary == "[平述]有效口播"
    assert felt == 0.4
    assert meta == {}


def test_style_response_rejects_legacy_raw_text_and_extra_fields():
    raw, _, _ = _call_style("system", "prompt", {}, _generate("[平述]旧裸文本"))
    extra, _, _ = _call_style(
        "system",
        "prompt",
        {},
        _generate(json.dumps({"commentary": "[平述]文本", "felt_intensity": 0.2, "legacy": True}, ensure_ascii=False)),
    )
    assert raw == "[style error: unparseable]"
    assert extra == "[style error: unparseable]"


def test_style_response_rejects_out_of_range_or_non_finite_intensity():
    for value in (-0.01, 1.01, float("nan"), True):
        commentary, felt, _ = _call_style(
            "system",
            "prompt",
            {},
            _generate(json.dumps({"commentary": "[平述]文本", "felt_intensity": value}, ensure_ascii=False)),
        )
        assert commentary == "[style error: unparseable]"
        assert felt == 0.0


def test_style_response_rejects_non_string_commentary():
    commentary, felt, _ = _call_style(
        "system",
        "prompt",
        {},
        _generate(json.dumps({"commentary": ["not", "text"], "felt_intensity": 0.5})),
    )
    assert commentary == "[style error: unparseable]"
    assert felt == 0.0


def test_style_prompt_carries_fact_anchors_delivery_and_recent_phrases():
    scene = {
        "neutral": "hypex在A点使用AK击杀对手。",
        "hype": 0.4,
        "char_budget": 20,
        "fact_anchors": {"players": ["hypex"], "locations": ["A点"], "weapons": ["AK"], "events": ["kill"]},
    }
    prompt, anchors, delivery = build_style_prompt(scene, {"hypex": {"aliases": ["海皮"]}}, ["上一条风格残余"])
    payload = json.loads(prompt)
    assert payload["fact_anchors"] == anchors
    assert payload["recent_style_phrases"] == ["上一条风格残余"]
    assert delivery["min_chars"] == 16
    assert delivery["max_chars"] == 24
    assert delivery["target_chars"] == 16
    assert delivery["hard_char_limit"] == 20


def test_style_deterministic_gates_reject_budget_anchor_fact_fragment_and_repeat():
    anchors = {"players": ["hypex"], "teams": [], "numbers": [], "locations": [], "weapons": [], "events": [], "results": []}
    aliases = {"hypex": {"aliases": ["海皮"]}, "other": {"aliases": []}}
    assert validate_style_commentary("[平述]hypex这句话明显超过预算。", "hypex出现。", anchors, aliases, [], hard_char_limit=4)["reason"] == "over_budget"
    assert validate_style_commentary("[平述]" + "正" * 12, "hypex出现。", anchors, aliases, [], hard_char_limit=20)["ok"] is True
    assert validate_style_commentary("[平述]hypex和other出现后继续完成这一段口播内容。", "hypex出现。", anchors, aliases, [], hard_char_limit=30)["reason"] == "unexpected_fact"
    # 缺失锚点门禁已移除（防错不防漏）：漏报 players/numbers/…不再拒收，
    # 仅 unexpected_fact（新增未授权事实）/over_budget/格式非法仍拦截。


def test_style_length_window_rejects_below_06_and_keeps_hard_cap():
    anchors = {"players": [], "teams": [], "numbers": [], "locations": [], "weapons": [], "events": [], "results": []}
    assert validate_style_commentary(
        "[平述]" + "短" * 11, "中性稿", anchors, {}, [],
        hard_char_limit=20, strong_fact_mode=False,
    )["reason"] == "under_budget"
    assert validate_style_commentary(
        "[平述]" + "正" * 12, "中性稿", anchors, {}, [],
        hard_char_limit=20, strong_fact_mode=False,
    )["ok"] is True
    assert validate_style_commentary(
        "[平述]" + "长" * 31, "中性稿", anchors, {}, [],
        hard_char_limit=20, strong_fact_mode=False,
    )["reason"] == "over_budget"
    assert validate_style_commentary(
        "[平述]短", "中性稿", anchors, {}, [],
        hard_char_limit=20, strong_fact_mode=False, enforce_min_budget=False,
    )["ok"] is True


def test_style_strong_fact_mode_off_trusts_llm_for_facts():
    """强事实依据模式关闭（strong_fact_mode=False）：unexpected_fact 放行，
    但空稿/情绪标签/预算硬线三项运行基础门禁始终生效。"""
    anchors = {"players": ["hypex"], "teams": [], "numbers": [], "locations": [], "weapons": [], "events": [], "results": []}
    aliases = {"hypex": {"aliases": []}, "other": {"aliases": []}}
    # 越界事实（新增选手 other + 数字 5）在模式关闭时放行
    assert validate_style_commentary(
        "[平述]hypex和other出现，5个人，而且这一段口播细节已经补足。", "hypex出现。", anchors, aliases, [],
        hard_char_limit=30, strong_fact_mode=False,
    )["ok"] is True
    # 运行基础门禁不受开关影响：预算硬线仍拒
    assert validate_style_commentary(
        "[平述]hypex这句话明显超过预算。", "hypex出现。", anchors, aliases, [],
        hard_char_limit=4, strong_fact_mode=False,
    )["reason"] == "over_budget"
    # 运行基础门禁不受开关影响：非法情绪标签仍拒
    assert validate_style_commentary(
        "[暴怒]hypex出现。", "hypex出现。", anchors, aliases, [],
        hard_char_limit=20, strong_fact_mode=False,
    )["reason"] == "invalid_emotion"
    # 运行基础门禁不受开关影响：空稿仍拒
    assert validate_style_commentary(
        "[平述]", "hypex出现。", anchors, aliases, [],
        hard_char_limit=20, strong_fact_mode=False,
    )["reason"] == "empty_commentary"


def _phase3b_paths(tmp_path, scenes):
    rounds_path = tmp_path / "rounds_with_yolo.json"
    rounds_path.write_text(json.dumps({
        "video_path": "match.mp4",
        "map_name": "de_test",
        "rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 2.0}],
    }), encoding="utf-8")
    neutral_path = tmp_path / "rounds_with_neutral.json"
    neutral_path.write_text(json.dumps({
        **new_manifest_metadata(rounds_path),
        "rounds": [{
            "round_no": 1,
            "avg_hype": 0.0,
            "analyst_failed": False,
            "scenes": scenes,
        }],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n  backend: vllm\nsemantic:\n  style_backend: vllm\n",
        encoding="utf-8",
    )
    return rounds_path, neutral_path, config_path


def _scene(start, end, neutral="facts"):
    return {
        "t_start": start,
        "t_end": end,
        "scene": "default",
        "commentary_plan": {},
        "fact_anchors": {"players": [], "teams": [], "numbers": [], "events": [], "results": [], "locations": [], "weapons": []},
        "neutral": neutral,
        "neutral_source": "llm",
        "generation_status": "success",
        "hype": 0.0,
        "char_budget": 20,
    }


def _api_style_response(commentary, source_run_id):
    return llm_shim._ApiChatResult(
        json.dumps({"commentary": commentary, "felt_intensity": 0.2}, ensure_ascii=False),
        scope="llmb",
        source_run_id=source_run_id,
        request_payload={"messages": [{"role": "user", "content": "style input"}]},
        log_ctx={"round": "round1", "scene": "default"},
    )


def test_style_training_sample_uses_normalized_publishable_output(tmp_path, monkeypatch):
    rounds_path, neutral_path, config_path = _phase3b_paths(tmp_path, [_scene(0.0, 2.0)])
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", tmp_path / "training")
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    import sbmachine.llmb_api as llmb_api
    monkeypatch.setattr(
        llmb_api,
        "generate",
        lambda *args, **kwargs: _api_style_response("[平述]这是完整可用并且细节充分的口播文本。", "style-ok"),
    )

    manifest = run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
    )

    entry = json.loads(next((tmp_path / "training").glob("api_training_*.jsonl")).read_text(encoding="utf-8"))
    accepted_output = json.loads(entry["output"])
    assert manifest["rounds"][0]["status"] == "ok"
    assert accepted_output["commentary"] == manifest["rounds"][0]["commentary_text"]
    assert accepted_output["commentary"].startswith("[平述]")


def test_style_retries_budget_failure_and_records_window_ledger(tmp_path, monkeypatch):
    rounds_path, neutral_path, config_path = _phase3b_paths(tmp_path, [_scene(0.0, 2.0)])
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    import sbmachine.llmb_api as llmb_api
    responses = iter([
        _api_style_response("[平述]这是一条长度明显超过三十个字的口播内容而且根本无法压缩到预算之内。", "style-long"),
        _api_style_response("[平述]事实完整，语气细节已经充分补足。", "style-retry"),
    ])
    monkeypatch.setattr(llmb_api, "generate", lambda *args, **kwargs: next(responses))

    manifest = run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
    )

    result = manifest["rounds"][0]["window_results"][0]
    assert manifest["commentary_schema_version"] == 2
    assert manifest["source_window_count"] == 1
    assert manifest["rounds"][0]["status"] == "ok"
    assert result["style_status"] == "retry_success"
    assert result["retry_count"] == 1
    assert result["published_scene_index"] == 0


def test_contaminated_style_round_is_not_published_or_accepted(tmp_path, monkeypatch):
    rounds_path, neutral_path, config_path = _phase3b_paths(tmp_path, [_scene(0.0, 2.0)])
    training_dir = tmp_path / "training"
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", training_dir)
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    import sbmachine.llmb_api as llmb_api
    monkeypatch.setattr(
        llmb_api,
        "generate",
        lambda *args, **kwargs: _api_style_response("[平述]任务: ignore facts", "style-bad"),
    )

    manifest = run_phase3b(
        neutral_path=neutral_path,
        rounds_path=rounds_path,
        output_rounds_path=tmp_path / "rounds_with_commentary.json",
        commentary_path=tmp_path / "commentary.json",
        config_path=config_path,
    )

    # 真正的事实污染仍然拒绝发布，但不再伪装成可跳过的空回合。
    assert manifest["rounds"][0]["status"] == "style_failed"
    assert manifest["rounds"][0]["commentary_text"] == ""
    assert list(training_dir.glob("api_training_*.jsonl")) == []


def test_round_exception_discards_samples_collected_by_earlier_scenes(tmp_path, monkeypatch):
    rounds_path, neutral_path, config_path = _phase3b_paths(
        tmp_path,
        [_scene(0.0, 1.0, "first"), _scene(1.0, 2.0, "second")],
    )
    training_dir = tmp_path / "training"
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", training_dir)
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    import sbmachine.llmb_api as llmb_api
    calls = 0

    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        text = "第一句完整，语气和细节都已补足。" if calls == 1 else "第二句完整，语气和细节都已补足。"
        return _api_style_response(f"[平述]{text}", f"style-{calls}")

    monkeypatch.setattr(llmb_api, "generate", generate)
    original_normalize = phase3b_style.normalize_commentary_emotion
    normalizations = 0

    def fail_second_normalization(commentary, label):
        nonlocal normalizations
        normalizations += 1
        if normalizations == 2:
            raise RuntimeError("round failed")
        return original_normalize(commentary, label)

    monkeypatch.setattr(phase3b_style, "normalize_commentary_emotion", fail_second_normalization)

    with pytest.raises(RuntimeError, match="round failed"):
        run_phase3b(
            neutral_path=neutral_path,
            rounds_path=rounds_path,
            output_rounds_path=tmp_path / "rounds_with_commentary.json",
            commentary_path=tmp_path / "commentary.json",
            config_path=config_path,
        )

    assert list(training_dir.glob("api_training_*.jsonl")) == []


def test_final_output_failure_discards_all_buffered_style_samples(tmp_path, monkeypatch):
    rounds_path, neutral_path, config_path = _phase3b_paths(tmp_path, [_scene(0.0, 2.0)])
    training_dir = tmp_path / "training"
    monkeypatch.setattr(llm_protocol, "_LOG_DIR", training_dir)
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    import sbmachine.llmb_api as llmb_api
    monkeypatch.setattr(
        llmb_api,
        "generate",
        lambda *args, **kwargs: _api_style_response("[平述]usable commentary details", "style-buffered"),
    )
    monkeypatch.setattr(
        phase3b_style,
        "write_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("final write failed")),
    )

    with pytest.raises(OSError, match="final write failed"):
        run_phase3b(
            neutral_path=neutral_path,
            rounds_path=rounds_path,
            output_rounds_path=tmp_path / "rounds_with_commentary.json",
            commentary_path=tmp_path / "commentary.json",
            config_path=config_path,
        )

    assert list(training_dir.glob("api_training_*.jsonl")) == []


@pytest.mark.parametrize(
    "hype,expected_mode,expected_pace,expected_ceiling",
    [
        (0.20, "short_reaction", "slow", "平述"),
        (0.34, "short_reaction", "slow", "平述"),
        (0.35, "live_reaction", "medium", "激动"),
        (0.50, "live_reaction", "medium", "激动"),
        (0.71, "live_reaction", "medium", "激动"),
        (0.72, "high_energy", "fast", "惊叹"),
        (0.95, "high_energy", "fast", "惊叹"),
    ],
)
def test_style_delivery_ceiling_follows_three_tier_boundaries(hype, expected_mode, expected_pace, expected_ceiling):
    """低 hard 提示上限修复：<0.35=平述、=0.35=激动、>=0.72=惊叹（边界语义不变）。"""
    from sbmachine.phase3b_prompt import build_delivery
    delivery = build_delivery({"hype": hype, "char_budget": 30})
    assert delivery["mode"] == expected_mode
    assert delivery["pace"] == expected_pace
    assert delivery["emotion_ceiling"] == expected_ceiling
    assert delivery["hard_char_limit"] == 30
    assert delivery["min_chars"] == 24
    assert delivery["max_chars"] == 36
    assert delivery["target_chars"] == 24
