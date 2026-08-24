"""cloud_prompts：云端 prompt 拼装 + §4.1 窗口类型判定表全覆盖 + 接入点分流断言。"""
from __future__ import annotations

import json

import pytest

from sbmachine import cloud_memory, cloud_prompts, llm_shim
from sbmachine.neutral_contract import new_manifest_metadata


@pytest.fixture(autouse=True)
def _reset_meme_profile(monkeypatch):
    monkeypatch.setattr(cloud_prompts, "_match_meme_profile", "")


@pytest.fixture(autouse=True)
def _clean_sessions():
    cloud_memory.clear_sessions()
    yield
    cloud_memory.clear_sessions()


def _kill_topic(**overrides) -> dict:
    action = {
        "type": "kill_topic",
        "semantic": "plain_kill",
        "priority": 0.4,
        "attacker": "A",
        "victims": ["B"],
        "pov_role": "killer",
        "pov_player": "A",
        "final_kill": False,
        "round_tags": [],
        "rule_tags": [],
    }
    action.update(overrides)
    return action


def _scene(actions=None, rule_state=None, round_won=None) -> dict:
    plan: dict = {"selected_actions": actions or []}
    if rule_state is not None:
        plan["rule_state"] = rule_state
    scene = {"commentary_plan": plan, "hype": 0.5, "fact_anchors": {"players": []}}
    if round_won is not None:
        scene["round_won"] = round_won
    return scene


def _snapshot(t_alive: int, ct_alive: int) -> dict:
    return {"kind": "snapshot", "teams": {"T": {"alive_count": t_alive, "hp_total": t_alive * 100}, "CT": {"alive_count": ct_alive, "hp_total": ct_alive * 100}}, "changed_teams": []}


# ── build_cloud_analyst_system ──

def test_cloud_analyst_system_has_json_contract():
    system = cloud_prompts.build_cloud_analyst_system()
    assert '{"neutral":"..."}' in system
    assert "ONLY source of facts" in system
    assert "Do not infer tactics" in system


# ── build_cloud_style_system ──

def test_cloud_style_system_five_sections_and_persona():
    config = {}
    system = cloud_prompts.build_cloud_style_system(config)
    for section in ("===== 身份 =====", "===== 语言 =====", "===== 事件类型分档 =====", "===== 事实边界 =====", "===== 输出契约 ====="):
        assert section in system
    assert "历史仅供风格与叙事参考" in system or "唯一事实来源" in system
    assert "{persona_hint}" not in system
    assert "{meme_profile_line}" not in system


def test_cloud_style_system_injects_meme_profile():
    config = {}
    system = cloud_prompts.build_cloud_style_system(config, "o系（研发）")
    assert "本场基调：o系（研发）" in system


# ── derive_window_type：判定表逐条 ──

def test_clutch_by_rule_state():
    scene = _scene([_kill_topic(pov_role="killer")], rule_state=_snapshot(1, 2))
    assert cloud_prompts.derive_window_type(scene) == "clutch"


def test_clutch_by_final_kill():
    scene = _scene([_kill_topic(final_kill=True)])
    assert cloud_prompts.derive_window_type(scene) == "clutch"


def test_highlight_multi_kill():
    scene = _scene([_kill_topic(victims=["B", "C"])])
    assert cloud_prompts.derive_window_type(scene) == "highlight"


def test_highlight_semantic_above_threshold():
    scene = _scene([_kill_topic(semantic="collateral", priority=0.9)])
    assert cloud_prompts.derive_window_type(scene) == "highlight"


def test_highlight_match_point_round():
    scene = _scene([_kill_topic(round_tags=["ct_match_point"])])
    assert cloud_prompts.derive_window_type(scene) == "highlight"


def test_plain_kill_stays_flat():
    scene = _scene([_kill_topic()])
    assert cloud_prompts.derive_window_type(scene) == "flat"


def test_fail_when_target_dies_without_target_kill():
    scene = _scene([_kill_topic(pov_role="victim")])
    assert cloud_prompts.derive_window_type(scene) == "fail"


def test_fail_on_food_rule_tag():
    scene = _scene([_kill_topic(rule_tags=["caught_switching"])])
    assert cloud_prompts.derive_window_type(scene) == "fail"


def test_fail_when_round_lost_even_with_kill():
    scene = _scene([_kill_topic(pov_role="victim"), _kill_topic(pov_role="killer", victims=["C"])], round_won=False)
    assert cloud_prompts.derive_window_type(scene) == "fail"


def test_not_fail_when_round_won_with_both_roles():
    scene = _scene([_kill_topic(pov_role="victim"), _kill_topic(pov_role="killer", victims=["C"])], round_won=True)
    assert cloud_prompts.derive_window_type(scene) == "flat"


def test_meme_death_when_fail_and_match_meme():
    cloud_prompts.build_cloud_style_system({}, "o系（研发）")
    scene = _scene([_kill_topic(pov_role="victim")])
    assert cloud_prompts.derive_window_type(scene) == "meme_death"


def test_flat_without_kills():
    scene = _scene([{"type": "utility_throw", "attacker": "A"}])
    assert cloud_prompts.derive_window_type(scene) == "flat"


def test_flat_empty_plan():
    assert cloud_prompts.derive_window_type(_scene([])) == "flat"


def test_observer_kill_stays_flat():
    scene = _scene([_kill_topic(pov_role="observer")])
    assert cloud_prompts.derive_window_type(scene) == "flat"


def test_pov_role_fallback_from_target_player():
    scene = _scene([_kill_topic(pov_role="unavailable", attacker="A", victims=["B"])])
    assert cloud_prompts.derive_window_type(scene, target_player="A") == "flat"
    scene_victim = _scene([_kill_topic(pov_role="unavailable", attacker="B", victims=["A"])])
    assert cloud_prompts.derive_window_type(scene_victim, target_player="A") == "fail"


# ── inject_window_type ──

def test_inject_window_type_appends_label():
    scene = _scene([_kill_topic(pov_role="victim")])
    injected = cloud_prompts.inject_window_type('{"neutral":"x"}', scene)
    assert injected.startswith('{"neutral":"x"}')
    assert "【窗口类型】fail" in injected


# ── compute_match_meme_profile：全表 ──

def _neutral_data(kills: int, deaths: int) -> dict:
    """构造 kills 次 attacker==target 的击杀、deaths 次 victim==target 的死亡。"""
    target = "A"
    actions = []
    remaining_deaths = deaths
    for index in range(max(kills, deaths)):
        attacker = target if index < kills else "B"
        victims = ["A"] if remaining_deaths > 0 else ["B"]
        remaining_deaths -= 1
        actions.append({
            "type": "kill_topic",
            "pov_player": target,
            "attacker": attacker,
            "victims": victims,
        })
    return {
        "rounds": [
            {
                "scenes": [
                    {
                        "commentary_plan": {"selected_actions": actions},
                        "fact_anchors": {"players": [target]},
                    }
                ]
            }
        ]
    }


def test_meme_211():
    assert cloud_prompts.compute_match_meme_profile(_neutral_data(2, 11)) == "211（高材生）"


def test_meme_o_series():
    assert cloud_prompts.compute_match_meme_profile(_neutral_data(0, 5)) == "o系（研发）"


def test_meme_i18():
    assert cloud_prompts.compute_match_meme_profile(_neutral_data(1, 18)) == "i18（典中典）"


def test_meme_i_series():
    assert cloud_prompts.compute_match_meme_profile(_neutral_data(1, 3)) == "i系"


def test_meme_z_series():
    assert cloud_prompts.compute_match_meme_profile(_neutral_data(2, 5)) == "z系（坐牢）"


def test_meme_none():
    assert cloud_prompts.compute_match_meme_profile(_neutral_data(3, 5)) == ""


def test_meme_profile_cached_for_inject():
    assert cloud_prompts.compute_match_meme_profile(_neutral_data(0, 5)) == "o系（研发）"
    scene = _scene([_kill_topic(pov_role="victim")])
    assert cloud_prompts.derive_window_type(scene) == "meme_death"


# ── 接入点分流断言：vllm 分支逐字保留 / api 分支走云端 ──

def _frame(t: float, events: dict | None = None, *, phase: str = "in_round") -> dict:
    data = {"when": {"video_time": t, "relative_sec": t, "phase": phase}, "who": {"view": "player", "pov_player": "p1"}}
    if events:
        data["events"] = events
    return data


def _round_record(round_no: int, frames: list[dict]) -> dict:
    start = float(frames[0]["when"]["video_time"])
    end = float(frames[-1]["when"]["video_time"])
    return {
        "round_no": round_no, "start_sec": start, "end_sec": end,
        "score_before": {"ct": 0, "t": 0}, "score_after": {"ct": 0, "t": 1},
        "demo_round_hint": round_no,
        "_phase2_yolo": {"key_frames": [
            {"time_sec": frame["when"]["video_time"], "gate_reason": "test", "background_info": frame, "has_frame": True}
            for frame in frames
        ]},
    }


def _analyst_fixtures(tmp_path):
    frames = [
        _frame(0.0), _frame(1.0), _frame(2.0), _frame(3.0), _frame(4.0),
        _frame(5.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47"}]}),
        _frame(6.0), _frame(7.0), _frame(8.0), _frame(9.0),
        _frame(10.0, {"c4": {"planted": True}}), _frame(11.0), _frame(12.0),
    ]
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(json.dumps({"video_path": "test.mp4", "map_name": "de_test", "rounds": [_round_record(1, frames)]}, ensure_ascii=False), encoding="utf-8")
    semantic_frames = [_frame(0.0), _frame(4.0), _frame(8.0), _frame(12.0)]
    semantic_frames[0]["where"] = {"players": [{"name": "raw_player", "side": "T", "hp": 100, "weapon": "AK", "callout": "Ramp"}]}
    semantic_frames[2]["events"] = {"kills": [{"attacker": "semantic_source", "victim": "B", "weapon": "AK-47", "tick": 512}]}
    semantic_frames[3]["events"] = {"kills": [{"attacker": "semantic_source", "victim": "C", "weapon": "AK-47", "tick": 613}]}
    semantic_path.write_text(json.dumps([{"round_no": 1, "frames": semantic_frames}], ensure_ascii=False), encoding="utf-8")
    return rounds_path, output_path, config_path, semantic_path


def test_vllm_analyst_branch_never_uses_cloud_system(tmp_path, monkeypatch):
    """backend=vllm 时 3a system 仍走共用 _build_analyst_system，云端拼装不得触发。"""
    from sbmachine.phase3a_analyst import run_phase3a
    rounds_path, output_path, config_path, semantic_path = _analyst_fixtures(tmp_path)
    config_path.write_text(
        "llm:\n  backend: vllm\nsemantic:\n  analyst_backend: vllm\npaths:\n  rounds_with_yolo_semantic_json: " + f'"{semantic_path.as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cloud_prompts, "build_cloud_analyst_system", lambda: (_ for _ in ()).throw(AssertionError("cloud system must not be used for vllm backend")))
    report = run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path, dry_run=True)
    assert report["mode"] == "phase3a_dry_run"


def test_vllm_style_branch_keeps_original_assembly(tmp_path, monkeypatch):
    """backend=vllm 时 3b system 拼装逐字保留（style_system + cs_rules + skill），无窗口类型注入。"""
    from sbmachine.phase3b_style import run_phase3b, _PROJECT_ROOT
    from sbmachine.phase3b_prompt import _load_persona
    from core.prompt_loader import load_prompt
    rounds_path = tmp_path / "rounds_with_yolo.json"
    rounds_path.write_text(json.dumps({"video_path": "match.mp4", "map_name": "de_test", "rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 2.0}]}), encoding="utf-8")
    neutral_path = tmp_path / "rounds_with_neutral.json"
    neutral_path.write_text(json.dumps({
        **new_manifest_metadata(rounds_path),
        "rounds": [{"round_no": 1, "avg_hype": 0.0, "analyst_failed": False, "scenes": [{
            "t_start": 0.0, "t_end": 2.0, "scene": "default", "commentary_plan": {},
            "fact_anchors": {"players": ["hypex"], "teams": [], "numbers": [], "events": [], "results": [], "locations": ["A点"], "weapons": ["AK"]},
            "neutral": "hypex在A点使用AK击杀对手。", "neutral_source": "llm", "generation_status": "success",
            "hype": 0.0, "char_budget": 20,
        }]}],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  backend: vllm\nsemantic:\n  style_backend: vllm\n", encoding="utf-8")

    captured = {}
    import sbmachine.llmb_api as llmb_api
    monkeypatch.setattr(llmb_api, "generate", lambda prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs: captured.update(system=system_prompt, user=prompt) or llm_shim._ApiChatResult(
        json.dumps({"commentary": "[平述]hypex在A点用AK击杀对手。", "felt_intensity": 0.2}, ensure_ascii=False),
        scope="llmb", source_run_id="vllm-ok", request_payload={}, log_ctx=log_ctx,
    ))
    monkeypatch.setattr(cloud_prompts, "build_cloud_style_system", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cloud style system must not be used for vllm backend")))
    monkeypatch.setattr(cloud_prompts, "inject_window_type", lambda *a, **k: (_ for _ in ()).throw(AssertionError("window type injection must not run for vllm backend")))

    run_phase3b(neutral_path=neutral_path, rounds_path=rounds_path, output_rounds_path=tmp_path / "commentary_rounds.json", commentary_path=tmp_path / "commentary.json", config_path=config_path)

    cs_rules_path = _PROJECT_ROOT / "Prompt" / "cs_rules.txt"
    cs_rules = cs_rules_path.read_text(encoding="utf-8").strip() if cs_rules_path.exists() else ""
    expected = "\n\n".join(filter(None, [load_prompt("style_system").replace("{persona_hint}", _load_persona()), cs_rules, ""]))
    assert captured["system"] == expected
    assert "【窗口类型】" not in captured["user"]


def test_cloud_style_branch_uses_cloud_system_and_window_type(tmp_path, monkeypatch):
    """backend=api 时 3b 用云端五段 system + user prompt 注入窗口类型，不拼共用 skill。"""
    from sbmachine.phase3b_style import run_phase3b
    rounds_path = tmp_path / "rounds_with_yolo.json"
    rounds_path.write_text(json.dumps({"video_path": "match.mp4", "map_name": "de_test", "rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 2.0}]}), encoding="utf-8")
    neutral_path = tmp_path / "rounds_with_neutral.json"
    neutral_path.write_text(json.dumps({
        **new_manifest_metadata(rounds_path),
        "rounds": [{"round_no": 1, "avg_hype": 0.0, "analyst_failed": False, "scenes": [{
            "t_start": 0.0, "t_end": 2.0, "scene": "default", "commentary_plan": {},
            "fact_anchors": {"players": ["hypex"], "teams": [], "numbers": [], "events": [], "results": [], "locations": ["A点"], "weapons": ["AK"]},
            "neutral": "hypex在A点使用AK击杀对手。", "neutral_source": "llm", "generation_status": "success",
            "hype": 0.0, "char_budget": 20,
        }]}],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  backend: vllm\nsemantic:\n  style_backend: api\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_cloud_execute(calls, scope="llmb", commentary="[平述]hypex在A点用AK击杀对手。"))

    run_phase3b(neutral_path=neutral_path, rounds_path=rounds_path, output_rounds_path=tmp_path / "commentary_rounds.json", commentary_path=tmp_path / "commentary.json", config_path=config_path)

    assert calls
    system = calls[-1]["messages"][0]["content"]
    assert "===== 身份 =====" in system
    assert "===== 事件类型分档 =====" in system
    assert "allowed_event_terms" not in system  # 云端不拼共用输入契约段落
    assert "【窗口类型】" in calls[-1]["messages"][-1]["content"]
    assert calls[-1]["secret_scope"] == "llmb"


def test_cloud_style_branch_sends_full_cloud_max_tokens_budget(tmp_path, monkeypatch):
    """云端 3b：max_tokens 直接放开为 cloud_style_output_max_tokens，不被字数公式截断。

    回归：style_runtime_config 曾因 STYLE_DEFAULTS key 过滤丢失 cloud_style_output_max_tokens，
    导致 phase3b 实际只发 1024（思考吃满截断 → unparseable）。"""
    from sbmachine.phase3b_style import run_phase3b
    rounds_path = tmp_path / "rounds_with_yolo.json"
    rounds_path.write_text(json.dumps({"video_path": "match.mp4", "map_name": "de_test", "rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 2.0}]}), encoding="utf-8")
    neutral_path = tmp_path / "rounds_with_neutral.json"
    neutral_path.write_text(json.dumps({
        **new_manifest_metadata(rounds_path),
        "rounds": [{"round_no": 1, "avg_hype": 0.0, "analyst_failed": False, "scenes": [{
            "t_start": 0.0, "t_end": 2.0, "scene": "default", "commentary_plan": {},
            "fact_anchors": {"players": ["hypex"], "teams": [], "numbers": [], "events": [], "results": [], "locations": ["A点"], "weapons": ["AK"]},
            "neutral": "hypex在A点使用AK击杀对手。", "neutral_source": "llm", "generation_status": "success",
            "hype": 0.0, "char_budget": 20,
        }]}],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n  backend: vllm\nsemantic:\n  style_backend: api\n  cloud_style_output_max_tokens: 4096\n",
        encoding="utf-8",
    )

    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_cloud_execute(calls, scope="llmb", commentary="[平述]hypex在A点用AK击杀对手。"))

    run_phase3b(neutral_path=neutral_path, rounds_path=rounds_path, output_rounds_path=tmp_path / "commentary_rounds.json", commentary_path=tmp_path / "commentary.json", config_path=config_path)

    assert calls
    assert all(call["max_tokens"] == 4096 for call in calls)


def test_cloud_analyst_branch_uses_cloud_system_and_is_stateless(tmp_path, monkeypatch):
    """backend=api 时 3a 用云端 system；3a 无上下文需求，走无状态路径（不建会话、无历史）。"""
    from sbmachine.phase3a_analyst import run_phase3a
    rounds_path, output_path, config_path, semantic_path = _analyst_fixtures(tmp_path)
    config_path.write_text(
        "llm:\n  backend: vllm\nsemantic:\n  analyst_backend: api\n  analyst_output_max_tokens: 256\n  window_max_sec: 10\n  window_min_sec: 3\npaths:\n  rounds_with_yolo_semantic_json: " + f'"{semantic_path.as_posix()}"\n',
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_cloud_execute(calls, scope="llma", neutral_from_projection=True))

    manifest = run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    assert calls
    system = calls[-1]["messages"][0]["content"]
    assert "ONLY source of facts" in system
    assert calls[0]["messages"][0]["content"] == cloud_prompts.build_cloud_analyst_system()
    # 3a 无会话：每条请求消息恒为 [system, user]，无 assistant 历史轮
    for call in calls:
        assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert cloud_memory._sessions == {}  # 不建会话、无锁竞争
    assert calls[-1]["secret_scope"] == "llma"
    assert manifest["rounds"][0]["scenes"]


def test_cloud_analyst_branch_sends_cloud_max_tokens_budget(tmp_path, monkeypatch):
    """云端 3a：llm_cfg.max_tokens 直接取 cloud_analyst_output_max_tokens，不被本地 256 截断。"""
    from sbmachine.phase3a_analyst import run_phase3a
    rounds_path, output_path, config_path, semantic_path = _analyst_fixtures(tmp_path)
    config_path.write_text(
        "llm:\n  backend: vllm\nsemantic:\n  analyst_backend: api\n  analyst_output_max_tokens: 256\n  cloud_analyst_output_max_tokens: 4096\n  window_max_sec: 10\n  window_min_sec: 3\npaths:\n  rounds_with_yolo_semantic_json: " + f'"{semantic_path.as_posix()}"\n',
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", _fake_cloud_execute(calls, scope="llma", neutral_from_projection=True))

    run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    assert calls
    assert all(call["max_tokens"] == 4096 for call in calls)


def test_budget_silence_is_expected_blank_not_retried(tmp_path, monkeypatch):
    """预算静默（budget_silence）是成本护栏的预期留白：不判 contract_error、不触发重试。"""
    from sbmachine.phase3a_analyst import run_phase3a
    rounds_path, output_path, config_path, semantic_path = _analyst_fixtures(tmp_path)
    config_path.write_text(
        "llm:\n  backend: vllm\nsemantic:\n  analyst_backend: api\n  analyst_output_max_tokens: 256\n  window_max_sec: 10\n  window_min_sec: 3\npaths:\n  rounds_with_yolo_semantic_json: " + f'"{semantic_path.as_posix()}"\n',
        encoding="utf-8",
    )

    def silence_execute(messages, llm_cfg, max_tokens=None, log_ctx=None, secret_scope=None, response_format=None):
        calls.append(1)
        return llm_shim._ApiChatResult(
            '{"neutral": ""}', scope=secret_scope, source_run_id="s", request_payload={"messages": messages},
            log_ctx=log_ctx, raw_response={"choices": [{"message": {"content": '{"neutral": ""}'}}]},
            finish_reason="stop", http_status=200, usage={"total_tokens": 0}, budget_silence=True,
        )

    calls = []
    monkeypatch.setattr(cloud_memory, "_execute_openai_chat", silence_execute)
    manifest = run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    assert len(calls) == 1  # 预算静默不重试
    scene = manifest["rounds"][0]["scenes"][0]
    assert scene["neutral_source"] == "intentional_empty"
    assert scene["retry_count"] == 0


def _fake_cloud_execute(calls: list, *, scope: str, commentary: str = "", neutral_from_projection: bool = False):
    def fake(messages, llm_cfg, max_tokens=None, log_ctx=None, secret_scope=None, response_format=None):
        calls.append({"messages": messages, "max_tokens": max_tokens, "secret_scope": secret_scope})
        if scope == "llmb":
            content = json.dumps({"commentary": commentary, "felt_intensity": 0.5}, ensure_ascii=False)
        else:
            if neutral_from_projection:
                projection = json.loads(messages[-1]["content"].strip().splitlines()[-1])
                neutral = "，".join(fact["canonical_text"] for fact in projection["required_facts"] if fact.get("required") is True)
                content = json.dumps({"neutral": neutral}, ensure_ascii=False)
            else:
                content = json.dumps({"neutral": "测试中性稿。"}, ensure_ascii=False)
        return llm_shim._ApiChatResult(content, scope=secret_scope, source_run_id="cloud-ok", request_payload={"messages": messages}, log_ctx=log_ctx, raw_response={"choices": [{"message": {"content": content}}]}, finish_reason="stop", http_status=200, usage={"total_tokens": 100})
    return fake
