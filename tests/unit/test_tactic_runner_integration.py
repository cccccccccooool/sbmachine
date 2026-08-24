import json

import pytest

from sbmachine import phase3a_analyst, phase3a_cloud_runner
from sbmachine.phase3a_analyst import run_phase3a
from sbmachine.scene_context import extract_actions
from sbmachine.tactic_book import compile_tactic_book, load_tactic_book
from sbmachine.tactic_matcher import match_window


_MAP_NAME = "de_tactic_fixture"


def _player(name: str, side: str, callout: str) -> dict:
    return {"name": name, "side": side, "hp": 100, "weapon": "AK-47", "callout": callout}


def _frame(t: float, players: list[dict], utilities: list[dict] | None = None) -> dict:
    return {
        "when": {"video_time": t, "relative_sec": t, "phase": "in_round"},
        "who": {"view": "player", "pov_player": players[0]["name"] if players else None},
        "where": {"players": players},
        "events": {"utilities": utilities or []},
    }


def _fake_a_case() -> dict:
    players = [
        _player("a_lurker", "T", "A_Short"),
        _player("b1", "T", "B_Main"),
        _player("b2", "T", "B_Main"),
        _player("b3", "T", "B_Main"),
    ]
    return {
        "rule_id": "fake_a_hit_b",
        "label": "假爆A真打B",
        "hint": "A小一人道具牵制，B区三人集结。",
        "matched_at": 2.0,
        "tactic": {
            "id": "fake_a_hit_b",
            "label": "假爆A真打B",
            "hint": "A小一人道具牵制，B区三人集结。",
            "side": "T",
            "when": [
                {"kind": "zone_count", "side": "T", "zone": {"callouts_any": ["A_Short"]}, "count": [1, 1]},
                {"kind": "zone_count", "side": "T", "zone": {"callouts_any": ["B_Main"]}, "count": [3, 5]},
                {
                    "kind": "event_count",
                    "event": "utility_throw",
                    "actor_side": "T",
                    "actor_zone": {"callouts_any": ["A_Short"]},
                    "types_any": ["Smoke Grenade", "Flashbang"],
                    "window_sec": 6,
                    "count": [2, None],
                },
            ],
            "priority": 10,
        },
        "frames": [
            _frame(0.0, players, [{"_event": "throw", "entity_id": 1, "throw_tick": 10, "thrower": "a_lurker", "type": "Smoke Grenade"}]),
            _frame(2.0, players, [{"_event": "throw", "entity_id": 2, "throw_tick": 20, "thrower": "a_lurker", "type": "Flashbang"}]),
            _frame(3.0, players),
        ],
    }


def _mid_stack_case() -> dict:
    players = [_player(f"t{index}", "T", "Mid") for index in range(4)]
    return {
        "rule_id": "t_mid_stack_retake",
        "label": "中路摆谱中期反清",
        "hint": "中路摆谱中期反清",
        "matched_at": 0.0,
        "tactic": {
            "id": "t_mid_stack_retake",
            "label": "中路摆谱中期反清",
            "side": "T",
            "when": [
                {
                    "kind": "zone_count",
                    "side": "T",
                    "zone": {"callouts_any": ["Mid", "TopMid", "BottomMid"]},
                    "count": [4, 5],
                }
            ],
            "priority": 5,
        },
        "frames": [_frame(0.0, players), _frame(3.0, players)],
    }


def _write_runner_inputs(tmp_path, frames: list[dict]) -> tuple:
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps(
            {
                "video_path": "fixture.mp4",
                "map_name": _MAP_NAME,
                "rounds": [{
                    "round_no": 1,
                    "start_sec": 0.0,
                    "end_sec": 6.0,
                    "score_before": {"ct": 0, "t": 0},
                    "score_after": {"ct": 0, "t": 1},
                    "demo_round_hint": 1,
                    "_phase2_yolo": {"key_frames": [
                        {
                            "time_sec": frame["when"]["video_time"],
                            "gate_reason": "test",
                            "background_info": frame,
                            "has_frame": True,
                        }
                        for frame in frames
                    ]},
                }],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    semantic_path.write_text(json.dumps([{"round_no": 1, "frames": frames}], ensure_ascii=False), encoding="utf-8")
    config_path.write_text(
        """
llm:
  backend: vllm
semantic:
  analyst_backend: vllm
  analyst_model: fixture
  analyst_output_max_tokens: 64
  analyst_concurrent_rounds: 1
  cloud_analyst_max_tokens: 64
  window_max_sec: 10
  window_min_sec: 3
""",
        encoding="utf-8",
    )
    return rounds_path, config_path


def _cloud_payload_from_prompt(prompt: str) -> dict:
    prefix = "Rule-layer projections only:\n"
    suffix = "\nReturn exactly {\"window_id\": string|null, \"neutral\": string}."
    assert prompt.startswith(prefix)
    assert prompt.endswith(suffix)
    return json.loads(prompt[len(prefix):-len(suffix)])


def _local_projection_from_prompt(prompt: str) -> dict:
    return json.loads(prompt.strip().splitlines()[-1])


def _required_neutral(projection: dict) -> str:
    return "，".join(
        fact["canonical_text"]
        for fact in projection["required_facts"]
        if fact.get("required") is True
    )


@pytest.mark.parametrize("case_factory", [_fake_a_case, _mid_stack_case], ids=["fake_a_hit_b", "t_mid_stack_retake"])
def test_disk_tactic_book_reaches_local_and_cloud_runners_with_the_same_hint(tmp_path, monkeypatch, case_factory):
    case = case_factory()
    database_root = tmp_path / "database"
    tactics_dir = database_root / "tactics"
    tactics_dir.mkdir(parents=True)
    (tactics_dir / f"{_MAP_NAME}.json").write_text(
        json.dumps({"version": 1, "map": _MAP_NAME, "tactics": [case["tactic"]]}, ensure_ascii=False),
        encoding="utf-8",
    )
    rounds_path, config_path = _write_runner_inputs(tmp_path, case["frames"])

    loaded = load_tactic_book(_MAP_NAME, database_root=database_root)
    assert [tactic.rule_id for tactic in loaded.tactics] == [case["rule_id"]]

    load_calls: list[str] = []

    def load_fixture_book(map_name: str):
        load_calls.append(map_name)
        return load_tactic_book(map_name, database_root=database_root)

    monkeypatch.setattr(phase3a_analyst, "load_tactic_book", load_fixture_book)
    monkeypatch.setattr(phase3a_cloud_runner, "load_tactic_book", load_fixture_book)

    local_prompts: list[str] = []

    def fake_local_generate(prompt, llm_cfg, **kwargs):
        local_prompts.append(prompt)
        return json.dumps(
            {"neutral": _required_neutral(_local_projection_from_prompt(prompt))},
            ensure_ascii=False,
        )

    import sbmachine.llma_api as llma_api

    monkeypatch.setattr(llma_api, "generate", fake_local_generate)
    local_manifest = run_phase3a(
        rounds_path=rounds_path,
        output_path=tmp_path / "local_neutral.json",
        config_path=config_path,
    )
    local_hint = local_manifest["rounds"][0]["scenes"][0]["commentary_plan"]["tactic_hint"]

    cloud_prompts: list[str] = []

    def fake_cloud_generate(prompt, llm_cfg, **kwargs):
        cloud_prompts.append(prompt)
        projection = _cloud_payload_from_prompt(prompt)["windows"][0]
        return json.dumps(
            {"window_id": "window-1", "neutral": _required_neutral(projection)},
            ensure_ascii=False,
        )

    monkeypatch.setattr(phase3a_cloud_runner, "generate_cloud_round", fake_cloud_generate)
    cloud_manifest = phase3a_cloud_runner.run_cloud_phase3a(
        rounds_path=rounds_path,
        output_path=tmp_path / "cloud_neutral.json",
        config_path=config_path,
    )
    cloud_hint = cloud_manifest["rounds"][0]["scenes"][0]["commentary_plan"]["tactic_hint"]
    cloud_payload = _cloud_payload_from_prompt(cloud_prompts[0])

    expected_hint = {
        "rule_id": case["rule_id"],
        "label": case["label"],
        "hint": case["hint"],
        "matched_at": case["matched_at"],
    }
    assert load_calls == [_MAP_NAME, _MAP_NAME]
    assert len(local_prompts) == 1
    # S3 A4: tactic hint 已移入 JSON 投影，不再有独立文本块。
    assert "tactic_hint" in local_prompts[0]
    assert case["label"] in local_prompts[0]
    assert local_hint == cloud_hint == expected_hint
    assert cloud_payload["windows"][0]["tactic_hint"] == {
        "rule_id": case["rule_id"],
        "label": case["label"],
        "hint": case["hint"],
    }
    assert "matched_at" not in cloud_payload["windows"][0]["tactic_hint"]


def test_he_throw_matches_tactic_without_expanding_scene_action_whitelist():
    book = compile_tactic_book(
        _MAP_NAME,
        {
            "version": 1,
            "map": _MAP_NAME,
            "tactics": [{
                "id": "he_probe",
                "label": "HE探点",
                "side": "T",
                "when": [{
                    "kind": "event_count",
                    "event": "utility_throw",
                    "actor_side": "T",
                    "types_any": ["HE Grenade"],
                    "window_sec": 3,
                    "count": [1, None],
                }],
                "priority": 1,
            }],
        },
    )
    frames = [_frame(1.0, [_player("t1", "T", "Ramp")], [
        {"_event": "throw", "entity_id": 11, "throw_tick": 100, "thrower": "t1", "type": "HE Grenade"},
        {"_event": "throw", "entity_id": 12, "throw_tick": 101, "thrower": "t1", "type": "Smoke Grenade"},
    ])]

    match = match_window(book, frames, context_frames=frames)
    public_utilities = [
        action["utility"] for action in extract_actions(frames, 1.0, 2.0)
        if action.get("type") == "utility_throw"
    ]

    assert match is not None
    assert match.rule_id == "he_probe"
    assert public_utilities == ["Smoke Grenade"]
