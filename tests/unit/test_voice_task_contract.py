"""voice_task_contract 纯结构校验器单元测试（计划书 §19 文件清单）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sbmachine.voice_task_contract import (
    CANDIDATE_POLICY_SPARSE_V1,
    MAX_CANDIDATES_PER_SCENE,
    SCHEMA_COMMENTARY_V3,
    SCHEMA_NEUTRAL_V4,
    SPEECH_METRIC_UNITS_V1,
    VOICE_TASK_CONTRACT_VERSION,
    validate_commentary_v3,
    validate_final_voice_task,
    validate_neutral_v4,
)


def load_fixture(name: str) -> dict:
    root = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    return json.loads((root / name).read_text(encoding="utf-8"))


def non_silent_scenes(payload: dict) -> list[dict]:
    return [
        scene
        for round_data in payload["rounds"]
        for scene in round_data["scenes"]
        if scene.get("neutral")
    ]


def silent_scenes(payload: dict) -> list[dict]:
    return [
        scene
        for round_data in payload["rounds"]
        for scene in round_data["scenes"]
        if not scene.get("neutral")
    ]


NEUTRAL_V4 = "phase3/neutral_v4.json"
COMMENTARY_V3 = "phase3/commentary_v3.json"
FINAL_VOICE_TASK = "phase4/final_voice_task.json"


class TestValidateNeutralV4:
    def test_fixture_passes(self):
        assert validate_neutral_v4(load_fixture(NEUTRAL_V4)) == []

    def test_fixture_has_full_non_silent_scene(self):
        scenes = non_silent_scenes(load_fixture(NEUTRAL_V4))
        assert scenes, "fixture must contain a non-silent scene exercising the full v4 shape"
        for key in ("neutral_source", "neutral_renderer", "rule_capsule", "fact_catalog", "required_fact_ids", "render_slot", "speech_budget"):
            assert key in scenes[0], f"non-silent scene is missing {key}"

    def test_wrong_schema_version(self):
        payload = load_fixture(NEUTRAL_V4)
        payload["schema_version"] = 3
        assert validate_neutral_v4(payload)

    def test_missing_rule_capsule(self):
        payload = load_fixture(NEUTRAL_V4)
        scene = non_silent_scenes(payload)[0]
        del scene["rule_capsule"]
        assert any("rule_capsule" in e for e in validate_neutral_v4(payload))

    def test_missing_fact_catalog(self):
        payload = load_fixture(NEUTRAL_V4)
        scene = non_silent_scenes(payload)[0]
        del scene["fact_catalog"]
        assert any("fact_catalog" in e for e in validate_neutral_v4(payload))

    def test_bad_fact_id_format(self):
        payload = load_fixture(NEUTRAL_V4)
        scene = non_silent_scenes(payload)[0]
        scene["fact_catalog"][0]["fact_id"] = "topic:kill"
        assert any("fact_id" in e for e in validate_neutral_v4(payload))

    def test_required_ids_not_in_catalog(self):
        payload = load_fixture(NEUTRAL_V4)
        scene = non_silent_scenes(payload)[0]
        scene["required_fact_ids"] = ["fact:v1:r001_w03:kill:00360:ffffffff"]
        assert any("required_fact_ids" in e for e in validate_neutral_v4(payload))

    def test_slot_tick_out_of_range(self):
        payload = load_fixture(NEUTRAL_V4)
        scene = non_silent_scenes(payload)[0]
        scene["render_slot"]["end_tick"] = scene["render_slot"]["start_tick"] + 99999
        assert any("render_slot" in e for e in validate_neutral_v4(payload))

    def test_missing_speech_budget(self):
        payload = load_fixture(NEUTRAL_V4)
        scene = non_silent_scenes(payload)[0]
        del scene["speech_budget"]
        assert any("speech_budget" in e for e in validate_neutral_v4(payload))

    def test_silent_scene_exempt(self):
        payload = load_fixture(NEUTRAL_V4)
        silent = silent_scenes(payload)
        assert silent, "fixture must contain a silent scene"
        assert validate_neutral_v4(payload) == []


class TestValidateCommentaryV3:
    def test_fixture_passes(self):
        assert validate_commentary_v3(load_fixture(COMMENTARY_V3)) == []

    def test_wrong_schema_version(self):
        payload = load_fixture(COMMENTARY_V3)
        payload["commentary_schema_version"] = 2
        assert validate_commentary_v3(payload)

    def test_missing_contract_version(self):
        payload = load_fixture(COMMENTARY_V3)
        del payload["voice_task_contract_version"]
        assert any("voice_task_contract_version" in e for e in validate_commentary_v3(payload))

    def test_green_contains_compact_rejected(self):
        payload = load_fixture(COMMENTARY_V3)
        task = next(t for t in payload["voice_tasks"] if t["risk_class"] == "green")
        compact = dict(task["candidates"][0], variant_id="compact", source="llmb_compact", text="紧凑稿")
        task["candidates"].append(compact)
        assert any("compact" in e for e in validate_commentary_v3(payload))

    def test_red_contains_primary_rejected(self):
        payload = load_fixture(COMMENTARY_V3)
        red = {
            "voice_task_id": "r999_w01",
            "window_id": "r999_w01",
            "render_slot": {"start_sec": 0.0, "end_sec": 3.0, "start_tick": 0, "end_tick": 90, "gap_policy": "independent_window"},
            "required_fact_ids": ["fact:v1:r999_w01:kill:00010:a1b2c3d4"],
            "speech_profile_id": "speech-profile-v1",
            "risk_class": "red",
            "selection_order": ["compact", "capsule"],
            "max_speed_factor": 1.5,
            "candidates": [
                {"variant_id": "primary", "source": "llmb", "text": "主稿", "felt_intensity": 0.5,
                 "spoken_units": 10, "safe_duration_upper_bound_at_base_speed_sec": 2.0,
                 "minimum_required_speed_factor": 1.0, "preserved_fact_ids": ["fact:v1:r999_w01:kill:00010:a1b2c3d4"]},
                {"variant_id": "capsule", "source": "rule_capsule", "text": "保底", "felt_intensity": 0.45,
                 "spoken_units": 6, "safe_duration_upper_bound_at_base_speed_sec": 1.2,
                 "minimum_required_speed_factor": 1.0, "preserved_fact_ids": ["fact:v1:r999_w01:kill:00010:a1b2c3d4"]},
            ],
            "semantic_state": "ok",
        }
        payload["voice_tasks"].append(red)
        assert any("red" in e and "primary" in e for e in validate_commentary_v3(payload))

    def test_preserved_missing_required_rejected(self):
        payload = load_fixture(COMMENTARY_V3)
        task = next(t for t in payload["voice_tasks"] if t["risk_class"] == "amber")
        task["candidates"][0]["preserved_fact_ids"] = []
        assert any("preserved_fact_ids" in e for e in validate_commentary_v3(payload))

    def test_too_many_candidates_rejected(self):
        payload = load_fixture(COMMENTARY_V3)
        task = next(t for t in payload["voice_tasks"] if t["risk_class"] == "amber")
        extra = dict(task["candidates"][0], variant_id="primary")
        for _ in range(MAX_CANDIDATES_PER_SCENE + 1):
            task["candidates"].append(dict(extra))
        assert any("candidates" in e for e in validate_commentary_v3(payload))

    def test_capsule_source_must_be_rule(self):
        payload = load_fixture(COMMENTARY_V3)
        task = next(t for t in payload["voice_tasks"] if t["risk_class"] == "amber")
        capsule = next(c for c in task["candidates"] if c["variant_id"] == "capsule")
        capsule["source"] = "llmb"
        assert any("rule_capsule" in e for e in validate_commentary_v3(payload))


class TestValidateFinalVoiceTask:
    def test_fixture_passes(self):
        payload = load_fixture(FINAL_VOICE_TASK)
        assert validate_final_voice_task(payload) == []

    def test_fixture_shape(self):
        payload = load_fixture(FINAL_VOICE_TASK)
        assert "rounds_final" in payload
        rounds = payload["rounds_final"]["rounds"]
        assert any(scene.get("voice_task_id") for r in rounds for scene in r["scenes"])

    def test_selected_variant_not_in_contract(self):
        payload = load_fixture(FINAL_VOICE_TASK)
        commentary = load_fixture(COMMENTARY_V3)
        scene = next(
            s for r in payload["rounds_final"]["rounds"] for s in r["scenes"]
            if s.get("fit_state") == "fit"
        )
        scene["selected_variant_id"] = "bogus"
        assert any("selected_variant_id" in e for e in validate_final_voice_task(payload, commentary))

    def test_fit_missing_fields(self):
        payload = load_fixture(FINAL_VOICE_TASK)
        scene = next(
            s for r in payload["rounds_final"]["rounds"] for s in r["scenes"]
            if s.get("fit_state") == "fit"
        )
        del scene["actual_duration_sec"]
        assert any("actual_duration_sec" in e for e in validate_final_voice_task(payload))

    def test_audio_tick_inverted(self):
        payload = load_fixture(FINAL_VOICE_TASK)
        scene = next(
            s for r in payload["rounds_final"]["rounds"] for s in r["scenes"]
            if s.get("fit_state") == "fit"
        )
        scene["audio_start_tick"], scene["audio_end_tick"] = scene["audio_end_tick"], scene["audio_start_tick"]
        assert any("audio" in e for e in validate_final_voice_task(payload))


class TestConstants:
    def test_versions(self):
        assert SCHEMA_NEUTRAL_V4 == 4
        assert SCHEMA_COMMENTARY_V3 == 3
        assert VOICE_TASK_CONTRACT_VERSION == 1
        assert CANDIDATE_POLICY_SPARSE_V1 == "sparse_v1"
        assert SPEECH_METRIC_UNITS_V1 == "speech_units_v1"

    def test_fixtures_are_strict_json(self):
        for name in (NEUTRAL_V4, COMMENTARY_V3, FINAL_VOICE_TASK):
            assert isinstance(load_fixture(name), dict)
