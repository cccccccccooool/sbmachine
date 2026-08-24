"""preflight v3/v4 契约接入与统一语音计量（count_spoken_chars）的单元测试。"""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import pytest

from sbmachine.common import count_spoken_chars
from sbmachine.preflight import (
    PublishContractError,
    validate_commentary_publishable,
    validate_final_manifest,
    validate_neutral_v4_publishable,
    validate_phase4_publishable,
)

SPOKEN_TEXT = "JDC击杀Tauson，CT拿下回合"


def _write_json(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _load(name: str) -> dict:
    root = Path(__file__).resolve().parents[1] / "fixtures"
    return json.loads((root / name).read_text(encoding="utf-8"))


def _neutral_v4_payload(load_fixture) -> dict:
    return copy.deepcopy(load_fixture("phase3/neutral_v4.json"))


def _active_v4_scene(payload: dict) -> dict:
    return next(scene for scene in payload["rounds"][0]["scenes"] if scene.get("neutral"))


def _write_neutral_v4(tmp_path: Path, payload: dict) -> Path:
    return _write_json(tmp_path, "rounds_with_neutral.json", payload)


# ─────────────────────────── B. neutral v4 ───────────────────────────


class TestNeutralV4Publishable:
    def test_fixture_passes(self, tmp_path, load_fixture):
        path = _write_neutral_v4(tmp_path, _neutral_v4_payload(load_fixture))
        validate_neutral_v4_publishable(path)

    def test_silent_scene_is_exempt_from_v4_fields(self, tmp_path):
        payload = {
            "schema_version": 4,
            "phase3a_mode": "rule_neutral_renderer",
            "run_id": uuid.uuid4().hex,
            "source_rounds_sha256": "fixture-only-sha256",
            "speech_metric_version": "speech_units_v1",
            "rounds": [
                {
                    "round_no": 1,
                    "scenes": [
                        {
                            "window_id": "r001_w01",
                            "t_start": 0.0,
                            "t_end": 5.0,
                            "neutral": "",
                            "neutral_source": "intentional_empty",
                        }
                    ],
                }
            ],
        }
        path = _write_neutral_v4(tmp_path, payload)
        validate_neutral_v4_publishable(path)

    def test_non_silent_scene_missing_v4_fields_is_rejected(self, tmp_path, load_fixture):
        payload = _neutral_v4_payload(load_fixture)
        del _active_v4_scene(payload)["rule_capsule"]
        path = _write_neutral_v4(tmp_path, payload)
        with pytest.raises(PublishContractError, match="rule_capsule"):
            validate_neutral_v4_publishable(path)

    def test_missing_fact_catalog_is_rejected(self, tmp_path, load_fixture):
        payload = _neutral_v4_payload(load_fixture)
        del _active_v4_scene(payload)["fact_catalog"]
        path = _write_neutral_v4(tmp_path, payload)
        with pytest.raises(PublishContractError, match="fact_catalog"):
            validate_neutral_v4_publishable(path)

    def test_missing_render_slot_is_rejected(self, tmp_path, load_fixture):
        payload = _neutral_v4_payload(load_fixture)
        del _active_v4_scene(payload)["render_slot"]
        path = _write_neutral_v4(tmp_path, payload)
        with pytest.raises(PublishContractError, match="render_slot"):
            validate_neutral_v4_publishable(path)

    def test_missing_required_fact_ids_is_rejected(self, tmp_path, load_fixture):
        payload = _neutral_v4_payload(load_fixture)
        del _active_v4_scene(payload)["required_fact_ids"]
        path = _write_neutral_v4(tmp_path, payload)
        with pytest.raises(PublishContractError, match="required_fact_ids"):
            validate_neutral_v4_publishable(path)

    def test_non_empty_neutral_without_source_is_rejected(self, tmp_path):
        payload = {
            "schema_version": 4,
            "phase3a_mode": "rule_neutral_renderer",
            "run_id": uuid.uuid4().hex,
            "source_rounds_sha256": "fixture-only-sha256",
            "rounds": [
                {
                    "round_no": 1,
                    "scenes": [
                        {
                            "window_id": "r001_w01",
                            "t_start": 0.0,
                            "t_end": 5.0,
                            "neutral": "有正文但不能豁免",
                        }
                    ],
                }
            ],
        }
        path = _write_neutral_v4(tmp_path, payload)
        with pytest.raises(PublishContractError, match="neutral_source"):
            validate_neutral_v4_publishable(path)

    def test_v3_delegates_to_existing_legacy_path(self, tmp_path):
        payload = {
            "schema_version": 3,
            "phase3a_mode": "llma_slicer_then_llma_analyze",
            "run_id": uuid.uuid4().hex,
            "source_rounds_sha256": "0" * 64,
            "rounds": [
                {
                    "round_no": 1,
                    "analyst_failed": False,
                    "scenes": [
                        {
                            "window_id": "r001_w01",
                            "t_start": 0.0,
                            "t_end": 5.0,
                            "neutral": "正常中性稿",
                            "neutral_source": "llm",
                            "generation_status": "success",
                        }
                    ],
                }
            ],
        }
        path = _write_neutral_v4(tmp_path, payload)
        validate_neutral_v4_publishable(path)

    def test_unsupported_schema_version_is_rejected(self, tmp_path):
        payload = _load("phase3/neutral_v4.json")
        payload["schema_version"] = 5
        path = _write_neutral_v4(tmp_path, payload)
        with pytest.raises(PublishContractError, match="unsupported neutral schema_version"):
            validate_neutral_v4_publishable(path)

    def test_missing_schema_version_is_rejected(self, tmp_path):
        payload = {
            "run_id": uuid.uuid4().hex,
            "rounds": [],
        }
        path = _write_neutral_v4(tmp_path, payload)
        with pytest.raises(PublishContractError, match="unsupported neutral schema_version"):
            validate_neutral_v4_publishable(path)


# ─────────────────────────── C. commentary v3 ───────────────────────────


def _rounds_with_commentary_v3() -> dict:
    commentary = _load("phase3/commentary_v3.json")
    scenes = []
    for task in commentary["voice_tasks"]:
        primary = next(c for c in task["candidates"] if c["variant_id"] == "primary")
        scenes.append(
            {
                "window_id": task["window_id"],
                "voice_task_id": task["voice_task_id"],
                "t_start": task["render_slot"]["start_sec"],
                "t_end": task["render_slot"]["end_sec"],
                "text": primary["text"],
                "emotion": "激动",
                "primary_variant_id": "primary",
            }
        )
    return {
        "source_neutral_sha256": commentary["source_neutral_sha256"],
        "rounds": [{"round_no": 1, "scenes": scenes}],
    }


class TestCommentaryV3Publishable:
    def test_fixture_passes(self, tmp_path):
        commentary_path = _write_json(tmp_path, "commentary.json", _load("phase3/commentary_v3.json"))
        _write_json(tmp_path, "rounds_with_commentary.json", _rounds_with_commentary_v3())
        validate_commentary_publishable(commentary_path)

    def test_missing_rounds_with_commentary_is_rejected(self, tmp_path):
        commentary_path = _write_json(tmp_path, "commentary.json", _load("phase3/commentary_v3.json"))
        with pytest.raises(PublishContractError, match="missing rounds_with_commentary"):
            validate_commentary_publishable(commentary_path)

    def test_preserved_fact_closure_error_is_rejected(self, tmp_path):
        commentary = _load("phase3/commentary_v3.json")
        task = next(t for t in commentary["voice_tasks"] if t["voice_task_id"] == "r001_w03")
        task["candidates"][0]["preserved_fact_ids"] = task["candidates"][0]["preserved_fact_ids"][:1]
        commentary_path = _write_json(tmp_path, "commentary.json", commentary)
        _write_json(tmp_path, "rounds_with_commentary.json", _rounds_with_commentary_v3())
        with pytest.raises(PublishContractError, match="preserved_fact_ids must cover"):
            validate_commentary_publishable(commentary_path)

    def test_source_hash_mismatch_is_rejected(self, tmp_path):
        commentary_path = _write_json(tmp_path, "commentary.json", _load("phase3/commentary_v3.json"))
        _write_json(tmp_path, "rounds_with_commentary.json", _rounds_with_commentary_v3())
        neutral_path = _write_json(tmp_path, "rounds_with_neutral.json", {"run_id": "other"})
        with pytest.raises(PublishContractError, match="source_neutral_sha256 does not match"):
            validate_commentary_publishable(commentary_path, neutral_path)

    def test_missing_primary_reference_is_rejected(self, tmp_path):
        commentary_path = _write_json(tmp_path, "commentary.json", _load("phase3/commentary_v3.json"))
        rounds_payload = _rounds_with_commentary_v3()
        rounds_payload["rounds"][0]["scenes"] = [
            scene for scene in rounds_payload["rounds"][0]["scenes"] if scene["voice_task_id"] != "r001_w03"
        ]
        _write_json(tmp_path, "rounds_with_commentary.json", rounds_payload)
        with pytest.raises(PublishContractError, match="r001_w03 is missing from rounds_with_commentary"):
            validate_commentary_publishable(commentary_path)

    def test_primary_reference_window_mismatch_is_rejected(self, tmp_path):
        commentary_path = _write_json(tmp_path, "commentary.json", _load("phase3/commentary_v3.json"))
        rounds_payload = _rounds_with_commentary_v3()
        scene = next(s for s in rounds_payload["rounds"][0]["scenes"] if s["voice_task_id"] == "r001_w03")
        scene["window_id"] = "other_window"
        _write_json(tmp_path, "rounds_with_commentary.json", rounds_payload)
        with pytest.raises(PublishContractError, match="window_id does not match"):
            validate_commentary_publishable(commentary_path)

    def test_candidate_set_mismatch_with_selection_policy_is_rejected(self, tmp_path):
        commentary = _load("phase3/commentary_v3.json")
        task = next(t for t in commentary["voice_tasks"] if t["voice_task_id"] == "r001_w03")
        task["candidates"] = [c for c in task["candidates"] if c["variant_id"] != "capsule"]
        commentary_path = _write_json(tmp_path, "commentary.json", commentary)
        _write_json(tmp_path, "rounds_with_commentary.json", _rounds_with_commentary_v3())
        with pytest.raises(PublishContractError, match="candidate set"):
            validate_commentary_publishable(commentary_path)

    def test_speech_metric_version_mismatch_is_rejected(self, tmp_path):
        commentary = _load("phase3/commentary_v3.json")
        commentary["speech_metric_version"] = "legacy_units_v0"
        commentary_path = _write_json(tmp_path, "commentary.json", commentary)
        _write_json(tmp_path, "rounds_with_commentary.json", _rounds_with_commentary_v3())
        with pytest.raises(PublishContractError, match="speech_metric_version must be"):
            validate_commentary_publishable(commentary_path)


# ─────────────────────────── A. 统一语音计量（v2） ───────────────────────────


def _v2_commentary_manifest(text: str) -> dict:
    output_chars = count_spoken_chars(text)
    rendered = f"[激动]{text}"
    return {
        "commentary_schema_version": 2,
        "source_neutral_run_id": uuid.uuid4().hex,
        "source_neutral_sha256": "0" * 64,
        "source_window_count": 1,
        "effective_style_config": {"style_budget_hard_tolerance": 0.0, "style_k_enabled": False},
        "rounds": [
            {
                "round_no": 1,
                "window_results": [
                    {
                        "window_id": "r001_w01",
                        "style_status": "ok",
                        "retry_count": 0,
                        "neutral_nonempty": True,
                        "neutral_source": "llm",
                        "published_scene_index": 0,
                        "t_start": 10.0,
                        "t_end": 15.0,
                        "char_budget": 20,
                        "output_chars": output_chars,
                    }
                ],
                "scenes": [
                    {
                        "window_id": "r001_w01",
                        "t_start": 10.0,
                        "t_end": 15.0,
                        "text": text,
                        "emotion": "激动",
                        "style_status": "ok",
                        "char_budget": 20,
                        "output_chars": output_chars,
                    }
                ],
                "status": "ok",
                "commentary_text": rendered,
                "emotion_segments": [{"emotion": "激动", "text": text, "order": 0}],
            }
        ],
    }


class TestUnifiedSpeechMetricV2:
    def test_output_chars_consistent_with_count_spoken_chars(self, tmp_path):
        assert len(SPOKEN_TEXT) != count_spoken_chars(SPOKEN_TEXT)
        commentary_path = _write_json(tmp_path, "commentary.json", _v2_commentary_manifest(SPOKEN_TEXT))
        validate_commentary_publishable(commentary_path)

    def test_len_based_output_chars_is_rejected(self, tmp_path):
        manifest = _v2_commentary_manifest(SPOKEN_TEXT)
        manifest["rounds"][0]["window_results"][0]["output_chars"] = len(SPOKEN_TEXT)
        manifest["rounds"][0]["scenes"][0]["output_chars"] = len(SPOKEN_TEXT)
        commentary_path = _write_json(tmp_path, "commentary.json", manifest)
        with pytest.raises(PublishContractError, match="output_chars does not match rendered scene text"):
            validate_commentary_publishable(commentary_path)

    def test_empty_round_is_publishable(self, tmp_path):
        # 旧产物的 status=empty 仍可读取；新 Phase3b 不再生成该状态。
        manifest = _v2_commentary_manifest(SPOKEN_TEXT)
        manifest["effective_style_config"] = {
            "style_budget_hard_tolerance": 0.0,
            "style_k_enabled": False,
            "style_empty_window_threshold": 0.30,
        }
        round_data = manifest["rounds"][0]
        round_data["status"] = "empty"
        round_data["commentary_text"] = ""
        round_data["emotion_segments"] = []
        round_data["scenes"] = []
        round_data["window_results"] = [
            {
                "window_id": "r001_w01",
                "style_status": "style_failed",
                "failure_reason": "over_budget",
                "retry_count": 0,
                "neutral_nonempty": True,
                "neutral_source": "llm",
                "published_scene_index": None,
                "t_start": 10.0,
                "t_end": 15.0,
                "char_budget": 20,
                "output_chars": None,
            }
        ]
        commentary_path = _write_json(tmp_path, "commentary.json", manifest)
        validate_commentary_publishable(commentary_path)


# ─────────────────────────── D. final manifest ───────────────────────────


def _final_rounds_final(load_fixture) -> dict:
    payload = copy.deepcopy(load_fixture("phase4/final_voice_task.json"))
    rounds_final = payload["rounds_final"]
    for round_data in rounds_final["rounds"]:
        round_data.setdefault("_phase4_audio", {"audio_path": "rounds/round_001.wav", "duration_sec": 18.0})
    return rounds_final


def _final_assemble_manifest(rounds_final: dict) -> dict:
    return {
        "rounds": [
            {
                "round_no": round_data["round_no"],
                "audio_path": "rounds/round_001.wav",
                "aligned": True,
                "segments": [],
            }
            for round_data in rounds_final["rounds"]
        ]
    }


def _v3_final_bundle(load_fixture) -> dict:
    rounds_final = _final_rounds_final(load_fixture)
    for round_data in rounds_final["rounds"]:
        round_data["scenes"] = [
            scene for scene in round_data.get("scenes", []) if scene.get("voice_task_id") == "r001_w03"
        ]
    return {
        "rounds_final": rounds_final,
        "assemble_manifest": _final_assemble_manifest(rounds_final),
    }


def _commentary_v3() -> dict:
    return _load("phase3/commentary_v3.json")


class TestFinalManifestV3:
    def test_valid_v3_bundle_passes(self, tmp_path, load_fixture):
        bundle = _v3_final_bundle(load_fixture)
        assert validate_final_manifest(bundle, commentary=_commentary_v3()) == []

    def test_selected_variant_not_in_candidates_is_rejected(self, tmp_path, load_fixture):
        bundle = _v3_final_bundle(load_fixture)
        scene = bundle["rounds_final"]["rounds"][0]["scenes"][0]
        scene["selected_variant_id"] = "bogus_variant"
        errors = validate_final_manifest(bundle, commentary=_commentary_v3())
        assert any("selected_variant_id is not among" in error for error in errors)

    def test_audio_end_tick_out_of_slot_is_rejected(self, tmp_path, load_fixture):
        bundle = _v3_final_bundle(load_fixture)
        scene = bundle["rounds_final"]["rounds"][0]["scenes"][0]
        scene["audio_end_tick"] = 500
        errors = validate_final_manifest(bundle, commentary=_commentary_v3())
        assert any("audio_end_tick must not exceed" in error for error in errors)

    def test_fit_state_requires_complete_selection_fields(self, tmp_path, load_fixture):
        bundle = _v3_final_bundle(load_fixture)
        scene = bundle["rounds_final"]["rounds"][0]["scenes"][0]
        del scene["audio_start_tick"]
        errors = validate_final_manifest(bundle, commentary=_commentary_v3())
        assert any("audio_start_tick" in error for error in errors)

    def test_v2_legacy_input_is_not_forced(self, load_fixture):
        payload = load_fixture("phase4/rounds_final.json")
        assert validate_final_manifest(payload, commentary=_commentary_v3()) == []

    def test_phase4_publishable_with_commentary_path(self, tmp_path, load_fixture):
        rounds_final = _final_rounds_final(load_fixture)
        for round_data in rounds_final["rounds"]:
            round_data["scenes"] = [
                scene for scene in round_data.get("scenes", []) if scene.get("voice_task_id") == "r001_w03"
            ]
        rounds_path = _write_json(tmp_path, "rounds_final.json", rounds_final)
        manifest_path = _write_json(tmp_path, "assemble_manifest.json", _final_assemble_manifest(rounds_final))
        commentary_path = _write_json(tmp_path, "commentary.json", _commentary_v3())
        validate_phase4_publishable(rounds_path, manifest_path, commentary_path)
