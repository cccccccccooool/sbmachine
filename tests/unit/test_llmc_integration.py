"""LLM-C 独立为 Phase3c 后，commentary v2 preflight 的严格逐窗门禁单元测试。

旧内嵌 LLM-C 已被移除（phase3b_style 不再整合回合稿）；发布门禁恢复唯一
严格逐窗校验：scenes 与成功 window_results 一对一、窗口身份/时间/风格/
预算/口播字数强校验。本文件验证旧式"整合段"产物被拒绝，以及旧
semantic.llmc 配置在 preflight_config 被 fail-closed。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from sbmachine.common import count_spoken_chars
from sbmachine.preflight import PublishContractError, preflight_config, validate_commentary_publishable
from audio_service.emotion import parse_emotional_text

SPOKEN_TEXT = "JDC击杀Tauson，CT拿下回合"


def _write_commentary(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "commentary.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def _strict_manifest(rounds: list[dict], **top) -> dict:
    """构造通过严格逐窗校验所需的 commentary v2 基座。"""
    return {
        "commentary_schema_version": 2,
        "source_neutral_run_id": uuid.uuid4().hex,
        "source_neutral_sha256": "0" * 64,
        "source_window_count": sum(len(r["window_results"]) for r in rounds),
        "effective_style_config": {
            "style_budget_hard_tolerance": 0.0,
            "style_k_enabled": False,
            "style_empty_window_threshold": 0.30,
        },
        "rounds": rounds,
        **top,
    }


def _window(window_id: str, *, text: str, emotion: str = "激动", t_start: float = 10.0, t_end: float = 15.0,
            budget: int | None = None, scene_index: int = 0) -> tuple[dict, dict]:
    """返回 (window_result, scene)，保证逐窗强校验可通过。"""
    budget = budget if budget is not None else max(8, count_spoken_chars(text))
    output_chars = count_spoken_chars(text)
    scene = {
        "window_id": window_id, "t_start": t_start, "t_end": t_end,
        "emotion": emotion, "text": text, "char_budget": budget,
        "output_chars": output_chars, "style_status": "ok",
    }
    window = {
        "window_id": window_id, "style_status": "ok", "retry_count": 0,
        "neutral_nonempty": True, "neutral_source": "llm",
        "published_scene_index": scene_index, "t_start": t_start, "t_end": t_end,
        "char_budget": budget, "output_chars": output_chars,
    }
    return window, scene


def _single_round(windows: list[tuple[dict, dict]]) -> dict:
    window_results = [w for w, _ in windows]
    scenes = [s for _, s in windows]
    rendered = "".join(f"[{s['emotion']}]{s['text']}" for s in scenes)
    segments = parse_emotional_text(rendered)
    return {
        "round_no": 1,
        "window_results": window_results,
        "scenes": scenes,
        "status": "ok",
        "commentary_text": rendered,
        "emotion_segments": [{"emotion": seg.emotion, "text": seg.text, "order": i} for i, seg in enumerate(segments)],
    }


class TestStrictWindowGate:
    def test_strict_one_to_one_scenes_pass(self, tmp_path):
        w1, s1 = _window("r001_w01", text=SPOKEN_TEXT)
        w2, s2 = _window("r001_w02", text="CT经济局翻盘", t_start=15.0, t_end=20.0, scene_index=1)
        manifest = _strict_manifest([_single_round([(w1, s1), (w2, s2)])])
        _write_commentary(tmp_path, manifest)
        validate_commentary_publishable(tmp_path / "commentary.json")

    def test_merged_integration_scenes_are_rejected(self, tmp_path):
        # 旧 LLM-C 整合段：一 scene 覆盖多窗（时间取并集、无逐窗审计字段）→ 严格门禁拒绝。
        w1, _ = _window("r001_w01", text=SPOKEN_TEXT)
        w2, _ = _window("r001_w02", text="CT经济局翻盘", t_start=15.0, t_end=20.0, scene_index=0)
        merged_scene = {
            "window_id": "r001_w01",  # 取来源首窗
            "t_start": 10.0, "t_end": 20.0,  # 覆盖并集
            "emotion": "激动", "text": SPOKEN_TEXT + "CT经济局翻盘", "style_status": "ok",
        }
        rendered = f"[激动]{SPOKEN_TEXT + 'CT经济局翻盘'}"
        round_data = {
            "round_no": 1,
            "window_results": [w1, w2],
            "scenes": [merged_scene],
            "status": "ok",
            "commentary_text": rendered,
            "emotion_segments": [],
        }
        manifest = _strict_manifest([round_data])
        commentary_path = _write_commentary(tmp_path, manifest)
        with pytest.raises(PublishContractError, match="does not match its scene"):
            validate_commentary_publishable(commentary_path)

    def test_window_exceeding_150x_hard_cap_is_rejected(self, tmp_path):
        # B 最终硬线 1.5×budget：output_chars > int(budget × 1.5) 拒绝（不叠加任何 tolerance）。
        budget = 10
        overlong = "长" * 16  # 16 > 15
        w1, s1 = _window("r001_w01", text=overlong, budget=budget)
        round_data = _single_round([(w1, s1)])
        manifest = _strict_manifest([round_data])
        commentary_path = _write_commentary(tmp_path, manifest)
        with pytest.raises(PublishContractError, match="exceeds char_budget"):
            validate_commentary_publishable(commentary_path)

    def test_top_level_llmc_key_is_ignored_by_publish_gate(self, tmp_path):
        # 顶层 llmc 键（旧 manifest 残留）不再被发布门禁读取/放行。
        w1, s1 = _window("r001_w01", text=SPOKEN_TEXT)
        manifest = _strict_manifest([_single_round([(w1, s1)])], llmc={"enabled": True, "round_budget_factor": 1.0})
        _write_commentary(tmp_path, manifest)
        validate_commentary_publishable(tmp_path / "commentary.json")


class TestLegacyLlmcConfigRejected:
    def test_preflight_rejects_legacy_semantic_llmc(self, tmp_path):
        config = {
            "phases": {"phase2_yolo": False, "phase3a_semantic": False, "phase3b_semantic": True},
            "semantic": {"llmc": {"enabled": True}},
        }
        report = preflight_config(config, root=tmp_path)
        assert report["config_valid"] is False
        assert any("semantic.llmc" in error for error in report["errors"])

    def test_preflight_rejects_phase4_strict_without_phase3c(self, tmp_path):
        # 半迁移态拦截：strict 发布 profile 必须同时开启 phase3c_render。
        config = {
            "phases": {"phase2_yolo": False, "phase3a_semantic": False, "phase3b_semantic": True,
                       "phase4_assemble": True},
            "phase4": {"publish_profile": "strict_av"},
        }
        report = preflight_config(config, root=tmp_path)
        assert report["config_valid"] is False
        assert any("strict Phase4 publish profiles require phase3c_render" in error for error in report["errors"])

    def test_phase4_default_off_does_not_require_phase3c(self, tmp_path):
        # 缺省 phase4_assemble（legacy v2 路径）不触发半迁移门禁。
        neutral = tmp_path / "rounds_with_neutral.json"
        neutral.write_text(json.dumps({"rounds": []}), encoding="utf-8")
        yolo = tmp_path / "rounds_with_yolo.json"
        yolo.write_text(json.dumps({"rounds": []}), encoding="utf-8")
        skill = tmp_path / "Prompt" / "skill" / "style_skill.md"
        skill.parent.mkdir(parents=True, exist_ok=True)
        skill.write_text("", encoding="utf-8")
        config = {
            "phases": {"phase2_yolo": False, "phase3a_semantic": False, "phase3b_semantic": True},
            "paths": {
                "rounds_with_neutral_json": str(neutral),
                "rounds_with_yolo_json": str(yolo),
                "style_skill": str(skill),
            },
        }
        report = preflight_config(config, root=tmp_path, only={"phase3b"})
        assert report["config_valid"] is True

    def test_phase3c_requires_mode_and_paths(self, tmp_path):
        config = {
            "phases": {"phase2_yolo": False, "phase3a_semantic": False, "phase3b_semantic": True,
                       "phase3c_render": True},
            "semantic": {"phase3c": {"mode": "bogus"}},
        }
        report = preflight_config(config, root=tmp_path)
        assert report["config_valid"] is False
        assert any("phase3c.mode" in error for error in report["errors"])
        assert any("llmb_draft_package_json" in error for error in report["errors"])
