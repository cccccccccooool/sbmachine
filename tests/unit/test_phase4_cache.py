"""TTS 缓存指纹单元测试：variant/text/speed/profile 任一变化都生成新缓存键。

设计依据：计划书 §10.5（缓存指纹包含 variant/text/speed/profile）与 §15.1（cache 单测）。
"""
from __future__ import annotations

import pytest

from audio_service import gpt_sovits_client


def _config() -> dict:
    return {
        "model": {"gpt_weights": "gpt.ckpt", "sovits_weights": "sovits.pth"},
        "reference": {
            "audio_path": "ref.wav",
            "prompt_text": "prompt-a",
            "prompt_lang": "zh",
            "text_lang": "zh",
        },
    }


def _base_key(monkeypatch, **kwargs) -> str:
    monkeypatch.setattr(gpt_sovits_client, "_emotion_speed_factors", lambda: {"激动": 1.0})
    return gpt_sovits_client.tts_cache_fingerprint(
        _config(),
        "[激动]text",
        speed_factor=1.0,
        variant_id="primary",
        profile_id="speech-profile-v1",
        **kwargs,
    )


def test_cache_fingerprint_changes_when_variant_changes(monkeypatch):
    baseline = _base_key(monkeypatch)
    assert gpt_sovits_client.tts_cache_fingerprint(
        _config(), "[激动]text", speed_factor=1.0, variant_id="compact", profile_id="speech-profile-v1"
    ) != baseline


def test_cache_fingerprint_changes_when_speed_changes(monkeypatch):
    baseline = _base_key(monkeypatch)
    assert gpt_sovits_client.tts_cache_fingerprint(
        _config(), "[激动]text", speed_factor=1.2, variant_id="primary", profile_id="speech-profile-v1"
    ) != baseline


def test_cache_fingerprint_changes_when_profile_changes(monkeypatch):
    baseline = _base_key(monkeypatch)
    assert gpt_sovits_client.tts_cache_fingerprint(
        _config(), "[激动]text", speed_factor=1.0, variant_id="primary", profile_id="speech-profile-v2"
    ) != baseline


def test_cache_fingerprint_changes_when_text_changes(monkeypatch):
    from sbmachine.phase4_assemble import _tts_cache_key

    fingerprint = _base_key(monkeypatch)
    baseline = _tts_cache_key("[激动]text", fingerprint)
    changed = _tts_cache_key("[激动]other text", fingerprint)
    assert changed != baseline


def test_cache_fingerprint_is_stable_for_identical_arguments(monkeypatch):
    assert _base_key(monkeypatch) == _base_key(monkeypatch)


def test_cache_fingerprint_legacy_call_remains_stable(monkeypatch):
    monkeypatch.setattr(gpt_sovits_client, "_emotion_speed_factors", lambda: {"激动": 1.0})
    first = gpt_sovits_client.tts_cache_fingerprint(_config(), "[激动]text")
    second = gpt_sovits_client.tts_cache_fingerprint(_config(), "[激动]text")
    assert first == second


def test_runtime_fingerprint_tracks_voice_reference(monkeypatch, tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"voice-a")
    config = {
        "model": {"gpt_weights": "gpt.ckpt", "sovits_weights": "sovits.pth"},
        "reference": {"audio_path": str(ref), "prompt_text": "prompt-a", "prompt_lang": "zh"},
    }
    baseline = gpt_sovits_client.tts_runtime_fingerprint(config, sample_rate_hz=1000)
    ref.write_bytes(b"voice-b")
    changed = gpt_sovits_client.tts_runtime_fingerprint(config, sample_rate_hz=1000)
    assert baseline["voice_fingerprint"] != changed["voice_fingerprint"]
    assert baseline["engine_fingerprint"] == changed["engine_fingerprint"]


def test_runtime_fingerprint_tracks_sample_rate(monkeypatch):
    baseline = gpt_sovits_client.tts_runtime_fingerprint(_config(), sample_rate_hz=1000)
    changed = gpt_sovits_client.tts_runtime_fingerprint(_config(), sample_rate_hz=32000)
    assert baseline["preprocess_fingerprint"] != changed["preprocess_fingerprint"]


def test_runtime_fingerprint_tracks_model_weights(monkeypatch, tmp_path):
    gpt_weights = tmp_path / "gpt.ckpt"
    gpt_weights.write_bytes(b"gpt-a")
    config = {
        "model": {"gpt_weights": str(gpt_weights), "sovits_weights": "sovits.pth"},
        "reference": {},
    }
    monkeypatch.delenv("GPT_SOVITS_GPT_WEIGHTS", raising=False)
    baseline = gpt_sovits_client.tts_runtime_fingerprint(config, sample_rate_hz=1000)
    gpt_weights.write_bytes(b"gpt-bb")
    changed = gpt_sovits_client.tts_runtime_fingerprint(config, sample_rate_hz=1000)
    assert baseline["engine_fingerprint"] != changed["engine_fingerprint"]
