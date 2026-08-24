"""speech_measure 单元测试：特征解析、测量输出、profile 读取与指纹。"""

from __future__ import annotations

import json

import pytest

from sbmachine import speech_measure
from sbmachine.common import count_spoken_chars
from sbmachine.speech_measure import (
    METRIC_VERSION,
    ProfileError,
    SpeechFeatureError,
    check_profile_match,
    estimate_required_speed_factor,
    load_profile,
    measure_text,
    parse_features,
    validate_profile_status,
)


def _validated_profile() -> dict:
    return {
        "profile_schema_version": 1,
        "profile_id": "speech-profile-v1",
        "status": "validated",
        "metric_version": "speech_units_v1",
        "engine_fingerprint": "engine-hash",
        "voice_fingerprint": "voice-hash",
        "preprocess_fingerprint": "pre-hash",
        "sample_rate_hz": 32000,
        "base_speed_factor": 1.0,
        "duration_estimator": {
            "kind": "nonnegative_linear_v1",
            "feature_order": ["zh_units", "english_words", "number_groups", "alnum_tokens", "pause_units"],
            "coefficients_sec": [0.18, 0.29, 0.31, 0.34, 0.12],
            "intercept_sec": 0.16,
        },
        "safety": {
            "method": "split_conformal_upper_v1",
            "coverage_target": 0.95,
            "upper_residual_sec": 0.24,
            "fixed_margin_sec": 0.08,
        },
    }


class TestParseFeatures:
    def test_pure_chinese(self):
        features = parse_features("JDC击杀Tauson")  # noqa
        assert features["zh_units"] >= 2
        assert features["english_words"] >= 2

    def test_pure_chinese_only(self):
        features = parse_features("CT拿下回合")
        assert features == {
            "zh_units": 4,
            "english_words": 1,
            "number_groups": 0,
            "alnum_tokens": 0,
            "pause_units": 0,
        }

    def test_english_player_name_is_one_word(self):
        features = parse_features("Tauson")
        assert features["english_words"] == 1
        assert features["zh_units"] == 0

    def test_mixed_zh_en(self):
        features = parse_features("JDC击杀Tauson，进入残局")
        assert features["english_words"] == 2
        assert features["zh_units"] == 6
        assert features["pause_units"] == 1

    def test_numbers(self):
        features = parse_features("比分3比2")
        assert features["zh_units"] == 3
        assert features["number_groups"] == 2

    def test_score_and_decimal(self):
        features = parse_features("3-2")
        assert features["number_groups"] == 1
        features = parse_features("1.5")
        assert features["number_groups"] == 1

    def test_c4_alnum_token(self):
        features = parse_features("C4")
        assert features["alnum_tokens"] == 1
        assert features["english_words"] == 0

    def test_weapon_alnum(self):
        features = parse_features("AK-47")
        assert features["alnum_tokens"] == 1
        assert features["number_groups"] == 0

    def test_punctuation_only(self):
        features = parse_features("……")
        assert features["pause_units"] == 2
        assert features["zh_units"] == 0

    def test_emotion_tag_excluded(self):
        plain = parse_features("【激动】JDC击杀")
        assert plain["zh_units"] == 2
        assert plain["english_words"] == 1

    def test_empty_text(self):
        assert parse_features("") == {
            "zh_units": 0,
            "english_words": 0,
            "number_groups": 0,
            "alnum_tokens": 0,
            "pause_units": 0,
        }

    def test_markdown_and_whitespace_noise(self):
        features = parse_features("  **JDC** 击杀\n")
        assert features["zh_units"] == 2
        assert features["english_words"] == 1

    def test_feature_stability(self):
        text = "JDC击杀Tauson，C4已安放"
        assert parse_features(text) == parse_features(text)

    def test_non_str_rejected(self):
        with pytest.raises(SpeechFeatureError):
            parse_features(123)


class TestMeasureText:
    def test_no_profile_no_safe_upper(self):
        result = measure_text("JDC击杀Tauson")
        assert result["metric_version"] == METRIC_VERSION
        assert result["safe_duration_upper_bound_at_base_speed_sec"] is None
        assert result["estimated_duration_sec"] is None
        assert result["profile_status"] == "unavailable"
        assert result["display_chars"] == count_spoken_chars("JDC击杀Tauson")

    def test_missing_profile_id(self):
        result = measure_text("abc", profile_id="does-not-exist")
        assert result["profile_status"] == "unavailable"
        assert result["safe_duration_upper_bound_at_base_speed_sec"] is None

    def test_validated_profile_gives_bounds(self, monkeypatch, tmp_path):
        profile = _validated_profile()
        root = tmp_path / "data" / "speech_profiles" / "speech-profile-v1"
        root.mkdir(parents=True)
        (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = measure_text("JDC击杀Tauson，进入残局", emotion="激动", profile_id="speech-profile-v1", speed_factor=1.0)
        assert result["profile_status"] == "validated"
        assert result["estimated_duration_sec"] is not None
        assert result["safe_duration_upper_bound_at_base_speed_sec"] is not None
        assert result["safe_duration_upper_bound_at_base_speed_sec"] >= result["estimated_duration_sec"]

    def test_speed_factor_scaling(self, monkeypatch, tmp_path):
        profile = _validated_profile()
        root = tmp_path / "data" / "speech_profiles" / "speech-profile-v1"
        root.mkdir(parents=True)
        (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        at1 = measure_text("JDC击杀Tauson", profile_id="speech-profile-v1", speed_factor=1.0)
        at2 = measure_text("JDC击杀Tauson", profile_id="speech-profile-v1", speed_factor=1.5)
        assert at2["estimated_duration_sec"] is not None and at1["estimated_duration_sec"] is not None
        assert abs(at2["estimated_duration_sec"] - at1["estimated_duration_sec"] / 1.5) < 0.001

    def test_invalid_speed_factor(self):
        with pytest.raises(SpeechFeatureError):
            measure_text("abc", speed_factor=0)

    def test_exploration_profile_no_safe_upper(self, monkeypatch, tmp_path):
        profile = _validated_profile()
        profile["status"] = "exploration"
        root = tmp_path / "data" / "speech_profiles" / "speech-profile-v1"
        root.mkdir(parents=True)
        (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = measure_text("abc", profile_id="speech-profile-v1")
        assert result["profile_status"] == "exploration"
        assert result["safe_duration_upper_bound_at_base_speed_sec"] is None


class TestLoadProfile:
    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert load_profile("nope") is None

    def test_corrupt_json(self, tmp_path, monkeypatch):
        root = tmp_path / "data" / "speech_profiles" / "p"
        root.mkdir(parents=True)
        (root / "profile.json").write_text("{not json", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert load_profile("p") is None

    def test_missing_fields(self, tmp_path, monkeypatch):
        root = tmp_path / "data" / "speech_profiles" / "p"
        root.mkdir(parents=True)
        (root / "profile.json").write_text(json.dumps({"profile_id": "p"}), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert load_profile("p") is None

    def test_wrong_schema_version(self, tmp_path, monkeypatch):
        profile = _validated_profile()
        profile["profile_schema_version"] = 99
        root = tmp_path / "data" / "speech_profiles" / "p"
        root.mkdir(parents=True)
        (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert load_profile("p") is None


class TestProfileStatusAndMatch:
    def test_valid_statuses(self):
        for status in ("validated", "exploration", "stale"):
            assert validate_profile_status({"status": status}) == status

    def test_invalid_status(self):
        with pytest.raises(ProfileError):
            validate_profile_status({"status": "bogus"})

    def test_fingerprint_match(self):
        profile = _validated_profile()
        assert check_profile_match(profile, engine_fingerprint="engine-hash", voice_fingerprint="voice-hash", preprocess_fingerprint="pre-hash")
        assert not check_profile_match(profile, engine_fingerprint="other", voice_fingerprint="voice-hash", preprocess_fingerprint="pre-hash")

    def test_fingerprint_non_dict(self):
        assert not check_profile_match(None, engine_fingerprint="a", voice_fingerprint="b", preprocess_fingerprint="c")


class TestEstimateSpeedFactor:
    def test_validated_returns_factor(self):
        profile = _validated_profile()
        factor = estimate_required_speed_factor("JDC击杀Tauson，进入残局", profile, slot_duration_sec=3.0)
        assert factor is not None
        assert factor >= 1.0

    def test_exploration_returns_none(self):
        profile = _validated_profile()
        profile["status"] = "exploration"
        assert estimate_required_speed_factor("abc", profile, slot_duration_sec=3.0) is None

    def test_non_validated_scaling_returns_none(self):
        profile = _validated_profile()
        profile["speed_scaling_verified"] = False
        assert estimate_required_speed_factor("abc", profile, slot_duration_sec=3.0) is None

    def test_none_profile(self):
        assert estimate_required_speed_factor("abc", None, slot_duration_sec=3.0) is None

    def test_bad_slot(self):
        profile = _validated_profile()
        with pytest.raises(SpeechFeatureError):
            estimate_required_speed_factor("abc", profile, slot_duration_sec=0)
