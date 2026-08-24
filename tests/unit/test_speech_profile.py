"""speech profile 标定工具单元测试：拆分、拟合、上界、验收、不可变。"""

from __future__ import annotations

import json

import pytest

from tools import calibrate_speech_profile as csp
from sbmachine.speech_measure import FEATURE_ORDER


def _sample(sample_id: str, text: str, duration: float, tags=None, speed_factor=None) -> dict:
    record = {"sample_id": sample_id, "text": text, "emotion": "平述", "pcm_duration_sec": duration, "source": "historical"}
    if tags:
        record["tags"] = tags
    if speed_factor is not None:
        record["speed_factor"] = speed_factor
    return record


def _manifest(tmp_path, samples) -> str:
    path = tmp_path / "calibration_manifest.jsonl"
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    return str(path)


class TestParseManifest:
    def test_requires_fields(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text(json.dumps({"sample_id": "x"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            csp.parse_manifest(path)

    def test_ok(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text(
            json.dumps({"sample_id": "x", "text": "你好", "pcm_duration_sec": 1.2}) + "\n",
            encoding="utf-8",
        )
        samples = csp.parse_manifest(path)
        assert len(samples) == 1
        assert samples[0]["sample_id"] == "x"


class TestSplitSamples:
    def test_near_duplicate_stays_in_same_set(self):
        a = _sample("a", "JDC击杀Tauson", 2.0)
        b = _sample("b", "JDC 击杀 Tauson", 2.1)
        c = _sample("c", "CT拿下回合", 1.5)
        fit, calibration, holdout, velocity = csp.split_samples([a, b, c])
        sets = [fit, calibration, holdout]
        groups = []
        for bucket in sets:
            texts = [s["text"] for s in bucket]
            groups.append(set(csp.text_fingerprint(t) for t in texts))
        for key in csp.text_fingerprint("JDC击杀Tauson"), csp.text_fingerprint("JDC 击杀 Tauson"):
            locations = [i for i, g in enumerate(groups) if key in g]
            assert len(locations) <= 1

    def test_velocity_subset_only_speed_samples(self):
        a = _sample("a", "JDC击杀Tauson", 2.0, speed_factor=1.25)
        b = _sample("b", "CT拿下回合", 1.5)
        fit, calibration, holdout, velocity = csp.split_samples([a, b])
        assert all(abs(float(v["speed_factor"]) - 1.25) < 1e-9 for v in velocity)
        assert all(not hasattr(s, "speed_factor") or float(s.get("speed_factor") or 1.0) == 1.0 for s in fit + calibration + holdout)

    def test_velocity_too_small_emptied(self):
        samples = [_sample("a", "甲", 1.0, speed_factor=1.25)]
        _, _, _, velocity = csp.split_samples(samples)
        assert velocity == []


class TestFitAndBounds:
    def test_coefficients_nonnegative(self):
        samples = [_sample(f"s{i}", "击杀" * 3 + f"选手{i}", 0.5 + 0.2 * (i % 5)) for i in range(60)]
        coefficients, intercept, feature_order = csp.fit_nonnegative_linear(samples)
        assert feature_order == list(FEATURE_ORDER)
        assert all(c >= 0 for c in coefficients)
        assert intercept >= 0

    def test_upper_residual_nonnegative(self):
        fit = [_sample(f"f{i}", "击杀" * 4, 0.8 + 0.05 * i) for i in range(30)]
        calibration = [_sample(f"c{i}", "击杀" * 4, 1.2 + 0.1 * i) for i in range(10)]
        coefficients, intercept, feature_order = csp.fit_nonnegative_linear(fit)
        residual = csp.calibrated_upper_residual(calibration, feature_order, coefficients, intercept, 0.95)
        assert residual >= 0

    def test_safe_upper_monotonic(self):
        fit = [_sample(f"f{i}", "长句" * 5, 1.0 + 0.1 * i) for i in range(30)]
        calibration = fit[:10]
        coefficients, intercept, feature_order = csp.fit_nonnegative_linear(fit)
        residual = csp.calibrated_upper_residual(calibration, feature_order, coefficients, intercept, 0.95)
        short = csp.predict_sec(_sample("s", "短", 0), feature_order, coefficients, intercept) + residual + 0.08
        long = csp.predict_sec(_sample("l", "很长" * 10, 0), feature_order, coefficients, intercept) + residual + 0.08
        assert long >= short


class TestAcceptance:
    COEFS = [0.18, 0.29, 0.31, 0.34, 0.12]
    INTERCEPT = 0.16

    @classmethod
    def _expected(cls, text: str) -> float:
        from sbmachine.speech_measure import parse_features

        features = parse_features(text)
        return cls.INTERCEPT + sum(c * features[k] for k, c in zip(FEATURE_ORDER, cls.COEFS))

    def test_under_160_is_exploration(self, tmp_path):
        samples = [_sample(f"s{i}", f"文{i}", 1.0) for i in range(50)]
        payload = csp.build_profile(
            "p",
            samples,
            engine_fingerprint="e",
            voice_fingerprint="v",
            preprocess_fingerprint="p",
            sample_rate_hz=32000,
            base_speed_factor=1.0,
            coverage_target=0.95,
            fixed_margin=0.08,
            force=True,
            out_dir=tmp_path / "out",
        )
        assert payload["status"] == "exploration"
        assert payload["dataset"]["total"] == 50

    def test_over_160_with_velocity_can_validate(self, tmp_path):
        templates = [
            "击杀文本{i}选手{j}",
            "回合文本{i}拿下",
            "C4安放{num}秒",
            "比分{num}选手{j}",
        ]
        samples = []
        counter = 0
        for template in templates:
            for i in range(45):
                text = template.format(i=counter, j=counter % 7, num=(counter % 9) + 1)
                duration = self._expected(text)
                samples.append(_sample(f"s{counter}", text, round(duration, 3), tags=["kill"]))
                counter += 1
        for i in range(25):
            text = f"验证文本{i}选手{i % 5}"
            duration = self._expected(text) / 1.25
            samples.append(_sample(f"v{i}", text, round(duration, 3), tags=["kill"], speed_factor=1.25))
        payload = csp.build_profile(
            "p",
            samples,
            engine_fingerprint="e",
            voice_fingerprint="v",
            preprocess_fingerprint="p",
            sample_rate_hz=32000,
            base_speed_factor=1.0,
            coverage_target=0.95,
            fixed_margin=0.2,
            force=True,
            out_dir=tmp_path / "out",
        )
        assert payload["status"] == "validated"
        assert payload["speed_scaling_verified"] is True
        assert payload["dataset"]["velocity_subset"] == 25

    def test_no_velocity_means_scaling_unverified(self, tmp_path):
        samples = [_sample(f"s{i}", f"回合文本{i}", 1.0) for i in range(170)]
        payload = csp.build_profile(
            "p",
            samples,
            engine_fingerprint="e",
            voice_fingerprint="v",
            preprocess_fingerprint="p",
            sample_rate_hz=32000,
            base_speed_factor=1.0,
            coverage_target=0.95,
            fixed_margin=0.5,
            force=True,
            out_dir=tmp_path / "out",
        )
        assert payload["speed_scaling_verified"] is False

    def test_refuses_overwrite_without_force(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir(parents=True)
        (out / "profile.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileExistsError):
            csp.build_profile(
                "p",
                [_sample("s1", "文本", 1.0)],
                engine_fingerprint="e",
                voice_fingerprint="v",
                preprocess_fingerprint="p",
                sample_rate_hz=32000,
                base_speed_factor=1.0,
                coverage_target=0.95,
                fixed_margin=0.08,
                force=False,
                out_dir=out,
            )

    def test_artifacts_written(self, tmp_path):
        samples = [_sample(f"s{i}", f"文本{i}", 1.0) for i in range(60)]
        payload = csp.build_profile(
            "p",
            samples,
            engine_fingerprint="e",
            voice_fingerprint="v",
            preprocess_fingerprint="p",
            sample_rate_hz=32000,
            base_speed_factor=1.0,
            coverage_target=0.95,
            fixed_margin=0.08,
            force=True,
            out_dir=tmp_path / "out",
        )
        assert (tmp_path / "out" / "profile.json").exists()
        assert (tmp_path / "out" / "report.json").exists()
        assert (tmp_path / "out" / "calibration_manifest.jsonl").exists()
        assert payload["profile_schema_version"] == 1


class TestSpeedScalingError:
    def test_p95_computed(self):
        samples = [_sample(f"v{i}", "文本", 1.0 + 0.01 * i, speed_factor=1.25) for i in range(25)]
        error = csp.velocity_scaling_error(samples, list(FEATURE_ORDER), [0.18, 0.29, 0.31, 0.34, 0.12], 0.16)
        assert error is not None
        assert error >= 0

    def test_empty_returns_none(self):
        assert csp.velocity_scaling_error([], list(FEATURE_ORDER), [0.1] * 5, 0.1) is None
