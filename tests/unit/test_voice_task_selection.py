"""Phase4 配音任务单（commentary v3）选择算法门禁测试。

覆盖计划书 §阶段2 门禁：
1. primary 原速适配 → 只调用一次 TTS。
2. primary 超时、compact 适配 → 调用两次 TTS。
3. primary/compact 超时、capsule 适配 → 三次。
4. 全部失败 → render_unfit，无最终 WAV/MP4 发布。
5. v2 输入行为不变。
6. 不存在任何 Phase4 到 Phase3b 的调用或写操作。
另测：profile 指纹不匹配 → profile_mismatch；audio_end_tick 越界 → render_unfit；
超长候选在 max_speed_factor 内重试；v3 缓存复用。
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from audio_service import gpt_sovits_client
from sbmachine import speech_measure
from sbmachine.phase4_assemble import run_phase4

PRIMARY_TEXT = "完整风格稿"
COMPACT_TEXT = "紧凑风格稿"
CAPSULE_TEXT = "规则层最短完整事实句"
FACT_IDS = ["fact:v1:r001_w03:kill:00360:a13f92c1"]
SOURCE_HASH = "fixture-v3-source-hash"


def _task(
    *,
    voice_task_id: str = "r001_w03",
    window_id: str = "r001_w03",
    start_sec: float = 12.0,
    end_sec: float = 15.0,
    start_tick: int = 360,
    end_tick: int = 450,
    selection_order=("primary", "compact", "capsule"),
    max_speed_factor: float = 1.5,
    profile_id: str = "speech-profile-v1",
    texts: dict[str, str] | None = None,
) -> dict:
    texts = texts or {"primary": PRIMARY_TEXT, "compact": COMPACT_TEXT, "capsule": CAPSULE_TEXT}
    candidates = [
        {
            "variant_id": "primary",
            "source": "llmb",
            "text": texts["primary"],
            "spoken_units": 18,
            "minimum_required_speed_factor": 1.15,
            "preserved_fact_ids": FACT_IDS,
        },
        {
            "variant_id": "compact",
            "source": "llmb_compact",
            "text": texts["compact"],
            "spoken_units": 14,
            "minimum_required_speed_factor": 1.0,
            "preserved_fact_ids": FACT_IDS,
        },
        {
            "variant_id": "capsule",
            "source": "rule_capsule",
            "text": texts["capsule"],
            "spoken_units": 11,
            "minimum_required_speed_factor": 1.0,
            "preserved_fact_ids": FACT_IDS,
        },
    ]
    if "compact" not in selection_order:
        candidates = [candidate for candidate in candidates if candidate["variant_id"] in selection_order]
    return {
        "voice_task_id": voice_task_id,
        "window_id": window_id,
        "render_slot": {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "gap_policy": "independent_window",
        },
        "required_fact_ids": FACT_IDS,
        "speech_profile_id": profile_id,
        "risk_class": "amber",
        "selection_order": list(selection_order),
        "max_speed_factor": max_speed_factor,
        "candidates": candidates,
        "semantic_state": "ok",
    }


def _write_wav(path: Path, duration_sec: float, *, sample_rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x01\x00" * round(duration_sec * sample_rate))


class FakeV3TTS:
    """按文本基速 + speed_factor 计算实际时长；时长 = base / speed_factor。"""

    def __init__(self, base_durations: dict[str, float], sample_rate: int = 1000) -> None:
        self.base_durations = dict(base_durations)
        self.sample_rate = sample_rate
        self.calls: list[dict] = []

    def synthesize(self, _runtime, text, output_path, *, budget_overage=1.0, speed_factor=1.0):
        self.calls.append({"text": text, "speed_factor": speed_factor, "budget_overage": budget_overage})
        base = self.base_durations.get(text)
        if base is None:
            raise RuntimeError(f"no fake duration for {text!r}")
        _write_wav(Path(output_path), base / speed_factor, sample_rate=self.sample_rate)
        return output_path


def _write_runtime_and_config(tmp_path: Path) -> tuple[Path, dict]:
    runtime = tmp_path / "tts_runtime.yaml"
    runtime.write_text(
        """
api: {}
model:
  gpt_weights: model-a.ckpt
  sovits_weights: sovits-a.pth
reference:
  audio_path: ref.wav
  prompt_text: reference prompt
  prompt_lang: zh
emotion_refs:
  激动:
    audio_path: excited.wav
    prompt_text: excited prompt
""",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
phase4:
  tts_config: {runtime.as_posix()}
  output_dir: {str(tmp_path / "rounds").replace("\\", "/")}
  tts_cache_dir: {str(tmp_path / "tts_cache").replace("\\", "/")}
  clip_cache_dir: {str(tmp_path / "clip_cache").replace("\\", "/")}
  make_video: false
  commentary_volume: 1.0
  game_audio_volume: 0.25
  sample_rate: 1000
""",
        encoding="utf-8",
    )
    runtime_config = gpt_sovits_client.read_config(runtime)
    return config, runtime_config


def _write_v3_inputs(tmp_path: Path, tasks: list[dict], *, scene_text: str = PRIMARY_TEXT) -> tuple[Path, Path]:
    rounds = tmp_path / "rounds_with_commentary.json"
    commentary = tmp_path / "commentary.json"
    rounds.write_text(
        json.dumps(
            {
                "video_path": "source.mp4",
                "map_name": "de_test",
                "source_neutral_run_id": "fixture-neutral-v4-run",
                "source_neutral_sha256": SOURCE_HASH,
                "rounds": [
                    {
                        "round_no": 1,
                        "start_sec": 10.0,
                        "end_sec": 20.0,
                        "score_before": {"ct": 0, "t": 0},
                        "score_after": {"ct": 0, "t": 1},
                        "scenes": [
                            {
                                "window_id": tasks[0]["window_id"],
                                "voice_task_id": tasks[0]["voice_task_id"],
                                "t_start": tasks[0]["render_slot"]["start_sec"],
                                "t_end": tasks[0]["render_slot"]["end_sec"],
                                "text": scene_text,
                                "emotion": "激动",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    commentary.write_text(
        json.dumps(
            {
                "commentary_schema_version": 3,
                "voice_task_contract_version": 1,
                "candidate_policy": "sparse_v1",
                "speech_metric_version": "speech_units_v1",
                "source_neutral_run_id": "fixture-neutral-v4-run",
                "source_neutral_sha256": SOURCE_HASH,
                "voice_tasks": tasks,
                "rounds": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return rounds, commentary


def _write_profile(tmp_path: Path, fingerprints: dict, *, status: str = "validated") -> None:
    root = tmp_path / "speech-profile-v1"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_schema_version": 1,
        "profile_id": "speech-profile-v1",
        "status": status,
        "metric_version": "speech_units_v1",
        "engine_fingerprint": fingerprints["engine_fingerprint"],
        "voice_fingerprint": fingerprints["voice_fingerprint"],
        "preprocess_fingerprint": fingerprints["preprocess_fingerprint"],
        "sample_rate_hz": 1000,
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
        "speed_scaling_verified": True,
        "dataset": {"total": 200, "fit": 120, "calibration": 40, "holdout": 40, "velocity_subset": 25, "manifest_sha256": "x"},
        "holdout_metrics": {
            "upper_bound_coverage": 0.975,
            "max_underestimate_sec": 0.11,
            "median_overestimate_ratio": 0.12,
            "tag_metrics": {},
            "velocity_scaling_p95": 0.01,
        },
        "acceptance_failures": [],
    }
    (root / "profile.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _run_v3(
    tmp_path: Path,
    monkeypatch,
    *,
    base_durations: dict[str, float],
    tasks: list[dict] | None = None,
    scene_text: str = PRIMARY_TEXT,
    write_profile: bool = True,
) -> tuple[dict, FakeV3TTS, dict]:
    tasks = tasks or [_task()]
    rounds, commentary = _write_v3_inputs(tmp_path, tasks, scene_text=scene_text)
    config_path, runtime_config = _write_runtime_and_config(tmp_path)
    if write_profile:
        fingerprints = gpt_sovits_client.tts_runtime_fingerprint(runtime_config, sample_rate_hz=1000)
        _write_profile(tmp_path, fingerprints)
    monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: tmp_path / profile_id)
    fake = FakeV3TTS(base_durations)
    monkeypatch.setattr("audio_service.gpt_sovits_client.synthesize_emotional", fake.synthesize)
    kwargs = {
        "rounds_path": rounds,
        "commentary_path": commentary,
        "output_rounds_path": tmp_path / "rounds_final.json",
        "manifest_path": tmp_path / "assemble_manifest.json",
        "config_path": config_path,
    }
    manifest = run_phase4(**kwargs)
    return manifest, fake, kwargs


class TestSelectionGates:
    def test_primary_fits_at_suggested_speed_calls_tts_once(self, tmp_path, monkeypatch):
        manifest, fake, kwargs = _run_v3(
            tmp_path,
            monkeypatch,
            base_durations={f"[激动]{PRIMARY_TEXT}": 2.0},
        )
        assert len(fake.calls) == 1
        assert fake.calls[0]["speed_factor"] == 1.15
        segment = manifest["rounds"][0]["segments"][0]
        assert segment["fit_state"] == "fit"
        assert segment["selected_variant_id"] == "primary"
        assert segment["selected_text"] == PRIMARY_TEXT
        assert segment["attempted_variants"] == ["primary"]
        assert segment["audio_start_tick"] == 360
        assert segment["audio_end_tick"] == 360 + round(2.0 / 1.15 * 30)
        assert segment["applied_speed_factor"] == 1.15
        assert (tmp_path / "rounds" / "round_001.wav").is_file()

    def test_primary_overlong_compact_fits_calls_tts_twice(self, tmp_path, monkeypatch):
        manifest, fake, _kwargs = _run_v3(
            tmp_path,
            monkeypatch,
            base_durations={f"[激动]{PRIMARY_TEXT}": 6.0, f"[激动]{COMPACT_TEXT}": 2.0},
        )
        assert len(fake.calls) == 2
        assert [call["text"] for call in fake.calls] == [f"[激动]{PRIMARY_TEXT}", f"[激动]{COMPACT_TEXT}"]
        segment = manifest["rounds"][0]["segments"][0]
        assert segment["fit_state"] == "fit"
        assert segment["selected_variant_id"] == "compact"
        assert segment["attempted_variants"] == ["primary", "compact"]

    def test_primary_and_compact_overlong_capsule_fits_calls_tts_three_times(self, tmp_path, monkeypatch):
        manifest, fake, _kwargs = _run_v3(
            tmp_path,
            monkeypatch,
            base_durations={
                f"[激动]{PRIMARY_TEXT}": 6.0,
                f"[激动]{COMPACT_TEXT}": 6.0,
                f"[激动]{CAPSULE_TEXT}": 2.0,
            },
        )
        assert len(fake.calls) == 3
        segment = manifest["rounds"][0]["segments"][0]
        assert segment["fit_state"] == "fit"
        assert segment["selected_variant_id"] == "capsule"
        assert segment["attempted_variants"] == ["primary", "compact", "capsule"]

    def test_all_candidates_fail_render_unfit_no_final_wav_or_mp4(self, tmp_path, monkeypatch):
        rounds_dir = tmp_path / "rounds"
        config_path, runtime_config = _write_runtime_and_config(tmp_path)
        config_text = config_path.read_text(encoding="utf-8").replace("make_video: false", "make_video: true")
        config_path.write_text(config_text, encoding="utf-8")
        rounds, commentary = _write_v3_inputs(tmp_path, [_task()])
        fingerprints = gpt_sovits_client.tts_runtime_fingerprint(runtime_config, sample_rate_hz=1000)
        _write_profile(tmp_path, fingerprints)
        monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: tmp_path / profile_id)
        fake = FakeV3TTS({f"[激动]{PRIMARY_TEXT}": 6.0, f"[激动]{COMPACT_TEXT}": 6.0, f"[激动]{CAPSULE_TEXT}": 6.0})
        monkeypatch.setattr("audio_service.gpt_sovits_client.synthesize_emotional", fake.synthesize)
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        manifest = run_phase4(**kwargs)

        assert len(fake.calls) == 3
        record = manifest["rounds"][0]
        assert record["aligned"] is False
        assert record["render_unfit"] is True
        assert record["audio_path"] == ""
        assert not (rounds_dir / "round_001.wav").exists()
        assert not (rounds_dir / "round_001.mp4").exists()
        segment = record["segments"][0]
        assert segment["fit_state"] == "render_unfit"
        assert segment["render_unfit_reason"] == "all candidates exceeded the fixed slot"
        assert segment["attempted_variants"] == ["primary", "compact", "capsule"]
        assert segment["selected_variant_id"] is None
        assert segment["audio_start_tick"] is None
        rounds_final = json.loads(kwargs["output_rounds_path"].read_text(encoding="utf-8"))
        scene = rounds_final["rounds"][0]["scenes"][0]
        assert scene["fit_state"] == "render_unfit"
        assert scene["render_unfit_reason"] == "all candidates exceeded the fixed slot"
        assert scene["audio_end_tick"] is None

    def test_overlong_candidate_retries_within_max_speed_factor(self, tmp_path, monkeypatch):
        manifest, fake, _kwargs = _run_v3(
            tmp_path,
            monkeypatch,
            base_durations={f"[激动]{PRIMARY_TEXT}": 4.5, f"[激动]{COMPACT_TEXT}": 2.0},
        )
        assert len(fake.calls) == 3
        speeds = [call["speed_factor"] for call in fake.calls]
        assert speeds[0] == 1.15
        assert 1.15 < speeds[1] <= 1.5
        assert speeds[2] == 1.0
        segment = manifest["rounds"][0]["segments"][0]
        assert segment["selected_variant_id"] == "compact"
        assert segment["attempted_variants"] == ["primary", "primary", "compact"]

    def test_v3_silent_round_writes_silence_and_skips(self, tmp_path, monkeypatch):
        rounds = tmp_path / "rounds_with_commentary.json"
        commentary = tmp_path / "commentary.json"
        rounds.write_text(
            json.dumps(
                {
                    "video_path": "source.mp4",
                    "map_name": "de_test",
                    "source_neutral_run_id": "fixture-neutral-v4-run",
                    "source_neutral_sha256": SOURCE_HASH,
                    "rounds": [
                        {
                            "round_no": 1,
                            "start_sec": 10.0,
                            "end_sec": 20.0,
                            "score_before": {"ct": 0, "t": 0},
                            "score_after": {"ct": 0, "t": 1},
                            "scenes": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        commentary.write_text(
            json.dumps(
                {
                    "commentary_schema_version": 3,
                    "voice_task_contract_version": 1,
                    "candidate_policy": "sparse_v1",
                    "speech_metric_version": "speech_units_v1",
                    "source_neutral_run_id": "fixture-neutral-v4-run",
                    "source_neutral_sha256": SOURCE_HASH,
                    "voice_tasks": [_task()],
                    "rounds": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        config_path, runtime_config = _write_runtime_and_config(tmp_path)
        fingerprints = gpt_sovits_client.tts_runtime_fingerprint(runtime_config, sample_rate_hz=1000)
        _write_profile(tmp_path, fingerprints)
        monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: tmp_path / profile_id)
        fake = FakeV3TTS({})
        monkeypatch.setattr("audio_service.gpt_sovits_client.synthesize_emotional", fake.synthesize)
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        manifest = run_phase4(**kwargs)

        assert fake.calls == []
        record = manifest["rounds"][0]
        assert record["skipped"] is True
        assert record["aligned"] is True
        assert (tmp_path / "rounds" / "round_001.wav").is_file()
        rounds_final = json.loads(kwargs["output_rounds_path"].read_text(encoding="utf-8"))
        assert "scenes" not in rounds_final["rounds"][0]

    def test_v3_dry_run_performs_no_writes(self, tmp_path, monkeypatch):
        _manifest, fake, kwargs = _run_v3(
            tmp_path,
            monkeypatch,
            base_durations={f"[激动]{PRIMARY_TEXT}": 2.0},
        )
        before = {
            str(path): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file() and path.suffix != ".wav"
        }
        result = run_phase4(**kwargs, dry_run=True)
        after = {
            str(path): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file() and path.suffix != ".wav"
        }
        assert after == before
        assert len(fake.calls) == 1
        assert result["rounds"][0]["segments"][0]["fit_state"] == "fit"

    def test_v3_cache_reuse_avoids_second_tts_call(self, tmp_path, monkeypatch):
        _manifest, fake, kwargs = _run_v3(
            tmp_path,
            monkeypatch,
            base_durations={f"[激动]{PRIMARY_TEXT}": 2.0},
        )
        run_phase4(**kwargs)
        assert len(fake.calls) == 1

    def test_mixed_round_writes_nothing_when_any_task_unfit(self, tmp_path, monkeypatch):
        good = _task(
            voice_task_id="r001_w02",
            window_id="r001_w02",
            start_sec=3.0,
            end_sec=6.0,
            start_tick=90,
            end_tick=180,
            selection_order=("primary", "capsule"),
            texts={"primary": "主稿", "compact": COMPACT_TEXT, "capsule": CAPSULE_TEXT},
        )
        good["risk_class"] = "green"
        bad = _task()
        rounds, commentary = _write_v3_inputs_two_scenes(tmp_path, [good, bad])
        config_path, runtime_config = _write_runtime_and_config(tmp_path)
        fingerprints = gpt_sovits_client.tts_runtime_fingerprint(runtime_config, sample_rate_hz=1000)
        _write_profile(tmp_path, fingerprints)
        monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: tmp_path / profile_id)
        fake = FakeV3TTS(
            {
                f"[激动]{PRIMARY_TEXT}": 6.0,
                f"[激动]{COMPACT_TEXT}": 6.0,
                f"[激动]{CAPSULE_TEXT}": 6.0,
                "[激动]主稿": 2.0,
            }
        )
        monkeypatch.setattr("audio_service.gpt_sovits_client.synthesize_emotional", fake.synthesize)
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        manifest = run_phase4(**kwargs)

        assert manifest["rounds"][0]["render_unfit"] is True
        assert manifest["rounds"][0]["aligned"] is False
        assert not (tmp_path / "rounds" / "round_001.wav").exists()
        segments = manifest["rounds"][0]["segments"]
        assert segments[0]["fit_state"] == "fit"
        assert segments[1]["fit_state"] == "render_unfit"


def _write_v3_inputs_two_scenes(tmp_path: Path, tasks: list[dict]) -> tuple[Path, Path]:
    rounds = tmp_path / "rounds_with_commentary.json"
    commentary = tmp_path / "commentary.json"
    rounds.write_text(
        json.dumps(
            {
                "video_path": "source.mp4",
                "map_name": "de_test",
                "source_neutral_run_id": "fixture-neutral-v4-run",
                "source_neutral_sha256": SOURCE_HASH,
                "rounds": [
                    {
                        "round_no": 1,
                        "start_sec": 0.0,
                        "end_sec": 20.0,
                        "score_before": {"ct": 0, "t": 0},
                        "score_after": {"ct": 0, "t": 1},
                        "scenes": [
                            {
                                "window_id": task["window_id"],
                                "voice_task_id": task["voice_task_id"],
                                "t_start": task["render_slot"]["start_sec"],
                                "t_end": task["render_slot"]["end_sec"],
                                "text": "主稿" if task["voice_task_id"] == "r001_w02" else PRIMARY_TEXT,
                                "emotion": "激动",
                            }
                            for task in tasks
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    commentary.write_text(
        json.dumps(
            {
                "commentary_schema_version": 3,
                "voice_task_contract_version": 1,
                "candidate_policy": "sparse_v1",
                "speech_metric_version": "speech_units_v1",
                "source_neutral_run_id": "fixture-neutral-v4-run",
                "source_neutral_sha256": SOURCE_HASH,
                "voice_tasks": tasks,
                "rounds": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return rounds, commentary


class TestV2Compatibility:
    def test_v2_input_behavior_unchanged(self, tmp_path, monkeypatch):
        rounds = tmp_path / "rounds_with_commentary.json"
        commentary = tmp_path / "commentary.json"
        rounds.write_text(
            json.dumps(
                {
                    "video_path": "source.mp4",
                    "map_name": "de_test",
                    "rounds": [
                        {
                            "round_no": 1,
                            "start_sec": 10.0,
                            "end_sec": 20.0,
                            "score_before": {"ct": 0, "t": 0},
                            "score_after": {"ct": 0, "t": 1},
                            "_phase3_semantic": {
                                "commentary_text": "[激动]A 用 AK 击杀 B",
                                "emotion_segments": [{"emotion": "激动", "text": "A 用 AK 击杀 B", "order": 0}],
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        commentary.write_text(
            json.dumps(
                {
                    "video_path": "source.mp4",
                    "map_name": "de_test",
                    "rounds": [
                        {
                            "round_no": 1,
                            "start_sec": 10.0,
                            "end_sec": 20.0,
                            "status": "ok",
                            "commentary_text": "[激动]A 用 AK 击杀 B",
                            "emotion_segments": [{"emotion": "激动", "text": "A 用 AK 击杀 B", "order": 0}],
                            "scenes": [
                                {
                                    "window_id": "r001_w01",
                                    "t_start": 12.0,
                                    "t_end": 15.0,
                                    "emotion": "激动",
                                    "text": "A 用 AK 击杀 B",
                                    "char_budget": 40,
                                    "output_chars": 14,
                                    "style_status": "ok",
                                }
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        config_path, _runtime_config = _write_runtime_and_config(tmp_path)
        calls = []

        def fake_synthesize(_runtime, text, output_path, budget_overage=None, speed_factor=None):
            calls.append(text)
            _write_wav(Path(output_path), 0.5)
            return output_path

        monkeypatch.setattr("audio_service.gpt_sovits_client.synthesize_emotional", fake_synthesize)
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        result = run_phase4(**kwargs)

        assert calls == ["[激动]A 用 AK 击杀 B"]
        segment = result["rounds"][0]["segments"][0]
        assert "voice_task_id" not in segment
        assert "fit_state" not in segment
        assert segment["audio_duration_sec"] == 0.5
        rounds_final = json.loads(kwargs["output_rounds_path"].read_text(encoding="utf-8"))
        assert "scenes" not in rounds_final["rounds"][0]


class TestProfileGate:
    def test_profile_missing_stops_with_profile_mismatch(self, tmp_path, monkeypatch):
        rounds, commentary = _write_v3_inputs(tmp_path, [_task()])
        config_path, _runtime_config = _write_runtime_and_config(tmp_path)
        monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: tmp_path / profile_id)
        fake = FakeV3TTS({f"[激动]{PRIMARY_TEXT}": 2.0})
        monkeypatch.setattr("audio_service.gpt_sovits_client.synthesize_emotional", fake.synthesize)
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        with pytest.raises(ValueError, match="profile_mismatch"):
            run_phase4(**kwargs)
        assert fake.calls == []

    def test_profile_fingerprint_mismatch_stops(self, tmp_path, monkeypatch):
        rounds, commentary = _write_v3_inputs(tmp_path, [_task()])
        config_path, runtime_config = _write_runtime_and_config(tmp_path)
        fingerprints = gpt_sovits_client.tts_runtime_fingerprint(runtime_config, sample_rate_hz=1000)
        fingerprints["voice_fingerprint"] = "a-different-voice"
        _write_profile(tmp_path, fingerprints)
        monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: tmp_path / profile_id)
        fake = FakeV3TTS({f"[激动]{PRIMARY_TEXT}": 2.0})
        monkeypatch.setattr("audio_service.gpt_sovits_client.synthesize_emotional", fake.synthesize)
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        with pytest.raises(ValueError, match="profile_mismatch"):
            run_phase4(**kwargs)
        assert fake.calls == []

    def test_non_validated_profile_stops(self, tmp_path, monkeypatch):
        rounds, commentary = _write_v3_inputs(tmp_path, [_task()])
        config_path, runtime_config = _write_runtime_and_config(tmp_path)
        fingerprints = gpt_sovits_client.tts_runtime_fingerprint(runtime_config, sample_rate_hz=1000)
        _write_profile(tmp_path, fingerprints, status="exploration")
        monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: tmp_path / profile_id)
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        with pytest.raises(ValueError, match="profile_mismatch"):
            run_phase4(**kwargs)


class TestContractCrossChecks:
    def test_task_window_mismatch_rejected(self, tmp_path, monkeypatch):
        task = _task()
        rounds, commentary = _write_v3_inputs(tmp_path, [task])
        payload = json.loads(rounds.read_text(encoding="utf-8"))
        payload["rounds"][0]["scenes"][0]["t_start"] = 13.0
        rounds.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        config_path, runtime_config = _write_runtime_and_config(tmp_path)
        fingerprints = gpt_sovits_client.tts_runtime_fingerprint(runtime_config, sample_rate_hz=1000)
        _write_profile(tmp_path, fingerprints)
        monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: tmp_path / profile_id)
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        with pytest.raises(ValueError, match="render_slot"):
            run_phase4(**kwargs)

    def test_source_hash_mismatch_rejected(self, tmp_path, monkeypatch):
        task = _task()
        rounds, commentary = _write_v3_inputs(tmp_path, [task])
        payload = json.loads(commentary.read_text(encoding="utf-8"))
        payload["source_neutral_sha256"] = "another-hash"
        commentary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        config_path, _runtime_config = _write_runtime_and_config(tmp_path)
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        with pytest.raises(ValueError, match="source_neutral_sha256 does not match"):
            run_phase4(**kwargs)

    def test_no_phase4_to_phase3b_calls_or_writes(self, tmp_path, monkeypatch):
        rounds, commentary = _write_v3_inputs(tmp_path, [_task()])
        config_path, runtime_config = _write_runtime_and_config(tmp_path)
        fingerprints = gpt_sovits_client.tts_runtime_fingerprint(runtime_config, sample_rate_hz=1000)
        _write_profile(tmp_path, fingerprints)
        monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: tmp_path / profile_id)
        fake = FakeV3TTS({f"[激动]{PRIMARY_TEXT}": 2.0})
        monkeypatch.setattr("audio_service.gpt_sovits_client.synthesize_emotional", fake.synthesize)
        before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        kwargs = {
            "rounds_path": rounds,
            "commentary_path": commentary,
            "output_rounds_path": tmp_path / "rounds_final.json",
            "manifest_path": tmp_path / "assemble_manifest.json",
            "config_path": config_path,
        }
        run_phase4(**kwargs)

        assert rounds.read_bytes() == before[str(rounds)]
        assert commentary.read_bytes() == before[str(commentary)]
        allowed = {
            str(tmp_path / "rounds_final.json"),
            str(tmp_path / "assemble_manifest.json"),
        }
        for path in tmp_path.rglob("*"):
            if not path.is_file() or str(path) in before:
                continue
            if str(path) in allowed:
                continue
            assert str(path).startswith(str(tmp_path / "rounds")) or str(path).startswith(
                str(tmp_path / "tts_cache")
            ), f"unexpected new file: {path}"


class TestFixedSlot:
    def test_audio_end_tick_conversion_matches_plan_example(self):
        from sbmachine.phase4_av import audio_end_tick, check_scene_slot_fit

        assert audio_end_tick(2.74, 360) == 442
        assert check_scene_slot_fit(2.74, 360, 450)
        assert not check_scene_slot_fit(3.02, 360, 450)
        assert not check_scene_slot_fit(1.0, 360, 389)

    def test_audio_overrun_becomes_render_unfit(self, tmp_path, monkeypatch):
        manifest, fake, _kwargs = _run_v3(
            tmp_path,
            monkeypatch,
            base_durations={f"[激动]{PRIMARY_TEXT}": 6.0, f"[激动]{COMPACT_TEXT}": 6.0, f"[激动]{CAPSULE_TEXT}": 6.0},
        )
        assert len(fake.calls) == 3
        segment = manifest["rounds"][0]["segments"][0]
        assert segment["fit_state"] == "render_unfit"
        assert segment["audio_end_tick"] is None
