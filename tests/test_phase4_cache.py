import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_service import gpt_sovits_client
from sbmachine import phase4_assemble, phase4_av
from sbmachine.phase4_assemble import run_phase4


def _write_wav(path: Path, duration_sec: float = 0.5, *, sample_rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x01\x00" * round(duration_sec * sample_rate))


def _write_phase4_inputs(
    rounds_path: Path,
    commentary_path: Path,
    *,
    clip_path: Path | None = None,
    scene_start: float = 12.0,
    scene_end: float = 15.0,
) -> None:
    round_data = {
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
    if clip_path is not None:
        round_data["segment_video"] = str(clip_path)
    rounds_path.write_text(
        json.dumps({"video_path": "source.mp4", "map_name": "de_test", "rounds": [round_data]}, ensure_ascii=False),
        encoding="utf-8",
    )
    commentary_path.write_text(
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
                                "t_start": scene_start,
                                "t_end": scene_end,
                                "emotion": "激动",
                                "text": "A 用 AK 击杀 B",
                                "char_budget": 40,
                                "output_chars": 14,
                                "style_status": "ok",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_config(
    path: Path,
    output_dir: Path,
    cache_dir: Path,
    clip_cache_dir: Path,
    *,
    make_video: bool = False,
    model: str = "model-a.ckpt",
) -> Path:
    runtime = path.parent / "tts_runtime.yaml"
    runtime.write_text(
        f"""
api: {{}}
model:
  gpt_weights: {model}
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
    path.write_text(
        f"""
phase4:
  tts_config: {runtime.as_posix()}
  output_dir: {output_dir.as_posix()}
  tts_cache_dir: {cache_dir.as_posix()}
  clip_cache_dir: {clip_cache_dir.as_posix()}
  make_video: {"true" if make_video else "false"}
  commentary_volume: 1.0
  game_audio_volume: 0.25
  sample_rate: 1000
""",
        encoding="utf-8",
    )
    return runtime


def _run(tmp_path: Path, monkeypatch, *, duration_sec: float = 0.5, make_video: bool = False, clip_path=None):
    rounds = tmp_path / "rounds_with_commentary.json"
    commentary = tmp_path / "commentary.json"
    output_rounds = tmp_path / "rounds_final.json"
    manifest = tmp_path / "manifest.json"
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "rounds"
    cache_dir = tmp_path / "tts_cache"
    _write_phase4_inputs(rounds, commentary, clip_path=clip_path)
    runtime = _write_config(config, output_dir, cache_dir, tmp_path / "clip_cache", make_video=make_video)
    calls = []

    def fake_synthesize(_runtime, text, output_path, budget_overage=None, speed_factor=None):
        calls.append(text)
        _write_wav(Path(output_path), duration_sec)
        return output_path

    monkeypatch.setattr("audio_service.gpt_sovits_client.synthesize_emotional", fake_synthesize)
    kwargs = {
        "rounds_path": rounds,
        "commentary_path": commentary,
        "output_rounds_path": output_rounds,
        "manifest_path": manifest,
        "config_path": config,
    }
    return kwargs, calls, output_dir, cache_dir, runtime


def test_phase4_reuses_scene_cache_and_places_audio_at_round_relative_time(tmp_path, monkeypatch):
    kwargs, calls, output_dir, cache_dir, _runtime = _run(tmp_path, monkeypatch)

    first = run_phase4(**kwargs)
    run_phase4(**kwargs)

    assert calls == ["[激动]A 用 AK 击杀 B"]
    assert len(list(cache_dir.glob("*.wav"))) == 1
    assert first["rounds"][0]["segments"][0]["relative_start_sec"] == 2.0
    assert first["rounds"][0]["segments"][0]["audio_duration_sec"] == 0.5
    with wave.open(str(output_dir / "round_001.wav"), "rb") as wav:
        assert wav.getnframes() == 10_000
        frames = wav.readframes(wav.getnframes())
    assert frames[: 2_000 * 2] == b"\x00" * (2_000 * 2)
    assert frames[2_000 * 2: 2_500 * 2] == b"\x01\x00" * 500
    assert frames[2_500 * 2:] == b"\x00" * (7_500 * 2)


def test_phase4_synthesizes_each_scene_independently(tmp_path, monkeypatch):
    kwargs, calls, output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch)
    commentary = kwargs["commentary_path"]
    payload = json.loads(commentary.read_text(encoding="utf-8"))
    item = payload["rounds"][0]
    item["commentary_text"] += "[平述]第二段"
    item["emotion_segments"].append({"emotion": "平述", "text": "第二段", "order": 1})
    item["scenes"].append({"t_start": 16.0, "t_end": 18.0, "emotion": "平述", "text": "第二段"})
    commentary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    rounds_payload = json.loads(kwargs["rounds_path"].read_text(encoding="utf-8"))
    semantic = rounds_payload["rounds"][0]["_phase3_semantic"]
    semantic["commentary_text"] += "[平述]第二段"
    semantic["emotion_segments"].append({"emotion": "平述", "text": "第二段", "order": 1})
    kwargs["rounds_path"].write_text(json.dumps(rounds_payload, ensure_ascii=False), encoding="utf-8")

    result = run_phase4(**kwargs)

    assert calls == ["[激动]A 用 AK 击杀 B", "[平述]第二段"]
    assert [segment["relative_start_sec"] for segment in result["rounds"][0]["segments"]] == [2.0, 6.0]
    assert (output_dir / "round_001_scene_001.wav").is_file()
    assert (output_dir / "round_001_scene_002.wav").is_file()


def test_phase4_cache_changes_when_model_changes(tmp_path, monkeypatch):
    kwargs, calls, _output_dir, cache_dir, runtime = _run(tmp_path, monkeypatch)

    run_phase4(**kwargs)
    runtime.write_text(runtime.read_text(encoding="utf-8").replace("model-a.ckpt", "model-b.ckpt"), encoding="utf-8")
    run_phase4(**kwargs)

    assert calls == ["[激动]A 用 AK 击杀 B", "[激动]A 用 AK 击杀 B"]
    assert len(list(cache_dir.glob("*.wav"))) == 2


def test_tts_fingerprint_covers_reference_prompt_language_and_speed(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-a")
    config = {
        "model": {"gpt_weights": "gpt.ckpt", "sovits_weights": "sovits.pth"},
        "reference": {
            "audio_path": str(reference),
            "prompt_text": "prompt-a",
            "prompt_lang": "zh",
            "text_lang": "zh",
        },
    }
    monkeypatch.setattr(gpt_sovits_client, "_emotion_speed_factors", lambda: {"激动": 1.0})
    baseline = gpt_sovits_client.tts_cache_fingerprint(config, "[激动]text")

    config["reference"]["prompt_text"] = "prompt-b"
    prompt_changed = gpt_sovits_client.tts_cache_fingerprint(config, "[激动]text")
    config["reference"]["prompt_text"] = "prompt-a"
    config["reference"]["prompt_lang"] = "en"
    language_changed = gpt_sovits_client.tts_cache_fingerprint(config, "[激动]text")
    config["reference"]["prompt_lang"] = "zh"
    reference.write_bytes(b"reference-b")
    reference_changed = gpt_sovits_client.tts_cache_fingerprint(config, "[激动]text")
    monkeypatch.setattr(gpt_sovits_client, "_emotion_speed_factors", lambda: {"激动": 1.2})
    speed_changed = gpt_sovits_client.tts_cache_fingerprint(config, "[激动]text")

    assert len({baseline, prompt_changed, language_changed, reference_changed, speed_changed}) == 5


def test_tts_fingerprint_hashes_existing_weight_contents(tmp_path, monkeypatch):
    gpt_weights = tmp_path / "gpt.ckpt"
    sovits_weights = tmp_path / "sovits.pth"
    gpt_weights.write_bytes(b"gpt-a")
    sovits_weights.write_bytes(b"sovits-a")
    config = {
        "model": {
            "gpt_weights": str(gpt_weights),
            "sovits_weights": str(sovits_weights),
        },
        "reference": {},
    }
    monkeypatch.delenv("GPT_SOVITS_GPT_WEIGHTS", raising=False)
    monkeypatch.delenv("GPT_SOVITS_SOVITS_WEIGHTS", raising=False)
    monkeypatch.setattr(gpt_sovits_client, "_emotion_speed_factors", lambda: {})

    baseline = gpt_sovits_client.tts_cache_fingerprint(config, "[平述]text")
    gpt_weights.write_bytes(b"gpt-b")
    gpt_changed = gpt_sovits_client.tts_cache_fingerprint(config, "[平述]text")
    gpt_weights.write_bytes(b"gpt-a")
    sovits_weights.write_bytes(b"sovits-b")
    sovits_changed = gpt_sovits_client.tts_cache_fingerprint(config, "[平述]text")

    assert len({baseline, gpt_changed, sovits_changed}) == 3


def test_phase4_requires_publishable_commentary_status(tmp_path, monkeypatch):
    kwargs, calls, _output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch)
    commentary = kwargs["commentary_path"]
    payload = json.loads(commentary.read_text(encoding="utf-8"))
    payload["rounds"][0]["status"] = "style_failed"
    commentary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="style_failed"):
        run_phase4(**kwargs)

    assert calls == []


def test_phase4_accepts_silent_commentary_round(tmp_path, monkeypatch):
    kwargs, calls, _output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch)
    rounds_payload = json.loads(kwargs["rounds_path"].read_text(encoding="utf-8"))
    rounds_payload["rounds"][0]["_phase3_semantic"] = {
        "commentary_text": "",
        "emotion_segments": [],
    }
    kwargs["rounds_path"].write_text(json.dumps(rounds_payload, ensure_ascii=False), encoding="utf-8")
    commentary_payload = json.loads(kwargs["commentary_path"].read_text(encoding="utf-8"))
    commentary_payload["rounds"][0].update({
        "status": "silent",
        "commentary_text": "",
        "emotion_segments": [],
        "scenes": [],
    })
    kwargs["commentary_path"].write_text(json.dumps(commentary_payload, ensure_ascii=False), encoding="utf-8")

    result = run_phase4(**kwargs)

    assert calls == []
    assert result["rounds"][0]["skipped"] is True


def test_phase4_rejects_commentary_from_different_source(tmp_path, monkeypatch):
    kwargs, calls, _output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch)
    commentary = kwargs["commentary_path"]
    payload = json.loads(commentary.read_text(encoding="utf-8"))
    payload["video_path"] = "another-match.mp4"
    commentary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="same video_path"):
        run_phase4(**kwargs)

    assert calls == []


def test_phase4_rejects_commentary_text_mismatch(tmp_path, monkeypatch):
    kwargs, calls, _output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch)
    commentary = kwargs["commentary_path"]
    payload = json.loads(commentary.read_text(encoding="utf-8"))
    payload["rounds"][0]["commentary_text"] = "[激动]different text"
    commentary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match rounds_with_commentary"):
        run_phase4(**kwargs)

    assert calls == []


def test_phase4_rejects_emotion_segment_text_mismatch(tmp_path, monkeypatch):
    kwargs, calls, _output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch)
    commentary = kwargs["commentary_path"]
    payload = json.loads(commentary.read_text(encoding="utf-8"))
    payload["rounds"][0]["emotion_segments"][0]["text"] = "different text"
    commentary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="emotion_segments do not match"):
        run_phase4(**kwargs)

    assert calls == []


def test_phase4_dry_run_performs_no_writes(tmp_path, monkeypatch):
    kwargs, calls, _output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch, make_video=True)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    result = run_phase4(**kwargs, dry_run=True)

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert calls == []
    assert result["total_rounds"] == 1


def test_clip_cache_key_uses_source_content_and_time_window(tmp_path, monkeypatch):
    source_a = tmp_path / "source-a.mp4"
    source_b = tmp_path / "source-b.mp4"
    source_a.write_bytes(b"match-a")
    source_b.write_bytes(b"match-b")
    cache_dir = tmp_path / "clip-cache"
    cut_calls = []

    def fake_cut(args):
        cut_calls.append(args)
        Path(args[-1]).write_bytes(b"clip")

    monkeypatch.setattr(phase4_assemble, "_run_ffmpeg", fake_cut)
    base_round = SimpleNamespace(round_no=1, start_sec=10.0, end_sec=20.0, segment_video="")
    shifted_round = SimpleNamespace(round_no=1, start_sec=11.0, end_sec=20.0, segment_video="")

    path_a = phase4_assemble._clip_for_round(base_round, source_a, cache_dir)
    path_b = phase4_assemble._clip_for_round(base_round, source_b, cache_dir)
    path_shifted = phase4_assemble._clip_for_round(shifted_round, source_a, cache_dir)
    reused = phase4_assemble._clip_for_round(base_round, source_a, cache_dir)

    assert len({path_a, path_b, path_shifted}) == 3
    assert reused == path_a
    assert len(cut_calls) == 3


def test_standalone_gpt_sovits_cli_uses_shared_lock(monkeypatch):
    events = []

    class RecordingLock:
        def __init__(self, path):
            events.append(("lock", Path(path)))

        def __enter__(self):
            events.append(("enter", None))
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", None))

    def fake_synthesize(_config, _text, _output):
        assert events[-1][0] == "enter"
        events.append(("synthesize", None))

    monkeypatch.setattr(gpt_sovits_client, "FileLock", RecordingLock)
    monkeypatch.setattr(gpt_sovits_client, "parse_args", lambda: SimpleNamespace(
        config="runtime.yaml",
        text="hello",
        output="demo.wav",
        emotional=False,
    ))
    monkeypatch.setattr(gpt_sovits_client, "read_config", lambda _path: {})
    monkeypatch.setattr(gpt_sovits_client, "synthesize", fake_synthesize)

    assert gpt_sovits_client.main() == 0
    assert events == [
        ("lock", gpt_sovits_client.PROJECT_ROOT / "output" / ".sovits.lock"),
        ("enter", None),
        ("synthesize", None),
        ("exit", None),
    ]


def test_phase4_rejects_legacy_commentary_without_scenes(tmp_path, monkeypatch):
    kwargs, calls, _output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch)
    commentary = kwargs["commentary_path"]
    payload = json.loads(commentary.read_text(encoding="utf-8"))
    del payload["rounds"][0]["scenes"]
    commentary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="scenes is required.*legacy"):
        run_phase4(**kwargs)

    assert calls == []


def test_phase4_fails_when_scene_audio_exceeds_window(tmp_path, monkeypatch):
    kwargs, _calls, _output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch, duration_sec=3.5)

    with pytest.raises(RuntimeError, match="scene audio exceeds its window"):
        run_phase4(**kwargs)


def test_gpt_sovits_rejects_non_wav_http_body(monkeypatch):
    class Response:
        content = b"<html>upstream error</html>"

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(gpt_sovits_client.requests, "post", lambda *args, **kwargs: Response())

    with pytest.raises(ValueError, match="decodable WAV"):
        gpt_sovits_client._synthesize_bytes({}, "text", {})


def test_mux_without_game_audio_uses_padded_commentary_and_no_shortest(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(phase4_av, "_probe_media", lambda _path: (False, 10.0))
    monkeypatch.setattr(phase4_av, "_run_ffmpeg", calls.append)

    phase4_av._mux_round_video(tmp_path / "clip.mp4", tmp_path / "speech.wav", tmp_path / "out.mp4")

    args = calls[0]
    audio_filter = args[args.index("-filter_complex") + 1]
    assert "[0:a" not in audio_filter
    assert "[1:a:0]" in audio_filter
    assert "apad" in audio_filter
    assert "atrim=duration=10.000000" in audio_filter
    assert "-shortest" not in args


def test_phase4_reuses_phase1_clip_when_make_video(tmp_path, monkeypatch):
    existing_clip = tmp_path / "clips" / "round_001.mp4"
    existing_clip.parent.mkdir()
    existing_clip.write_bytes(b"clip")
    kwargs, _calls, output_dir, _cache_dir, _runtime = _run(
        tmp_path,
        monkeypatch,
        make_video=True,
        clip_path=existing_clip,
    )
    cut_calls = []
    mux_inputs = []

    def fake_mux(clip_path, audio_path, output_path, game_vol=0.25, comm_vol=1.0):
        mux_inputs.append(Path(clip_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"mp4")
        return output_path

    monkeypatch.setattr("sbmachine.phase4_assemble._run_ffmpeg", cut_calls.append)
    monkeypatch.setattr("sbmachine.phase4_assemble._mux_round_video", fake_mux)

    result = run_phase4(**kwargs)

    assert cut_calls == []
    assert mux_inputs == [existing_clip]
    assert result["rounds"][0]["video_path"] == str(output_dir / "round_001.mp4")


def test_phase4_progress_sink_failure_does_not_change_business_result(tmp_path, monkeypatch):
    """Catches an observational callback exception aborting TTS/assembly."""
    kwargs, _calls, _output_dir, _cache_dir, _runtime = _run(tmp_path, monkeypatch)

    result = run_phase4(
        **kwargs,
        progress_sink=lambda *_args: (_ for _ in ()).throw(RuntimeError("progress unavailable")),
    )

    assert result["rounds"][0]["round_no"] == 1
