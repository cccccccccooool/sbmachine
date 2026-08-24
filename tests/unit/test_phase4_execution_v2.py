import json
import wave
from pathlib import Path

from sbmachine.phase4_assemble import (
    _RenderUnitPlan,
    _strict_select_render_unit,
    run_phase4,
)
from sbmachine.phase4_av import assemble_scene_canvas_v2
from sbmachine.schemas import MatchPackage, RoundRecord, save_match
from tests.support.phase4_v2 import b_package, c_package


def _write_wav(path: Path, frames: int, sample_rate: int = 32000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def test_strict_tts_uses_c_final_text_and_bounded_speed_retry(tmp_path):
    plan = _RenderUnitPlan(
        round_no=1,
        unit_id="r001_u01",
        sequence=1,
        final_text="C final text",
        emotion="calm",
        text_source="llmc",
        slot_start_sec=0.0,
        slot_end_sec=1.0,
        slot_start_tick=0,
        slot_end_tick=30,
        required_speed_factor=1.0,
        max_speed_factor=1.5,
        speech_profile_id="",
        timeline_id="timeline-test",
        render_package_artifact_identity="artifact-test",
    )
    calls = []

    def fake_synthesize(_runtime, text, output_path, *, speed_factor=1.0, **_kwargs):
        calls.append((text, speed_factor))
        duration = 1.2 if speed_factor < 1.1 else 1.0
        _write_wav(Path(output_path), int(round(duration * 32000)))

    def fake_fingerprint(_runtime, text, **kwargs):
        return f"{text}:{kwargs.get('speed_factor')}"

    result = _strict_select_render_unit(
        plan,
        tts_runtime={},
        cache_dir=tmp_path / "cache",
        audio_path=tmp_path / "scene.wav",
        synthesize_emotional=fake_synthesize,
        fingerprint_fn=fake_fingerprint,
        sample_rate=32000,
    )

    assert result["fit_state"] == "fit"
    assert result["final_text"] == "C final text"
    assert result["attempted_speed_factors"] == [1.0, 1.2]
    assert calls == [("[calm]C final text", 1.0), ("[calm]C final text", 1.2)]


def test_strict_execution_dry_run_reads_only_c_final_text(tmp_path, monkeypatch):
    b_payload = b_package()
    c_path = tmp_path / "render.json"
    c_path.write_text(json.dumps(c_package(b_payload, final_text="C-owned text")), encoding="utf-8")
    rounds_path = tmp_path / "rounds.json"
    save_match(rounds_path, MatchPackage(video_path="", rounds=[RoundRecord(1, 0.0, 1.0)]))
    tts_config = tmp_path / "tts.yaml"
    tts_config.write_text("{}", encoding="utf-8")
    config = {
        "phase4": {
            "publish_profile": "strict_av",
            "clip_mode": "strict_decode",
            "make_video": False,
            "tts_config": str(tts_config),
            "sample_rate": 32000,
        },
    }
    monkeypatch.setattr("sbmachine.phase4_assemble.load_config", lambda _path: config)

    manifest = run_phase4(
        rounds_path=rounds_path,
        commentary_path=None,
        render_package_path=c_path,
        output_rounds_path=tmp_path / "out.json",
        manifest_path=tmp_path / "manifest.json",
        config_path=tmp_path / "config.yaml",
        dry_run=True,
    )

    assert manifest["rounds"][0]["segments"][0]["final_text"] == "C-owned text"
    assert manifest["media_sync_status"] == "not_checked"
    assert manifest["delivery_status"] == "fail"


def test_canvas_reports_sample_coordinates_and_rejects_overrun(tmp_path):
    audio = tmp_path / "scene.wav"
    _write_wav(audio, 250, sample_rate=1000)
    result = assemble_scene_canvas_v2(
        [{"unit_id": "u1", "audio_asset": str(audio), "slot_start_sec": 10.25, "slot_end_sec": 10.5}],
        tmp_path / "round.wav",
        10.0,
        11.0,
        default_sample_rate=1000,
    )
    unit = result["units"][0]
    assert unit["asset_frame_count"] == 250
    assert unit["round_canvas_start_sample"] == 250
    assert unit["round_canvas_end_sample"] == 500
    assert unit["timeline_start_sample"] == 10250

    overrun = tmp_path / "overrun.wav"
    _write_wav(overrun, 251, sample_rate=1000)
    try:
        assemble_scene_canvas_v2(
            [{"unit_id": "u1", "audio_asset": str(overrun), "slot_start_sec": 10.25, "slot_end_sec": 10.5}],
            tmp_path / "bad.wav",
            10.0,
            11.0,
            default_sample_rate=1000,
        )
    except RuntimeError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("strict canvas accepted an out-of-slot asset")

