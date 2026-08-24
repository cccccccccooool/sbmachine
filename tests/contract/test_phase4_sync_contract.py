import json

from sbmachine.preflight import validate_phase4_execution_v2, validate_phase4_publishable


def _manifest():
    segment = {
        "unit_id": "u1",
        "sequence": 1,
        "final_text": "final",
        "emotion": "calm",
        "text_source": "llmc",
        "render_slot": {"start_sec": 0.0, "end_sec": 1.0, "start_tick": 0, "end_tick": 30},
        "actual_duration_sec": 1.0,
        "asset_frame_count": 32000,
        "sample_rate": 32000,
        "applied_speed_factor": 1.0,
        "fit_state": "fit",
        "attempted_speed_factors": [1.0],
        "round_canvas_start_sample": 0,
        "round_canvas_end_sample": 32000,
        "round_slot_end_sample": 32000,
        "round_canvas_limit_sample": 32000,
        "timeline_start_sample": 0,
        "timeline_end_sample": 32000,
        "slot_timeline_end_sample": 32000,
        "timeline_canvas_end_sample": 32000,
    }
    return {
        "phase4_execution_contract_version": 2,
        "sync_schema_version": 2,
        "publish_profile": "strict_av",
        "render_package_artifact_identity": "artifact-test",
        "media_sync_status": "pass",
        "content_gate_status": "pass",
        "delivery_status": "pass",
        "media_checks": {
            "source_identity": "pass",
            "decoded_pts_monotone": "pass",
            "clip_boundary": "pass",
            "audio_within_slot": "pass",
            "canvas_bounds": "pass",
            "subtitle_within_audio": "not_required",
        },
        "content_checks": {
            "package_status": "ready",
            "text_sources": ["llmc"],
            "fact_check_scope": "disabled",
            "blocked_rounds": 0,
        },
        "rounds": [{
            "round_no": 1,
            "audio_path": "round.wav",
            "skipped": False,
            "aligned": True,
            "media_sync_status": "pass",
            "content_gate_status": "pass",
            "delivery_status": "pass",
            "segments": [segment],
        }],
    }


def _package():
    return {
        "contract": "commentary_render_package_v2",
        "artifact_identity": "artifact-test",
        "package_status": "ready",
        "content_policy": {"phase3c_mode": "optional", "fact_check_scope": "disabled"},
    }


def test_v2_manifest_dispatches_through_publishable_gate(tmp_path):
    manifest = _manifest()
    assert validate_phase4_execution_v2(
        manifest,
        rounds_final={"rounds": [{"round_no": 1}]},
        render_package=_package(),
    ) == []
    rounds_path = tmp_path / "rounds.json"
    manifest_path = tmp_path / "manifest.json"
    rounds_path.write_text(json.dumps({"rounds": [{"round_no": 1}]}), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        validate_phase4_publishable(rounds_path, manifest_path)
    except Exception as exc:
        assert "commentary_render_package_v2" in str(exc)
    else:
        raise AssertionError("strict v2 publishable validation accepted a missing render package")
