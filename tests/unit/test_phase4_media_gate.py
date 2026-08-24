from copy import deepcopy

from sbmachine.preflight import validate_phase4_execution_v2


def _segment(text_source="llmb_passthrough"):
    return {
        "unit_id": "r001_u01",
        "sequence": 1,
        "final_text": "final text",
        "emotion": "calm",
        "text_source": text_source,
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


def _manifest(*, profile="strict_av", content="degraded", source="llmb_passthrough"):
    return {
        "phase4_execution_contract_version": 2,
        "sync_schema_version": 2,
        "publish_profile": profile,
        "render_package_artifact_identity": "artifact-test",
        "media_sync_status": "pass",
        "content_gate_status": content,
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
            "text_sources": [source],
            "fact_check_scope": "disabled",
            "blocked_rounds": 0,
        },
        "rounds": [{
            "round_no": 1,
            "audio_path": "round.wav",
            "skipped": False,
            "aligned": True,
            "media_sync_status": "pass",
            "content_gate_status": content,
            "delivery_status": "pass",
            "segments": [_segment(source)],
        }],
    }


def _render_package(*, mode="optional", scope="disabled"):
    return {
        "contract": "commentary_render_package_v2",
        "artifact_identity": "artifact-test",
        "package_status": "ready",
        "content_policy": {"phase3c_mode": mode, "fact_check_scope": scope},
    }


def test_strict_av_allows_passthrough_content_when_media_passes():
    errors = validate_phase4_execution_v2(
        _manifest(),
        rounds_final={"rounds": [{"round_no": 1}]},
        render_package=_render_package(),
    )
    assert errors == []


def test_strict_c_rejects_passthrough_and_weak_fact_scope():
    manifest = _manifest(profile="strict_c", content="pass")
    errors = validate_phase4_execution_v2(
        manifest,
        rounds_final={"rounds": [{"round_no": 1}]},
        render_package=_render_package(),
    )
    assert any("strong fact scope" in error for error in errors)
    assert any("llmb_passthrough" in error for error in errors)


def test_aligned_is_derived_from_media_status():
    manifest = _manifest()
    manifest["rounds"][0]["aligned"] = False
    errors = validate_phase4_execution_v2(
        manifest,
        rounds_final={"rounds": [{"round_no": 1}]},
        render_package=_render_package(),
    )
    assert any("aligned must be derived" in error for error in errors)
