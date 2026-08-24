from __future__ import annotations

import hashlib
import json

from sbmachine.phase3c_llmc import _build_render_package_v2


def b_unit(
    unit_id: str = "r001_u01",
    *,
    start_sec: float = 0.0,
    end_sec: float = 1.0,
    text: str = "B draft text",
    emotion: str = "calm",
) -> dict:
    start_tick = int(round(start_sec * 30))
    end_tick = int(round(end_sec * 30))
    return {
        "unit_id": unit_id,
        "sequence": 1,
        "draft_text": text,
        "emotion_binding": {"emotion": emotion, "authority": "policy"},
        "allowed_fact_ids": [f"fact:{unit_id}"],
        "carry_in_fact_ids": [],
        "fact_catalog": {f"fact:{unit_id}": {"kind": "event", "value": "known"}},
        "render_slot": {
            "slot_id": unit_id,
            "timeline_id": "timeline-test",
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_tick": start_tick,
            "end_tick": end_tick,
        },
        "speech_capacity": {
            "slot_sec": end_sec - start_sec,
            "safe_upper_sec": end_sec - start_sec,
            "required_speed_factor": 1.0,
            "draft_hard_speed_factor": 1.5,
        },
    }


def b_package(*, text: str = "B draft text") -> dict:
    package = {
        "contract": "llmb_draft_package_v2",
        "producer": "phase3b",
        "run_id": "run-test",
        "source": {
            "neutral_run_id": "run-test",
            "neutral_sha256": "neutral-sha",
            "timeline_id": "timeline-test",
            "source_video_sha256": "video-sha",
        },
        "tts_policy": {
            "profile_status": "not_required",
            "speech_profile_id": "",
            "require_validated_profile": False,
            "max_speed_factor": 1.5,
        },
        "render_timebase_fps": 30.0,
        "rounds": [{
            "round_id": "r001",
            "round_no": 1,
            "status": "ready",
            "units": [b_unit(text=text)],
        }],
    }
    body = dict(package)
    package["artifact_identity"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return package


def c_package(
    b_payload: dict,
    *,
    mode: str = "optional",
    integration_status: str = "llmc_accepted",
    final_text: str = "C final text",
    fact_scope: str = "disabled",
) -> dict:
    round_data = b_payload["rounds"][0]
    result = {
        "round_data": round_data,
        "integration_status": integration_status,
        "texts_by_unit": {"r001_u01": final_text},
        "r_c_by_unit": {"r001_u01": 1.0},
    }
    return _build_render_package_v2(
        b_payload,
        mode,
        [result],
        fact_scope=fact_scope,
    )

