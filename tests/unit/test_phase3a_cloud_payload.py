from sbmachine.phase3a_cloud_payload import build_cloud_round_payload
from sbmachine.phase3a_cloud_prompt import cloud_system_prompt


def test_cloud_payload_has_no_raw_dem_timeline_or_roster():
    payload, windows = build_cloud_round_payload(
        round_no=1, map_name="de_test", windows=[{
            "t_start": 10.0, "t_end": 12.0, "scene": "未下包",
            "main_topic": {"kind": "utility", "summary": "A投出烟雾"},
            "selected_actions": [{"type": "utility_throw", "utility": "Smoke"}],
            "state_block": "开局状态：A(T·AK·Ramp·100血)",
            "rule_state": {"kind": "snapshot", "teams": {
                "T": {"alive_count": 3, "hp_total": 268},
                "CT": {"alive_count": 2, "hp_total": 143},
            }, "changed_teams": ["T"]},
        }],
    )

    assert "round" not in payload
    assert payload["contract_version"] == 5
    assert payload["windows"][0]["projection_version"] == 2
    assert payload["windows"][0]["required_facts"]
    # S3: utility summary 从 selected_actions 确定性重建（非空）。
    assert isinstance(windows["window-1"]["main_topic"]["summary"], str)
    assert len(windows["window-1"]["main_topic"]["summary"]) > 0
    assert windows["window-1"]["main_topic"]["kind"] == "utility"
    assert "roster" not in payload and "timeline" not in payload
    assert "where" not in str(payload)
    assert "state_block" not in str(payload) and "Ramp" not in str(payload)
    assert "hp_total" not in str(payload)
    assert windows["window-1"]["rule_state"]["teams"]["T"] == {"alive_count": 3}


def test_cloud_system_prompt_treats_windows_as_ordered_round_context():
    prompt = cloud_system_prompt().casefold()

    assert "ordered" in prompt
    assert "scene" in prompt
    assert "round" in prompt
