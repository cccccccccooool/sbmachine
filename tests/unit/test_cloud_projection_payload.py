from sbmachine.phase3a_cloud_payload import build_cloud_round_payload


def test_cloud_payload_accepts_only_rule_window_projections():
    payload, windows = build_cloud_round_payload(
        round_no=1,
        map_name="de_test",
        windows=[{
            "t_start": 10.0, "t_end": 15.0, "scene": "未下包",
            "main_topic": {"kind": "tactic", "summary": "假爆A真打B：A小一人道具牵制，B区三人集结。"},
            "selected_actions": [{"type": "utility_throw", "utility": "Smoke", "x": 1, "destination": {"x": 2}}], "state_block": "状态变化：A 转移B_Main",
            "tactic_hint": {"rule_id": "fake_a_hit_b", "label": "假爆A真打B", "hint": "A小一人道具牵制，B区三人集结。", "matched_at": 12.0},
        }],
    )

    assert payload["windows"][0]["id"] == "window-1"
    assert windows["window-1"]["tactic_hint"]["rule_id"] == "fake_a_hit_b"
    assert windows["window-1"]["selected_actions"] == [{"type": "utility_throw", "utility": "Smoke"}]
    text = str(payload)
    assert "roster" not in text and "timeline" not in text
    assert "where" not in text and "event_ledger" not in text and "evidence" not in text
    assert payload["windows"][0]["required_facts"][0]["anchors"]["events"] == ["utility_throw"]
    assert '"x"' not in text and '"y"' not in text and '"z"' not in text


def test_cloud_payload_exposes_ordered_round_rule_context_without_raw_timestamps():
    payload, _ = build_cloud_round_payload(
        round_no=7,
        map_name="de_test",
        windows=[
            {
                "t_start": 10.0,
                "t_end": 15.0,
                "scene": "unplanted",
                "main_topic": {"kind": "utility", "summary": "raw Ramp must not pass"},
                "selected_actions": [{"type": "utility_throw", "utility": "Smoke"}],
            },
            {
                "t_start": 15.0,
                "t_end": 20.0,
                "scene": "bomb",
                "main_topic": {"kind": "tactic", "summary": "verified tactic"},
                "selected_actions": [],
                "tactic_hint": {
                    "rule_id": "fake_a_hit_b",
                    "label": "fake A hit B",
                    "hint": "verified diversion",
                    "matched_at": 18.0,
                },
            },
        ],
    )

    assert payload["round_context"] == {
        "round_no": 7,
        "map_name": "de_test",
        "window_count": 2,
        "selection_policy": "at_most_one",
    }
    assert [
        (window["id"], window["order"], window["scene"])
        for window in payload["windows"]
    ] == [("window-1", 1, "unplanted"), ("window-2", 2, "bomb")]
    text = str(payload)
    assert "t_start" not in text and "t_end" not in text and "10.0" not in text
    assert "Ramp" not in text and "matched_at" not in text
