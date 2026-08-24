from types import SimpleNamespace

from sbmachine.phase2_yolo import build_semantic_frames


class _Demo:
    map_name = "de_inferno"
    capabilities = {"orientation": True, "round_scores": True}

    @staticmethod
    def round_by_no(round_no):
        assert round_no == 7
        return {
            "start_tick": 100,
            "freeze_end_tick": 200,
            "end_tick": 900,
            "winner": "CT",
            "reason": "CTWin",
            "ct_score": 4,
            "t_score": 3,
        }


def test_semantic_export_adds_demo_capabilities_and_round_result_without_changing_frames():
    background = {
        "when": {"video_time": 1.0},
        "who": {"pov_player": "A"},
        "where": {"players": []},
        "events": {"kills": [], "score_ocr": {"status": "debug"}},
    }
    frame = SimpleNamespace(background_info=background, has_frame=False)
    round_record = SimpleNamespace(
        round_no=2,
        demo_round_hint=7,
        phase2_yolo=SimpleNamespace(key_frames=[frame]),
    )

    result = build_semantic_frames(SimpleNamespace(rounds=[round_record]), demo=_Demo())

    assert result[0]["round_no"] == 2
    assert result[0]["demo_round_no"] == 7
    assert result[0]["map_name"] == "de_inferno"
    assert result[0]["capabilities"]["orientation"] is True
    assert result[0]["round_result"]["ct_score"] == 4
    assert result[0]["frames"][0]["when"]["score_before"] == {"ct": 3, "t": 3}
    assert result[0]["frames"][0]["has_frame"] is False
    assert "score_ocr" not in result[0]["frames"][0]["events"]
