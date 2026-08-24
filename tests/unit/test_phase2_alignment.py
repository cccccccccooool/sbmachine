from pathlib import Path
from types import SimpleNamespace

import pytest

from sbmachine.phase1_slice import build_rounds_from_segments
from sbmachine.phase2_background import _c4_planted_at, _demo_round_hint, build_background_info, resolve_demo_round_hints
from sbmachine.phase2_timeline import build_timeline
from sbmachine.phase2_yolo import should_sample_alignment_ocr, should_sample_score_ocr
from sbmachine.schemas import load_match, save_match
from sbmachine.time_align import C4VisualTracker, RoundTimeAlign


def test_phase1_keeps_round_aligner_metadata_without_prephase_hud_evidence(tmp_path):
    match = build_rounds_from_segments(
        Path("match.mp4"),
        [{
            "start_sec": 100.0,
            "end_sec": 150.0,
            "demo_round_hint": 7,
            "align_offset": -5504.0,
            "align_method": "duration_dp",
            "align_confidence": 0.7,
            "hud_observations": [{"time_sec": 101.0, "regions": [], "timer": {"value": "1:20"}}],
        }],
        "de_test",
    )
    output = tmp_path / "rounds.json"
    save_match(output, match)
    restored = load_match(output).rounds[0]

    assert (restored.demo_round_hint, restored.align_offset) == (7, -5504.0)
    assert (restored.align_method, restored.align_confidence) == ("duration_dp", 0.7)
    assert "hud_observations" not in output.read_text(encoding="utf-8")


def test_unknown_phase2_fields_are_ignored(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(
        '{"video_path":"match.mp4","rounds":[{"round_no":1,"start_sec":0,"end_sec":1,'
        '"_phase2_yolo":{"key_frames":[{"time_sec":0,"gate_reason":"demo_only",'
        '"has_frame":false,"unknown_field":"ignored"}],"unknown_data":"ignored"},'
        '"unknown_round_field":"ignored"}]}',
        encoding="utf-8",
    )
    output = tmp_path / "current.json"

    match = load_match(source)
    save_match(output, match)
    payload = output.read_text(encoding="utf-8")

    assert match.rounds[0].phase2_yolo.key_frames[0].has_frame is False
    assert '"_phase2_yolo"' in payload
    assert "unknown_field" not in payload
    assert "unknown_data" not in payload
    assert "unknown_round_field" not in payload


def test_unmatched_demo_hint_is_inferred_from_nearest_anchor():
    records = [
        SimpleNamespace(round_no=1, demo_round_hint="unmatched"),
        SimpleNamespace(round_no=2, demo_round_hint=7),
    ]

    resolve_demo_round_hints(records, [{"round_no": 6}, {"round_no": 7}])

    assert _demo_round_hint(records[0]) == 6


def test_unmatched_demo_hint_without_anchors_starts_from_first_demo_round():
    records = [
        SimpleNamespace(round_no=1, demo_round_hint="unmatched"),
        SimpleNamespace(round_no=2, demo_round_hint=None),
    ]

    resolve_demo_round_hints(records, [{"round_no": 1}, {"round_no": 2}])

    assert [record.demo_round_hint for record in records] == [1, 2]


def test_phase2_timeline_uses_fixed_visual_rate_without_demo_events():
    assert build_timeline(100.0, 103.0, interval_sec=1.0) == [
        (100.0, True), (101.0, True), (102.0, True), (103.0, True),
    ]


def test_alignment_ocr_uses_initial_and_complete_periodic_windows():
    sample = lambda time_sec: should_sample_alignment_ocr(
        time_sec, 100.0, 158.0, initial_sec=10.0, period_sec=20.0, window_sec=5.0
    )

    assert sample(100.0) is True
    assert sample(109.0) is True
    assert sample(110.0) is False
    assert sample(114.0) is False
    assert sample(115.0) is False
    assert sample(130.0) is True
    assert sample(134.0) is True
    assert sample(135.0) is False
    assert sample(150.0) is True
    assert sample(154.0) is True

    short_tail_sample = lambda time_sec: should_sample_alignment_ocr(
        time_sec, 100.0, 153.0, initial_sec=10.0, period_sec=20.0, window_sec=5.0
    )
    assert short_tail_sample(150.0) is False


def test_c4_state_is_derived_from_dem_ticks_not_alignment_lock():
    meta = {"bomb_planted_tick": 3200, "bomb_defused_tick": 5000, "end_tick": 0}

    assert _c4_planted_at(meta, 3199) is False
    assert _c4_planted_at(meta, 3200) is True
    assert _c4_planted_at(meta, 4999) is True
    assert _c4_planted_at(meta, 5000) is False


def test_rejected_timer_outlier_does_not_poison_monotone_guard():
    align = RoundTimeAlign({"freeze_end_tick": 640}, 64.0, provisional_offset=-5504.0)
    align.add_anchor(100.0, "1:40")
    align.add_anchor(101.0, "0:04")
    align.add_anchor(102.0, "1:34")

    assert not any("non-monotone anchor timer=1:34" in warning for warning in align.warnings)

def test_no_evidence_time_conversion_uses_one_invertible_offset():
    align = RoundTimeAlign({"freeze_end_tick": 640}, 64.0)

    tick = align.to_tick(123.25)

    assert align.to_video_time(tick) == 123.25


def test_first_anchor_is_checked_against_provisional_offset():
    align = RoundTimeAlign({"freeze_end_tick": 640}, 64.0, provisional_offset=-5760.0)

    align.add_anchor(100.0, "1:45")

    assert align.offsets == []
    assert any("drop first anchor" in warning for warning in align.warnings)

    align.add_anchor(100.0, "1:55")
    assert align.offsets == [-5760.0]


def test_first_anchor_without_provisional_requires_a_second_anchor():
    align = RoundTimeAlign({"freeze_end_tick": 640}, 64.0)

    align.add_anchor(100.0, "1:55")
    assert align.offsets == []

    align.add_anchor(101.0, "1:54")
    assert align.offsets == [-5760.0, -5760.0]


def _timer_observation(video_time, timer_sec, confidence=0.9):
    return {
        "kind": "timer", "video_time": float(video_time), "timer_sec": int(timer_sec),
        "normalized": f"{timer_sec // 60}:{timer_sec % 60:02d}", "parse_status": "parsed",
        "alignment_status": "pending" if confidence >= 0.35 else "state_rejected",
        "ocr_confidence": confidence,
    }


def test_offline_timer_cluster_locks_and_ignores_wrong_provisional_offset():
    align = RoundTimeAlign({"round_no": 6, "freeze_end_tick": 640}, 64.0, provisional_offset=999999.0)
    for video_time, timer_sec in ((100, 115), (101, 114), (102, 113), (103, 112)):
        align.observe_timer(_timer_observation(video_time, timer_sec))

    result = align.solve(source_start_sec=100, source_end_sec=130)

    assert result["status"] == "locked"
    assert result["timer_accepted"] == 4
    assert align.to_tick(100) == 640
    assert align.relative_sec_for_tick(align.to_tick(103)) == pytest.approx(3.0)


def test_confirmed_reset_tightens_effective_start_and_marks_foreign_tail():
    align = RoundTimeAlign({"round_no": 6, "freeze_end_tick": 640}, 64.0)
    for video_time, timer_sec in ((96, 3), (97, 2), (98, 115), (99, 114), (100, 113), (101, 112)):
        align.observe_timer(_timer_observation(video_time, timer_sec))

    result = align.solve(source_start_sec=96, source_end_sec=130)

    assert result["status"] == "locked"
    assert result["confirmed_resets"] == [98.0]
    assert result["effective_start_sec"] == 98.0
    assert result["foreign_tail"] == {"start_sec": 96.0, "end_sec": 97.0}


def test_single_bad_low_timer_does_not_create_reset_or_change_mapping():
    align = RoundTimeAlign({"round_no": 1, "freeze_end_tick": 0}, 64.0)
    for video_time, timer_sec in ((100, 100), (101, 99), (102, 5), (103, 97), (104, 96), (105, 95)):
        align.observe_timer(_timer_observation(video_time, timer_sec))

    result = align.solve(source_start_sec=100, source_end_sec=120)

    assert result["status"] == "locked"
    assert result["confirmed_resets"] == []
    assert result["timer_rejected"] >= 1


def test_unresolved_alignment_fails_closed_without_formal_mapping():
    align = RoundTimeAlign({"round_no": 1, "freeze_end_tick": 0}, 64.0)
    align.observe_timer(_timer_observation(100, 115))
    result = align.solve(source_start_sec=100, source_end_sec=110)
    assert result["status"] == "alignment_unresolved"
    assert align.is_locked is False
    assert align.observations[0]["alignment_status"] == "state_rejected"


def test_repeated_timer_sequence_cannot_establish_mapping():
    align = RoundTimeAlign({"round_no": 1, "freeze_end_tick": 0}, 64.0)
    for video_time in (100, 101, 102, 103):
        align.observe_timer(_timer_observation(video_time, 115))
    result = align.solve(source_start_sec=100, source_end_sec=110)
    assert result["status"] == "alignment_unresolved"
    assert result["timer_accepted"] == 0


def test_slow_drift_updates_slope_but_stays_near_demo_tick_rate():
    align = RoundTimeAlign({"round_no": 1, "freeze_end_tick": 0}, 64.0)
    for video_time, timer_sec in ((100.0, 115), (101.01, 114), (102.02, 113), (103.03, 112), (104.04, 111)):
        align.observe_timer(_timer_observation(video_time, timer_sec))
    result = align.solve(source_start_sec=100, source_end_sec=120)
    assert result["status"] == "locked"
    assert result["slope_tick_per_sec"] == pytest.approx(64.0 / 1.01, rel=0.01)


def test_persistent_offset_step_reanchors_to_latest_stable_cluster():
    align = RoundTimeAlign({"round_no": 1, "freeze_end_tick": 0}, 64.0)
    for video_time, timer_sec in ((100, 115), (101, 114), (102, 113), (110, 115), (111, 114), (112, 113)):
        align.observe_timer(_timer_observation(video_time, timer_sec))
    result = align.solve(source_start_sec=100, source_end_sec=120)
    assert result["status"] == "locked"
    assert result["state_history"] == ["UNANCHORED", "ACQUIRING", "DEGRADED", "LOST", "REANCHORING", "LOCKED"]
    assert align.to_tick(110) == 0


def test_c4_visual_requires_stable_two_of_three_onset():
    tracker = C4VisualTracker(window=3, min_present=2)
    region = [{"confidence": 0.8}]
    assert tracker.update(1, region)["transition"] == "none"
    assert tracker.update(2, [])["transition"] == "none"
    assert tracker.update(3, region)["transition"] == "candidate_onset"
    assert tracker.update(4, [])["transition"] == "none"


def test_c4_false_onset_in_nonplant_round_marks_source_unsupported():
    align = RoundTimeAlign({"round_no": 1, "freeze_end_tick": 0, "bomb_planted_tick": None}, 64.0)
    for video_time, timer_sec in ((100, 115), (101, 114), (102, 113)):
        align.observe_timer(_timer_observation(video_time, timer_sec))
    c4 = [
        {"video_time": 100, "source_supported": True, "transition": "none"},
        {"video_time": 101, "source_supported": True, "transition": "candidate_onset"},
    ]
    result = align.solve(source_start_sec=100, source_end_sec=120, c4_observations=c4)
    assert result["status"] == "locked"
    assert result["c4_evidence"] == "unsupported"
    assert result["c4_support_reason"] == "stable_onset_in_nonplant_round"


def test_c4_conflict_degrades_alignment_without_changing_dem_fact():
    meta = {"round_no": 1, "freeze_end_tick": 0, "bomb_planted_tick": 640}
    align = RoundTimeAlign(meta, 64.0, anchor_tolerance_sec=2.0)
    for video_time, timer_sec in ((100, 115), (101, 114), (102, 113)):
        align.observe_timer(_timer_observation(video_time, timer_sec))
    result = align.solve(
        source_start_sec=100,
        source_end_sec=120,
        c4_observations=[{
            "video_time": 120, "source_supported": True, "transition": "candidate_onset",
        }],
    )
    assert result["status"] == "locked"
    assert result["c4_evidence"] == "conflict"
    assert result["alignment_confidence"] == "degraded"
    assert meta["bomb_planted_tick"] == 640


def test_score_scheduler_is_independent_from_periodic_timer_windows():
    assert should_sample_score_ocr(100, 100, consensus_frames=5) is True
    assert should_sample_score_ocr(104, 100, consensus_frames=5) is True
    assert should_sample_score_ocr(105, 100, consensus_frames=5) is False
    assert should_sample_score_ocr(130, 100, consensus_frames=5, reset_frames_remaining=2) is True


def test_background_relative_and_phase_ignore_rejected_single_frame_timer():
    class FakeDemo:
        def state_at(self, *_args, **_kwargs): return []
        def match_player(self, *_args, **_kwargs): return SimpleNamespace(score=0.0, name="", steamid="")
        def kills_between(self, *_args): return []
        def utilities_between(self, *_args): return []

    meta = {"round_no": 1, "freeze_end_tick": 640}
    align = RoundTimeAlign(meta, 64.0)
    for video_time, timer_sec in ((100, 115), (101, 114), (102, 113)):
        align.observe_timer(_timer_observation(video_time, timer_sec))
    assert align.solve(source_start_sec=100, source_end_sec=120)["status"] == "locked"

    background, tick = build_background_info(
        demo=FakeDemo(), round_meta=meta, align=align, video_time=103,
        pov_ocr_result={"raw_text": "", "engine": "fake"},
        timer_ocr_result={"normalized": "0:05", "alignment_status": "state_rejected"},
        prev_tick=None,
    )

    assert background["when"]["timer"] == ""
    assert background["when"]["relative_sec"] == pytest.approx((tick - 640) / 64.0)
    assert background["when"]["relative_sec"] == pytest.approx(3.0)


def test_background_excludes_lower_boundary_utility_and_sanitizes_legacy_score():
    class FakeDemo:
        def state_at(self, *_args, **_kwargs): return []
        def match_player(self, *_args, **_kwargs): return SimpleNamespace(score=0.0, name="", steamid="")
        def kills_between(self, *_args): return []
        def utilities_between(self, *_args):
            return [
                {"_event": "detonate", "det_tick": 0, "name": "old"},
                {"_event": "detonate", "det_tick": 64, "name": "new"},
            ]

    meta = {"round_no": 1, "freeze_end_tick": 0}
    align = RoundTimeAlign(meta, 64.0)
    for video_time, timer_sec in ((100, 115), (101, 114), (102, 113)):
        align.observe_timer(_timer_observation(video_time, timer_sec))
    assert align.solve(source_start_sec=100, source_end_sec=120)["status"] == "locked"

    background, tick = build_background_info(
        demo=FakeDemo(), round_meta=meta, align=align, video_time=101,
        pov_ocr_result={"raw_text": "", "engine": "fake"},
        timer_ocr_result={"normalized": "1:54", "alignment_status": "accepted"},
        score_ocr_result={"ct": 12, "t": 1, "raw": "STAGE112-1MATCH"},
        prev_tick=0,
    )

    assert tick == 64
    assert [item["name"] for item in background["events"]["utilities"]] == ["new"]
    score = background["events"]["score_ocr"]
    assert score["pair_status"] == "legacy_unverified"
    assert score["ct"] is None and score["t"] is None
