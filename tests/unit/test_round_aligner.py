from sbmachine.round_aligner import _needleman_wunsch_align, align_segments


def test_round_aligner_skeleton_keeps_unmatched_when_no_evidence():
    video_segments = [{"start_sec": 0.0, "end_sec": 10.0}]
    demo_rounds = []

    result = align_segments(video_segments, demo_rounds, tick_rate=64.0)

    assert result[0].demo_round_hint == "unmatched"
    assert result[0].align_method == "unmatched"


def test_duration_dp_backtracks_from_best_subsequence_tail():
    assert _needleman_wunsch_align([10.0], [10.0, 12.0], gap_penalty=8.0) == [0]


def test_align_l0_score_tie_returns_first_seen_round_no():
    from sbmachine.round_aligner import align_l0_score

    # ct+t+1: 帧1 -> round 3, 帧2 -> round 5, 各出现一次(平票)
    # 平票时须返回先遇到的键(round 3)
    frames = [
        {"ct": 1, "t": 1},   # round_no = 3
        {"ct": 2, "t": 2},   # round_no = 5
    ]
    assert align_l0_score({}, frames) == 3

    # 反向顺序 -> 先遇到 5
    frames_rev = [
        {"ct": 2, "t": 2},   # round_no = 5
        {"ct": 1, "t": 1},   # round_no = 3
    ]
    assert align_l0_score({}, frames_rev) == 5


def test_align_l0_score_picks_mode():
    from sbmachine.round_aligner import align_l0_score

    frames = [
        {"ct": 1, "t": 1},   # round 3
        {"ct": 2, "t": 2},   # round 5
        {"ct": 2, "t": 2},   # round 5
    ]
    assert align_l0_score({}, frames) == 5


def test_align_l0_score_none_when_empty():
    from sbmachine.round_aligner import align_l0_score

    assert align_l0_score({}, []) is None
    assert align_l0_score({}, [{"ct": None, "t": 1}]) is None
