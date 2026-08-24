from sbmachine.phase2_quality import OcrBudget, build_alignment_warning, coalesce_yolo_gaps


def test_yolo_missing_samples_are_reported_as_non_blocking_time_ranges():
    warnings = coalesce_yolo_gaps([1.0, 3.0, 6.0, 15.0], max_gap_sec=3.0)

    assert warnings == [
        {
            "type": "yolo_no_detection",
            "start_sec": 1.0,
            "end_sec": 6.0,
            "sample_count": 3,
            "message": "YOLO 未检测到 UI；该时段继续输出 DEM 时间轴事实",
        },
        {
            "type": "yolo_no_detection",
            "start_sec": 15.0,
            "end_sec": 15.0,
            "sample_count": 1,
            "message": "YOLO 未检测到 UI；该时段继续输出 DEM 时间轴事实",
        },
    ]


def test_ocr_budget_counts_roi_and_enforces_normal_and_hard_limits():
    budget = OcrBudget(10, normal_ratio=1.0, degraded_extra_ratio=0.2)
    assert budget.can_consume(1) is True
    budget.consume("timer", 8, variant_calls=12)
    budget.consume("score", 2, variant_calls=4)
    assert budget.actual_roi_calls == 10
    assert budget.can_consume(1) is False
    assert budget.can_consume(2, degraded=True) is True
    budget.consume("timer", 2)
    assert budget.can_consume(1, degraded=True) is False
    assert budget.budget_exhausted is True
    assert budget.summary()["hard_limit"] == 12
    assert budget.summary()["change_percent"] == 20.0


def test_alignment_warning_keeps_failure_classes_separate():
    warning = build_alignment_warning(
        {"status": "locked", "score_fact_support": "unsupported", "score_evidence": "none", "c4_evidence": "unsupported"},
        [
            {"parse_status": "not_scheduled", "alignment_status": "not_scheduled"},
            {"parse_status": "ocr_empty", "alignment_status": "pending"},
            {"parse_status": "parsed", "alignment_status": "accepted"},
        ],
        [{"pair_status": "incomplete"}],
        {"actual_roi_calls": 3, "budget_exhausted": False},
    )
    assert warning["timer"]["not_scheduled"] == 1
    assert warning["timer"]["ocr_empty"] == 1
    assert warning["timer"]["accepted"] == 1
    assert warning["score"]["score_fact_support"] == "unsupported"
