import numpy as np

from sbmachine import phase2_ocr


def test_read_ocr_text_stops_after_accepted_variant(monkeypatch):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    calls = []

    class FakeOcr:
        def __call__(self, image):
            calls.append(image)
            return [[None, "1:23", 0.9]], None

    monkeypatch.setattr(phase2_ocr, "_get_rapid_ocr", lambda: FakeOcr())
    monkeypatch.setattr(
        phase2_ocr,
        "_variants",
        lambda crop: [("original", crop), ("enhanced", crop)],
    )

    result = phase2_ocr.read_ocr_text(
        frame,
        {"box": [0, 0, 20, 20]},
        accept_pattern=r"\d{1,2}:\d{2}",
    )

    assert result["raw_text"] == "1:23"
    assert result["variant"] == "original"
    assert len(calls) == 1


def test_read_ocr_text_keeps_fallbacks_until_result_is_accepted(monkeypatch):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    responses = iter(("noise", "2-1"))
    calls = []

    class FakeOcr:
        def __call__(self, image):
            calls.append(image)
            return [[None, next(responses), 0.9]], None

    monkeypatch.setattr(phase2_ocr, "_get_rapid_ocr", lambda: FakeOcr())
    monkeypatch.setattr(
        phase2_ocr,
        "_variants",
        lambda crop: [("original", crop), ("enhanced", crop), ("unused", crop)],
    )

    result = phase2_ocr.read_ocr_text(
        frame,
        {"box": [0, 0, 20, 20]},
        accept_pattern=r"\d+\s*[:\-]\s*\d+",
    )

    assert result["raw_text"] == "2-1"
    assert result["variant"] == "enhanced"
    assert len(calls) == 2


def test_pov_ocr_skips_when_yolo_region_has_no_white_text(monkeypatch):
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    region = {"label": "pov_name", "box": [10, 10, 70, 30], "confidence": 0.9}
    monkeypatch.setattr(phase2_ocr, "pov_white_text_ratio", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(phase2_ocr, "read_ocr_text", lambda *_args, **_kwargs: AssertionError("OCR must not run"))

    result, source, used_region = phase2_ocr._detect_pov_ocr(
        frame, {"regions": [region]}, {"white_ratio_threshold": 0.01}, 0
    )

    assert source == "yolo_pov_white_gate"
    assert used_region == region
    assert result["engine"] == "skipped:pov_white_gate"


def test_pov_ocr_reads_yolo_region_when_white_text_gate_passes(monkeypatch):
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    region = {"label": "pov_name", "box": [10, 10, 70, 30], "confidence": 0.9}
    monkeypatch.setattr(phase2_ocr, "pov_white_text_ratio", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        phase2_ocr,
        "read_ocr_text",
        lambda *_args, **_kwargs: {"raw_text": "Player", "confidence": 0.9, "engine": "fake"},
    )

    result, source, used_region = phase2_ocr._detect_pov_ocr(
        frame, {"regions": [region]}, {"white_ratio_threshold": 0.01, "min_confidence": 0.35}, 0
    )

    assert source == "yolo_pov_region"
    assert used_region == region
    assert result["raw_text"] == "Player"
    assert result["white_ratio"] == 1.0


def test_timer_observation_is_strict_and_normalizes_fullwidth_colon():
    valid = phase2_ocr.parse_timer_observation(" 1\uFF1A23 ", video_time=10, confidence=0.9)
    assert (valid["normalized"], valid["timer_sec"], valid["parse_status"]) == ("1:23", 83, "parsed")
    assert phase2_ocr.parse_timer_observation("0:00", video_time=0, confidence=0.9)["timer_sec"] == 0
    assert phase2_ocr.parse_timer_observation("1:55", video_time=0, confidence=0.9)["timer_sec"] == 115
    assert phase2_ocr.parse_timer_observation("1:60", video_time=0, confidence=0.9)["parse_status"] == "parse_rejected"
    assert phase2_ocr.parse_timer_observation("2:00", video_time=0, confidence=0.9)["parse_status"] == "parse_rejected"


def test_low_confidence_timer_remains_candidate_but_is_state_rejected():
    result = phase2_ocr.parse_timer_observation("1:23", video_time=10, confidence=0.1, min_confidence=0.35)
    assert result["parse_status"] == "parsed"
    assert result["alignment_status"] == "state_rejected"


def test_top_hud_never_enters_score_pair(monkeypatch):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    monkeypatch.setattr(phase2_ocr, "read_ocr_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not OCR top_hud")))
    result = phase2_ocr._detect_score_ocr(
        frame,
        {"regions": [{"label": "top_hud", "box": [0, 0, 200, 30], "confidence": 0.99}]},
        {},
        0,
    )
    assert result["pair_status"] == "incomplete"
    assert result["left"] is None and result["right"] is None


def test_score_pair_reads_two_single_side_regions(monkeypatch):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    responses = iter(("4", "1"))
    monkeypatch.setattr(
        phase2_ocr,
        "read_ocr_text",
        lambda *_args, **_kwargs: {"raw_text": next(responses), "confidence": 0.9, "variant_inference_calls": 1},
    )
    result = phase2_ocr._detect_score_ocr(
        frame,
        {"regions": [
            {"label": "score", "box": [150, 0, 170, 20], "confidence": 0.8},
            {"label": "score", "box": [30, 0, 50, 20], "confidence": 0.9},
        ]},
        {},
        0,
    )
    assert (result["left"]["value"], result["right"]["value"]) == (4, 1)
    assert result["ct"] is None and result["t"] is None


def test_score_pair_consensus_accepts_three_of_five_and_rejects_tie():
    def observation(left, right, confidence=0.9):
        return {
            "left": {"value": left, "ocr_confidence": confidence},
            "right": {"value": right, "ocr_confidence": confidence},
            "pair_status": "pending_consensus",
        }

    consensus = phase2_ocr.ScorePairConsensus(window=5)
    result = None
    for pair in ((4, 1), (4, 1), (3, 1), (4, 1), (5, 1)):
        result = consensus.update(observation(*pair))
    assert result["pair_status"] == "accepted_for_alignment"
    assert (result["left"]["value"], result["right"]["value"]) == (4, 1)

    tied = phase2_ocr.ScorePairConsensus(window=4, min_votes=2)
    for pair in ((4, 1), (3, 1), (4, 1), (3, 1)):
        result = tied.update(observation(*pair))
    assert result["pair_status"] == "conflict"


def test_score_side_aliases_are_exact_and_long_hud_text_is_rejected():
    for raw in ("I", "i", "L", "l", "+", "/", "\uFF0F"):
        result = phase2_ocr.parse_score_side(raw, confidence=0.9, roi_source="test")
        assert (result["value"], result["parse_status"]) == (1, "parsed")
    for raw in ("STAGE1|2-1MATCH", "STAGE112-1MATCH", "112", "31"):
        result = phase2_ocr.parse_score_side(raw, confidence=0.9, roi_source="test", max_value=30)
        assert result["value"] is None
        assert result["parse_status"] == "parse_rejected"


def test_score_pair_consensus_accepts_four_of_five():
    consensus = phase2_ocr.ScorePairConsensus(window=5)
    result = None
    for left, right in ((4, 1), (4, 1), (3, 1), (4, 1), (4, 1)):
        result = consensus.update({
            "left": {"value": left, "ocr_confidence": 0.9},
            "right": {"value": right, "ocr_confidence": 0.9},
            "pair_status": "pending_consensus",
        })
    assert result["pair_status"] == "accepted_for_alignment"
    assert (result["left"]["value"], result["right"]["value"]) == (4, 1)


def test_legacy_score_payload_is_unverified_and_cannot_become_side_pair():
    result = phase2_ocr.normalize_score_observation({
        "ct": 12, "t": 1, "raw": "STAGE112-1MATCH", "source": "legacy",
    }, video_time=10)
    assert result["pair_status"] == "legacy_unverified"
    assert result["observation_status"] == "legacy_unverified"
    assert result["left"] is None and result["right"] is None
    assert result["ct"] is None and result["t"] is None
    assert (result["legacy_ct"], result["legacy_t"]) == (12, 1)
