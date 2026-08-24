from fractions import Fraction

from sbmachine.media_clock import (
    MediaClockAdapter,
    round_half_even,
    seconds_to_pts,
    seconds_to_sample,
)


def test_fraction_rounding_is_half_even():
    assert round_half_even(Fraction(1, 2)) == 0
    assert round_half_even(Fraction(3, 2)) == 2
    assert round_half_even(Fraction(5, 2)) == 2


def test_seconds_map_to_pts_with_origin_and_nonzero_stream_start():
    assert seconds_to_pts(1.5, "1/2", 100) == 103
    assert seconds_to_pts(1.5, "1/2", 100, timeline_origin_sec=0.5) == 102


def test_sample_mapping_and_adapter_use_integer_endpoints():
    assert seconds_to_sample(0.5, 32000) == 16000
    adapter = MediaClockAdapter.from_probe({
        "time_base": "1/90000",
        "stream_start_pts": 9000,
        "stream_start_time_sec": 0.1,
        "source_sha256": "video-sha",
    }, sample_rate=32000)
    interval = adapter.map_interval(0.1, 1.1)
    assert interval["expected_start_pts"] == 18000
    assert interval["expected_end_pts"] == 108000
    assert interval["timeline_start_sample"] == 3200
    assert interval["timeline_end_sample"] == 35200
