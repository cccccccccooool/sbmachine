import numpy as np
import pytest

from vision_service.region_crops import (
    box_from_norm,
    clip_box,
    crop_frame,
    mask_regions,
    region_type,
    screen_side_for_label,
)


@pytest.mark.parametrize(("label", "expected"), [
    ("left_player_hud_group", "player_hud_group"),
    ("killfeed_area", "killfeed"),
    ("round_timer", "timer"),
    ("c4_status", "c4"),
    ("radar", "minimap"),
    ("score_area", "top_hud"),
    ("pov_marker_bar", "pov_name"),
    ("future_widget", "unknown"),
    (None, "unknown"),
])
def test_region_type_maps_supported_aliases_and_fails_closed(label, expected):
    assert region_type(label) == expected


@pytest.mark.parametrize(("label", "expected"), [
    ("left_player_hud_group", "left"),
    ("right_player_hud_group", "right"),
    ("timer", "unknown"),
])
def test_screen_side_for_label_uses_only_explicit_prefix(label, expected):
    assert screen_side_for_label(label) == expected


def test_clip_box_rounds_pads_and_clamps_to_frame_bounds():
    assert clip_box([1.6, 2.4, 9.6, 8.4], 10, 10, padding=2) == [0, 0, 10, 10]


@pytest.mark.parametrize("box", [[1, 2, 3], [4, 4, 2, 8], [1, 1, 1, 3]])
def test_clip_box_rejects_malformed_or_empty_boxes(box):
    with pytest.raises(ValueError):
        clip_box(box, 10, 10)


def test_box_from_norm_scales_each_axis_independently():
    assert box_from_norm([0.1, 0.2, 0.9, 0.8], 1920, 1080) == [192.0, 216.0, 1728.0, 864.0]


def test_crop_frame_returns_an_independent_copy():
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape((6, 8, 3))

    crop = crop_frame(frame, [2, 1, 6, 5])
    crop[:] = 0

    assert crop.shape == (4, 4, 3)
    assert np.any(frame[1:5, 2:6] != 0)


def test_mask_regions_masks_valid_boxes_and_skips_invalid_boxes():
    frame = np.full((5, 6, 3), 255, dtype=np.uint8)

    masked = mask_regions(
        frame,
        [{"box": [1, 1, 4, 4]}, {"box": [2, 2, 2, 3]}, {"missing": "box"}],
        color=(1, 2, 3),
    )

    assert np.all(masked[1:4, 1:4] == np.array([1, 2, 3]))
    assert np.all(masked[0, 0] == 255)
    assert np.all(frame == 255)