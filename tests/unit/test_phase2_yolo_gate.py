import sys
import types

import numpy as np
import pytest

from sbmachine.phase2_yolo_gate import YoloGate


class FakeModel:
    def __init__(self, result=None):
        self.result = result
        self.calls = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, frame, **kwargs):
        self.calls.append({"frame": frame, "kwargs": kwargs})
        return [self.result]


class FakeBox:
    def __init__(self, class_id, confidence, box):
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])
        self.xyxy = np.array([box], dtype=float)


def make_gate(result=None, *, skip_labels=()):
    gate = object.__new__(YoloGate)
    gate.config = {"white_frame_mean_threshold": 245}
    gate.model = FakeModel(result)
    gate.conf_threshold = 0.35
    gate.skip_labels = set(skip_labels)
    gate.pov_name_labels = {"pov_name", "pov_name_area", "pov_player_bar", "pov_marker_bar"}
    gate.ocr_labels = {"timer", "timer_area", "round_timer"}
    gate.c4_labels = {"c4", "c4_area", "c4_status"}
    gate.coordinate_only_labels = {"minimap", "radar", "minimap_area", "top_hud", "top_ui", "score", "score_area"}
    return gate


def test_init_requires_a_configured_model_path():
    with pytest.raises(ValueError, match="model_path"):
        YoloGate({})


def test_init_rejects_a_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        YoloGate({"model_path": str(tmp_path / "missing.pt")})


def test_init_loads_existing_model_on_cpu(monkeypatch, tmp_path):
    model_path = tmp_path / "ui.pt"
    model_path.write_bytes(b"fake")
    model = FakeModel()
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = lambda path: model
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    gate = YoloGate({"model_path": str(model_path), "conf_threshold": 0.6})

    assert gate.model_path == str(model_path)
    assert gate.conf_threshold == 0.6
    assert model.device == "cpu"


def test_decide_short_circuits_near_white_frames_before_model_call():
    gate = make_gate()

    decision = gate.decide(np.full((4, 4, 3), 255, dtype=np.uint8))

    assert decision.reason == "flash_or_white_frame"
    assert decision.tags == ["flash"]
    assert gate.model.calls == []


def test_decide_returns_no_signal_when_all_detections_are_below_threshold():
    result = types.SimpleNamespace(names={0: "timer"}, boxes=[FakeBox(0, 0.2, [1, 2, 3, 4])])
    gate = make_gate(result)

    decision = gate.decide(np.zeros((4, 4, 3), dtype=np.uint8))

    assert decision.reason == "no_ui_yolo_signal"
    assert decision.confidence == 0.0
    assert decision.background["regions"] == []


def test_decide_routes_supported_regions_and_preserves_unknown_detections():
    result = types.SimpleNamespace(
        names={0: "timer", 1: "c4_status", 2: "left_player_hud_group", 3: "unknown_widget"},
        boxes=[
            FakeBox(0, 0.91, [1, 2, 3, 4]),
            FakeBox(1, 0.82, [5, 6, 7, 8]),
            FakeBox(2, 0.73, [9, 10, 11, 12]),
            FakeBox(3, 0.64, [13, 14, 15, 16]),
        ],
    )
    gate = make_gate(result)

    decision = gate.decide(np.zeros((20, 20, 3), dtype=np.uint8))

    assert decision.reason == "ui_yolo_signal"
    assert decision.confidence == pytest.approx(0.91)
    background = decision.background
    assert [item["label"] for item in background["ocr_regions"]] == ["timer"]
    assert [item["label"] for item in background["c4_regions"]] == ["c4_status"]
    assert [item["screen_side"] for item in background["player_hud_groups"]] == ["left"]
    assert [item["label"] for item in background["loose_detections"]] == ["unknown_widget"]
    assert "send_to_c4_detector" in background["c4_regions"][0]
    assert "planted" not in background["c4_regions"][0]


def test_decide_skip_label_retains_structured_background():
    result = types.SimpleNamespace(names={0: "top_hud"}, boxes=[FakeBox(0, 0.8, [1, 1, 3, 3])])
    gate = make_gate(result, skip_labels={"top_hud"})

    decision = gate.decide(np.zeros((4, 4, 3), dtype=np.uint8))

    assert decision.reason == "ui_yolo_skip_label"
    assert decision.background["coordinate_only_regions"][0]["label"] == "top_hud"


def test_structure_background_marks_pov_for_ocr_and_keeps_role_resolution_fail_closed():
    gate = make_gate()

    background = gate.structure_background([
        {"label": "pov_name", "confidence": 0.9, "box": [1, 2, 3, 4]},
        {"label": "minimap", "confidence": 0.8, "box": [5, 6, 7, 8]},
    ])

    assert background["ocr_regions"][0]["send_to_ocr"] is True
    assert background["coordinate_only_regions"][0]["type"] == "minimap"
    assert background["role_resolution"]["status"] == "region_router_only"