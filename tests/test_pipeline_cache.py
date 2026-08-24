import json
import subprocess
from pathlib import Path

from sbmachine import run_all
from sbmachine import phase1_preprocess_slice


def test_video_marking_passes_slicer_v2_args_and_returns_segments(tmp_path, monkeypatch):
    video = tmp_path / "match.mp4"
    model = tmp_path / "model.pt"
    demo_dir = tmp_path / "demo"
    out_jsonl = tmp_path / "staging" / "detector_rows.jsonl"
    out_segments = tmp_path / "staging" / "segments.json"
    video.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    demo_dir.mkdir()
    (demo_dir / "rounds.json").write_text("[]", encoding="utf-8")
    calls = []

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            calls.append(cmd)
        def poll(self):
            return 0
        returncode = 0

    monkeypatch.setattr("sbmachine.run_all.subprocess.Popen", FakeProcess)

    detections, segments = run_all._run_video_marking(
        {
            "video": str(video),
            "demo_output_dir": str(demo_dir),
            "hud_detections_jsonl": str(out_jsonl),
            "segments_out_json": str(out_segments),
        },
        {
            "model": str(model),
            "interval_sec": 0.5,
            "smooth_window": 7,
            "min_live_sec": 12,
            "bridge_gap_sec": 2,
            "device": "cpu",
            "workers": 3,
        },
        use_gpu_guard=False,
    )

    cmd = calls[0]
    assert detections.name == "detector_rows.jsonl"
    assert segments.name == "segments.json"
    assert cmd[cmd.index("--segment-output") + 1] == str(segments)
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--workers") + 1] == "3"
    assert "--hud-model" not in cmd
    assert "--timer-roi" not in cmd
    assert "--demo-rounds" in cmd


def test_preprocess_uses_video_marking_segments_before_configured_segments(monkeypatch, tmp_path):
    monkeypatch.setattr(run_all, "PACKAGE_ROOT", tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    video = tmp_path / "match.mp4"
    video.write_text("", encoding="utf-8")
    configured_segments = tmp_path / "configured_segments.json"
    slicer_segments = tmp_path / "slicer_segments.json"
    slicer_model = tmp_path / "model.pt"
    configured_segments.write_text(json.dumps({"segments": []}), encoding="utf-8")
    slicer_segments.write_text(json.dumps({"segments": []}), encoding="utf-8")
    slicer_model.write_text("", encoding="utf-8")
    (config_dir / "pipeline.yaml").write_text(
        f"""
runtime:
  manage_services: true
  use_gpu_guard: false
phases:
  demo_parse: false
  video_marking: true
  preprocess_slice: true
  phase2_yolo: false
  phase3a_semantic: false
  phase3b_semantic: false
  phase4_assemble: false
paths:
  video: {video.as_posix()}
  segments_json: {configured_segments.as_posix()}
  rounds_json: {(tmp_path / "rounds.json").as_posix()}
  round_list_json: {(tmp_path / "round_list.json").as_posix()}
  segments_out_json: {(tmp_path / "segments_out.json").as_posix()}
slicer:
  model: {slicer_model.as_posix()}
""",
        encoding="utf-8",
    )
    seen = {}

    def fake_video_marking(paths, slicer_config, use_gpu_guard, log_path=None, **kwargs):
        detector = tmp_path / "detector.jsonl"
        detector.write_text("", encoding="utf-8")
        return detector, slicer_segments

    monkeypatch.setattr(run_all, "_run_video_marking", fake_video_marking)

    def fake_preprocess(**kwargs):
        seen.update(kwargs)
        kwargs["output_rounds_path"].parent.mkdir(parents=True, exist_ok=True)
        round_data = {"round_no": 1, "start_sec": 0.0, "end_sec": 10.0}
        kwargs["output_rounds_path"].write_text(
            json.dumps({"video_path": str(video), "rounds": [round_data]}), encoding="utf-8"
        )
        kwargs["output_list_path"].write_text(json.dumps({"rounds": [round_data]}), encoding="utf-8")
        kwargs["output_segments_path"].write_text(
            json.dumps([{"start_sec": 0.0, "end_sec": 10.0}]), encoding="utf-8"
        )
        return None

    monkeypatch.setattr(run_all, "run_preprocess_slice", fake_preprocess)
    monkeypatch.setattr(run_all, "_run_phases_subprocess", lambda *args, **kwargs: None)

    run_all.run_all(config_dir)

    assert seen["segments_path"] == slicer_segments
    assert seen["detections_path"] == tmp_path / "detector.jsonl"
