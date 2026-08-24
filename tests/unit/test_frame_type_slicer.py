import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.slicing import run_frame_type_slicer as slicer


def _args(tmp_path, **overrides):
    values = {
        "video": str(tmp_path / "video.mp4"),
        "model": str(tmp_path / "model.pt"),
        "replay_model": "",
        "replay_roi": "",
        "replay_threshold": 0.65,
        "frame_output": str(tmp_path / "frames.jsonl"),
        "segment_output": str(tmp_path / "segments.json"),
        "demo_rounds": "",
        "markers": "",
        "interval_sec": 1.0,
        "start_sec": 0.0,
        "end_sec": None,
        "smooth_window": 1,
        "game_label": "game",
        "live_label": "game",
        "replay_label": "replay_marker",
        "min_live_sec": 0.0,
        "bridge_gap_sec": 3.0,
        "device": "cpu",
        "workers": 2,
        "progress_every": 25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _stub_model(monkeypatch):
    monkeypatch.setattr(slicer, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(slicer, "load_checkpoint", lambda *_args: (object(), ["game", "break"], 224, {}))


def test_replay_rows_split_live_segments_and_legacy_rows_default_to_live():
    rows = [
        {"time_sec": 0.0, "smooth_label": "game"},
        {"time_sec": 1.0, "smooth_label": "game"},
        {"time_sec": 2.0, "smooth_label": "game", "is_replay": True},
        {"time_sec": 3.0, "smooth_label": "game", "is_replay": True},
        {"time_sec": 4.0, "smooth_label": "game"},
        {"time_sec": 5.0, "smooth_label": "game"},
    ]

    segments = slicer.build_segments_v2(
        rows,
        live_label="game",
        min_live_sec=0.0,
        bridge_gap_sec=3.0,
    )

    assert [(segment["start_sec"], segment["end_sec"]) for segment in segments] == [(0.0, 1.0), (4.0, 5.0)]
    assert segments[0]["bridge_decisions"][-1]["reason"] == "replay_gap"


def test_unknown_duration_falls_back_to_one_worker_and_reads_to_eof(tmp_path, monkeypatch):
    _stub_model(monkeypatch)
    monkeypatch.setattr(slicer, "get_video_duration_sec", lambda _path: None)
    monkeypatch.setattr(slicer, "predict_frame", lambda *_args: {"label": "game", "confidence": 0.9})
    seen_end_secs = []

    def fake_frames(_path, _interval_sec, _start_sec, end_sec):
        seen_end_secs.append(end_sec)
        yield 0.0, object()

    monkeypatch.setattr(slicer, "iter_video_frames", fake_frames)

    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_context", lambda *_args: pytest.fail("multiprocessing must not start"))
    args = _args(tmp_path)

    frame_output, segment_output = slicer.run(args)

    assert seen_end_secs == [None]
    assert frame_output.read_text(encoding="utf-8").count("\n") == 1
    payload = json.loads(segment_output.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 1
    assert "segments" in payload


class _FakeProcess:
    def __init__(self, *, alive=False, exitcode=0):
        self._alive = alive
        self.exitcode = exitcode
        self.terminated = False

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False

    def join(self):
        pass


class _FakeContext:
    def __init__(self, result_queue, processes):
        self.result_queue = result_queue
        self.processes = iter(processes)

    def Queue(self):
        return self.result_queue

    def Process(self, **_kwargs):
        return next(self.processes)


def _run_with_context(tmp_path, monkeypatch, context):
    _stub_model(monkeypatch)
    monkeypatch.setattr(slicer, "get_video_duration_sec", lambda _path: 10.0)

    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_context", lambda _method: context)
    args = _args(tmp_path)
    with pytest.raises(RuntimeError) as exc_info:
        slicer.run(args)
    assert not (tmp_path / "frames.jsonl").exists()
    assert not (tmp_path / "segments.json").exists()
    return str(exc_info.value)


def test_queue_empty_uses_stdlib_exception_and_reports_dead_worker(tmp_path, monkeypatch):
    class EmptyQueue:
        def get(self, **_kwargs):
            raise queue.Empty

    context = _FakeContext(EmptyQueue(), [_FakeProcess(exitcode=7), _FakeProcess(exitcode=0)])

    message = _run_with_context(tmp_path, monkeypatch, context)

    assert message == "Worker process 0 died unexpectedly with exitcode 7"


def test_worker_error_does_not_write_success_outputs(tmp_path, monkeypatch):
    class ErrorQueue:
        def get(self, **_kwargs):
            return 0, -1, "model failed"

    processes = [_FakeProcess(alive=True), _FakeProcess(alive=True)]
    context = _FakeContext(ErrorQueue(), processes)

    message = _run_with_context(tmp_path, monkeypatch, context)

    assert message == "Worker 0 crashed: model failed"
    assert all(process.terminated for process in processes)


def test_worker_ranges_share_one_grid_without_duplicate_boundaries():
    ranges = slicer.worker_time_ranges(0.0, 10.0, 1.0, 3)
    samples = []
    for start, end in ranges:
        samples.extend([start + index for index in range(int(end - start) + 1)])

    assert samples == list(range(11))
    assert len(samples) == len(set(samples))


def test_demo_alignment_uses_manifest_tick_rate(tmp_path, monkeypatch):
    rounds_path = tmp_path / "rounds.json"
    rounds_path.write_text("[]", encoding="utf-8")
    seen = {}

    def fake_align(_segments, _rounds, tick_rate):
        seen["tick_rate"] = tick_rate
        return []

    monkeypatch.setattr(slicer, "validate_demo_manifest", lambda _path: {"tick_rate": 128.0})
    monkeypatch.setattr("sbmachine.round_aligner.align_segments", fake_align)

    slicer.validate_segments_with_demo(
        [{"start_sec": 0.0, "end_sec": 10.0}], [], rounds_path
    )

    assert seen["tick_rate"] == 128.0


def test_missing_demo_never_preserves_a_positional_round_hint():
    segments = [{"start_sec": 0.0, "end_sec": 10.0, "demo_round_hint": 1}]

    slicer.validate_segments_with_demo(segments, [], None)

    assert segments[0]["demo_round_hint"] == "unmatched"
    assert segments[0]["align_method"] == "unmatched"


def test_dual_output_publish_rolls_back_if_second_replace_fails(tmp_path, monkeypatch):
    frame_output = tmp_path / "frames.jsonl"
    segment_output = tmp_path / "segments.json"
    frame_output.write_text("old frames\n", encoding="utf-8")
    segment_output.write_text('{"old": true}', encoding="utf-8")
    real_replace = slicer.os.replace

    def fail_segment_publish(source, target):
        source_path = Path(source)
        if source_path.name.startswith(".segments.json.") and source_path.suffix == ".tmp":
            raise OSError("segment promotion failed")
        return real_replace(source, target)

    monkeypatch.setattr(slicer.os, "replace", fail_segment_publish)

    with pytest.raises(OSError, match="segment promotion failed"):
        slicer.write_outputs_atomically(
            frame_output,
            segment_output,
            [{"time_sec": 0.0}],
            {"segments": []},
        )

    assert frame_output.read_text(encoding="utf-8") == "old frames\n"
    assert segment_output.read_text(encoding="utf-8") == '{"old": true}'
