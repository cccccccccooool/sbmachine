from __future__ import annotations

from sbmachine.progress_events import ProgressEvent, ProgressEventReader, ProgressEventWriter
from sbmachine.run_all import _forward_progress_event


def test_writer_and_reader_preserve_a_valid_stage_progress_event(tmp_path):
    """Catches JSONL transport dropping the actual completed/total payload."""
    path = tmp_path / "diagnostics" / "progress" / "phase3.jsonl"
    writer = ProgressEventWriter(path, run_id="run-1")

    assert writer.emit(
        event="stage_progress",
        stage="phase3a",
        completed=1,
        total=2,
        unit="round",
    )

    reader = ProgressEventReader(path, run_id="run-1")
    assert reader.read_available() == [
        ProgressEvent(
            run_id="run-1",
            sequence=1,
            event="stage_progress",
            stage="phase3a",
            completed=1,
            total=2,
            unit="round",
            detail=None,
        )
    ]


def test_reader_rejects_wrong_run_and_out_of_order_events_without_returning_them(tmp_path):
    """Catches foreign or stale JSONL records from altering the current run's UI."""
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            (
                '{"schema_version":1,"run_id":"other","sequence":1,"event":"stage_start","stage":"phase3a"}',
                '{"schema_version":1,"run_id":"run-1","sequence":2,"event":"stage_start","stage":"phase3a"}',
                '{"schema_version":1,"run_id":"run-1","sequence":1,"event":"stage_progress","stage":"phase3a","completed":1,"total":2,"unit":"round"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    reader = ProgressEventReader(path, run_id="run-1")

    assert [event.sequence for event in reader.read_available()] == [2]
    assert reader.summary["wrong_run_id"] == 1
    assert reader.summary["out_of_order"] == 1


def test_parent_forwards_child_work_complete_as_validation_progress():
    """Catches a child work-complete event turning into an unauthorized done event."""
    calls: list[tuple] = []
    callbacks = {"on_stage_progress": lambda *args: calls.append(args)}

    _forward_progress_event(
        callbacks,
        ProgressEvent(
            run_id="run-1",
            sequence=4,
            event="stage_work_complete",
            stage="phase3a",
            completed=2,
            total=2,
            unit="round",
            detail=None,
        ),
    )

    assert calls == [("phase3a", 2, 2, "round", "处理完成，等待门禁")]
