from __future__ import annotations

from sbmachine import display as display_module


class _FakeLive:
    def __init__(self) -> None:
        self.events = []

    def stop(self) -> None:
        self.events.append("stop")

    def start(self, *, refresh: bool) -> None:
        self.events.append(("start", refresh))


class _FakeConsole:
    def __init__(self, value: str = "continue") -> None:
        self.value = value
        self.printed = []
        self.inputs = []

    def print(self, value) -> None:
        self.printed.append(value)

    def input(self, prompt: str) -> str:
        self.inputs.append(prompt)
        return self.value


def _clock(monkeypatch, seconds: float) -> None:
    monkeypatch.setattr(display_module.time, "monotonic", lambda: seconds)


def test_determinate_progress_uses_completed_units_not_global_elapsed(monkeypatch):
    """Catches a bar that is driven by elapsed time instead of 2/4 work units."""
    _clock(monkeypatch, 0.0)
    view = display_module._Display(["phase2"])
    view.on_stage_start("phase2")

    _clock(monkeypatch, 42.0)
    view.on_stage_progress("phase2", 2, 4, "round", None)

    assert view._bar("phase2").plain == "█" * 10 + "▒" * 10
    assert "2/4 round" in view._status_text("phase2").plain


def test_completed_work_waits_for_done_before_green_completion(monkeypatch):
    """Catches treating completed == total as a published, green stage."""
    _clock(monkeypatch, 0.0)
    view = display_module._Display(["phase3a"])
    view.on_stage_start("phase3a")
    view.on_stage_progress("phase3a", 3, 3, "round", None)

    assert view.status["phase3a"] == "running"
    assert view._bar("phase3a").plain == "█" * 20
    assert "校验" in view._status_text("phase3a").plain


def test_indeterminate_progress_uses_the_stage_clock_without_fake_percentage(monkeypatch):
    """Catches a late-starting stage inheriting a full global elapsed-time bar."""
    _clock(monkeypatch, 0.0)
    view = display_module._Display(["phase2", "phase3a"])
    view.on_stage_start("phase2")

    _clock(monkeypatch, 42.0)
    view.on_stage_start("phase3a")

    bar = view._bar("phase3a").plain
    assert bar != "█" * 20
    assert bar.count("█") <= 1
    assert "运行中" in view._status_text("phase3a").plain


def test_invalid_and_terminal_progress_events_do_not_mutate_display_state(monkeypatch):
    """Catches overflow or post-done events reviving a stage or falsifying its ratio."""
    _clock(monkeypatch, 0.0)
    view = display_module._Display(["phase4"])
    view.on_stage_start("phase4")
    view.on_stage_progress("phase4", 1, 2, "round", None)
    view.on_stage_progress("phase4", 3, 2, "round", None)
    view.on_stage_done("phase4")
    view.on_stage_progress("phase4", 2, 2, "round", None)

    assert view.status["phase4"] == "done"
    assert view.completed["phase4"] == 1
    assert view.total["phase4"] == 2


def test_empty_round_prompt_pauses_live_and_restores_it():
    console = _FakeConsole("continue")
    live = _FakeLive()

    choice = display_module._prompt_empty_rounds(console, live, "空回合：round_no=[1]")

    assert choice == "continue"
    assert live.events == ["stop", ("start", True)]
    assert len(console.printed) == 1
    assert "continue / retry / cancel" in console.inputs[0]
