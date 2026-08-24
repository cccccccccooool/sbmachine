"""empty_round_decision（Phase3b→Phase4 空回合三选一决策钩子）的单元测试。"""
from __future__ import annotations

import io
import json

from sbmachine.empty_round_decision import decide_empty_rounds, has_empty_rounds
from sbmachine import run_all as run_all_module


class _FakePrompt:
    def __init__(self, value: str) -> None:
        self.buf = io.StringIO(value)

    def __call__(self, _text: str) -> str:
        return self.buf.readline()


def test_has_empty_rounds_false_without_empty():
    manifest = {"rounds": [{"round_no": 1, "status": "ok"}, {"round_no": 2, "status": "silent"}]}
    assert has_empty_rounds(manifest) is False


def test_has_empty_rounds_true_with_empty():
    manifest = {"rounds": [{"round_no": 1, "status": "ok"}, {"round_no": 2, "status": "empty"}]}
    assert has_empty_rounds(manifest) is True


def test_decide_returns_continue_when_no_empty():
    manifest = {"rounds": [{"round_no": 1, "status": "ok"}]}
    assert decide_empty_rounds(manifest, prompt=_FakePrompt("retry\n")) == "continue"


def test_decide_continue():
    manifest = {"rounds": [{"round_no": 1, "status": "empty"}]}
    assert decide_empty_rounds(manifest, prompt=_FakePrompt("continue\n")) == "continue"


def test_decide_retry():
    manifest = {"rounds": [{"round_no": 1, "status": "empty"}]}
    assert decide_empty_rounds(manifest, prompt=_FakePrompt("retry\n")) == "retry"


def test_decide_cancel():
    manifest = {"rounds": [{"round_no": 1, "status": "empty"}]}
    assert decide_empty_rounds(manifest, prompt=_FakePrompt("cancel\n")) == "cancel"


def test_decide_shortcut_retry_digit():
    manifest = {"rounds": [{"round_no": 1, "status": "empty"}]}
    assert decide_empty_rounds(manifest, prompt=_FakePrompt("2\n")) == "retry"


def test_decide_shortcut_cancel_letter():
    manifest = {"rounds": [{"round_no": 1, "status": "empty"}]}
    assert decide_empty_rounds(manifest, prompt=_FakePrompt("x\n")) == "cancel"


def test_decide_garbage_falls_back_to_continue():
    manifest = {"rounds": [{"round_no": 1, "status": "empty"}]}
    assert decide_empty_rounds(manifest, prompt=_FakePrompt("不认识的输入\n")) == "continue"


def test_run_all_injects_tui_prompt_callback(tmp_path):
    commentary_path = tmp_path / "commentary.json"
    commentary_path.write_text(
        json.dumps({"rounds": [{"round_no": 1, "status": "empty"}]}),
        encoding="utf-8",
    )
    prompts = []

    def tui_prompt(text: str) -> str:
        prompts.append(text)
        return "retry"

    action = run_all_module._decide_empty_rounds_from_commentary(
        {"paths": {"commentary_json": str(commentary_path)}},
        {"prompt_empty_rounds": tui_prompt},
    )

    assert action == "retry"
    assert len(prompts) == 1
    assert "round_no=[1]" in prompts[0]
