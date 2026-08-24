"""phase3b 诊断落盘：_init_style_diagnostics / _write_style_diagnostic 写脱敏 JSONL。"""
from __future__ import annotations

import json

from sbmachine import phase3b_style


def _reset_diagnostics():
    phase3b_style._DIAGNOSTICS_DIR = None
    phase3b_style._DIAGNOSTICS_RUN_ID = ""


def test_style_diagnostics_write_summarized_jsonl(tmp_path, monkeypatch):
    _reset_diagnostics()
    monkeypatch.setattr(phase3b_style, "_DIAGNOSTICS_LOCK", __import__("threading").Lock())
    try:
        phase3b_style._init_style_diagnostics(tmp_path, "run-ab12")
        meta = {
            "finish_reason": "length",
            "http_status": 200,
            "reasoning_chars": 1200,
            "usage": {
                "prompt_tokens": 5000,
                "completion_tokens": 688,
                "prompt_cache_hit_tokens": 4800,
                "prompt_cache_miss_tokens": 200,
                "total_tokens": 5688,
                "completion_tokens_details": {"reasoning_tokens": 688},
            },
        }
        phase3b_style._write_style_diagnostic(
            2, "r002_w05", 4, 1, 4096, False, "response_error", meta,
        )
        phase3b_style._write_style_diagnostic(
            2, "r002_w05", 4, 2, 4096, True, "", {"finish_reason": "stop"},
        )

        lines = (tmp_path / "diagnostics" / "phase3b" / "run-ab12_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["run_id"] == "run-ab12"
        assert first["round_no"] == 2
        assert first["window_id"] == "r002_w05"
        assert first["attempt"] == 1
        assert first["max_tokens"] == 4096
        assert first["validation_ok"] is False
        assert first["validation_reason"] == "response_error"
        assert first["finish_reason"] == "length"
        assert first["reasoning_chars"] == 1200
        assert first["usage"]["reasoning_tokens"] == 688
        # 脱敏：不落 prompt/响应正文
        raw = lines[0]
        assert "messages" not in raw and "content" not in raw

        second = json.loads(lines[1])
        assert second["validation_ok"] is True
        assert "usage" not in second  # meta 无 usage 时不写该键
    finally:
        _reset_diagnostics()


def test_style_diagnostics_noop_before_init(tmp_path):
    _reset_diagnostics()
    phase3b_style._write_style_diagnostic(1, "r001_w01", 0, 0, 128, False, "x", {})
    assert list(tmp_path.iterdir()) == []
