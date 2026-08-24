"""原子事实 ID 稳定性与冲突测试（§7.3）。"""

from __future__ import annotations

import pytest

from sbmachine.commentary_planner import _fact_fingerprint, build_atomic_fact_units


def _kill_row(tick: int, attacker: str, victim: str, *, weapon: str = "AK-47", suppressed: bool = False) -> dict:
    row = {
        "event_id": f"kill:{tick}:{attacker}:{victim}",
        "type": "kill",
        "hard_fact": True,
        "event_tick": tick,
        "attacker": attacker,
        "victim": victim,
        "weapon": weapon,
    }
    if suppressed:
        row["suppressed_reason"] = "lower_priority"
    return row


def _plan(ledger: list[dict], *, t_start: float = 12.0, t_end: float = 15.0, kind: str = "kill_topic") -> dict:
    return {
        "version": 2,
        "ownership": {"t_start": t_start, "t_end": t_end, "include_end": False},
        "main_topic": {"kind": kind, "summary": "测试", "priority_class": "highlight"},
        "selected_actions": [],
        "event_ledger": ledger,
    }


class TestFactIdStability:
    def test_array_reorder_keeps_ids(self):
        ledger_a = [_kill_row(360, "JDC", "Tauson"), _kill_row(400, "s1mple", "device")]
        ledger_b = [_kill_row(400, "s1mple", "device"), _kill_row(360, "JDC", "Tauson")]
        result_a = build_atomic_fact_units("r001_w03", _plan(ledger_a))
        result_b = build_atomic_fact_units("r001_w03", _plan(ledger_b))
        assert result_a["required_fact_ids"] == result_b["required_fact_ids"]

    def test_fact_id_format(self):
        result = build_atomic_fact_units("r001_w03", _plan([_kill_row(360, "JDC", "Tauson")]))
        fact_id = result["required_fact_ids"][0]
        assert fact_id.startswith("fact:v1:r001_w03:kill:00360:")
        assert len(fact_id.rsplit(":", 1)[1]) == 8

    def test_tick_padded_to_5_digits(self):
        result = build_atomic_fact_units("r001_w03", _plan([_kill_row(90, "JDC", "Tauson")]))
        assert "kill:00090:" in result["required_fact_ids"][0]

    def test_derived_origin_and_range(self):
        ledger = [_kill_row(360, "JDC", "Tauson"), {
            "event_id": "round_end:450",
            "type": "round_end",
            "hard_fact": True,
            "event_tick": 450,
            "winner": "CT",
        }]
        result = build_atomic_fact_units("r001_w03", _plan(ledger, t_start=12.0, t_end=15.0))
        units = {u["fact_id"]: u for u in result["fact_units"]}
        kill_unit = [u for u in result["fact_units"] if u["kind"] == "kill"][0]
        round_unit = [u for u in result["fact_units"] if u["kind"] == "round_result"][0]
        assert kill_unit["origin"] == "event"
        assert kill_unit["source_tick_range"] == [360, 360]
        assert round_unit["origin"] == "derived"
        assert round_unit["source_tick_range"] == [360, 450]
        assert kill_unit["anchor_tick"] == 360
        assert round_unit["anchor_tick"] == 450

    def test_duplicate_events_deduplicated(self):
        ledger = [_kill_row(360, "JDC", "Tauson"), _kill_row(360, "JDC", "Tauson")]
        result = build_atomic_fact_units("r001_w03", _plan(ledger))
        assert len(result["fact_units"]) == 1
        assert len(result["required_fact_ids"]) == 1

    def test_weapon_does_not_change_id(self):
        ledger_a = [_kill_row(360, "JDC", "Tauson", weapon="AK-47")]
        ledger_b = [_kill_row(360, "JDC", "Tauson", weapon="AWP")]
        result_a = build_atomic_fact_units("r001_w03", _plan(ledger_a))
        result_b = build_atomic_fact_units("r001_w03", _plan(ledger_b))
        assert result_a["required_fact_ids"] == result_b["required_fact_ids"]

    def test_collision_fails_closed(self, monkeypatch):
        real = _fact_fingerprint

        def forced_collision(kind, anchor_tick, row):
            fp, payload = real(kind, anchor_tick, row)
            return "deadbeef", payload

        monkeypatch.setattr("sbmachine.commentary_planner._fact_fingerprint", forced_collision)
        ledger = [
            _kill_row(360, "JDC", "Tauson"),
            _kill_row(360, "JDC", "device"),
        ]
        with pytest.raises(ValueError, match="collision"):
            build_atomic_fact_units("r001_w03", _plan(ledger))

    def test_suppressed_events_excluded(self):
        ledger = [_kill_row(360, "JDC", "Tauson", suppressed=True)]
        result = build_atomic_fact_units("r001_w03", _plan(ledger))
        assert result["fact_units"] == []
        assert result["required_fact_ids"] == []

    def test_silence_returns_empty(self):
        result = build_atomic_fact_units("r001_w03", _plan([], kind="silence"))
        assert result["fact_units"] == []
        assert result["target_units"] == 0

    def test_legacy_topic_ids_not_present(self):
        result = build_atomic_fact_units("r001_w03", _plan([_kill_row(360, "JDC", "Tauson")]))
        for fact_id in result["required_fact_ids"]:
            assert not fact_id.startswith("topic:")

    def test_required_sorted_by_priority_then_tick(self):
        ledger = [
            _kill_row(360, "JDC", "Tauson"),
            {"event_id": "round_end:450", "type": "round_end", "hard_fact": True, "event_tick": 450, "winner": "CT"},
        ]
        result = build_atomic_fact_units("r001_w03", _plan(ledger))
        kinds = [u["kind"] for u in result["fact_units"]]
        assert kinds == ["kill", "round_result"]
