"""规则中性句渲染器测试（§8.2/§8.4）：排序、连接、capsule 完整性、无截断。"""

from __future__ import annotations

import pytest

from sbmachine.rule_neutral_renderer import (
    NEUTRAL_RENDERER_POLICY,
    NEUTRAL_SOURCE,
    RendererError,
    render_capsule,
    render_neutral,
    validate_preserved_facts,
)


def _task(fact_units, required_fact_ids=None, **extra):
    task = {
        "window_id": "r001_w03",
        "fact_units": fact_units,
        "required_fact_ids": required_fact_ids or [u["fact_id"] for u in fact_units],
        "fact_anchors": {},
        "target_units": 14,
        "hard_units": 21,
    }
    task.update(extra)
    return task


def _unit(fact_id, kind, clause, capsule, priority, anchor_tick, required=True, **extra):
    unit = {
        "fact_id": fact_id,
        "kind": kind,
        "origin": "event",
        "anchor_tick": anchor_tick,
        "source_tick_range": [anchor_tick, anchor_tick],
        "canonical_clause": clause,
        "capsule_clause": capsule,
        "required": required,
        "priority": priority,
        "attacker": extra.pop("attacker", None),
        "victim": extra.pop("victim", None),
        "winner": extra.pop("winner", None),
        "side": extra.pop("side", None),
    }
    return unit


def _kill_unit(fact_id="fact:v1:r001_w03:kill:00360:a13f92c1", attacker="JDC", victim="Tauson", anchor_tick=360):
    return _unit(
        fact_id, "kill", f"{attacker}击杀{victim}", f"{attacker}击杀{victim}", 100, anchor_tick,
        attacker=attacker, victim=victim,
    )


def _round_unit(fact_id="fact:v1:r001_w03:round_result:00450:b721da80", winner="CT", anchor_tick=450):
    return _unit(
        fact_id, "round_result", f"{winner}方赢下回合", f"{winner}胜", 90, anchor_tick, winner=winner,
    )


class TestRenderNeutral:
    def test_single_fact_uses_full_clause(self):
        unit = _kill_unit()
        out = render_neutral(_task([unit]))
        assert out["neutral"] == "JDC击杀Tauson"
        assert out["neutral_source"] == NEUTRAL_SOURCE
        assert out["renderer_policy"] == NEUTRAL_RENDERER_POLICY
        assert out["preserved_fact_ids"] == [unit["fact_id"]]

    def test_multi_fact_joiner(self):
        kill = _kill_unit()
        rnd = _round_unit()
        out = render_neutral(_task([kill, rnd]))
        assert "JDC击杀Tauson" in out["neutral"]
        assert "赢下回合" in out["neutral"]
        assert out["neutral"].count("，") >= 1

    def test_no_causal_or_contrast_words(self):
        kill = _kill_unit()
        rnd = _round_unit()
        out = render_neutral(_task([kill, rnd]))
        for word in ("因为", "所以", "但是", "可惜", "竟然"):
            assert word not in out["neutral"]

    def test_sorting_is_deterministic(self):
        units = [
            _round_unit(),
            _kill_unit(),
            _kill_unit("fact:v1:r001_w03:kill:00400:cccccccc", attacker="s1mple", victim="device", anchor_tick=400),
        ]
        a = render_neutral(_task(units))
        b = render_neutral(_task(list(reversed(units))))
        assert a["neutral"] == b["neutral"]
        assert a["neutral"].index("JDC击杀Tauson") < a["neutral"].index("s1mple击杀device")

    def test_missing_required_fails(self):
        kill = _kill_unit()
        task = _task([kill], required_fact_ids=["fact:v1:r001_w03:round_result:00450:b721da80"])
        with pytest.raises(RendererError, match="missing"):
            render_neutral(task)

    def test_empty_units_fails(self):
        with pytest.raises(RendererError):
            render_neutral(_task([]))


class TestRenderCapsule:
    def test_covers_all_required(self):
        kill = _kill_unit()
        rnd = _round_unit()
        task = _task([kill, rnd])
        capsule = render_capsule(task)
        assert "JDC击杀Tauson" in capsule
        assert "CT胜" in capsule
        assert all(fid in kill["fact_id"] or fid in rnd["fact_id"] for fid in task["required_fact_ids"])

    def test_no_truncation_of_long_clause(self):
        long_clause = "ZywOo在B点完成一次五杀后继续推进" * 3
        unit = _unit(
            "fact:v1:r001_w03:kill:00360:d0d0d0d0", "kill", long_clause, long_clause, 100, 360,
            attacker="ZywOo", victim="对方",
        )
        capsule = render_capsule(_task([unit]))
        assert capsule == long_clause

    def test_capsule_is_complete_sentences(self):
        kill = _kill_unit()
        rnd = _round_unit()
        capsule = render_capsule(_task([kill, rnd]))
        assert not capsule.endswith("…")
        assert "..." not in capsule
        assert capsule.count("，") == 1


class TestValidatePreservedFacts:
    def test_all_preserved(self):
        kill = _kill_unit()
        rnd = _round_unit()
        result = validate_preserved_facts("JDC击杀Tauson，CT随后赢下回合", [kill, rnd], [kill["fact_id"], rnd["fact_id"]])
        assert result["missing_required"] == []
        assert len(result["preserved_fact_ids"]) == 2

    def test_missing_player_detected(self):
        kill = _kill_unit()
        result = validate_preserved_facts("JDC击杀对手", [kill], [kill["fact_id"]])
        assert result["missing_required"] == [kill["fact_id"]]
        assert result["preserved_fact_ids"] == []

    def test_unexpected_fact_detected(self):
        kill = _kill_unit()
        text = "JDC击杀Tauson，随后拆除C4"
        result = validate_preserved_facts(text, [kill], [kill["fact_id"]])
        assert "bomb_defused" in result["unexpected_fact_ids"]

    def test_unexpected_latin_entity_detected(self):
        kill = _kill_unit()
        text = "JDC击杀Tauson，device也在场"
        result = validate_preserved_facts(text, [kill], [kill["fact_id"]])
        assert "device" in result["unexpected_fact_ids"]
        assert result["missing_required"] == []

    def test_silence_validation(self):
        result = validate_preserved_facts("", [], [])
        assert result["preserved_fact_ids"] == []
        assert result["missing_required"] == []
        assert result["unexpected_fact_ids"] == []
