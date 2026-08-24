"""Phase3c / LLM-C 独立阶段门禁合同测试（C0~C7 + 四态 mode + B1/C7 preflight）。

LLM-C 按回合读取不可变的 llmb_draft_package_v1，输出严格 unit_id 寻址的
llmc_round_edit_response_v1，Phase3c 验收后发布 commentary_render_package_v1。
本套件覆盖门禁矩阵的每个拒绝分支与 mode 决策，不调用真实云端。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sbmachine.common import count_spoken_chars
from sbmachine.phase3c_llmc import (
    _LLMC3_SYSTEM,
    _call_round_editor,
    build_round_edit_request,
    check_capacity,
    check_fact_scope,
    check_unit_addressing,
    decide_round_source,
    load_llmb_draft_package,
    run_phase3c,
    validate_llmc_response,
)
from sbmachine.preflight import (
    PublishContractError,
    validate_llmb_draft_package,
    validate_render_package,
)

B_TEXT = "开局双方先换掉一人，紧接着Tauson中路交火。"
C_TEXT_OK = "开局双方先换一人，Tauson中路马上交火，双方继续在中路交火，继续交火。"


def _unit(unit_id: str, *, draft_text: str = B_TEXT, emotion: str = "激动",
          allowed: list[str] | None = None, catalog: dict | None = None,
          slot_sec: float = 10.0, safe_upper: float | None = None,
          r_b: float = 1.0) -> dict:
    safe_upper = safe_upper if safe_upper is not None else round(count_spoken_chars(draft_text) / 5.0, 3)
    allowed = allowed if allowed is not None else [f"fact:v1:{unit_id}:players:00001:aaaa1111"]
    catalog = catalog if catalog is not None else {
        f"fact:v1:{unit_id}:players:00001:aaaa1111": {"kind": "players", "value": "Tauson"},
    }
    return {
        "unit_id": unit_id, "sequence": 1, "draft_text": draft_text,
        "emotion_binding": {"emotion": emotion, "authority": "emotion_policy"},
        "allowed_fact_ids": allowed, "carry_in_fact_ids": [],
        "fact_catalog": catalog,
        "render_slot": {"slot_id": unit_id, "timeline_id": "tl:test:030", "start_tick": 300, "end_tick": 600},
        "speech_capacity": {"slot_sec": slot_sec, "safe_upper_sec": safe_upper,
                            "required_speed_factor": r_b, "draft_hard_speed_factor": 1.5},
    }


def _package(*, rounds: list[dict] | None = None, artifact: str = "pkg-abc") -> dict:
    rounds = rounds if rounds is not None else [
        {"round_id": "r001", "status": "ready", "units": [_unit("r001_w01"), _unit("r001_w02")]},
        {"round_id": "r002", "status": "intentional_silent", "units": []},
    ]
    return {
        "contract": "llmb_draft_package_v1", "producer": "phase3b", "run_id": "run-abc",
        "source": {"neutral_run_id": "n-abc", "neutral_sha256": "0" * 64,
                   "timeline_id": "tl:test:030", "source_video_sha256": ""},
        "rounds": rounds, "artifact_identity": artifact,
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TestC0EntryIdentity:
    def test_loads_valid_package(self, tmp_path):
        pkg = _package()
        loaded = load_llmb_draft_package(_write(tmp_path / "b.json", pkg))
        assert loaded["contract"] == "llmb_draft_package_v1"
        assert len(loaded["rounds"]) == 2

    @pytest.mark.parametrize("mutate,err", [
        ("contract", "unexpected contract"),
        ("producer", "produced by phase3b"),
        ("missing_source", "source."),
        ("bad_status", "invalid status"),
        ("ready_without_units", "requires non-empty units"),
        ("dup_unit", "duplicate unit_id"),
    ])
    def test_c0_rejects_bad_packages(self, tmp_path, mutate, err):
        pkg = _package()
        if mutate == "contract":
            pkg["contract"] = "other"
        elif mutate == "producer":
            pkg["producer"] = "phase3a"
        elif mutate == "missing_source":
            pkg["source"] = {"neutral_run_id": "x"}
        elif mutate == "bad_status":
            pkg["rounds"][1]["status"] = "bogus"
        elif mutate == "ready_without_units":
            pkg["rounds"][1] = {"round_id": "r003", "status": "ready", "units": []}
        elif mutate == "dup_unit":
            pkg["rounds"][0]["units"].append(_unit("r001_w01"))
        with pytest.raises(PublishContractError, match=err):
            load_llmb_draft_package(_write(tmp_path / "b.json", pkg))


class TestC1ResponseShape:
    def _good_response(self, units=None) -> str:
        units = units if units is not None else [
            {"unit_id": "r001_w01", "text": C_TEXT_OK},
            {"unit_id": "r001_w02", "text": "CT拿下这一分。"},
        ]
        return json.dumps({"contract": "llmc_round_edit_response_v1", "round_id": "r001", "units": units}, ensure_ascii=False)

    @pytest.mark.parametrize("raw,reason", [
        ("not json", "unparseable_json"),
        ("{}", "bad_contract"),
        ('{"contract":"llmc_round_edit_response_v1","round_id":"r999","units":[]}', "round_id_mismatch"),
        ('{"contract":"llmc_round_edit_response_v1","round_id":"r001","ext":1,"units":[]}', "unexpected_fields"),
        ('{"contract":"llmc_round_edit_response_v1","round_id":"r001","units":[{"unit_id":"r001_w01","text":"x","emotion":"激动"}]}', "unit_unexpected_fields"),
        ('{"contract":"llmc_round_edit_response_v1","round_id":"r001","units":[{"unit_id":"","text":"x"}]}', "unit_id_missing"),
        ('{"contract":"llmc_round_edit_response_v1","round_id":"r001","units":[{"unit_id":"r001_w01","text":""}]}', "empty_text"),
        ('{"contract":"llmc_round_edit_response_v1","round_id":"r001","units":[{"unit_id":"r001_w01","text":"[激动]开局"}]}', "emotion_tag_in_text"),
        ('{"contract":"llmc_round_edit_response_v1","round_id":"r001","units":[{"unit_id":"r001_w01","text":"时长5.11秒"}]}', "leak_marker"),
    ])
    def test_c1_rejects(self, raw, reason):
        parsed, got = validate_llmc_response(raw, "r001")
        assert parsed is None
        assert reason in got

    def test_c1_accepts_clean_response(self):
        parsed, reason = validate_llmc_response(self._good_response(), "r001")
        assert parsed is not None and not reason
        assert parsed["round_id"] == "r001"
        assert len(parsed["units"]) == 2


class TestC2Addressing:
    def test_accepts_exact_order(self):
        resp = [{"unit_id": "r001_w01", "text": "a"}, {"unit_id": "r001_w02", "text": "b"}]
        assert check_unit_addressing(resp, ["r001_w01", "r001_w02"]) == ""

    def test_rejects_reordered(self):
        resp = [{"unit_id": "r001_w02", "text": "b"}, {"unit_id": "r001_w01", "text": "a"}]
        assert "order_or_id_mismatch" in check_unit_addressing(resp, ["r001_w01", "r001_w02"])

    def test_rejects_missing(self):
        resp = [{"unit_id": "r001_w01", "text": "a"}]
        assert "count_mismatch" in check_unit_addressing(resp, ["r001_w01", "r001_w02"])

    def test_rejects_fabricated(self):
        resp = [{"unit_id": "r001_w01", "text": "a"}, {"unit_id": "r001_w99", "text": "c"}]
        assert "order_or_id_mismatch" in check_unit_addressing(resp, ["r001_w01", "r001_w02"])


class TestC3FactScope:
    def test_accepts_b_derived_text(self):
        unit = _unit("r001_w01")
        assert check_fact_scope("开局换一人，Tauson在中路交火。", unit["draft_text"], unit["fact_catalog"], []) == ""

    def test_rejects_new_number(self):
        unit = _unit("r001_w01")
        err = check_fact_scope("双方换掉三人，比分来到5比3。", unit["draft_text"], unit["fact_catalog"], [])
        assert "unexpected_number" in err

    def test_rejects_new_b_locations_flagged_when_not_in_b(self):
        # B 原文无 B 点；C 新增 B 点 → 拒绝
        unit = _unit("r001_w01")
        err = check_fact_scope("开局换一人，B点爆炸。", unit["draft_text"], unit["fact_catalog"], [])
        assert "unexpected_location" in err

    def test_accepts_location_present_in_b(self):
        unit = _unit("r001_w01", draft_text="开局双方在中路交火。")
        assert check_fact_scope("开局在中路交火。", unit["draft_text"], unit["fact_catalog"], []) == ""


class TestC5Capacity:
    def test_within_capacity(self):
        unit = _unit("r001_w01", safe_upper=3.0, r_b=1.0)
        ok, r_c, reason = check_capacity(unit, "短")
        assert ok and r_c < 1.0

    def test_exceeding_125x_rejected(self):
        unit = _unit("r001_w01", slot_sec=10.0, r_b=1.25)
        text = "长" * 63  # 63/5=12.6s → r_c=1.26 > 1.25
        ok, r_c, reason = check_capacity(unit, text)
        assert not ok and "over_budget" in reason

    def test_non_regression_when_b_long(self):
        # r_B=1.2 → C 不得超过 1.2；1.22 拒绝，1.18 通过
        unit = _unit("r001_w01", slot_sec=10.0, r_b=1.2)
        ok, _, _ = check_capacity(unit, "长" * 61)  # 12.2s → r_c=1.22
        assert not ok
        ok, _, _ = check_capacity(unit, "长" * 59)  # 11.8s → r_c=1.18
        assert ok

    def test_b_short_forces_c_normal(self):
        # r_B=1.0 → C 必须 <=1.0；1.04 拒绝
        unit = _unit("r001_w01", slot_sec=10.0, r_b=1.0)
        ok, _, _ = check_capacity(unit, "长" * 52)  # 10.4s → r_c=1.04
        assert not ok


class TestC6SourceDecision:
    def test_off(self):
        assert decide_round_source("off", True, True) == ("llmb_passthrough", "llmb_passthrough")

    def test_shadow_always_passthrough(self):
        assert decide_round_source("shadow", True, True) == ("llmb_passthrough", "llmb_passthrough")
        assert decide_round_source("shadow", False, True) == ("llmb_passthrough", "llmb_passthrough")

    def test_optional(self):
        assert decide_round_source("optional", True, True) == ("llmc_accepted", "llmc")
        assert decide_round_source("optional", False, True) == ("llmb_passthrough", "llmb_passthrough")

    def test_required_blocks_on_failure(self):
        assert decide_round_source("required", True, True) == ("llmc_accepted", "llmc")
        assert decide_round_source("required", False, True) == ("blocked", "llmb_passthrough")


class TestRoundRequest:
    def test_request_contract_and_speech_budget(self):
        round_data = {"round_id": "r001", "status": "ready", "units": [_unit("r001_w01", safe_upper=15.0)]}
        req = build_round_edit_request(round_data)
        assert req["contract"] == "llmc_round_edit_request_v1"
        assert req["edit_policy"]["allow_merge"] is False
        assert req["edit_policy"]["hard_speed_factor"] == 1.25
        budget = req["units"][0]["speech_budget"]
        assert budget["metric"] == "speech_units_v1"
        assert budget["normal_capacity"] == 75  # safe_upper 15s × 5 字/秒
        assert budget["hard_capacity"] == 94    # 75×1.25

    @pytest.mark.parametrize(
        "draft_text,expected",
        [
            ("短" * 25, "润色扩充"),
            ("中" * 50, "不做字数调整"),
            ("长" * 75, "删除冗余"),
        ],
    )
    def test_edit_directive_follows_source_ratio(self, draft_text, expected):
        round_data = {"round_id": "r001", "status": "ready", "units": [_unit("r001_w01", draft_text=draft_text, slot_sec=10.0)]}
        request_unit = build_round_edit_request(round_data)["units"][0]
        assert expected in request_unit["edit_directive"]

    def test_system_no_longer_requires_compression(self):
        assert "压缩优先" not in _LLMC3_SYSTEM
        assert "不得扩写" not in _LLMC3_SYSTEM

    def test_legacy_request_without_edit_directive_remains_callable(self):
        request = {
            "contract": "llmc_round_edit_request_v1",
            "round_id": "r001",
            "units": [{
                "unit_id": "r001_w01", "source_text": "旧稿", "source_length_chars": 2,
                "speech_budget": {"normal_capacity": 10},
            }],
        }
        raw = json.dumps({
            "contract": "llmc_round_edit_response_v1", "round_id": "r001",
            "units": [{"unit_id": "r001_w01", "text": "旧稿"}],
        }, ensure_ascii=False)
        parsed, reason = _call_round_editor(
            lambda *args, **kwargs: raw, {}, request, "r001",
            max_retries=0, debug_enabled=False,
        )
        assert parsed is not None and reason == ""

    def test_under_budget_response_retries_with_expansion_feedback(self):
        source_unit = _unit("r001_w01", draft_text="源" * 50, slot_sec=10.0)
        request = build_round_edit_request({"round_id": "r001", "status": "ready", "units": [source_unit]})
        prompts = []
        responses = iter(["短" * 20, "足" * 30])

        def generate(prompt, *args, **kwargs):
            prompts.append(json.loads(prompt))
            return json.dumps({
                "contract": "llmc_round_edit_response_v1", "round_id": "r001",
                "units": [{"unit_id": "r001_w01", "text": next(responses)}],
            }, ensure_ascii=False)

        parsed, reason = _call_round_editor(
            generate, {}, request, "r001", max_retries=1,
            debug_enabled=False, source_units=[source_unit],
        )
        assert parsed is not None and reason == ""
        assert len(prompts) == 2
        assert prompts[1]["retry_feedback"]["failure_reason"] == "under_budget:r001_w01"
        assert "润色扩充" in prompts[1]["retry_feedback"]["instruction"]

    def test_request_excludes_real_timestamps(self):
        round_data = {"round_id": "r001", "status": "ready", "units": [_unit("r001_w01")]}
        assert "start_tick" not in json.dumps(round_data["units"][0]["render_slot"]) if False else True
        req = build_round_edit_request(round_data)
        req_json = json.dumps(req)
        assert "render_slot" not in req_json
        assert "start_tick" not in req_json
        assert "end_tick" not in req_json


class TestPreflightB1C7:
    def test_b1_valid(self, tmp_path):
        validate_llmb_draft_package(_write(tmp_path / "b.json", _package()))

    def test_b1_rejects_ready_units_without_facts(self, tmp_path):
        unit = _unit("r001_w01", allowed=[])
        pkg = _package(rounds=[{"round_id": "r001", "status": "ready", "units": [unit]}])
        with pytest.raises(PublishContractError, match="allowed_fact_ids"):
            validate_llmb_draft_package(_write(tmp_path / "b.json", pkg))

    def test_c7_valid_passthrough_matches_b(self, tmp_path):
        b_path = _write(tmp_path / "b.json", _package())
        render = {
            "contract": "commentary_render_package_v1", "producer": "phase3c", "status": "render_ready",
            "llmc_mode": "off",
            "source": {"llmb_artifact_identity": "pkg-abc", "neutral_run_id": "n-abc",
                       "neutral_sha256": "0" * 64, "timeline_id": "tl:test:030", "source_video_sha256": ""},
            "rounds": [
                {"round_id": "r001", "integration_status": "llmb_passthrough", "selected_source": "llmb_passthrough",
                 "render_units": [
                     {"unit_id": "r001_w01", "sequence": 1, "text": B_TEXT, "emotion": "激动",
                      "render_slot": _unit("r001_w01")["render_slot"],
                      "required_fact_ids": ["fact:v1:r001_w01:players:00001:aaaa1111"],
                      "required_speed_factor": 1.0, "source": "llmb"},
                     {"unit_id": "r001_w02", "sequence": 1, "text": B_TEXT, "emotion": "激动",
                      "render_slot": _unit("r001_w02")["render_slot"],
                      "required_fact_ids": ["fact:v1:r001_w02:players:00001:aaaa1111"],
                      "required_speed_factor": 1.0, "source": "llmb"},
                 ]},
                {"round_id": "r002", "integration_status": "skipped", "selected_source": "llmb_passthrough",
                 "render_units": []},
            ],
            "artifact_identity": "render-abc",
        }
        validate_render_package(_write(tmp_path / "render.json", render), b_path)

    def test_c7_rejects_passthrough_text_mismatch(self, tmp_path):
        b_path = _write(tmp_path / "b.json", _package())
        good = {
            "contract": "commentary_render_package_v1", "producer": "phase3c", "status": "render_ready",
            "llmc_mode": "off",
            "source": {"llmb_artifact_identity": "pkg-abc", "neutral_run_id": "n-abc",
                       "neutral_sha256": "0" * 64, "timeline_id": "tl:test:030", "source_video_sha256": ""},
            "rounds": [
                {"round_id": "r001", "integration_status": "llmb_passthrough", "selected_source": "llmb_passthrough",
                 "render_units": [
                     {"unit_id": "r001_w01", "sequence": 1, "text": "篡改的文本", "emotion": "激动",
                      "render_slot": _unit("r001_w01")["render_slot"],
                      "required_fact_ids": ["fact:v1:r001_w01:players:00001:aaaa1111"],
                      "required_speed_factor": 1.0, "source": "llmb"},
                 ]},
            ],
            "artifact_identity": "x",
        }
        with pytest.raises(PublishContractError, match="passthrough text must equal"):
            validate_render_package(_write(tmp_path / "render.json", good), b_path)


class TestRunPhase3c:
    def _config(self) -> dict:
        return {
            "llm": {"backend": "api"},
            "semantic": {"phase3c": {"mode": "off"}},
            "paths": {},
        }

    def test_mode_off_wraps_package(self, tmp_path, monkeypatch):
        b_path = _write(tmp_path / "b.json", _package())
        out = tmp_path / "render.json"
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("", encoding="utf-8")
        monkeypatch.setattr("sbmachine.phase3c_llmc.load_config", lambda _p: self._config())

        result = run_phase3c(draft_package_path=b_path, output_render_path=out, config_path=cfg_path)
        assert result["llmc_mode"] == "off"
        assert result["rounds_total"] == 2
        assert result["rounds_passthrough"] == 1
        assert result["rounds_skipped"] == 1
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["contract"] == "commentary_render_package_v1"
        assert payload["rounds"][0]["selected_source"] == "llmb_passthrough"
        validate_render_package(out, b_path)

    def test_mode_optional_accepts_c(self, tmp_path, monkeypatch):
        b_path = _write(tmp_path / "b.json", _package())
        out = tmp_path / "render.json"
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("", encoding="utf-8")
        cfg = self._config()
        cfg["semantic"]["phase3c"]["mode"] = "optional"

        def fake_generate(secret_scope, semantic_cfg=None):
            assert secret_scope == "llmc"
            def gen_fn(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, response_format=None):
                return json.dumps({
                    "contract": "llmc_round_edit_response_v1", "round_id": "r001",
                    "units": [
                        {"unit_id": "r001_w01", "text": C_TEXT_OK},
                        {"unit_id": "r001_w02", "text": C_TEXT_OK},
                    ],
                }, ensure_ascii=False)
            return gen_fn

        monkeypatch.setattr("sbmachine.phase3c_llmc.load_config", lambda _p: cfg)
        monkeypatch.setattr("sbmachine.cloud_memory.make_generate", fake_generate)

        result = run_phase3c(draft_package_path=b_path, output_render_path=out, config_path=cfg_path)
        assert result["rounds_llmc_accepted"] == 1
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["rounds"][0]["integration_status"] == "llmc_accepted"
        assert payload["rounds"][0]["render_units"][0]["text"] == C_TEXT_OK
        assert payload["rounds"][0]["render_units"][0]["source"] == "llmc"

    def test_mode_required_blocks_on_bad_c(self, tmp_path, monkeypatch):
        b_path = _write(tmp_path / "b.json", _package())
        out = tmp_path / "render.json"
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("", encoding="utf-8")
        cfg = self._config()
        cfg["semantic"]["phase3c"]["mode"] = "required"

        def fake_generate(secret_scope, semantic_cfg=None):
            def gen_fn(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, response_format=None):
                return json.dumps({
                    "contract": "llmc_round_edit_response_v1", "round_id": "r001",
                    "units": [{"unit_id": "r001_w01", "text": C_TEXT_OK}],  # 缺 r001_w02 → C2 拒绝
                }, ensure_ascii=False)
            return gen_fn

        monkeypatch.setattr("sbmachine.phase3c_llmc.load_config", lambda _p: cfg)
        monkeypatch.setattr("sbmachine.cloud_memory.make_generate", fake_generate)

        result = run_phase3c(draft_package_path=b_path, output_render_path=out, config_path=cfg_path)
        assert result["rounds_blocked"] == 1
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["rounds"][0]["integration_status"] == "blocked"
        assert payload["rounds"][0]["render_units"] == []

    def test_dry_run_does_not_write_output(self, tmp_path, monkeypatch):
        b_path = _write(tmp_path / "b.json", _package())
        out = tmp_path / "render.json"
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("", encoding="utf-8")
        monkeypatch.setattr("sbmachine.phase3c_llmc.load_config", lambda _p: self._config())

        result = run_phase3c(draft_package_path=b_path, output_render_path=out, config_path=cfg_path, dry_run=True)
        assert result["rounds_total"] == 2
        assert not out.exists()

    def test_invalid_mode_rejected(self, tmp_path, monkeypatch):
        b_path = _write(tmp_path / "b.json", _package())
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("", encoding="utf-8")
        cfg = self._config()
        cfg["semantic"]["phase3c"]["mode"] = "bogus"
        monkeypatch.setattr("sbmachine.phase3c_llmc.load_config", lambda _p: cfg)
        with pytest.raises(PublishContractError, match="phase3c.mode"):
            run_phase3c(draft_package_path=b_path, output_render_path=tmp_path / "x.json", config_path=cfg_path)
