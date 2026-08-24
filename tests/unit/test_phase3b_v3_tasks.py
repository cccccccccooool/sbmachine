"""Phase3b commentary v3 稀疏候选（voice task）生产侧测试。

覆盖（§5.1 两次判定 / §9.4 形状 / §9.7 契约）：
- green：只生成 primary+capsule，无 compact 调用（mock LLM-B 调用计数）。
- amber：primary+compact+capsule，LLM-B 最多两次调用。
- red 预判：直接 compact+capsule，primary 不进生产候选。
- 终判降级：primary 生成后实测变 red → primary 只进诊断。
- profile 未 validated：不交付 v3（输出 v2），risk_class=unknown 影子诊断。
- preserved_fact_ids 由校验器计算（mock 返回不含 required 事实的文本 → 候选 semantic_state != ok）。
- capsule 候选 source=rule_capsule 且情绪不超过硬事实档位。
- commentary v3 输出通过 sbmachine.voice_task_contract.validate_commentary_v3。
"""
import json
from pathlib import Path

import pytest

from sbmachine import llmb_api, phase3b_style, speech_measure
from sbmachine.phase3b_style import run_phase3b
from sbmachine.voice_task_contract import validate_commentary_v3

PROFILE_ID = "speech-profile-v1"

# ── 复用 §11.5 示例的线性估计器（0.16 + 0.18*zh + 0.29*en + 0.31*num + 0.34*alnum + 0.12*pause）──
_PROFILE = {
    "profile_schema_version": 1,
    "profile_id": PROFILE_ID,
    "status": "validated",
    "metric_version": "speech_units_v1",
    "engine_fingerprint": "test-engine",
    "voice_fingerprint": "test-voice",
    "preprocess_fingerprint": "test-preprocess",
    "sample_rate_hz": 32000,
    "base_speed_factor": 1.0,
    "speed_scaling_verified": True,
    "duration_estimator": {
        "kind": "nonnegative_linear_v1",
        "feature_order": ["zh_units", "english_words", "number_groups", "alnum_tokens", "pause_units"],
        "coefficients_sec": [0.18, 0.29, 0.31, 0.34, 0.12],
        "intercept_sec": 0.16,
    },
    "safety": {
        "method": "split_conformal_upper_v1",
        "coverage_target": 0.95,
        "upper_residual_sec": 0.24,
        "fixed_margin_sec": 0.08,
    },
}

_KILL_ID = "fact:v1:r001_w03:kill:00360:a13f92c1"
_ROUND_ID = "fact:v1:r001_w03:round_result:00450:b721da80"

# U = safe upper（基准速度）：estimated + 0.24 + 0.08 = estimated + 0.32
# 槽位 12s→15s，S=3.0；green: U<=3；amber: 3<U<=4.5；red: U>4.5
# （U 值由 speech_measure.parse_features + 线性估计器实测核对；文本不含 T方 等
#   会被 build_fact_anchors 补充成额外 team anchor 的片段）
_NEUTRAL_GREEN = "JDC击杀Tauson，CT随后赢下回合"               # zh=8 en=3 pause=1 → U=2.91 → green
_NEUTRAL_AMBER = "JDC击杀Tauson，CT随后赢下回合，一人顽强防守"  # zh=14 en=3 pause=2 → U=4.11 → amber
_NEUTRAL_RED = "JDC击杀Tauson，CT随后赢下回合，一人顽强防守到底不放弃"  # zh=19 en=3 pause=2 → U=5.01 → red
_CAPSULE = "JDC击杀Tauson，CT胜"
_PRIMARY_OK = "JDC击杀Tauson，CT随后赢下回合"   # U=2.91 → green 终判
_PRIMARY_AMBER = "JDC击杀Tauson，CT随后赢下回合，一人顽强防守"  # U=4.11 → amber 终判
_PRIMARY_RED = "JDC击杀Tauson，CT随后赢下回合，一人顽强防守到底不放弃"  # U=5.01 → red 终判
_COMPACT_OK = "JDC击杀Tauson，CT胜。"
_PRIMARY_MISSING_ROUND = "JDC击杀Tauson，CT这边处理掉了一人"  # 过 anchors 验收、终判 amber，但缺 round_result 关键 token

_CONFIG = """llm:
  backend: vllm
semantic:
  style_backend: vllm
  voice_task:
    enabled: true
    candidate_policy: sparse_v1
    speech_profile_id: {profile_id}
    engine_fingerprint: test-engine
    voice_fingerprint: test-voice
    preprocess_fingerprint: test-preprocess
"""


def _fact_catalog():
    return [
        {
            "fact_id": _KILL_ID,
            "kind": "kill",
            "origin": "event",
            "anchor_tick": 360,
            "source_tick_range": [360, 360],
            "canonical_clause": "JDC击杀Tauson",
            "required": True,
            "priority": 100,
            "attacker": "JDC",
            "victim": "Tauson",
        },
        {
            "fact_id": _ROUND_ID,
            "kind": "round_result",
            "origin": "derived",
            "anchor_tick": 450,
            "source_tick_range": [360, 450],
            "canonical_clause": "CT拿下回合",
            "required": True,
            "priority": 90,
            "winner": "CT",
        },
    ]


def _v4_scene(*, neutral: str, window_id: str = "r001_w03", t_start: float = 12.0, t_end: float = 15.0,
              capsule: str = _CAPSULE, hype: float = 0.0, char_budget: int = 15) -> dict:
    return {
        "window_id": window_id,
        "t_start": t_start,
        "t_end": t_end,
        "scene": "default",
        "neutral": neutral,
        "neutral_source": "rule_template",
        "neutral_renderer": {"selected": "rule_template", "policy": "template_default_v1"},
        "rule_capsule": capsule,
        "fact_catalog": _fact_catalog(),
        "required_fact_ids": [_KILL_ID, _ROUND_ID],
        "fact_anchors": {
            "players": ["JDC", "Tauson"],
            "teams": ["CT"],
            "numbers": [],
            "events": ["kill"],
            "results": ["round_won"],
            "locations": [],
            "weapons": [],
        },
        "render_slot": {
            "start_sec": t_start,
            "end_sec": t_end,
            "start_tick": int(round(t_start * 30)),
            "end_tick": int(round(t_end * 30)),
            "continuity_group_id": None,
            "gap_policy": "independent_window",
        },
        "speech_budget": {"target_units": 14, "hard_units": 21, "profile_id": PROFILE_ID},
        "char_budget": char_budget,
        "hype": hype,
    }


def _write_v4_neutral(tmp_path: Path, scenes: list[dict], rounds_path: Path) -> Path:
    neutral_path = tmp_path / "rounds_with_neutral.json"
    neutral_path.write_text(json.dumps({
        "schema_version": 4,
        "phase3a_mode": "rule_neutral_renderer",
        "run_id": "test-v4-run",
        "source_rounds_sha256": "test-only-sha256",
        "speech_metric_version": "speech_units_v1",
        "video_path": "match.mp4",
        "map_name": "de_test",
        "rounds": [{
            "round_no": 1,
            "start_sec": 0.0,
            "end_sec": 90.0,
            "avg_hype": 0.0,
            "analyst_failed": False,
            "scenes": scenes,
        }],
    }), encoding="utf-8")
    return neutral_path


def _write_rounds(tmp_path: Path) -> Path:
    rounds_path = tmp_path / "rounds_with_yolo.json"
    rounds_path.write_text(json.dumps({
        "video_path": "match.mp4",
        "map_name": "de_test",
        "rounds": [{"round_no": 1, "start_sec": 0.0, "end_sec": 90.0}],
    }), encoding="utf-8")
    return rounds_path


@pytest.fixture
def profile_dir(tmp_path, monkeypatch):
    """在 tmp_path 下提供 speech profile，并接管 speech_measure 的 profile 根目录。"""
    root = tmp_path / "speech_profiles"
    (root / PROFILE_ID).mkdir(parents=True, exist_ok=True)
    (root / PROFILE_ID / "profile.json").write_text(json.dumps(_PROFILE, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(speech_measure, "_profile_root", lambda profile_id: root / profile_id)
    return root


@pytest.fixture
def v4_runner(tmp_path, monkeypatch, profile_dir):
    """构造 v4 生产环境，返回 (run, 断言辅助)。"""
    monkeypatch.setattr(phase3b_style, "_PROJECT_ROOT", tmp_path)
    rounds_path = _write_rounds(tmp_path)

    state = {"calls": [], "prompts": [], "variant_text": {}}

    def generate(*args, **kwargs):
        prompt = args[0] if args else kwargs.get("prompt", "")
        state["prompts"].append(prompt)
        payload = json.loads(prompt)
        variant = payload["delivery"]["variant_kind"]
        state["calls"].append(variant)
        text = state["variant_text"].get(variant, _PRIMARY_OK)
        return json.dumps({"commentary": f"[平述]{text}", "felt_intensity": 0.2}, ensure_ascii=False)

    monkeypatch.setattr(llmb_api, "generate", generate)

    def run(scenes, config_text=None, profile_status="validated"):
        if profile_status != "validated":
            profile_path = profile_dir / PROFILE_ID / "profile.json"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["status"] = profile_status
            profile_path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
        neutral_path = _write_v4_neutral(tmp_path, scenes, rounds_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_text or _CONFIG.format(profile_id=PROFILE_ID), encoding="utf-8")
        commentary_path = tmp_path / "commentary.json"
        manifest = run_phase3b(
            neutral_path=neutral_path,
            rounds_path=rounds_path,
            output_rounds_path=tmp_path / "rounds_with_commentary.json",
            commentary_path=commentary_path,
            config_path=config_path,
        )
        return manifest, commentary_path, tmp_path / "rounds_with_commentary.json", state

    return run


def _voice_task(manifest):
    assert manifest["commentary_schema_version"] == 3
    tasks = manifest.get("voice_tasks") or []
    assert len(tasks) == 1
    return tasks[0]


def _voice_task_diagnostics(run_id, tmp_path):
    entries = []
    diag_path = tmp_path / "diagnostics" / "phase3b" / f"{run_id}_voice_task.jsonl"
    if diag_path.exists():
        entries = [json.loads(line) for line in diag_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return entries


def test_v4_green_only_primary_and_capsule(tmp_path, monkeypatch, v4_runner):
    manifest, _, _, state = v4_runner([_v4_scene(neutral=_NEUTRAL_GREEN)])

    task = _voice_task(manifest)
    assert task["risk_class"] == "green"
    assert task["selection_order"] == ["primary", "capsule"]
    assert [c["variant_id"] for c in task["candidates"]] == ["primary", "capsule"]
    assert state["calls"] == ["primary"]
    primary = task["candidates"][0]
    assert primary["source"] == "llmb"
    assert set(primary["preserved_fact_ids"]) == {_KILL_ID, _ROUND_ID}
    # delivery 使用 v4 speech_budget 的 target_units/hard_units（§9.1）
    payload = json.loads(state["prompts"][0])
    assert payload["delivery"]["variant_kind"] == "primary"
    assert payload["delivery"]["target_units"] == 14
    assert payload["delivery"]["hard_units"] == 21
    assert payload["delivery"]["slot_duration_sec"] == 3.0
    assert payload["delivery"]["max_speed_factor"] == 1.5
    assert set(payload["required_fact_ids"]) == {_KILL_ID, _ROUND_ID}
    assert validate_commentary_v3(manifest) == []


def test_v4_amber_primary_compact_capsule(tmp_path, monkeypatch, v4_runner):
    state = {"calls": [], "prompts": [], "variant_text": {}}
    state["variant_text"] = {"primary": _PRIMARY_AMBER, "compact": _COMPACT_OK}
    monkeypatch.setattr(llmb_api, "generate", _variant_generate(state))

    manifest, _, _, _ = v4_runner([_v4_scene(neutral=_NEUTRAL_GREEN)], config_text=None)

    task = _voice_task(manifest)
    assert task["risk_class"] == "amber"
    assert task["selection_order"] == ["primary", "compact", "capsule"]
    assert [c["variant_id"] for c in task["candidates"]] == ["primary", "compact", "capsule"]
    # LLM-B 最多两次调用（primary + compact）
    assert state["calls"] == ["primary", "compact"]
    compact = task["candidates"][1]
    assert compact["source"] == "llmb_compact"
    assert set(compact["preserved_fact_ids"]) == {_KILL_ID, _ROUND_ID}
    assert compact["minimum_required_speed_factor"] == 1.0
    # compact 调用携带压缩约束（§9.1）
    compact_prompt = json.loads(state["prompts"][1])
    assert compact_prompt["delivery"]["variant_kind"] == "compact"
    assert "required" in compact_prompt["compact_constraint"]
    assert validate_commentary_v3(manifest) == []


def _variant_generate(state):
    def generate(*args, **kwargs):
        prompt = args[0] if args else kwargs.get("prompt", "")
        state["prompts"].append(prompt)
        payload = json.loads(prompt)
        variant = payload["delivery"]["variant_kind"]
        state["calls"].append(variant)
        text = state["variant_text"].get(variant, _PRIMARY_OK)
        return json.dumps({"commentary": f"[平述]{text}", "felt_intensity": 0.2}, ensure_ascii=False)
    return generate


def test_v4_red_prejudge_generates_compact_directly(tmp_path, monkeypatch, v4_runner):
    state = {"calls": [], "prompts": [], "variant_text": {"compact": _COMPACT_OK}}
    monkeypatch.setattr(llmb_api, "generate", _variant_generate(state))

    manifest, _, _, _ = v4_runner([_v4_scene(neutral=_NEUTRAL_RED)])

    task = _voice_task(manifest)
    assert task["risk_class"] == "red"
    assert task["selection_order"] == ["compact", "capsule"]
    assert [c["variant_id"] for c in task["candidates"]] == ["compact", "capsule"]
    # 预判 red：primary 不生成、不进生产候选，只直接生成 compact
    assert state["calls"] == ["compact"]
    diag = _voice_task_diagnostics(manifest["source_neutral_run_id"], tmp_path)
    pre = next(e for e in diag if e.get("semantic_state") == "pre_judge")
    assert pre["pre_risk_class"] == "red"
    assert validate_commentary_v3(manifest) == []


def test_v4_final_judge_primary_turns_red_and_stays_in_diagnostics_only(tmp_path, monkeypatch, v4_runner):
    state = {"calls": [], "prompts": [], "variant_text": {"primary": _PRIMARY_RED, "compact": _COMPACT_OK}}
    monkeypatch.setattr(llmb_api, "generate", _variant_generate(state))

    # 预判 green（neutral U=2.91 <= 3），但 primary 实际文本 U=5.30 > 4.5
    manifest, _, _, _ = v4_runner([_v4_scene(neutral=_NEUTRAL_GREEN, char_budget=30)])

    task = _voice_task(manifest)
    assert task["risk_class"] == "red"
    assert task["selection_order"] == ["compact", "capsule"]
    assert [c["variant_id"] for c in task["candidates"]] == ["compact", "capsule"]
    assert state["calls"] == ["primary", "compact"]
    diag = _voice_task_diagnostics(manifest["source_neutral_run_id"], tmp_path)
    primary_diag = next(e for e in diag if e.get("variant_id") == "primary" and e.get("semantic_state") == "timing_red")
    assert primary_diag["pre_risk_class"] == "green"
    assert primary_diag["risk_class"] == "red"
    assert validate_commentary_v3(manifest) == []


def test_v4_profile_not_validated_writes_unknown_shadow_and_outputs_v2(tmp_path, monkeypatch, v4_runner):
    manifest, commentary_path, _, _ = v4_runner(
        [_v4_scene(neutral=_NEUTRAL_GREEN)], profile_status="exploration",
    )

    # 不交付正式 commentary v3：输出沿用 v2 单稿路径
    assert manifest["commentary_schema_version"] == 2
    assert json.loads(commentary_path.read_text(encoding="utf-8"))["commentary_schema_version"] == 2
    # risk_class=unknown 影子诊断
    diag = _voice_task_diagnostics(manifest["source_neutral_run_id"], tmp_path)
    assert diag
    assert all(e["risk_class"] == "unknown" for e in diag)
    assert diag[0]["semantic_state"] == "unknown"


def test_v4_preserved_fact_ids_computed_by_validator(tmp_path, monkeypatch, v4_runner):
    state = {"calls": [], "prompts": [], "variant_text": {
        "primary": _PRIMARY_MISSING_ROUND, "compact": _COMPACT_OK,
    }}
    monkeypatch.setattr(llmb_api, "generate", _variant_generate(state))

    # primary 过 anchors 验收但缺 round_result 关键 token → 校验器判定未覆盖 → 不进生产
    manifest, _, _, _ = v4_runner([_v4_scene(neutral=_NEUTRAL_AMBER)])

    task = _voice_task(manifest)
    assert [c["variant_id"] for c in task["candidates"]] == ["compact", "capsule"]
    assert all(c["variant_id"] != "primary" for c in task["candidates"])
    diag = _voice_task_diagnostics(manifest["source_neutral_run_id"], tmp_path)
    primary_diag = next(e for e in diag if e.get("variant_id") == "primary")
    assert primary_diag["semantic_state"] != "ok"
    assert "missing_required" in primary_diag["reason"]
    # 生产候选的 preserved_fact_ids 全部覆盖 required
    for candidate in task["candidates"]:
        assert set(candidate["preserved_fact_ids"]) == {_KILL_ID, _ROUND_ID}
    assert validate_commentary_v3(manifest) == []


def test_v4_capsule_candidate_rule_source_and_emotion_floor(tmp_path, monkeypatch, v4_runner):
    state = {"calls": [], "prompts": [], "variant_text": {}}
    monkeypatch.setattr(llmb_api, "generate", _variant_generate(state))

    # 硬事实强度 0.9（高），capsule 情绪不得超过 0.45 上限
    manifest, _, _, _ = v4_runner([_v4_scene(neutral=_NEUTRAL_GREEN, hype=0.9)])

    task = _voice_task(manifest)
    capsule = next(c for c in task["candidates"] if c["variant_id"] == "capsule")
    assert capsule["source"] == "rule_capsule"
    assert capsule["felt_intensity"] <= 0.45
    assert capsule["felt_intensity"] <= 0.9
    assert capsule["felt_intensity"] == 0.45
    assert capsule["text"] == _CAPSULE
    assert capsule["spoken_units"] > 0
    # capsule 不消耗任何额外 LLM-B 调用
    assert state["calls"] == ["primary"]
    assert validate_commentary_v3(manifest) == []


def test_v4_output_passes_voice_task_contract_and_rounds_reference(tmp_path, monkeypatch, v4_runner):
    state = {"calls": [], "prompts": [], "variant_text": {"primary": _PRIMARY_AMBER, "compact": _COMPACT_OK}}
    monkeypatch.setattr(llmb_api, "generate", _variant_generate(state))

    manifest, _, rounds_with_commentary_path, _ = v4_runner([_v4_scene(neutral=_NEUTRAL_GREEN)])

    assert validate_commentary_v3(manifest) == []
    # rounds_with_commentary v3：source hash + 每 scene 引用 voice_task_id/primary（§9.5/§10.2）
    rounds_payload = json.loads(rounds_with_commentary_path.read_text(encoding="utf-8"))
    assert rounds_payload["source_neutral_sha256"] == manifest["source_neutral_sha256"]
    scene = rounds_payload["rounds"][0]["scenes"][0]
    assert scene["voice_task_id"] == "r001_w03"
    assert scene["primary_variant_id"] == "primary"
    assert scene["text"] == next(
        c["text"] for c in manifest["voice_tasks"][0]["candidates"] if c["variant_id"] == "primary"
    )
