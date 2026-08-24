"""阶段 3：LLM-A 两阶段并发（analyst_window_concurrency）契约测试。

覆盖：
- 缺省/显式 1：并发度 1 兼容模式——回合 worker 内按窗口顺序同步调用（无并发）
- 并发 2：有界窗口并发派发，结果仍按 window_id 顺序归并（确定性顺序）
- 并发模式与串行模式的发布产物等价（window_id/neutral/neutral_source/generation_status）
- 熔断：连续 http_client_error 达阈值中止整阶段（并发模式下同样生效）
- llma_input.json 仍按时间顺序（窗口顺序）
"""
import json
import threading
import time
from pathlib import Path

import pytest

import sbmachine.llma_api as llma_api
from sbmachine import llm_shim
from sbmachine.phase3a_analyst import Phase3aCircuitBreak, run_phase3a


def _frame(t: float, events: dict | None = None, *, phase: str = "in_round") -> dict:
    data = {
        "when": {"video_time": t, "relative_sec": t, "phase": phase},
        "who": {"view": "player", "pov_player": "p1"},
    }
    if events:
        data["events"] = events
    return data


def _round_record(round_no: int, frames: list[dict]) -> dict:
    start = float(frames[0]["when"]["video_time"])
    end = float(frames[-1]["when"]["video_time"])
    return {
        "round_no": round_no,
        "start_sec": start,
        "end_sec": end,
        "score_before": {"ct": 0, "t": 0},
        "score_after": {"ct": 0, "t": 1},
        "demo_round_hint": round_no,
        "_phase2_yolo": {
            "key_frames": [
                {
                    "time_sec": frame["when"]["video_time"],
                    "gate_reason": "test",
                    "background_info": frame,
                    "has_frame": True,
                }
                for frame in frames
            ],
        },
    }


def _kill_frame(t: float, tick: int) -> dict:
    # 每个击杀帧用不同的 victim，避免被 dead_this_round 判为 corpse_shoot
    return _frame(t, {"kills": [{"attacker": "A", "victim": f"B{tick}", "weapon": "AK-47", "tick": tick}]})


def _contract_neutral_from_prompt(prompt: str) -> str:
    projection = json.loads(prompt.strip().splitlines()[-1])
    return "，".join(
        fact["canonical_text"]
        for fact in projection["required_facts"]
        if fact.get("required") is True
    )


def _write_inputs(tmp_path: Path, *, window_concurrency: int | None = None,
                  client_error_threshold: int | None = None,
                  frames: list[dict] | None = None) -> tuple[Path, Path, Path]:
    if frames is None:
        # 20 秒回合、每秒一个击杀帧：切出多个窗口，多数窗口会发起 LLM-A 请求。
        frames = [_kill_frame(float(t), t * 100) for t in range(0, 21)]
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps({"video_path": "test.mp4", "map_name": "de_test", "rounds": [_round_record(1, frames)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    semantic_frames = [
        frame.copy() for frame in frames
    ]
    semantic_path.write_text(
        json.dumps([{"round_no": 1, "frames": semantic_frames}], ensure_ascii=False),
        encoding="utf-8",
    )
    semantic_lines = [
        "llm:",
        "  backend: vllm",
        "semantic:",
        "  analyst_backend: vllm",
        "  analyst_model: qwen3",
        "  analyst_output_max_tokens: 256",
        "  analyst_concurrent_rounds: 1",
        "  window_max_sec: 3",
        "  window_min_sec: 3",
    ]
    if window_concurrency is not None:
        semantic_lines.append(f"  analyst_window_concurrency: {window_concurrency}")
    if client_error_threshold is not None:
        semantic_lines.append(f"  analyst_client_error_threshold: {client_error_threshold}")
    semantic_lines.append(f'paths:\n  rounds_with_yolo_semantic_json: "{semantic_path.as_posix()}"')
    semantic_lines.append("debug:")
    semantic_lines.append("  phase3: false")
    config_path.write_text("\n".join(semantic_lines), encoding="utf-8")
    return rounds_path, output_path, config_path


class _Probe:
    """记录每次 LLM-A 请求的线程与并发度。"""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.threads: set[str] = set()
        self.max_active = 0
        self.calls: list[dict] = []
        self._active = 0
        self._lock = threading.Lock()
        self.counter = [0]

    def generate(self, prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.threads.add(threading.current_thread().name)
            self.calls.append({"thread": threading.current_thread().name, "prompt": prompt})
        time.sleep(self.delay)
        try:
            return llm_shim._ApiChatResult(
                json.dumps({"neutral": _contract_neutral_from_prompt(prompt)}, ensure_ascii=False),
                scope="llma",
                source_run_id="run-conc",
                request_payload={"messages": [{"role": "user", "content": prompt}]},
                log_ctx=log_ctx,
            )
        finally:
            with self._lock:
                self._active -= 1


def _scenes_signature(manifest: dict) -> list[dict]:
    sig = []
    for scene in manifest["rounds"][0]["scenes"]:
        entry = {
            "window_id": scene["window_id"],
            "neutral": scene["neutral"],
            "neutral_source": scene["neutral_source"],
            "generation_status": scene["generation_status"],
            "hype": scene.get("hype"),
            "scream_eligible": scene.get("scream_eligible"),
            "char_budget": scene.get("char_budget"),
        }
        if "speech_budget" in scene:
            sb = scene["speech_budget"]
            entry["speech_budget"] = {
                k: sb.get(k) for k in ("target_units", "hard_units", "profile_id")
            }
        sig.append(entry)
    return sig


def _run(tmp_path, monkeypatch, probe, *, window_concurrency=None, frames=None):
    rounds_path, output_path, config_path = _write_inputs(
        tmp_path, window_concurrency=window_concurrency, frames=frames
    )
    monkeypatch.setattr(llm_shim, "_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(llma_api, "generate", probe.generate)
    manifest = run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)
    return manifest, output_path


def test_window_concurrency_default_is_serial(tmp_path, monkeypatch):
    """缺省（无 analyst_window_concurrency）→ 并发度 1 兼容模式：单线程顺序调用。"""
    probe = _Probe(delay=0.03)
    manifest, _ = _run(tmp_path, monkeypatch, probe)

    assert probe.max_active == 1
    assert len(probe.threads) == 1
    window_ids = [scene["window_id"] for scene in manifest["rounds"][0]["scenes"]]
    assert window_ids == sorted(window_ids)
    assert len(probe.calls) >= 1


def test_window_concurrency_one_is_serial(tmp_path, monkeypatch):
    """显式 analyst_window_concurrency: 1 → 同样单线程顺序调用。"""
    probe = _Probe(delay=0.03)
    manifest, _ = _run(tmp_path, monkeypatch, probe, window_concurrency=1)

    assert probe.max_active == 1
    assert len(probe.threads) == 1
    window_ids = [scene["window_id"] for scene in manifest["rounds"][0]["scenes"]]
    assert window_ids == sorted(window_ids)


def test_window_concurrency_two_parallel_and_ordered(tmp_path, monkeypatch):
    """并发 2：多线程并行派发，scenes 仍按 window_id 顺序归并。"""
    probe = _Probe(delay=0.08)
    manifest, _ = _run(tmp_path, monkeypatch, probe, window_concurrency=2)

    assert probe.max_active == 2
    assert len(probe.threads) >= 2
    window_ids = [scene["window_id"] for scene in manifest["rounds"][0]["scenes"]]
    assert window_ids == sorted(window_ids)
    assert len(probe.calls) >= 2


def test_window_concurrency_matches_serial_output(tmp_path, monkeypatch):
    """并发模式与串行模式的发布产物等价（窗口/稿件/来源/状态完全一致）。"""
    serial_probe = _Probe(delay=0.01)
    serial_manifest, _ = _run(tmp_path, monkeypatch, serial_probe)

    parallel_probe = _Probe(delay=0.01)
    parallel_tmp = tmp_path / "parallel"
    parallel_tmp.mkdir()
    rounds_path, output_path, config_path = _write_inputs(
        parallel_tmp, window_concurrency=2
    )
    monkeypatch.setattr(llm_shim, "_LOG_DIR", parallel_tmp / "logs")
    monkeypatch.setattr(llma_api, "generate", parallel_probe.generate)
    parallel_manifest = run_phase3a(
        rounds_path=rounds_path, output_path=output_path, config_path=config_path
    )

    assert _scenes_signature(parallel_manifest) == _scenes_signature(serial_manifest)
    assert parallel_probe.max_active == 2
    assert serial_probe.max_active == 1


def test_window_concurrency_llma_input_keeps_window_order(tmp_path, monkeypatch):
    """llma_input.json 的窗口投影仍按时间顺序（窗口顺序）。"""
    probe = _Probe(delay=0.02)
    _, output_path = _run(tmp_path, monkeypatch, probe, window_concurrency=2)

    artifact = json.loads((output_path.with_name("llma_input.json")).read_text(encoding="utf-8"))
    windows = artifact["rounds"][0]["windows"]
    window_ids = [w["window_id"] for w in windows]
    assert window_ids == sorted(window_ids)
    assert len(windows) >= 1


def test_window_concurrency_breaker_aborts_stage(tmp_path, monkeypatch):
    """并发模式下 Phase3aCircuitBreak 熔断仍生效：连续 http_client_error 中止整阶段。"""
    import requests

    class Fake401Response:
        status_code = 401

    def raise_401(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs):
        exc = requests.HTTPError("Unauthorized")
        exc.response = Fake401Response()
        raise exc

    rounds_path, output_path, config_path = _write_inputs(
        tmp_path, window_concurrency=2, client_error_threshold=1
    )
    monkeypatch.setattr(llm_shim, "_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(llma_api, "generate", raise_401)

    with pytest.raises(Phase3aCircuitBreak):
        run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)
    assert not output_path.exists()


def _variant_frames() -> list[dict]:
    """构造 hard 逐窗不同的确定性帧序列：首窗单杀（0.4）、末窗 5 连杀（1.0）。

    对应 2026-08-17 已固化的 r004 污染样本"末窗高 hard 传染全回合"回归夹具；
    若归并再发生"外层 sc_hype 残留"，本夹具会让全部窗口被写成末窗值 1.0。
    """
    frames = [_kill_frame(1.0, 100)]
    frames.append(_frame(8.5, {"kills": [
        {"attacker": "A", "victim": f"MX{i}", "weapon": "AK-47", "tick": 85_000 + i}
        for i in range(5)
    ]}))
    return sorted(frames, key=lambda x: x["when"]["video_time"])


def _player_frames() -> list[dict]:
    """带 where.players 的逐秒击杀帧：确保每个非静默窗口都注入 player_state。"""
    frames = []
    for t in range(0, 21):
        frame = _kill_frame(float(t), t * 100)
        frame["where"] = {"players": [
            {"name": "JDC", "side": "CT", "hp": max(1, 100 - t), "weapon": "M4", "callout": "Mid"},
            {"name": "REZ", "side": "T", "hp": 100, "weapon": "AK", "callout": "A"},
        ]}
        frames.append(frame)
    return frames


def _llma_input_states(output_path: Path) -> list[str | None]:
    artifact = json.loads((output_path.with_name("llma_input.json")).read_text(encoding="utf-8"))
    return [w["projection"].get("player_state") for w in artifact["rounds"][0]["windows"]]


def test_window_hard_values_stay_per_window_no_leak(tmp_path, monkeypatch):
    """逐窗 hard 不串写：首窗单杀（0.4）不得被末窗 5 连杀（1.0）污染成同值。"""
    probe = _Probe(delay=0.01)
    manifest, _ = _run(tmp_path, monkeypatch, probe, frames=_variant_frames())

    scene_hypes = [s.get("hype") for s in manifest["rounds"][0]["scenes"]]
    assert len(scene_hypes) >= 2
    # 核心回归：各窗 hard 不得被"末窗残留值"写成全同；
    # 首窗独立值 0.4 在修复前会被末窗残留（0.0/高值）覆盖而失败（红）。
    assert scene_hypes[0] == 0.4, scene_hypes
    assert len(set(scene_hypes)) > 1, scene_hypes
    # 资格同样来自逐窗快照，不重算、不串写
    for s in manifest["rounds"][0]["scenes"]:
        assert isinstance(s.get("scream_eligible"), bool)


def test_window_hard_signature_serial_matches_parallel(tmp_path, monkeypatch):
    """串行与并行的逐窗 hard/资格/预算签名一致（回归 hard 传播修复）。"""
    serial_probe = _Probe(delay=0.01)
    serial_manifest, _ = _run(tmp_path, monkeypatch, serial_probe, frames=_variant_frames())

    parallel_probe = _Probe(delay=0.01)
    parallel_tmp = tmp_path / "parallel_hard"
    parallel_tmp.mkdir()
    rounds_path, output_path, config_path = _write_inputs(
        parallel_tmp, window_concurrency=2, frames=_variant_frames()
    )
    monkeypatch.setattr(llm_shim, "_LOG_DIR", parallel_tmp / "logs")
    monkeypatch.setattr(llma_api, "generate", parallel_probe.generate)
    parallel_manifest = run_phase3a(
        rounds_path=rounds_path, output_path=output_path, config_path=config_path
    )

    assert _scenes_signature(parallel_manifest) == _scenes_signature(serial_manifest)
    assert parallel_probe.max_active >= 2


def test_window_concurrency_player_state_serial_matches_parallel(tmp_path, monkeypatch):
    """串行与并发的 player_state 序列一致；阶段 1 串行生成，窗口并发不改变内容。"""
    serial_probe = _Probe(delay=0.01)
    _, serial_output = _run(tmp_path, monkeypatch, serial_probe, frames=_player_frames())
    serial_states = _llma_input_states(serial_output)

    parallel_probe = _Probe(delay=0.01)
    parallel_tmp = tmp_path / "parallel_ps"
    parallel_tmp.mkdir()
    rounds_path, output_path, config_path = _write_inputs(
        parallel_tmp, window_concurrency=2, frames=_player_frames()
    )
    monkeypatch.setattr(llm_shim, "_LOG_DIR", parallel_tmp / "logs")
    monkeypatch.setattr(llma_api, "generate", parallel_probe.generate)
    run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)
    parallel_states = _llma_input_states(output_path)

    assert parallel_states == serial_states
    assert any(state and state.startswith("首次快照：") for state in serial_states)
    assert "hp_total" not in json.dumps(
        json.loads((serial_output.with_name("llma_input.json")).read_text(encoding="utf-8"))
    )
