"""LLM-A 语料收集（data/llma/）：无论成败都收 input + LLM 输出。"""
from __future__ import annotations

import json

import pytest

from sbmachine import llm_shim, phase3a_analyst
from tests.unit.test_phase3a_audit import _contract_neutral_from_prompt, _frame, _round_record


def _setup(tmp_path, monkeypatch, *, corpus_dir):
    """构造单回合 2 窗输入；返回 (rounds_path, output_path, config_path, semantic_path)。"""
    frames = [
        _frame(0.0),
        _frame(2.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47"}]}),
        _frame(5.0, {"kills": [{"attacker": "C", "victim": "D", "weapon": "AK-47"}]}),
        _frame(10.0),
    ]
    rounds_path = tmp_path / "rounds_with_yolo.json"
    semantic_path = tmp_path / "rounds_with_yolo_semantic.json"
    output_path = tmp_path / "rounds_with_neutral.json"
    config_path = tmp_path / "config.yaml"
    rounds_path.write_text(
        json.dumps({"video_path": "test.mp4", "map_name": "de_test", "rounds": [_round_record(1, frames)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    semantic_frames = [
        _frame(0.0),
        _frame(2.0, {"kills": [{"attacker": "A", "victim": "B", "weapon": "AK-47", "tick": 256}]}),
        _frame(5.0, {"kills": [{"attacker": "C", "victim": "D", "weapon": "AK-47", "tick": 512}]}),
        _frame(10.0),
    ]
    semantic_path.write_text(json.dumps([{"round_no": 1, "frames": semantic_frames}], ensure_ascii=False), encoding="utf-8")
    config_path.write_text(
        f"""
llm:
  backend: vllm
semantic:
  analyst_backend: vllm
  analyst_model: qwen3
  analyst_output_max_tokens: 256
  analyst_concurrent_rounds: 1
  window_max_sec: 10
  window_min_sec: 3
paths:
  rounds_with_yolo_semantic_json: "{semantic_path.as_posix()}"
debug:
  phase3: false
""",
        encoding="utf-8",
    )
    return rounds_path, output_path, config_path, semantic_path


def _enable_corpus(monkeypatch, corpus_dir: Path, run_id: str = "run-test") -> Path:
    """启用语料收集并指向测试目录（忽略实际 run_id，固定写指定文件）。"""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    target = corpus_dir / f"{run_id}.jsonl"
    monkeypatch.setattr(
        phase3a_analyst, "_init_llma_corpus",
        lambda rid: setattr(phase3a_analyst, "_LLMA_CORPUS_PATH", target),
    )
    return target


def test_corpus_collects_both_success_and_failure_windows(tmp_path, monkeypatch):
    """成败窗口都收：成功窗 output=neutral，失败窗 output=原始响应，均含 input 投影。"""
    corpus_dir = tmp_path / "llma"
    target = _enable_corpus(monkeypatch, corpus_dir)
    rounds_path, output_path, config_path, semantic_path = _setup(tmp_path, monkeypatch, corpus_dir=corpus_dir)

    call_count = [0]

    def fake_generate(prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return llm_shim._ApiChatResult(
                json.dumps({"neutral": _contract_neutral_from_prompt(prompt)}, ensure_ascii=False),
                scope="llma", source_run_id="run-test",
                request_payload={"messages": [{"role": "user", "content": prompt}]},
                log_ctx=log_ctx, finish_reason="stop",
            )
        return llm_shim._ApiChatResult(
            '{"neutral":"未完成', scope="llma", source_run_id="run-test",
            request_payload={"messages": [{"role": "user", "content": prompt}]},
            log_ctx=log_ctx, finish_reason="length",
            raw_response={"choices": [{"message": {"content": '{"neutral":"未完成'}, "finish_reason": "length"}]},
        )

    import sbmachine.llma_api as llma_api
    monkeypatch.setattr(llma_api, "generate", fake_generate)

    phase3a_analyst.run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # 两个窗口都采集，含失败窗口
    entries = [json.loads(line) for line in lines]
    statuses = [e["status"] for e in entries]
    assert "success" in statuses and "truncated" in statuses
    success = next(e for e in entries if e["status"] == "success")
    failed = next(e for e in entries if e["status"] == "truncated")
    # 成功窗：output=最终 neutral（非空）；失败窗：output=原始响应
    assert success["output"]
    assert failed["output"] == '{"neutral":"未完成'
    # 都带 input 投影与元信息
    for e in entries:
        assert e["input"]["projection_version"] == 2
        assert e["round_no"] == 1
        assert e["window_id"].startswith("r001_w")
        assert "char_budget" in e


def test_corpus_disabled_when_dry_run(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "llma"
    target = _enable_corpus(monkeypatch, corpus_dir, run_id="run-dry")
    rounds_path, output_path, config_path, semantic_path = _setup(tmp_path, monkeypatch, corpus_dir=corpus_dir)

    phase3a_analyst.run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path, dry_run=True)

    assert not target.exists() or target.read_text(encoding="utf-8").strip() == ""


def test_corpus_append_failure_does_not_break_pipeline(tmp_path, monkeypatch):
    """语料写盘失败（如磁盘只读）不影响主链：_append_llma_corpus 内部吞异常。"""
    # 指向一个不可写目标：父路径是文件（非目录），open 必然抛 OSError
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setattr(phase3a_analyst, "_LLMA_CORPUS_PATH", blocker / "corpus.jsonl")
    rounds_path, output_path, config_path, semantic_path = _setup(tmp_path, monkeypatch, corpus_dir=tmp_path)

    import sbmachine.llma_api as llma_api
    monkeypatch.setattr(
        llma_api, "generate",
        lambda prompt, llm_cfg, system_prompt=None, max_tokens=None, log_ctx=None, **kwargs: llm_shim._ApiChatResult(
            json.dumps({"neutral": "测试中性稿。"}, ensure_ascii=False),
            scope="llma", source_run_id="run-x",
            request_payload={"messages": [{"role": "user", "content": prompt}]},
            log_ctx=log_ctx, finish_reason="stop",
        ),
    )

    manifest = phase3a_analyst.run_phase3a(rounds_path=rounds_path, output_path=output_path, config_path=config_path)
    assert manifest["rounds"]  # 主链未中断
