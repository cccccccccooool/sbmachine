"""6657 风格离线录像解说 AI 项目
项目功能：搭建一条「整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音」的离线生成流水线。
本文件功能：pytest 测试配置，将项目根目录加入 sys.path，并提供测试骨架共享 fixture。
"""
import json
import sys
from pathlib import Path

import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_UNIT_FILES = {
    "runtime": {
        "test_display_progress.py", "test_env_secrets.py", "test_pipeline_cache.py",
        "test_progress_events.py", "test_runtime_transaction.py", "test_service_manager.py",
    },
    "demo_phase1": {
        "test_demo_parser.py", "test_demo_query.py", "test_demo_query_ranges.py",
        "test_frame_type_slicer.py", "test_round_segmenter.py",
    },
    "phase2": {
        "test_hype_score.py", "test_hype_score_perf.py", "test_kill_semantics.py",
        "test_map_template_initializer.py", "test_phase2_alignment.py", "test_phase2_ocr.py",
        "test_phase2_quality.py", "test_phase2_yolo_gate.py", "test_region_crops.py",
        "test_round_aligner.py", "test_scene_context.py", "test_vlm_clients.py",
    },
    "phase3a": {
        "test_cloud_projection_payload.py", "test_commentary_planner.py", "test_debug_phase3a.py",
        "test_llm_projection.py", "test_llm_shim.py", "test_manual_notes.py", "test_manual_notes_planner.py",
        "test_neutral_contract.py", "test_ordered_scan_optimizations.py", "test_phase3_payload.py",
        "test_phase3a_audit.py", "test_phase3a_cloud_payload.py", "test_phase3a_cloud_runner.py",
        "test_phase3a_manual_notes.py", "test_phase3a_recovery.py", "test_phase3a_response.py",
        "test_phase3a_windowed.py", "test_preflight_neutral.py", "test_prompt_safety.py",
        "test_tactic_authoring.py", "test_tactic_matcher.py", "test_tactic_projection.py",
        "test_tactic_runner_integration.py",
    },
    "phase3b": {"test_emotion_policy.py", "test_llm_shim.py", "test_phase3b_response.py", "test_phase3b_silence.py"},
    "phase4": {
        "test_phase4_cache.py",
        "test_voice_task_selection.py",
    },
    "mcp": {"test_mcp_server.py"},
    "data_tools": {
        "test_auto_material_pipeline.py", "test_cnb_dagu_bootstrap.py", "test_export_api_sft.py",
        "test_label_commentary_pairs.py", "test_one_click_train.py", "test_prepare_tiny_llm_data.py",
        "test_training_build_info.py", "test_training_data_builder.py", "test_vllm_runtime_tools.py",
    },
}

INTEGRATION_FILES = {
    "test_auto_material_pipeline.py", "test_cnb_dagu_bootstrap.py", "test_demo_parser.py",
    "test_frame_type_slicer.py", "test_map_template_initializer.py", "test_one_click_train.py",
    "test_phase3a_cloud_runner.py", "test_phase4_cache.py", "test_runtime_transaction.py",
    "test_tactic_runner_integration.py", "test_vllm_runtime_tools.py",
}


def pytest_collection_modifyitems(items):
    """按测试文件显式映射业务单元，不移动历史测试文件。"""
    for item in items:
        path = Path(str(item.path))
        if "contract" in path.parts:
            item.add_marker(pytest.mark.contract)
        for marker_name, file_names in TEST_UNIT_FILES.items():
            if path.name in file_names:
                item.add_marker(getattr(pytest.mark, marker_name))
        if path.name in INTEGRATION_FILES:
            item.add_marker(pytest.mark.integration)

def pytest_configure(config):
    config.addinivalue_line("markers", "ffmpeg: tests that require ffmpeg binaries")
    config.addinivalue_line("markers", "slow: slow tests that are not part of the quick CPU suite")


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def load_fixture(fixtures_dir):
    def _load(name: str):
        path = fixtures_dir / name
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        return json.loads(path.read_text(encoding="utf-8-sig"))

    return _load


@pytest.fixture
def fake_backends(monkeypatch):
    from tests.fakes import FakeLLM, FakeTTS, FakeVLM

    llma = FakeLLM(["{\"neutral\":\"fake neutral\"}"])
    llmb = FakeLLM(["{\"commentary\":\"fake commentary\",\"emotion_segments\":[]}"])
    vlm = FakeVLM()
    tts = FakeTTS()

    targets = [
        ("sbmachine.llma_api", "generate", llma.generate),
        ("sbmachine.llmb_api", "generate", llmb.generate),
        ("audio_service.gpt_sovits_client", "synthesize", tts.synthesize),
    ]
    for module_name, attr, replacement in targets:
        try:
            module = __import__(module_name, fromlist=[attr])
        except Exception:
            continue
        monkeypatch.setattr(module, attr, replacement, raising=False)

    return {"llma": llma, "llmb": llmb, "vlm": vlm, "tts": tts}


@pytest.fixture(autouse=True)
def _no_llma_corpus_writes(monkeypatch):
    """测试期禁止往真实仓库 data/llma 写语料；需验证语料收集的测试单独启用。"""
    import sbmachine.phase3a_analyst as phase3a_analyst

    monkeypatch.setattr(phase3a_analyst, "_LLMA_CORPUS_PATH", None)
    monkeypatch.setattr(phase3a_analyst, "_init_llma_corpus", lambda run_id: None)
    yield
