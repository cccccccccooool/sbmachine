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
    from tests.fakes import FakeLLM, FakeTTS

    llma = FakeLLM(["{\"neutral\":\"fake neutral\"}"])
    llmb = FakeLLM(["{\"commentary\":\"fake commentary\",\"emotion_segments\":[]}"])
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
