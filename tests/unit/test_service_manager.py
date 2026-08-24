from types import SimpleNamespace

from sbmachine.compose_manager import ComposeManager
from sbmachine.service_manager import ServiceManager


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_health_requires_exact_200_and_expected_service_identity():
    vlm_identity = {"service": "vlm"}
    assert ServiceManager._matches_identity(
        FakeResponse(200, {"status": "ok", "service": "ai-6657-vlm"}),
        vlm_identity,
    ) is True
    assert ServiceManager._matches_identity(
        FakeResponse(201, {"status": "ok", "service": "ai-6657-vlm"}),
        vlm_identity,
    ) is False
    assert ServiceManager._matches_identity(
        FakeResponse(200, {"status": "ok", "service": "other"}),
        vlm_identity,
    ) is False


def test_vllm_and_sovits_health_validate_their_own_api_shape():
    models = FakeResponse(200, {"object": "list", "data": [{"id": "analyst"}, {"id": "style"}]})
    assert ServiceManager._matches_identity(
        models,
        {"service": "vllm", "models": ["analyst", "style"]},
    ) is True
    assert ServiceManager._matches_identity(
        models,
        {"service": "vllm", "models": ["analyst", "missing"]},
    ) is False
    models = FakeResponse(200, {"object": "list", "data": [{"id": "qwen3"}]})
    assert ServiceManager._matches_identity(models, {"service": "vllm", "model": "qwen3"}) is True
    assert ServiceManager._matches_identity(models, {"service": "vllm", "model": "other"}) is False
    openapi = FakeResponse(200, {"paths": {"/tts": {"post": {}}}})
    assert ServiceManager._matches_identity(openapi, {"service": "sovits"}) is True


def test_compose_health_probe_passes_service_identity(monkeypatch):
    config = {
        "llm": {"base_url": "http://127.0.0.1:8000/v1"},
        "semantic": {"model": "qwen3"},
        "phases": {"phase3a_semantic": True, "phase3b_semantic": True},
    }
    manager = ComposeManager(config)
    monkeypatch.setattr(manager, "_compose", lambda *args: SimpleNamespace(returncode=0))
    captured = {}

    def fake_poll(url, timeout, interval=2.0, identity=None):
        captured.update({"url": url, "identity": identity})
        return True

    monkeypatch.setattr(ServiceManager, "_poll_health", staticmethod(fake_poll))

    manager.up_one("talk_service")

    assert captured == {
        "url": "http://127.0.0.1:8000/v1/models",
        "identity": {"service": "vllm", "models": ["qwen3"]},
    }


def test_health_identity_requires_every_active_local_phase_model():
    manager = ServiceManager({
        "llm": {"backend": "vllm"},
        "semantic": {
            "analyst_backend": "vllm",
            "style_backend": "vllm",
            "analyst_model": "analyst-model",
            "style_model": "style-model",
        },
        "phases": {"phase3a_semantic": True, "phase3b_semantic": True},
    })

    assert manager._health_identity("vllm") == {
        "service": "vllm",
        "models": ["analyst-model", "style-model"],
    }


def test_health_identity_ignores_inactive_or_api_phase_models():
    manager = ServiceManager({
        "semantic": {
            "analyst_backend": "vllm",
            "style_backend": "api",
            "analyst_model": "analyst-model",
            "style_model": "external-style-model",
        },
        "phases": {"phase3a_semantic": True, "phase3b_semantic": True},
    })

    assert manager._health_identity("vllm")["models"] == ["analyst-model"]
