from pathlib import Path

import pytest
import yaml

from sbmachine import llm_shim
from sbmachine.common import PROJECT_ROOT


def test_vllm_transport_ignores_global_api_overrides(monkeypatch):
    monkeypatch.setattr(
        llm_shim,
        "_load_secrets",
        lambda: {
            "api_key": "remote-secret",
            "base_url": "https://remote.example/v1",
            "model": "remote-model",
        },
    )

    base_url, api_key, model = llm_shim._resolve_chat_target(
        {
            "_transport_backend": "vllm",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "qwen3",
        }
    )

    assert base_url == "http://127.0.0.1:8000/v1"
    assert api_key == "EMPTY"
    assert model == "qwen3"


def test_api_transport_keeps_global_api_overrides(monkeypatch):
    monkeypatch.setattr(
        llm_shim,
        "_load_secrets",
        lambda: {
            "api_key": "remote-secret",
            "base_url": "https://remote.example/v1",
            "model": "remote-model",
        },
    )

    assert llm_shim._resolve_chat_target(
        {
            "_transport_backend": "api",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "qwen3",
        }
    ) == ("https://remote.example/v1", "remote-secret", "remote-model")


def test_talk_runtime_never_prebakes_or_implicitly_downloads_models():
    dockerfile = (PROJECT_ROOT / "docker" / "talk.Dockerfile").read_text(encoding="utf-8")
    entrypoint = (PROJECT_ROOT / "docker" / "talk_entrypoint.sh").read_text(encoding="utf-8")

    assert "snapshot_download" not in dockerfile
    assert "AI6657_PRELOAD_TALK_MODELS" not in dockerfile
    assert 'prepare)' in entrypoint
    assert 'verify)' in entrypoint
    serve_block = entrypoint.split('  serve|"")', 1)[1].split('  *)', 1)[0]
    assert "prepare_model" not in serve_block
    assert "verify_model" in serve_block


def test_talk_compose_service_is_an_opt_in_addon_without_workspace_mount():
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    talk = compose["services"]["talk_service"]

    assert talk["profiles"] == ["talk"]
    assert talk["gpus"] == "all"
    assert ".:/workspace" not in talk["volumes"]

