from sbmachine import llm_shim
from sbmachine import llm_protocol


def test_load_secrets_uses_process_environment_before_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "AI6657_CLOUD_API_KEY=dotenv-cloud-key\n"
        "AI6657_CLOUD_LLMA_API_KEY=dotenv-analyst-key\n"
        "AI6657_CLOUD_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_protocol, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("AI6657_CLOUD_API_KEY", "process-key")
    monkeypatch.delenv("AI6657_CLOUD_LLMA_API_KEY", raising=False)

    secrets = llm_shim._load_secrets()

    assert secrets["api_key"] == "process-key"
    assert secrets["llma"]["api_key"] == "dotenv-analyst-key"
    assert secrets["model"] == "dotenv-model"


def test_load_secrets_legacy_keys_fall_back_to_cloud_scope(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "AI6657_API_KEY=legacy-key\n"
        "AI6657_BASE_URL=https://legacy.example/v1\n"
        "AI6657_LLM_MODEL=legacy-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_protocol, "_PROJECT_ROOT", tmp_path)

    secrets = llm_shim._load_secrets()

    assert secrets["api_key"] == "legacy-key"
    assert secrets["base_url"] == "https://legacy.example/v1"
    assert secrets["model"] == "legacy-model"
    assert secrets["llma"]["base_url"] == "https://legacy.example/v1"
    assert secrets["llmb"]["model"] == "legacy-model"


def test_load_secrets_scope_overrides_cloud_and_legacy_scoped_keys_warn(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "AI6657_LLMA_API_KEY=legacy-scoped-key\n"
        "AI6657_LLMB_MODEL=legacy-scoped-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_protocol, "_PROJECT_ROOT", tmp_path)

    secrets = llm_shim._load_secrets()

    assert secrets["llma"]["api_key"] == "legacy-scoped-key"
    assert secrets["llmb"]["model"] == "legacy-scoped-model"
    assert any("legacy" in w.lower() for w in secrets["warnings"])


def test_load_secrets_local_defaults_to_loopback():
    secrets = llm_shim._load_secrets()
    assert secrets["local"]["base_url"] == "http://127.0.0.1:8000/v1"


def test_remote_api_requires_key():
    try:
        llm_shim._resolve_api_key("https://api.example.com/v1", "")
    except ValueError as exc:
        assert "API key is required" in str(exc)
    else:
        raise AssertionError("missing remote API key must fail before the request")


def test_local_api_allows_placeholder_key():
    assert llm_shim._resolve_api_key("http://127.0.0.1:8000/v1", "") == "EMPTY"
