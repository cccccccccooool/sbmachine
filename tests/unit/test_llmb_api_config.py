"""llmb_api.style_runtime_config 的云端特化键透传单测。

回归：cloud_style_output_max_tokens 曾被 STYLE_DEFAULTS 的 key 集合过滤掉，
导致 phase3b 云端 max_tokens 永远回退本地 1024（思考吃满截断 → unparseable）。
"""
from __future__ import annotations

from sbmachine import llmb_api


def _config(**semantic_overrides) -> dict:
    semantic = {"style_backend": "api"}
    semantic.update(semantic_overrides)
    return {"llm": {"backend": "api"}, "semantic": semantic}


def test_style_runtime_config_forwards_cloud_style_output_max_tokens():
    llm_cfg, style_cfg = llmb_api.style_runtime_config(
        _config(cloud_style_output_max_tokens=4096)
    )
    assert style_cfg["cloud_style_output_max_tokens"] == 4096
    assert style_cfg["style_output_max_tokens"] == llmb_api.STYLE_DEFAULTS["style_output_max_tokens"]


def test_style_runtime_config_keeps_local_default_when_cloud_key_absent():
    _, style_cfg = llmb_api.style_runtime_config(_config())
    assert "cloud_style_output_max_tokens" not in style_cfg
    assert style_cfg["style_output_max_tokens"] == llmb_api.STYLE_DEFAULTS["style_output_max_tokens"]


def test_style_runtime_config_coerces_cloud_key_to_int():
    _, style_cfg = llmb_api.style_runtime_config(
        _config(cloud_style_output_max_tokens="4096")
    )
    assert style_cfg["cloud_style_output_max_tokens"] == 4096
    assert isinstance(style_cfg["cloud_style_output_max_tokens"], int)


def test_style_runtime_config_ignores_falsy_cloud_key():
    _, style_cfg = llmb_api.style_runtime_config(
        _config(cloud_style_output_max_tokens=0)
    )
    assert "cloud_style_output_max_tokens" not in style_cfg
