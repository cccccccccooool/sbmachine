"""配置文件直白化：单份配置、无 profile 覆盖层、backend 即唯一开关。"""
from pathlib import Path

import pytest

from core.config_loader import ConfigError, load_config


def _write_configs(tmp_path: Path) -> Path:
    (tmp_path / "pipeline.yaml").write_text(
        "phases:\n  phase3a_semantic: true\n  phase3b_semantic: true\n",
        encoding="utf-8",
    )
    (tmp_path / "llm.yaml").write_text(
        "semantic:\n  analyst_backend: api\n  style_backend: api\n"
        "  analyst_window_concurrency: 4\n  style_concurrent_scenes: 4\n",
        encoding="utf-8",
    )
    return tmp_path


def test_load_config_single_flat_semantic(tmp_path):
    config_dir = _write_configs(tmp_path)
    merged = load_config(config_dir)
    semantic = merged["semantic"]
    assert semantic["analyst_backend"] == "api"
    assert semantic["analyst_window_concurrency"] == 4
    # 无 profile 机制：semantic 下不应再有 profiles 覆盖层
    assert "profiles" not in semantic


def test_load_config_rejects_conflicting_scalars(tmp_path):
    config_dir = _write_configs(tmp_path)
    (config_dir / "extra.yaml").write_text(
        "semantic:\n  analyst_window_concurrency: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="semantic.analyst_window_concurrency"):
        load_config(config_dir)
