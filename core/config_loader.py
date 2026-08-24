"""6657 风格离线录像解说 AI 项目。

项目功能：搭建一个“整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS
语音”的离线生成流水线。
本文件功能：统一配置文件加载器。

启动方式：被 sbmachine/common.py 的 load_config() 间接调用，也被各阶段脚本直接导入。
输入数据流：config/ 目录下的所有 *.yaml 文件（按字母序合并）或单个 yaml 文件。
输出数据流：返回一个合并后的 dict 配置对象，供调用方读取各模块参数。
用法用途：通过 load_config(path_or_dir) 统一加载项目配置；传入目录则合并所有 yaml，传入文件则加载单个文件。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "config"


class ConfigError(ValueError):
    """当彼此独立的配置文件对同一键给出冲突取值时抛出。"""


def _record_origins(value: Any, path: tuple[str, ...], source: str, origins: dict[tuple[str, ...], str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _record_origins(child, (*path, str(key)), source, origins)
    else:
        origins[path] = source


def _merge_config(
    target: dict,
    incoming: dict,
    *,
    source: str,
    origins: dict[tuple[str, ...], str],
    path: tuple[str, ...] = (),
) -> None:
    for key, value in incoming.items():
        key_path = (*path, str(key))
        if key not in target:
            target[key] = deepcopy(value)
            _record_origins(value, key_path, source, origins)
            continue

        current_value = target[key]
        if isinstance(current_value, dict) and isinstance(value, dict):
            _merge_config(current_value, value, source=source, origins=origins, path=key_path)
            continue
        if current_value == value:
            continue

        previous_source = origins.get(key_path, "an earlier config file")
        dotted_path = ".".join(key_path)
        raise ConfigError(
            f"config conflict at '{dotted_path}': {previous_source} defines {current_value!r}, "
            f"but {source} defines {value!r}"
        )


def _load_yaml(path: Path) -> dict:
    import yaml

    try:
        with path.open(encoding="utf-8") as config_file:
            yaml_data = yaml.safe_load(config_file) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(yaml_data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return yaml_data


def _resolve_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _normalize_demo_path(config: dict) -> None:
    paths = config.setdefault("paths", {})
    demo = config.setdefault("demo", {})
    if not isinstance(paths, dict) or not isinstance(demo, dict):
        raise ConfigError("config sections 'paths' and 'demo' must be mappings")

    output_value = paths.get("demo_output_dir")
    parsed_value = demo.get("parsed_dir")
    output_path = _resolve_path(output_value)
    parsed_path = _resolve_path(parsed_value)
    if output_path is not None and parsed_path is not None and output_path != parsed_path:
        raise ConfigError(
            "demo path conflict: paths.demo_output_dir "
            f"({output_value!r}) != demo.parsed_dir ({parsed_value!r})"
        )

    canonical = output_value if output_path is not None else parsed_value
    if canonical is not None:
        paths["demo_output_dir"] = canonical
        demo["parsed_dir"] = canonical


def load_config(path_or_dir=None) -> dict:
    """加载项目配置。

    被 sbmachine/common.py 的 load_config() 调用，也被各阶段脚本直接导入。

    Parameters
    ----------
    path_or_dir : Path | str | None
        配置路径。为目录时合并所有 *.yaml；为文件时加载单个 yaml；为 None 时默认 config/。

    Returns
    -------
    dict
        合并后的配置字典，键为各 yaml 顶层段名（如 pipeline/llm/yolo 等）。
    """
    config_path = _DEFAULT_CONFIG_DIR if path_or_dir is None else Path(path_or_dir)

    if not config_path.exists():
        raise ConfigError(f"config path does not exist: {config_path}")

    if config_path.is_dir():
        merged_config: dict = {}
        origins: dict[tuple[str, ...], str] = {}
        for yaml_file in sorted(config_path.glob("*.yaml")):
            yaml_data = _load_yaml(yaml_file)
            # 直接加载 train.yaml 时保留文件内的原始 schema；整体加载目录时，
            # 将其包进 train 命名空间，避免 train.profile 覆盖顶层 profile。
            if yaml_file.name == "train.yaml":
                yaml_data = {"train": yaml_data}
            _merge_config(merged_config, yaml_data, source=yaml_file.name, origins=origins)
        _normalize_demo_path(merged_config)
        return merged_config

    loaded_config = _load_yaml(config_path)
    _normalize_demo_path(loaded_config)
    return loaded_config
