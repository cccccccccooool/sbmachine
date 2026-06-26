"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：统一配置文件加载器。

启动方式：被 sbmachine/common.py 的 load_config() 间接调用，也被各阶段脚本直接导入。
输入数据流：config/ 目录下的所有 *.yaml 文件（按字母序合并）或单个 yaml 文件。
输出数据流：返回一个合并后的 dict 配置对象，供调用方读取各模块参数。
用法用途：通过 load_config(path_or_dir) 统一加载项目配置；传入目录则合并所有 yaml，传入文件则加载单个文件。
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "config"


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
        合并后的配置字典，键为各 yaml 顶层段名（如 vision/llm/tts 等）。
    """
    import yaml

    p = _DEFAULT_CONFIG_DIR if path_or_dir is None else Path(path_or_dir)

    if p.is_dir():
        import warnings
        merged: dict = {}
        for yaml_file in sorted(p.glob("*.yaml")):
            with yaml_file.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for key in data:
                if key in merged:
                    warnings.warn(f"config key '{key}' overridden by {yaml_file.name}", stacklevel=2)
            merged.update(data)
        return merged

    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
