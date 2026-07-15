"""从 Prompt/ 目录加载提示词文本。

启动方式：被 phase3a_analyst.py、phase3b_style.py 等运行时模块导入调用。
输入数据流：Prompt/ 目录下的运行时 .txt 文件（如 analyst_system.txt）。
输出数据流：返回文件内容字符串，供调用方做 .replace() 模板填充。
用法用途：通过 load_prompt(name) 读取 Prompt/{name}.txt；若文件不存在则抛出 FileNotFoundError。

用法示例：
    from core.prompt_loader import load_prompt
    text = load_prompt("analyst_system")     # 加载 Prompt/analyst_system.txt
    filled = load_prompt("analyst_round").replace("{json_payload}", json_str)
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_DIR = _PROJECT_ROOT / "Prompt"


def load_prompt(name: str) -> str:
    """读取 Prompt/{name}.txt 文件内容并返回去除首尾空白的字符串。

    被以下模块调用：sbmachine/phase3a_analyst.py（加载 analyst_system/analyst_round）、
    sbmachine/phase3b_style.py（加载 style_system）。

    Parameters
    ----------
    name : str
        提示词文件名（不含 .txt 后缀），如 "analyst_system"。

    Returns
    -------
    str
        文件内容（strip 后）。

    Raises
    ------
    FileNotFoundError
            当 Prompt/{name}.txt 不存在时抛出，错误信息包含目录下可用的文件列表。
    """
    path = _PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}. Available: {[p.stem for p in _PROMPT_DIR.glob('*.txt')]}")
    return path.read_text(encoding="utf-8").strip()
