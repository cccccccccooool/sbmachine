"""
启动方式：被 sbmachine/phase2_vision.py、phase3a_analyst.py、phase3b_style.py、phase2_vlm_client.py 等模块导入调用。
输入数据流：Prompt/ 目录下的 .txt 文件（如 vlm_scene.txt、commentary.txt 等）。
输出数据流：返回文件内容字符串，供调用方做 .replace() 模板填充。
用法用途：通过 load_prompt(name) 读取 Prompt/{name}.txt；若文件不存在则抛出 FileNotFoundError。
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_DIR = _PROJECT_ROOT / "Prompt"


def load_prompt(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}. Available: {[p.stem for p in _PROMPT_DIR.glob('*.txt')]}")
    return path.read_text(encoding="utf-8").strip()
