"""公共工具函数。提供路径解析、JSON读写、YAML加载以及配置文件的集成读取支持。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(value: str | Path | None, *, base: Path | None = None) -> Path | None:
    """将相对路径解析为绝对路径。"""
    if value is None or value == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or PROJECT_ROOT) / path


def require_path(value: str | Path | None, name: str, *, base: Path | None = None) -> Path:
    """解析路径并在为空时报错。"""
    path = resolve_path(value, base=base)
    if path is None:
        raise ValueError(f"缺少必填参数:{name}")
    return path


def read_json(path: Path) -> Any:
    """读取 JSON 文件并返回解析后的对象。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    """将对象写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_yaml(path: Path) -> dict:
    """读取 YAML 文件，不存在时返回空 dict。"""
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_config(path_or_dir: Path | str | None = None) -> dict:
    """从 config/ 目录（合并所有 yaml 文件）或单个 yaml 文件中加载配置。     内部委托给 core.config_loader.load_config。"""
    from core.config_loader import load_config as _load
    return _load(path_or_dir)


def profile_value(config: dict, section: str, profile: str, default: Any = None) -> Any:
    """从配置中按 profile 查找值。被各阶段脚本用于读取模型 profile 参数。"""
    data = config.get(section, {})
    if isinstance(data, dict):
        profiles = data.get("profiles", {})
        if isinstance(profiles, dict) and profile in profiles:
            return profiles[profile]
    return default



def _output_cap(llm_config: dict, max_tokens: int | None) -> int | None:
    """输出 token 上限：显式参数 > 配置 max_tokens > 无上限(None)。
    封死失控生成（思考链/复读跑满 num_ctx 撞超时）——同时治速度和输出端爆上下文。"""
    cap = int(max_tokens or llm_config.get("max_tokens", 0) or 0)
    return cap if cap > 0 else None


def generate_commentary(
    prompt: str,
    llm_config: dict,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
) -> str:
    """调用 LLM API（无历史，分析模型专用）。"""

    if system_prompt is None:
        trained = PROJECT_ROOT / "training" / "persona_system_prompt.txt"
        if trained.exists():
            system_prompt = trained.read_text(encoding="utf-8").strip()
        else:
            fallback = PROJECT_ROOT / "Prompt" / "commentary_persona.txt"
            if fallback.exists():
                system_prompt = fallback.read_text(encoding="utf-8").strip()
            else:
                system_prompt = "你是 6657 风格的 CS2 中文解说,输出带 [情绪] 标签的口播解说。"

    cap = _output_cap(llm_config, max_tokens)
    backend = llm_config.get("backend", "api")

    # 增加直观的 API 正在请求提示（防多线程终端死锁，改用原生 print）
    ctx_hint = ""
    if log_ctx:
        r = log_ctx.get("round", "")
        s = log_ctx.get("scene", "")
        if r and s: ctx_hint = f" [{r} - {s}]"
        elif r: ctx_hint = f" [{r}]"
    
    timeout = llm_config.get("timeout_sec", 300)
    print(f"  >> [LLM API] 正在请求 {backend} 后端{ctx_hint}... (timeout: {timeout}s)", flush=True)

    from sbmachine.llm_shim import generate_commentary_openai
    return generate_commentary_openai(prompt, llm_config, system_prompt, max_tokens=cap, log_ctx=log_ctx)


def generate_commentary_chat(
    messages: list[dict],
    llm_config: dict,
    max_tokens: int | None = None,
) -> str:
    """调用 LLM API（带对话历史，风格模型专用）。"""
    cap = _output_cap(llm_config, max_tokens)
    from sbmachine.llm_shim import generate_commentary_chat_openai
    return generate_commentary_chat_openai(messages, llm_config, max_tokens=cap)
