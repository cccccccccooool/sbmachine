"""公共工具函数。提供路径解析、JSON读写、YAML加载、hype规则加载以及配置文件的集成读取支持。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
import requests

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



# ── hype rules（模块级缓存，避免热路径每局读磁盘） ──
_HYPE_RULES_CACHE: dict | None = None


def load_hype_rules() -> dict:
    """加载 Prompt/json/hype_rules.json，结果模块级缓存（进程内单例）。"""
    global _HYPE_RULES_CACHE
    if _HYPE_RULES_CACHE is None:
        path = PROJECT_ROOT / "Prompt" / "json" / "hype_rules.json"
        _HYPE_RULES_CACHE = json.loads(path.read_text(encoding="utf-8"))
    return _HYPE_RULES_CACHE


def load_json_library(path: Path) -> dict[str, list[str]]:
    """加载 bucket→片段列表 结构的 JSON 库（catchphrase / commentary_demos 共用）。

    Returns an empty dict when the file is missing or unparseable.
    Each entry may be a plain string or a dict with a 'text' key.
    Keys starting with '_' are treated as metadata and skipped.
    """
    if not path.exists():
        return {}
    try:
        lib = json.loads(path.read_text(encoding="utf-8"))
        return {
            bucket: [e.get("text", "") if isinstance(e, dict) else str(e) for e in entries]
            for bucket, entries in lib.items()
            if not bucket.startswith("_") and isinstance(entries, list)
        }
    except Exception:
        return {}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type(requests.HTTPError),
    reraise=True
)
def _post_with_retry(url: str, payload: dict, timeout: int):
    """带重试的 HTTP POST 请求。

    使用 tenacity 重试 3 次，仅对连接/超时类异常重试（HTTPError 包括 4xx 不重试），指数退避 2~10s。
    """
    # 强制忽略所有系统代理（Clash/v2ray 等），防止 127.0.0.1 被代理拦截导致 Connection Reset
    proxies = {"http": None, "https": None}
    response = requests.post(url, json=payload, timeout=timeout, proxies=proxies)
    response.raise_for_status()  # 4xx 抒 HTTPError 不会被 tenacity 重试
    return response


def _output_cap(llm_config: dict, max_tokens: int | None) -> int | None:
    """输出 token 上限：显式参数 > 配置 max_tokens > 无上限(None)。
    封死失控生成（思考链/复读跑满 num_ctx 撞超时）——同时治速度和输出端爆上下文。"""
    cap = int(max_tokens or llm_config.get("max_tokens", 0) or 0)
    return cap if cap > 0 else None


def resolve_backend(config: dict, stage: str) -> str:
    semantic = config.get("semantic", {}) if isinstance(config.get("semantic", {}), dict) else {}
    llm = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    env_name = f"AI6657_{stage.upper()}_BACKEND"
    return str(os.getenv(env_name) or semantic.get(f"{stage}_backend") or llm.get("backend") or "ollama").lower()
