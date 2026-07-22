"""公共工具函数。提供路径解析、JSON 读写、hype 规则加载以及配置文件的集成读取支持。"""
from __future__ import annotations

import json
import os
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


def require_debug_output(path: Path, name: str) -> Path:
    """防止单独运行的阶段覆盖已发布的流水线产物目录。"""
    resolved = path.resolve()
    for published_dir in (PROJECT_ROOT / "output" / "demo", PROJECT_ROOT / "output" / "sbmachine"):
        try:
            resolved.relative_to(published_dir.resolve())
        except ValueError:
            continue
        raise ValueError(
            f"{name} points at published output ({path}); use run.py or configure a custom debug output"
        )
    return path


def read_json(path: Path) -> Any:
    """读取 JSON 文件并返回解析后的对象。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    """将对象写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_config(path_or_dir: Path | str | None = None) -> dict:
    """从 config/ 目录（递归合并所有 yaml 文件）或单个 yaml 文件中加载配置。"""
    from core.config_loader import load_config as _load
    return _load(path_or_dir)


# ── hype rules（模块级缓存，避免热路径每局读磁盘） ──
_HYPE_RULES_CACHE: dict | None = None
_CS_GAME_RULES_CACHE: dict | None = None


def load_hype_rules() -> dict:
    """加载 Prompt/json/hype_rules.json，结果模块级缓存（进程内单例）。"""
    global _HYPE_RULES_CACHE
    if _HYPE_RULES_CACHE is None:
        path = PROJECT_ROOT / "Prompt" / "json" / "hype_rules.json"
        _HYPE_RULES_CACHE = json.loads(path.read_text(encoding="utf-8"))
    return _HYPE_RULES_CACHE


def load_cs_game_rules() -> dict:
    """加载 Phase3 的确定性 CS2 规则，热路径只读一次。"""
    global _CS_GAME_RULES_CACHE
    if _CS_GAME_RULES_CACHE is None:
        path = PROJECT_ROOT / "Prompt" / "json" / "cs_game_rules.json"
        _CS_GAME_RULES_CACHE = json.loads(path.read_text(encoding="utf-8"))
    return _CS_GAME_RULES_CACHE


def _output_cap(llm_config: dict, max_tokens: int | None) -> int | None:
    """输出 token 上限：显式参数 > 配置 max_tokens > 无上限(None)。
    封死失控生成（思考链/复读跑满 num_ctx 撞超时）——同时治速度和输出端爆上下文。"""
    cap = int(max_tokens or llm_config.get("max_tokens", 0) or 0)
    return cap if cap > 0 else None


def resolve_backend(config: dict, stage: str) -> str:
    """解析某阶段的后端：环境变量 > semantic 分阶段配置 > llm.backend > 默认 vllm。"""
    semantic = config.get("semantic", {}) if isinstance(config.get("semantic", {}), dict) else {}
    llm = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    env_name = f"AI6657_{stage.upper()}_BACKEND"
    return str(os.getenv(env_name) or semantic.get(f"{stage}_backend") or llm.get("backend") or "vllm").lower()

def _phase3_role_enabled(phases: dict, key: str) -> bool:
    """Return whether one Phase3 role is enabled, including the legacy switch."""
    if key in phases:
        return bool(phases[key])
    return bool(phases.get("phase3_semantic", True))


def phase3_backend_plan(config: dict) -> dict[str, str]:
    """Map every enabled Phase3 role to its configured transport backend.

    This is the single source of truth for deciding whether the optional local
    vLLM/talk component is needed.  Environment overrides remain honoured via
    :func:`resolve_backend`.
    """
    phases = config.get("phases", {})
    phases = phases if isinstance(phases, dict) else {}
    plan: dict[str, str] = {}
    for role, phase_key in (("analyst", "phase3a_semantic"), ("style", "phase3b_semantic")):
        if _phase3_role_enabled(phases, phase_key):
            plan[role] = resolve_backend(config, role)
    return plan


def talk_component_requirement(config: dict) -> dict[str, object]:
    """Describe whether this run needs the optional local vLLM talk add-on."""
    plan = phase3_backend_plan(config)
    vllm_roles = [role for role, backend in plan.items() if backend == "vllm"]
    if vllm_roles:
        reason = "vllm selected for enabled Phase3 role(s): " + ", ".join(vllm_roles)
    elif plan:
        reason = "all enabled Phase3 role(s) use API or another non-vLLM backend"
    else:
        reason = "no Phase3 role is enabled"
    return {
        "required": bool(vllm_roles),
        "roles": vllm_roles,
        "active_backends": plan,
        "reason": reason,
    }
