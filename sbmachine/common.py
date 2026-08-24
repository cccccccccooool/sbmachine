"""公共工具函数。提供路径解析、JSON 读写、hype 规则加载以及配置文件的集成读取支持。"""
from __future__ import annotations

import json
import os
import re
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


# ── 阶段产物统一输出目录与固定文件名 ──
# 只要给定一个输出目录（paths.output_dir，默认 output/sbmachine），未单独
# 指定的产物路径自动派生成 <output_dir>/<固定文件名>，无需逐个写死。
DEFAULT_OUTPUT_DIR = "output/sbmachine"

PRODUCT_FILENAMES: dict[str, str] = {
    "rounds_json": "rounds.json",
    "round_list_json": "round_list.json",
    "segments_out_json": "segments.json",
    "rounds_with_yolo_json": "rounds_with_yolo.json",
    "rounds_with_yolo_semantic_json": "rounds_with_yolo_semantic.json",
    "rounds_with_neutral_json": "rounds_with_neutral.json",
    "llma_input_json": "llma_input.json",
    "rounds_with_commentary_json": "rounds_with_commentary.json",
    "llmb_draft_package_json": "llmb_draft_package.json",
    "commentary_render_package_json": "commentary_render_package.json",
    "rounds_final_json": "rounds_final.json",
    "commentary_json": "commentary.json",
    "assemble_manifest_json": "assemble_manifest.json",
}


def ensure_output_paths(config: dict) -> dict:
    """把未显式配置的产物路径按统一输出目录 + 固定文件名派生补齐。"""
    paths = config.setdefault("paths", {})
    output_dir = str(paths.get("output_dir") or DEFAULT_OUTPUT_DIR).rstrip("/\\")
    for key, filename in PRODUCT_FILENAMES.items():
        if not paths.get(key):
            paths[key] = f"{output_dir}/{filename}"
    return config


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


def debug_output_dir(run_id: str) -> Path:
    """返回 output/debug/<run_id>/ 目录路径。"""
    return PROJECT_ROOT / "output" / "debug" / run_id


def resolve_backend(config: dict, stage: str) -> str:
    """解析某阶段的后端：环境变量 > semantic 分阶段配置 > llm.backend > 默认 vllm。"""
    semantic = config.get("semantic", {}) if isinstance(config.get("semantic", {}), dict) else {}
    llm = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    env_name = f"AI6657_{stage.upper()}_BACKEND"
    return str(os.getenv(env_name) or semantic.get(f"{stage}_backend") or llm.get("backend") or "vllm").lower()


_SPOKEN_LATIN_RE = re.compile(r"[A-Za-z0-9]+")


def count_spoken_chars(text: str) -> int:
    """口播字数：中文/符号逐字符计 1，连续英文或数字序列计 1（按词朗读）。

    例："blameF击杀Tauson" → blameF(1) + 击杀(2) + Tauson(1) = 4。
    用于 Phase3a/3b 的字符预算与超长校验，避免英文按字母数占用预算。
    """
    if not text:
        return 0
    return len(_SPOKEN_LATIN_RE.sub("", text)) + len(_SPOKEN_LATIN_RE.findall(text))
