"""Phase 3b — 风格模型的提示词组装辅助（persona、绰号注入、JSON/尾句提取）。"""
from __future__ import annotations

import json
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── persona ──

def _load_persona() -> str:
    path = _PROJECT_ROOT / "Prompt" / "persona.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


# ── prompt assembly ──

def _load_player_aliases() -> dict:
    path = _PROJECT_ROOT / "database" / "player_aliases.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _aliases_hint(neutral_text: str, aliases: dict) -> str:
    """仅为中性稿里真正出现过的选手注入绰号，避免无关绰号污染提示词。"""
    if not aliases:
        return ""
    lines = []
    nt_lower = neutral_text.lower()
    for std_name, info in aliases.items():
        if not isinstance(info, dict):
            continue
        if std_name.lower() not in nt_lower:
            continue
        alts = info.get("aliases", [])
        if not alts:
            continue
        alt_str = "、".join(f'"{a}"' for a in alts)
        desc = info.get("desc", "")
        line = f"  · {std_name} → 可叫 {alt_str}"
        if desc:
            line += f"（{desc}）"
        lines.append(line)
    if not lines:
        return ""
    return "选手绰号参考（可自然替换，不强制）：\n" + "\n".join(lines)


_LEAK_MARKERS = ("```json", '"scenes"', '"t_start"', "【中性稿】", "【场景信息】", "【当前对局状态】")
_CONTAMINATION_MARKERS = ("任务", "注：", "字数", "根据以上信息", "【任务", "【核心")


def _strip_tags(s: str) -> str:
    return re.sub(r"\[[^\]]{1,4}\]", "", s).strip()


def _extract_json_obj(raw: str) -> dict | None:
    """只解析恰好一个 JSON 对象；旧版代码围栏与前后散文一律拒绝。"""
    try:
        value = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _extract_tail(commentary: str, max_chars: int = 80) -> str:
    stripped = _strip_tags(commentary)
    parts = [p for p in re.split(r"(?<=[。！？])", stripped) if p.strip()]
    if not parts:
        return stripped[-40:] if len(stripped) > 40 else stripped
    tail = parts[-1]
    if len(parts) >= 2:
        candidate = parts[-2] + tail
        if len(candidate) <= max_chars:
            tail = candidate
    if len(tail) > max_chars:
        tail = tail[-40:]
    return tail

