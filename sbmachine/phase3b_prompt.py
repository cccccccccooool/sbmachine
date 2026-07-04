"""Phase 3b prompt helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path

from sbmachine.common import load_hype_rules

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_emotion_constraint(
    round_emotion: str,
    avg_hype: float,
    global_emotion: dict[str, float],
) -> str:
    rules = load_hype_rules()
    em_cfg = rules["emotions"]
    ec = rules["emotion_constraints"]

    sad_val = global_emotion.get("沮丧", 0.0)
    ang_val = global_emotion.get("愤怒", 0.0)
    sad_thr = float(em_cfg["沮丧"]["threshold"])
    ang_thr = float(em_cfg["愤怒"]["threshold"])

    lines = []

    if ang_val >= ang_thr:
        if round_emotion != "尖叫":
            lines.append(ec["愤怒"]["hint_zh"].replace("{emotion_val}", f"{ang_val:.2f}"))
    if sad_val >= sad_thr:
        if round_emotion not in ("激动", "尖叫"):
            lines.append(ec["沮丧"]["hint_zh"].replace("{emotion_val}", f"{sad_val:.2f}"))

    hype_str = f"{avg_hype:.2f}"
    if round_emotion in ec:
        tmpl = ec[round_emotion]["hint_zh"]
        lines.append(tmpl.replace("{hype}", hype_str).replace("{emotion_val}", hype_str))

    return "\n".join(lines) if lines else ""


# ── catchphrase few-shot ──

def _hype_bucket(hype: float, rules: dict) -> str:
    em = rules["emotions"]
    if hype >= float(em["尖叫"]["threshold"]):
        return "击杀/激动"
    if hype >= float(em["激动"]["threshold"]):
        return "残局/紧张"
    return "开场/平述"


def _few_shot_hint(catchphrases: dict[str, list[str]], hype: float, n: int = 4) -> str:
    rules = load_hype_rules()
    bucket = _hype_bucket(hype, rules)
    phrases = catchphrases.get(bucket, [])[:n]
    if not phrases:
        return ""
    return (
        "可自然化用以下口头禅（不要堆砌，不要每句都用）：\n"
        + "\n".join(f"  · {p}" for p in phrases)
    )


# ── commentary demos few-shot（仅 API 路径）──

def _demo_hint(demos: dict[str, list[str]], hype: float, n: int = 2) -> str:
    rules = load_hype_rules()
    bucket = _hype_bucket(hype, rules)
    samples = demos.get(bucket, [])[:n]
    if not samples:
        return ""
    return (
        "下面是 6657 在类似场面的真实解说片段，只学语气、节奏、用词，绝不照搬其中人名/事件/数据：\n"
        + "\n".join(f"  · {s}" for s in samples)
    )


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
    """Only inject aliases for players actually mentioned in the neutral text."""
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


_LEAK_MARKERS = ('```json', '"scenes"', '"t_start"', '【中性稿】', '【场景信息】', '【当前对局状态】')
_CONTAMINATION_MARKERS = ("任务", "注：", "字数", "根据以上信息", "【任务", "【核心")


def _strip_tags(s: str) -> str:
    return re.sub(r"\[[^\]]{1,4}\]", "", s).strip()


def _extract_json_obj(raw: str) -> dict | None:
    """剥 ```json``` 围栏 + 取最外层 { } 再解析；失败返回 None（FIX-4）。"""
    s = raw.strip()
    if "```" in s:
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        s = s[start:end + 1]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_tail(commentary: str, max_chars: int = 80) -> str:
    stripped = _strip_tags(commentary)
    parts = [p for p in re.split(r'(?<=[。！？])', stripped) if p.strip()]
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

