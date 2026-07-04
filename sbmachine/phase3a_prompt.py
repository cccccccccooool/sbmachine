"""Phase 3a prompt and response helpers."""
from __future__ import annotations

import json
import re

from core.prompt_loader import load_prompt
from sbmachine.phase3a_payload import _dumps_compact


_ANALYST_JSON_CONTRACT = (
    '严格输出单个 JSON 对象：{"scenes":[{"t_start":float,"t_end":float,"scene":str,"neutral":str}]}；'
    "t_start/t_end 填入 prompt 窗口列表中的对应值（不要自己算）；不加 markdown 代码块，不加任何额外文本。"
)


def _build_analyst_system() -> str:
    """analyst 的 ollama system 字段 = 中性事实规则 + JSON 契约（非 6657 persona）。
    放进 system，ollama 截断输入时输出契约不丢（FIX-0）。"""
    return load_prompt("analyst_system") + "\n\n" + _ANALYST_JSON_CONTRACT


def _format_window_list(windows: list[tuple[float, float]], beats: list[dict]) -> str:
    """生成给 LLM 的窗口列表文本，每行标注窗内事件。"""
    lines = []
    for idx, (lo, hi) in enumerate(windows):
        events_in_window: list[str] = []
        for beat in beats:
            t = float((beat.get("when") or {}).get("video_time", 0))
            if not (lo <= t < hi):
                continue
            ev = beat.get("events") or {}
            for k in (ev.get("kills") or []):
                if k.get("is_corpse_shoot"):
                    continue
                attacker = k.get("attacker", "?")
                weapon = k.get("weapon", "?")
                victim = k.get("victim", "?")
                callout = k.get("callout") or ""
                loc = f"@{callout}" if callout else ""
                events_in_window.append(f"■击杀 {attacker}({weapon})→{victim}{loc}")
            c4 = ev.get("c4") or {}
            if c4.get("planted"):
                events_in_window.append("■C4 planted")
            if c4.get("begin_defuse_tick"):
                events_in_window.append("■拆弹开始")
        event_str = " / ".join(events_in_window) if events_in_window else "无"
        lines.append(f"窗{idx + 1}: t∈[{lo:.1f}, {hi:.1f}]  事件: {event_str}")
    return "\n".join(lines)


def _build_analyst_prompt(payload: dict, windows: list[tuple[float, float]] | None = None) -> str:
    """user prompt = 窗口列表 + 紧凑 JSON payload。"""
    template = load_prompt("analyst_round")
    beats = payload.get("keyframes", [])
    if windows is None:
        windows = [(float(payload.get("start_sec", 0)), float(payload.get("end_sec", 0)))]
    n = len(windows)
    window_list = _format_window_list(windows, beats)
    return (template
            .replace("{N}", str(n))
            .replace("{window_list}", window_list)
            .replace("{json_payload}", _dumps_compact(payload)))


# ── 切段机器（segment_long_rounds=true 时启用；切段+合并全在 phase3a 内部，下游透明） ──

def _build_segment_prompt(seg_payload: dict, i: int, k: int, lo: float, hi: float,
                           windows: list[tuple[float, float]] | None = None,
                           state_so_far: str = "") -> str:
    """段 prompt = 分段元信息头（身份+时间窗）+ 跨段前情（i>1 时）+ 整局模板。
    state_so_far: _build_round_state_so_far() 输出，仅第 2 段起注入，不写入任何 JSON 产物。
    """
    hi_disp = float(seg_payload.get("end_sec", lo)) if hi == float("inf") else hi
    meta = (
        f"【分段说明】本局共切 {k} 段，当前第 {i} 段，时间窗 t∈[{lo:.0f},{hi_disp:.0f})。"
        f"只输出 t_start 落在本窗内的 scene；不写「本局开始/结束」类整局收尾语。"
    )
    if state_so_far:
        meta += f"\n\n【本局前情】（第 {i}/{k} 段，前段已发生）\n{state_so_far}"
    # 传入本段对应的子窗口（若无则用全段单窗兜底）
    seg_windows = windows or [(lo, hi_disp)]
    return meta + "\n\n" + _build_analyst_prompt(seg_payload, windows=seg_windows)


def _build_round_state_so_far(beats: list[dict]) -> str:
    """从已覆盖的 beats 提取前情摘要（击杀 + 植弹），注入分段 prompt 的跨段上下文。
    去重：同一 victim 的击杀只报告一次（防重叠帧重复报告）。
    仅注入 LLM-A 的 prompt，不出现在任何 JSON 产物中。
    """
    seen_victims: set[str] = set()
    kill_lines: list[str] = []
    bomb_planted = False

    for beat in beats:
        ev = beat.get("events") or {}
        for k in (ev.get("kills") or []):
            if k.get("is_corpse_shoot"):
                continue
            victim = str(k.get("victim", "?"))
            if victim in seen_victims:
                continue
            seen_victims.add(victim)
            attacker = k.get("attacker", "?")
            weapon = k.get("weapon", "?")
            callout = k.get("callout") or ""
            loc = f" 在 {callout}" if callout else ""
            kill_lines.append(f"- 击杀: {attacker}({weapon}){loc} 击杀 {victim}")
        c4 = ev.get("c4") or {}
        if c4.get("planted"):
            bomb_planted = True

    if not kill_lines and not bomb_planted:
        return ""

    parts = kill_lines[:]
    if bomb_planted:
        parts.append("- 炸弹已植入")
    return "\n".join(parts)


def _parse_scenes_response(text: str) -> list[dict] | None:
    """Try to parse LLM output as JSON scenes array. Returns None on failure."""
    stripped = text.strip()
    # 剥离 ```json ... ``` 围栏（预呓文常带）
    if "```" in stripped:
        m = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
        if m:
            stripped = m.group(1).strip()
    # 取最外层 { }（容忍"好的/我现在"类预呓文前后缀）
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        stripped = stripped[start:end + 1]
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and isinstance(data.get("scenes"), list):
            scenes = data["scenes"]
            if scenes and all(isinstance(s, dict) for s in scenes):
                return scenes
    except (json.JSONDecodeError, ValueError):
        pass
    return None

