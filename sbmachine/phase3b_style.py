"""Phase 3b — 风格模型：中性稿 + hype hint → 6657 口播 + [情绪] 标签。"""
from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from core.prompt_loader import load_prompt
from audio_service.emotion import parse_emotional_text
from sbmachine.common import load_config, load_hype_rules, load_json_library, require_path, write_json
from sbmachine.schemas import EmotionSegment, SemanticData, load_match, save_match

_PROJECT_ROOT = Path(__file__).resolve().parents[1]




def _dominant_scene_emotion(hype: float) -> str:
    """Map scene hype score to emotion label (平淡/激动/尖叫). Same logic as phase3a."""
    em = load_hype_rules()["emotions"]
    if hype >= float(em["尖叫"]["threshold"]):
        return "尖叫"
    if hype >= float(em["激动"]["threshold"]):
        return "激动"
    return "平淡"


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


# A-1: stateless generate，不再用 /api/chat 累积历史
def _call_style(system: str, user_prompt: str, llm_cfg: dict, round_no: int = 0, debug: bool = False,
                max_tokens: int | None = None, log_ctx: dict | None = None) -> tuple[str, float]:
    """Returns (commentary_text, felt_intensity). Uses /api/generate (stateless)."""
    try:
        from sbmachine.common import generate_commentary
        raw = generate_commentary(user_prompt, llm_cfg, system, max_tokens=max_tokens, log_ctx=log_ctx)
    except Exception as exc:
        return f"[style error: {exc}]", 0.0

    data = _extract_json_obj(raw)
    if data is not None and "commentary" in data:
        commentary = str(data.get("commentary", ""))
        if (not commentary.strip()
                or any(mk in commentary for mk in _LEAK_MARKERS)
                or commentary.lstrip().startswith("{")):
            commentary, felt = "[style error: contract-leak]", 0.0
        else:
            try:
                felt = float(data.get("felt_intensity", 0.0) or 0.0)
            except (TypeError, ValueError):
                felt = 0.0
    else:
        # 模型输出裸口播文本（无 JSON 壳）时降级直接使用，避免丢弃有效解说
        stripped = raw.strip()
        if (stripped
                and not stripped.lstrip().startswith("{")
                and not any(mk in stripped for mk in _LEAK_MARKERS)):
            commentary, felt = stripped, 0.0
        else:
            commentary, felt = "[style error: unparseable]", 0.0

    # A-5: debug dump 改存 system_prompt + user_prompt，不再存 messages_input
    if debug:
        debug_dir = _PROJECT_ROOT / "output" / "debug_phase3"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dump = {
            "round_no":      round_no,
            "model":         llm_cfg.get("model", ""),
            "phase":         "3b_style",
            "system_prompt": system,
            "user_prompt":   user_prompt,
            "response_raw":  raw,
            "commentary":    commentary,
            "felt_intensity": felt,
        }
        out = debug_dir / f"r{round_no:03d}_3b_style.json"
        out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")

    return commentary, felt


# A-3: 提取尾巴，≤80字，按句界切
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


# ── main runner ──

def run_phase3b(
    *,
    neutral_path: Path,
    rounds_path: Path,
    output_rounds_path: Path,
    commentary_path: Path,
    config_path: Path,
    dry_run: bool = False,
) -> dict:
    import os
    from sbmachine.match_memory import MatchMemory

    config = load_config(config_path)
    debug_enabled = bool(config.get("debug", {}).get("phase3", False) or os.getenv("AI6657_DEBUG_PHASE3"))
    llm_cfg = dict(config.get("llm", {}))
    is_api = llm_cfg.get("backend", "ollama") == "api"   # 仅 API 路径走风格升级，本地零改动
    style_model = config.get("semantic", {}).get("style_model") or config.get("semantic", {}).get("model", "")
    if style_model:
        llm_cfg["model"] = style_model

    catchphrase_path = _PROJECT_ROOT / config.get("paths", {}).get(
        "catchphrase_library", "Prompt/json/catchphrase_library.json"
    )
    catchphrases = load_json_library(catchphrase_path)

    demos: dict[str, list[str]] = {}
    if is_api:
        demos_path = _PROJECT_ROOT / config.get("paths", {}).get(
            "commentary_demos", "Prompt/json/commentary_demos.json"
        )
        demos = load_json_library(demos_path)
    aliases = _load_player_aliases()
    persona = _load_persona()

    neutral_data = json.loads(neutral_path.read_text(encoding="utf-8"))
    neutral_by_round: dict[int, dict] = {
        int(r["round_no"]): r
        for r in neutral_data.get("rounds", [])
    }

    match = load_match(rounds_path)
    profile = str(config.get("profile", "style"))

    demo_rounds: list[dict] = []
    try:
        from sbmachine.common import resolve_path
        pd = resolve_path(config.get("demo", {}).get("parsed_dir", "output/demo"))
        if pd and (pd / "rounds.json").exists():
            demo_rounds = json.loads((pd / "rounds.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    memory = MatchMemory.init(map_name=match.map_name, total_rounds_est=len(match.rounds))
    manifest_rounds = []
    errors: list[dict] = []

    cs_rules_path = _PROJECT_ROOT / "Prompt" / "cs_rules.txt"
    cs_rules = cs_rules_path.read_text(encoding="utf-8").strip() if cs_rules_path.exists() else ""
    # A-1: system_content 固定，不再随 scene 滚动；API 路径用人味升级版 prompt
    style_prompt_name = "style_system_api" if is_api else "style_system"
    system_content = "\n\n".join(filter(None, [
        load_prompt(style_prompt_name).replace("{persona_hint}", persona),
        cs_rules,
    ]))

    # A-3: 跨 scene 跨局持久，不重置
    last_tail: str = ""

    for rnd in tqdm(match.rounds, desc="Phase3b style", unit="round"):
        global_emotion = memory.emotion_snapshot()

        round_data = neutral_by_round.get(rnd.round_no, {})
        scenes = round_data.get("scenes", [])
        round_emotion = round_data.get("round_emotion", "平淡")
        peak_hype = float(round_data.get("peak_hype", 0.0))
        avg_hype = float(round_data.get("avg_hype", 0.0))
        analyst_failed = bool(round_data.get("analyst_failed", False))

        if dry_run:
            commentary, felt_intensity = f"[平述]第{rnd.round_no}局解说占位。", avg_hype
            scenes_manifest = []
        elif analyst_failed or not scenes:
            print(f"[phase3b] round {rnd.round_no} skipped: analyst failed or no scenes", file=sys.stderr)
            commentary, felt_intensity = f"[平述]（第{rnd.round_no}局中性稿缺失，跳过解说）", 0.0
            scenes_manifest = []
        else:
            memory_ctx = memory.render()

            scene_commentaries: list[str] = []
            scene_commentaries_meta: list[dict] = []   # 改动5: scene 级输出元数据（t_start/t_end/emotion/text）
            felt_intensity = 0.0

            for scene in scenes:
                scene_neutral = scene.get("neutral", "")
                if not scene_neutral.strip():
                    continue
                scene_hype = float(scene.get("hype", avg_hype))
                char_budget = int(scene.get("char_budget", 100))
                scene_name = scene.get("scene", "")
                t_start = float(scene.get("t_start", rnd.start_sec))
                t_end = float(scene.get("t_end", rnd.end_sec))
                duration = max(1.0, t_end - t_start)

                scene_emotion = _dominant_scene_emotion(scene_hype)
                constraint = _build_emotion_constraint(scene_emotion, scene_hype, global_emotion)
                few_shot = _few_shot_hint(catchphrases, scene_hype)
                alias_hint = _aliases_hint(scene_neutral, aliases)
                demo_hint = _demo_hint(demos, scene_hype) if is_api else ""

                # API 路径不把绝对秒数喂给模型（防复述"867秒"），只留时长+字数预算
                if is_api:
                    scene_info = (
                        f"（约{duration:.0f}秒，字数预算约{char_budget}字）"
                        + (f"\n阶段：{scene_name}" if scene_name and scene_name != "full" else "")
                    )
                else:
                    scene_info = (
                        f"时间：{t_start:.1f}~{t_end:.1f}秒（{duration:.0f}秒，字数预算约{char_budget}字）"
                        + (f"\n阶段：{scene_name}" if scene_name and scene_name != "full" else "")
                    )

                # A-4: user_prompt 替代 chat_messages；A-3: 注入 last_tail
                user_prompt = "\n\n".join(filter(None, [
                    f"【当前对局状态】\n{memory_ctx}",
                    f"【上一句解说】\n{last_tail}" if last_tail else "",
                    f"【情绪约束】\n{constraint}" if constraint else "",
                    f"【口癖参考】\n{few_shot}" if few_shot else "",
                    f"【风格范例】\n{demo_hint}" if demo_hint else "",
                    f"【选手绰号】\n{alias_hint}" if alias_hint else "",
                    f"【场景信息】\n{scene_info}",
                    f"【中性稿】\n{scene_neutral}",
                ]))

                # 按字数预算推输出上限：封死复读/思考跑满 num_ctx 撞超时。
                # 输出含 JSON 包壳 + commentary 正文；中文 ~1.6 tok/字，留壳+余量。
                scene_max_tokens = max(160, min(768, int(char_budget * 2.2) + 80))

                log_ctx = {"round": f"round{rnd.round_no}", "scene": scene_name}
                scene_commentary, scene_felt = _call_style(
                    system_content,
                    user_prompt,
                    llm_cfg,
                    round_no=rnd.round_no,
                    debug=debug_enabled,
                    max_tokens=scene_max_tokens,
                    log_ctx=log_ctx,
                )

                if not scene_commentary.startswith("[style error:"):
                    prev = scene_commentaries[-1] if scene_commentaries else ""
                    if scene_commentary and _strip_tags(scene_commentary) == _strip_tags(prev):
                        # R-3: 相邻去重保留，无 transcript 无需 pop
                        print(f"[phase3b] round {rnd.round_no} scene '{scene_name}' adjacent-repeat, skipping", file=sys.stderr)
                    else:
                        scene_commentaries.append(scene_commentary)
                        felt_intensity = scene_felt
                        # 污染检测：含 instruction 标记则置空，阻断毒化传播
                        if any(m in scene_commentary for m in _CONTAMINATION_MARKERS):
                            last_tail = ""
                        else:
                            last_tail = _extract_tail(scene_commentary)  # A-3
                        # 记录 scene 级输出（用于 manifest scenes 字段）
                        scene_commentaries_meta.append({
                            "t_start": t_start,
                            "t_end":   t_end,
                            "emotion": scene_emotion,
                            "text":    _strip_tags(scene_commentary),
                        })
                else:
                    print(f"[phase3b] round {rnd.round_no} scene '{scene_name}' style error, skipping", file=sys.stderr)
                    errors.append({
                        "round": f"round{rnd.round_no}",
                        "round_no": rnd.round_no,
                        "scene": scene_name,
                        "error": scene_commentary,
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    })

            commentary = "".join(scene_commentaries) if scene_commentaries else f"[平述]（第{rnd.round_no}局所有场景解说失败）"
            scenes_manifest = scene_commentaries_meta  # 改动5: 仅非静默场景有产出

        if felt_intensity > 0.0:
            felt_clamp = float(load_hype_rules().get("felt_intensity_clamp", 0.2))
            delta = max(-felt_clamp, min(felt_clamp, felt_intensity - avg_hype))
            effective_hype = max(0.0, min(1.0, avg_hype + delta))
        else:
            effective_hype = avg_hype
        memory.update(rnd, demo_rounds, round_hype=effective_hype)

        parsed = parse_emotional_text(commentary)
        rnd.phase3_semantic = SemanticData(
            model_profile=profile,
            model_name=str(llm_cfg.get("model", "")),
            commentary_text=commentary,
            emotion_segments=[EmotionSegment(seg.emotion, seg.text, i) for i, seg in enumerate(parsed)],
        )
        manifest_rounds.append({
            "round_no":        rnd.round_no,
            "start_sec":       rnd.start_sec,
            "end_sec":         rnd.end_sec,
            "commentary_text": commentary,
            "hype_avg":        round(avg_hype, 3),
            "felt_intensity":  round(felt_intensity, 3),
            "emotion_segments": [seg.__dict__ for seg in rnd.phase3_semantic.emotion_segments],
            "scenes":          scenes_manifest,  # 改动5: scene 级时间戳+情绪+口播文本（纯增量）
        })

    if errors:
        err_path = _PROJECT_ROOT / "logs" / "error.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if err_path.exists():
            try:
                existing = json.loads(err_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.extend(errors)
        err_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    save_match(output_rounds_path, match)
    manifest = {
        "video_path":    match.video_path,
        "map_name":      match.map_name,
        "model_profile": profile,
        "model_name":    str(llm_cfg.get("model", "")),
        "rounds":        manifest_rounds,
    }
    write_json(commentary_path, manifest)
    return manifest
