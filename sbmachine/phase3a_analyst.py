"""Phase 3a — 分析模型：事件行 → 中性解说稿 + hype 曲线。"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from sbmachine.common import load_config, require_path, resolve_backend, write_json
from sbmachine.hype_score import _compute_char_budget, _scene_hype, _speech_rate_config, compute_hype, dominant_round_emotion
from sbmachine.phase3a_payload import _ANALYST_PROMPT_TOKEN_BUDGET, _CHARS_PER_TOKEN, _dumps_compact, _semantic_payload, _slim_payload_for_prompt
from sbmachine.phase3a_prompt import _build_analyst_prompt, _build_analyst_system, _build_segment_prompt, _parse_scenes_response
from sbmachine.phase3a_windows import _segment_windows, build_scene_windows
from sbmachine.schemas import load_match

_PROJECT_ROOT = Path(__file__).resolve().parents[1]












def _analyst_scenes_segmented(
    slim_full: dict,
    llm_cfg: dict,
    gen_fn,
    system: str,
    round_no: int,
    debug: bool,
    overlap: int,
    scene_windows: list[tuple[float, float]] | None = None,
) -> list[dict] | None:
    """逐段分析 + 归属窗合并。返回与 _parse_scenes_response 同形的 parsed_scenes，全段失败→None。
    scene_windows: build_scene_windows 产出的确定性窗口，用于精确分段 prompt 注入。
    """
    seg_windows = _segment_windows(slim_full.get("keyframes", []), _ANALYST_PROMPT_TOKEN_BUDGET, overlap)
    if not seg_windows:
        return None
    k = len(seg_windows)
    base = {key: slim_full[key] for key in ("round_no", "start_sec", "end_sec", "demo_round_hint") if key in slim_full}
    all_frames = slim_full.get("keyframes", [])
    merged: list[dict] = []
    covered_beats: list[dict] = []   # 已覆盖段的 beats，用于跨段前情
    for idx, w in enumerate(seg_windows):
        seg_payload = {**base, "keyframes": w["frames"]}
        # 跨段前情：第 2 段起注入前段已发生事件摘要
        state_so_far = ""
        if idx > 0 and covered_beats:
            state_so_far = _build_round_state_so_far(covered_beats)
        # 本分析段对应的确定性子窗口
        if scene_windows:
            # 只取落在本段时间范围内的子窗口
            lo_f, hi_f = w["lo"], w["hi"]
            sub_windows: list[tuple[float, float]] = [
                (a, b) for (a, b) in scene_windows if a < hi_f and b > lo_f
            ] or [(lo_f, hi_f if hi_f != float("inf") else float(base.get("end_sec", lo_f)))]
        else:
            lo_f = w["lo"]
            hi_f = w["hi"] if w["hi"] != float("inf") else float(base.get("end_sec", lo_f))
            sub_windows = [(lo_f, hi_f)]
        prompt = _build_segment_prompt(
            seg_payload, idx + 1, k, w["lo"], w["hi"],
            windows=sub_windows,
            state_so_far=state_so_far,
        )
        raw = _call_analyst(prompt, llm_cfg, gen_fn, system_prompt=system, round_no=round_no, debug=debug, seg=idx + 1)
        if raw == _ANALYST_FAILED:
            # 跳过失败段，但仍累计 covered_beats
            covered_beats.extend(w["frames"])
            continue
        for sc in (_parse_scenes_response(raw) or []):
            t0 = float(sc.get("t_start", -1))
            if w["lo"] <= t0 < w["hi"]:          # ★归属窗去重：重叠帧仅上下文，边界事件归前段
                merged.append(sc)
        covered_beats.extend(w["frames"])
    if not merged:
        return None
    merged.sort(key=lambda s: float(s.get("t_start", 0)))
    for a, b in zip(merged, merged[1:]):         # 单调缝合：t_end≤下个 t_start（守音画同步）
        if float(a.get("t_end", 0)) > float(b.get("t_start", 0)):
            a["t_end"] = b["t_start"]
    return merged




_ANALYST_FAILED = "__ANALYST_FAILED__"


# ── scene helpers ──



def _call_analyst(prompt: str, llm_cfg: dict, gen_fn, system_prompt: str | None = None, round_no: int = 0, debug: bool = False, seg: int = 0) -> str:
    try:
        log_ctx = {"round": f"round{round_no}"}
        if seg > 0:
            log_ctx["scene"] = f"seg{seg}"
        result = gen_fn(prompt, llm_cfg, system_prompt=system_prompt, log_ctx=log_ctx)
    except Exception as exc:
        print(f"[phase3a] round {round_no} analyst error: {exc}", file=sys.stderr)
        return _ANALYST_FAILED

    # ── debug dump ──
    if debug:
        debug_dir = _PROJECT_ROOT / "output" / "debug_phase3"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dump = {
            "round_no":  round_no,
            "seg":       seg,
            "model":     llm_cfg.get("model", ""),
            "phase":     "3a_analyst",
            "prompt":    prompt,
            "response":  result,
        }
        name = f"r{round_no:03d}_s{seg}_3a_analyst.json" if seg else f"r{round_no:03d}_3a_analyst.json"
        out = debug_dir / name
        out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    # ────────────────────────────────────────────────────────────────────────
    return result


# ── main runner ──

def run_phase3a(
    *,
    rounds_path: Path,
    output_path: Path,
    config_path: Path,
    demo_rounds_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    import os
    config = load_config(config_path)
    debug_enabled = bool(config.get("debug", {}).get("phase3", False) or os.getenv("AI6657_DEBUG_PHASE3"))
    llm_cfg = dict(config.get("llm", {}))
    analyst_model = config.get("semantic", {}).get("analyst_model") or config.get("semantic", {}).get("model", "")
    if analyst_model:
        llm_cfg["model"] = analyst_model
    # 分析模型输出 JSON scenes 数组（多 scene 较长），单独给较大上限封住失控生成。
    llm_cfg["max_tokens"] = int(config.get("semantic", {}).get("analyst_max_tokens", 3072))
    backend = resolve_backend(config, "analyst")
    if backend == "api":
        from sbmachine import llma_api as _llma_backend
    else:
        from sbmachine import llma_local as _llma_backend
    gen_fn = _llma_backend.generate

    match = load_match(rounds_path)

    demo_rounds: list[dict] = []
    try:
        from sbmachine.common import resolve_path
        pd = resolve_path(config.get("demo", {}).get("parsed_dir", "output/demo"))
        if pd:
            if not demo_rounds and (pd / "rounds.json").exists():
                demo_rounds = json.loads((pd / "rounds.json").read_text(encoding="utf-8"))
    except Exception as exc:  # IO error or JSON corruption in optional demo file
        print(f"[phase3a] warning: could not load demo rounds.json: {exc}", file=sys.stderr)
    if demo_rounds_path and demo_rounds_path.exists():
        demo_rounds = json.loads(demo_rounds_path.read_text(encoding="utf-8"))

    tick_rate = 64.0
    try:
        from sbmachine.common import resolve_path
        pd = resolve_path(config.get("demo", {}).get("parsed_dir", "output/demo"))
        if pd and (pd / "demo_meta.json").exists():
            tick_rate = float(json.loads((pd / "demo_meta.json").read_text()).get("tick_rate", 64))
    except Exception as exc:  # IO error or JSON corruption in optional demo_meta.json
        print(f"[phase3a] warning: could not read demo_meta.json (tick_rate defaults to {tick_rate}): {exc}", file=sys.stderr)

    concurrent_rounds = max(1, int(config.get("semantic", {}).get("analyst_concurrent_rounds", 1)))
    analyst_system = _build_analyst_system()   # 常量，循环外算一次
    segment_on = bool(config.get("semantic", {}).get("segment_long_rounds", False))
    segment_overlap = max(0, int(config.get("semantic", {}).get("segment_overlap_frames", 2)))
    window_max_sec = float(config.get("semantic", {}).get("window_max_sec", 10.0))
    window_min_sec = float(config.get("semantic", {}).get("window_min_sec", 3.0))

    def _process_round(rnd) -> dict:
        payload = _semantic_payload(rnd)
        beats = payload.get("keyframes", [])      # 全帧 → compute_hype（不可降采样）
        hypes = compute_hype(beats, demo_rounds, tick_rate)

        # 取峰值而非均值定回合情绪：一局的情绪身份由它的最高光时刻决定，
        # 均值会被大量平淡 beat 稀释掉 ace/残局这种峰值（如纯走位局里一个1v3）。
        peak_hype = max(hypes) if hypes else 0.0
        avg_hype = round(sum(hypes) / len(hypes), 3) if hypes else 0.0
        round_emotion = dominant_round_emotion(peak_hype)
        speech_rate = _speech_rate_config()

        # ── 确定性切窗：由 demo 事件锚点决定，不依赖 LLM ──
        scene_wins = build_scene_windows(
            beats,
            rnd.start_sec,
            rnd.end_sec,
            window_max_sec=window_max_sec,
            window_min_sec=window_min_sec,
        )

        # 长回合决策树：三分支都产出与 _parse_scenes_response 同形的 parsed_scenes（或 None）。
        if dry_run:
            parsed_scenes = [{
                "t_start": rnd.start_sec, "t_end": rnd.end_sec,
                "scene": "full", "neutral": f"[dry-run] 第{rnd.round_no}局中性稿占位。",
            }]
        else:
            slim_full = _slim_payload_for_prompt(payload, downsample=False)   # 仅瘦字段不降帧
            est_tok = len(_dumps_compact(slim_full)) / _CHARS_PER_TOKEN
            if est_tok <= _ANALYST_PROMPT_TOKEN_BUDGET:
                # 绝大多数局：单次调用（全帧已落预算内）
                raw = _call_analyst(
                    _build_analyst_prompt(slim_full, windows=scene_wins), llm_cfg, gen_fn,
                    system_prompt=analyst_system, round_no=rnd.round_no, debug=debug_enabled)
                parsed_scenes = None if raw == _ANALYST_FAILED else _parse_scenes_response(raw)
            elif segment_on:
                # 超预算 + 开关 ON：切段无损，逐段分析后合并（附传确定性子窗口）
                parsed_scenes = _analyst_scenes_segmented(
                    slim_full, llm_cfg, gen_fn, analyst_system, rnd.round_no, debug_enabled,
                    segment_overlap, scene_windows=scene_wins)
            else:
                # 超预算 + 开关 OFF（默认）：二次压缩降采样，单次调用
                slim_ds = _slim_payload_for_prompt(payload)
                raw = _call_analyst(
                    _build_analyst_prompt(slim_ds, windows=scene_wins), llm_cfg, gen_fn,
                    system_prompt=analyst_system, round_no=rnd.round_no, debug=debug_enabled)
                parsed_scenes = None if raw == _ANALYST_FAILED else _parse_scenes_response(raw)

        # FIX-2+3：空/过短/预呓文/不可解析/全段失败 一律判失败，绝不把原始响应当 neutral。
        analyst_failed = parsed_scenes is None

        if parsed_scenes is not None:
            # ── 将 LLM 输出按确定性窗口对齐 ──
            # 建立 LLM 输出的 neutral 索引（按顺序，窗口数对齐时直接映射）
            llm_neutral: dict[int, str] = {}    # win_idx → neutral
            llm_scene: dict[int, str] = {}      # win_idx → scene name
            if len(parsed_scenes) == len(scene_wins):
                # LLM 按序输出，1-to-1 映射
                for wi, sc in enumerate(parsed_scenes):
                    llm_neutral[wi] = sc.get("neutral", "")
                    llm_scene[wi] = sc.get("scene", "")
            else:
                # 数量不匹配：按 t_start 最近窗匹配；同窗多 scene neutral 拼接
                win_starts = [w[0] for w in scene_wins]
                for sc in parsed_scenes:
                    t0 = float(sc.get("t_start", -1))
                    if not scene_wins:
                        break
                    wi = min(range(len(win_starts)), key=lambda i: abs(win_starts[i] - t0))
                    neu = sc.get("neutral", "")
                    if wi not in llm_neutral:
                        llm_neutral[wi] = neu
                        llm_scene[wi] = sc.get("scene", "")
                    elif neu.strip():
                        llm_neutral[wi] = llm_neutral[wi] + "。" + neu  # 同窗拼接

            scenes_out = []
            for wi, (t_start, t_end) in enumerate(scene_wins):
                duration = max(1.0, t_end - t_start)
                sc_hype = _scene_hype(beats, hypes, t_start, t_end)
                sc_emotion = dominant_round_emotion(sc_hype)
                char_budget = _compute_char_budget(duration, sc_emotion, speech_rate)
                neutral = llm_neutral.get(wi, "")

                # 事件窗兜底：LLM 漏写时自动生成 neutral
                if not neutral.strip():
                    fallback_parts: list[str] = []
                    for beat in beats:
                        t = float((beat.get("when") or {}).get("video_time", 0))
                        if not (t_start <= t < t_end):
                            continue
                        ev = beat.get("events") or {}
                        for k in (ev.get("kills") or []):
                            if k.get("is_corpse_shoot"):
                                continue
                            attacker = k.get("attacker", "?")
                            callout = k.get("callout") or ""
                            weapon = k.get("weapon", "?")
                            victim = k.get("victim", "?")
                            loc = f"在{callout}" if callout else ""
                            fallback_parts.append(f"{attacker}{loc}用{weapon}击杀{victim}")
                        c4 = ev.get("c4") or {}
                        if c4.get("planted"):
                            site = ""
                            for p in (beat.get("where", {}) or {}).get("players", []):
                                co = p.get("callout") or ""
                                if co:
                                    site = co
                                    break
                            fallback_parts.append(f"T方在{site}完成下包" if site else "T方完成下包")
                    if fallback_parts:
                        neutral = "【事件自动生成】" + "；".join(fallback_parts)

                scenes_out.append({
                    "t_start":    t_start,
                    "t_end":      t_end,
                    "scene":      llm_scene.get(wi, ""),
                    "neutral":    neutral,
                    "hype":       sc_hype,
                    "char_budget": char_budget,
                })
        else:
            # fallback: single scene covering full round
            duration = max(1.0, rnd.end_sec - rnd.start_sec)
            char_budget = _compute_char_budget(duration, round_emotion, speech_rate)
            scenes_out = [{
                "t_start":    rnd.start_sec,
                "t_end":      rnd.end_sec,
                "scene":      "full",
                "neutral":    "",   # 失败 → 空稿，phase3b 走占位分支；绝不落预呓文垃圾
                "hype":       peak_hype,
                "char_budget": char_budget,
            }]

        return {
            "round_no":        rnd.round_no,
            "start_sec":       rnd.start_sec,
            "end_sec":         rnd.end_sec,
            "demo_round_hint": rnd.demo_round_hint,
            "round_emotion":   round_emotion,
            "peak_hype":       peak_hype,
            "avg_hype":        avg_hype,
            "analyst_failed":  analyst_failed,
            "scenes":          scenes_out,
        }

    result_rounds = []
    with ThreadPoolExecutor(max_workers=concurrent_rounds) as pool:
        futures = {pool.submit(_process_round, rnd): rnd.round_no for rnd in match.rounds}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Phase3a analyst", unit="round"):
            result_rounds.append(fut.result())
    result_rounds.sort(key=lambda r: r["round_no"])

    if result_rounds and not dry_run:
        total = len(result_rounds)
        failed = sum(1 for r in result_rounds if r.get("analyst_failed"))
        print(f"[phase3a] analyst success {total - failed}/{total} rounds", file=sys.stderr)
        if failed == total or failed / total > 0.5:
            print(
                f"[phase3a] FATAL: {failed}/{total} rounds failed (>{50 if failed < total else 0}% threshold). "
                "Check LLM endpoint / API key. Exiting non-zero to prevent empty-shell output.",
                file=sys.stderr,
            )
            sys.exit(1)

    manifest = {
        "video_path": match.video_path,
        "map_name":   match.map_name,
        "model":      llm_cfg.get("model", ""),
        "rounds":     result_rounds,
    }
    write_json(output_path, manifest)
    return manifest
