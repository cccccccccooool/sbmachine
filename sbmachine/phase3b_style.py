"""Phase 3b — 风格模型：中性稿 + hype hint → 6657 口播 + [情绪] 标签。"""
from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from core.prompt_loader import load_prompt
from audio_service.emotion import parse_emotional_text
from sbmachine.common import load_config, load_hype_rules, resolve_backend, write_json
from sbmachine.neutral_contract import validate_neutral_manifest
from sbmachine.emotion_policy import EmotionPolicy, normalize_commentary_emotion, weighted_intensity
from sbmachine.phase3b_prompt import _CONTAMINATION_MARKERS, _LEAK_MARKERS, _aliases_hint, _extract_json_obj, _extract_tail, _load_persona, _load_player_aliases, _strip_tags
from sbmachine.llm_shim import accept_api_response
from sbmachine.schemas import EmotionSegment, SemanticData, load_match, save_match

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _ValidatedStyleCommentary(str):
    """校验通过的口播文本，附带原始响应，供最终 accept 反馈给后端。"""

    def __new__(cls, value: str, raw_response: object) -> "_ValidatedStyleCommentary":
        result = super().__new__(cls, value)
        result.raw_response = raw_response
        return result

    def accept(self, output: str) -> None:
        accept_api_response(self.raw_response, output=output)


# 无状态生成：不累积任何对话历史。
def _call_style(system: str, user_prompt: str, llm_cfg: dict, gen_fn, round_no: int = 0, debug: bool = False,
                max_tokens: int | None = None, log_ctx: dict | None = None) -> tuple[str, float]:
    """返回 (口播文本, felt_intensity)；走 /api/generate，无状态。"""
    try:
        raw = gen_fn(user_prompt, llm_cfg, system, max_tokens=max_tokens, log_ctx=log_ctx)
    except Exception as exc:
        return f"[style error: {exc}]", 0.0

    data = _extract_json_obj(raw)
    if data is None or set(data) != {"commentary", "felt_intensity"}:
        commentary, felt = "[style error: unparseable]", 0.0
    else:
        raw_commentary = data.get("commentary")
        commentary = raw_commentary if isinstance(raw_commentary, str) else ""
        if (not isinstance(raw_commentary, str)
                or not commentary.strip()
                or any(marker in commentary for marker in _LEAK_MARKERS)
                or commentary.lstrip().startswith("{")):
            error = "contract-leak" if isinstance(raw_commentary, str) else "unparseable"
            commentary, felt = f"[style error: {error}]", 0.0
        else:
            try:
                raw_felt = data["felt_intensity"]
                if isinstance(raw_felt, bool):
                    raise ValueError("boolean intensity")
                felt = float(raw_felt)
                if not math.isfinite(felt) or not 0.0 <= felt <= 1.0:
                    raise ValueError("intensity outside [0, 1]")
                commentary = _ValidatedStyleCommentary(commentary, raw)
            except (TypeError, ValueError):
                commentary, felt = "[style error: unparseable]", 0.0

    # debug 模式落盘：记录本次无状态请求与响应原文。
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

    config = load_config(config_path)
    debug_enabled = bool(config.get("debug", {}).get("phase3", False) or os.getenv("AI6657_DEBUG_PHASE3"))
    llm_cfg = dict(config.get("llm", {}))
    backend = resolve_backend(config, "style")
    if backend not in {"api", "vllm"}:
        raise ValueError(f"unsupported style backend: {backend}; use vllm or api")
    from sbmachine import llmb_api as _llmb_backend
    gen_fn = _llmb_backend.generate
    style_model = config.get("semantic", {}).get("style_model") or config.get("semantic", {}).get("model", "")
    if style_model:
        llm_cfg["model"] = style_model

    aliases = _load_player_aliases()
    persona = _load_persona()

    neutral_data = validate_neutral_manifest(
        json.loads(neutral_path.read_text(encoding="utf-8")), rounds_path,
    )
    neutral_by_round: dict[int, dict] = {
        int(r["round_no"]): r
        for r in neutral_data.get("rounds", [])
    }

    match = load_match(rounds_path)
    profile = str(config.get("profile", "style"))

    manifest_rounds = []
    errors: list[dict] = []
    accepted_run_samples: list[tuple[_ValidatedStyleCommentary, str]] = []

    cs_rules_path = _PROJECT_ROOT / "Prompt" / "cs_rules.txt"
    cs_rules = cs_rules_path.read_text(encoding="utf-8").strip() if cs_rules_path.exists() else ""
    system_content = "\n\n".join(filter(None, [
        load_prompt("style_system").replace("{persona_hint}", persona),
        cs_rules,
        _llmb_backend.load_style_skill(config),
    ]))

    emotion_policy = EmotionPolicy.from_rules(load_hype_rules())

    for rnd in tqdm(match.rounds, desc="Phase3b style", unit="round"):
        round_data = neutral_by_round.get(rnd.round_no, {})
        scenes = round_data.get("scenes", [])
        avg_hype = float(round_data.get("avg_hype", 0.0))
        analyst_failed = bool(round_data.get("analyst_failed", False))
        round_final_intensity = avg_hype
        round_status = "ok"
        last_tail = ""
        accepted_samples: list[tuple[_ValidatedStyleCommentary, str]] = []

        if dry_run:
            commentary, felt_intensity = "", 0.0
            scenes_manifest = []
            round_status = "dry_run"
        elif analyst_failed or not scenes:
            reason = "analyst_failed" if analyst_failed else "no_scenes"
            print(f"[phase3b] round {rnd.round_no} skipped: {reason}", file=sys.stderr)
            commentary, felt_intensity = "", 0.0
            scenes_manifest = []
            round_status = reason
        else:

            scene_commentaries: list[str] = []
            scene_commentaries_meta: list[dict] = []
            felt_samples: list[tuple[float, float]] = []
            final_intensity_samples: list[tuple[float, float]] = []

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
                alias_hint = _aliases_hint(scene_neutral, aliases)

                # 任何后端都不向风格模型暴露绝对时间，只给时长和字数预算。
                scene_info = (
                    f"（约{duration:.0f}秒，字数预算约{char_budget}字）"
                    + (f"\n阶段：{scene_name}" if scene_name and scene_name != "full" else "")
                )

                # 只保留上一句的句尾，用于衔接语气，不累积完整历史。
                user_prompt = "\n\n".join(filter(None, [
                    f"【上一句解说】\n{last_tail}" if last_tail else "",
                    f"【选手绰号】\n{alias_hint}" if alias_hint else "",
                    f"【场景信息】\n{scene_info}",
                    f"【中性稿】\n{scene_neutral}",
                ]))

                # 按字数预算推输出上限：封死复读/思考跑满 num_ctx 撞超时。
                # 输出含 JSON 包壳 + commentary 正文；中文 ~1.6 tok/字，留壳+余量。
                scene_max_tokens = max(160, min(768, int(char_budget * 2.2) + 80))

                log_ctx = {
                    "run_id": neutral_data["run_id"],
                    "round": f"round{rnd.round_no}",
                    "scene": scene_name,
                }
                scene_commentary, scene_felt = _call_style(
                    system_content,
                    user_prompt,
                    llm_cfg,
                    gen_fn,
                    round_no=rnd.round_no,
                    debug=debug_enabled,
                    max_tokens=scene_max_tokens,
                    log_ctx=log_ctx,
                )

                if not scene_commentary.startswith("[style error:"):
                    prev = scene_commentaries[-1] if scene_commentaries else ""
                    if scene_commentary and _strip_tags(scene_commentary) == _strip_tags(prev):
                        # 抑制与上一句完全相同的相邻复读。
                        print(f"[phase3b] round {rnd.round_no} scene '{scene_name}' adjacent-repeat, skipping", file=sys.stderr)
                    else:
                        scream_eligible = bool(scene.get("scream_eligible", False))
                        decision = emotion_policy.decide(
                            hard_intensity=scene_hype,
                            llmb_intensity=scene_felt,
                            scream_eligible=scream_eligible,
                        )
                        validated_commentary = scene_commentary
                        scene_commentary = normalize_commentary_emotion(scene_commentary, decision.label)
                        if (
                            not _strip_tags(scene_commentary)
                            or any(marker in scene_commentary for marker in _CONTAMINATION_MARKERS)
                        ):
                            print(f"[phase3b] round {rnd.round_no} scene '{scene_name}' contaminated, skipping", file=sys.stderr)
                            errors.append({
                                "round": f"round{rnd.round_no}",
                                "round_no": rnd.round_no,
                                "scene": scene_name,
                                "error": "[style error: contaminated]",
                                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                            })
                            last_tail = ""
                            continue
                        if isinstance(validated_commentary, _ValidatedStyleCommentary):
                            accepted_samples.append((validated_commentary, json.dumps({
                                "commentary": scene_commentary,
                                "felt_intensity": scene_felt,
                            }, ensure_ascii=False, separators=(",", ":"))))
                        scene_commentaries.append(scene_commentary)
                        felt_samples.append((scene_felt, duration))
                        final_intensity_samples.append((decision.score, duration))
                        last_tail = _extract_tail(scene_commentary)
                        # 记录 scene 级输出（用于 manifest scenes 字段）
                        scene_commentaries_meta.append({
                            "t_start": t_start,
                            "t_end": t_end,
                            "emotion": decision.label,
                            "emotion_score": decision.score,
                            "hard_intensity": round(scene_hype, 3),
                            "llmb_intensity": round(scene_felt, 3),
                            "scream_eligible": scream_eligible,
                            "text": _strip_tags(scene_commentary),
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

            if scene_commentaries:
                commentary = "".join(scene_commentaries)
                round_status = "ok"
            else:
                commentary = ""
                has_neutral_input = any(str(scene.get("neutral") or "").strip() for scene in scenes)
                round_status = "style_failed" if has_neutral_input else "silent"
            felt_intensity = weighted_intensity(felt_samples)
            round_final_intensity = weighted_intensity(final_intensity_samples)
            scenes_manifest = scene_commentaries_meta


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
            "status":          round_status,
            "hype_avg":        round(avg_hype, 3),
            "felt_intensity":  round(felt_intensity, 3),
            "final_intensity": round(round_final_intensity, 3),
            "emotion_segments": [seg.__dict__ for seg in rnd.phase3_semantic.emotion_segments],
            "scenes":          scenes_manifest,
        })
        if round_status == "ok":
            accepted_run_samples.extend(accepted_samples)

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

    # commentary 只承载解说产物；phase2 视觉时间线（YOLO 框、DEM 投影）的权威副本在
    # rounds_with_yolo.json，此处不透传，防止产物膨胀。phase4 与发布契约均不读 _phase2_yolo。
    for rnd in match.rounds:
        rnd.phase2_yolo = None
    save_match(output_rounds_path, match)
    manifest = {
        "video_path":    match.video_path,
        "map_name":      match.map_name,
        "model_profile": profile,
        "model_name":    str(llm_cfg.get("model", "")),
        "rounds":        manifest_rounds,
    }
    write_json(commentary_path, manifest)
    for response, output in accepted_run_samples:
        response.accept(output)
    return manifest
