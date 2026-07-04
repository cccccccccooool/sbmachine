"""第二阶段（多模态视觉感知）。利用 YOLO 检测 HUD、进行 POV 以及比分 OCR 提取、结合 VLM 进行画面内容分析。"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from core.prompt_loader import load_prompt
from sbmachine.common import load_config, require_path, resolve_path
from sbmachine.demo_query import DemoQuery
from sbmachine.phase2_background import _demo_round_hint, build_background_info, parse_vlm_json
from sbmachine.phase2_debug import DebugWriter
from sbmachine.phase2_ocr import _detect_pov_ocr, _detect_score_ocr, _first_timer_region, _maskable_regions, _resolve_ocr_box, read_ocr_text
from sbmachine.phase2_timeline import build_timeline
from sbmachine.phase2_yolo_gate import YoloUiDetector
from sbmachine.schemas import KeyFrame, VisionData, load_match, save_match
from sbmachine.time_align import RoundTimeAlign
from vision_service.region_crops import mask_regions











def run_phase2(
    *,
    rounds_path: Path,
    output_path: Path,
    config_path: Path,
    video_path: Path | None = None,
    dry_run: bool = False,
    debug_dir: Path | None = None,
) -> None:
    config = load_config(config_path)
    match = load_match(rounds_path)
    actual_video = video_path or resolve_path(match.video_path)
    if actual_video is None and not dry_run:
        raise ValueError("video path is required")

    vision_config = config.get("vision", {})
    yolo_config = vision_config.get("yolo", {})
    vlm_config = vision_config.get("vlm", {})
    demo_config = config.get("demo", {})
    pov_ocr_config = vision_config.get("pov_ocr", {})
    timer_ocr_config = vision_config.get("timer_ocr", {})
    score_ocr_config = vision_config.get("score_ocr", {})
    sampling_config = vision_config.get("sampling", {})
    demo_interval_sec = float(sampling_config.get("demo_interval_sec", vision_config.get("sample_interval_sec", 1.0)))
    vlm_interval_sec = float(sampling_config.get("vlm_interval_sec", 3.0))
    dense_pre_sec = float(sampling_config.get("dense_pre_sec", 3.0))
    dense_post_sec = float(sampling_config.get("dense_post_sec", 1.5))
    dense_fps = float(sampling_config.get("dense_fps", 2))
    yolo_enabled = bool(yolo_config.get("enabled", True))
    crop_padding = int(vision_config.get("crop_padding_px", 4))
    global_mask_padding = int(vision_config.get("global_mask_padding_px", 8))
    plant_empty_timer_frames = int(demo_config.get("plant_empty_timer_frames", 3))
    pov_match_min_score = float(pov_ocr_config.get("min_match_score", 0.6))
    spectator_min_frames = int(pov_ocr_config.get("spectator_min_frames", 2))

    dbg = DebugWriter(debug_dir)
    dbg.open()

    parsed_demo_dir = resolve_path(demo_config.get("parsed_dir", "output/demo"))
    demo = None if dry_run else DemoQuery.load(parsed_demo_dir or Path("output/demo"))

    if dry_run:
        yolo = None
        vlm = None
    else:
        yolo = YoloUiDetector(yolo_config) if yolo_enabled else None
        if str(vlm_config.get("backend", "local")) == "api":
            from sbmachine.vlm_api import VlmApiClient as _Vlm
        else:
            from sbmachine.vlm_local import VlmClient as _Vlm
        vlm = _Vlm.global_scene(vlm_config)

    for round_record in tqdm(match.rounds, desc="Phase2", unit="round"):
        tqdm.write(f"[Round {round_record.round_no}] {round_record.start_sec:.1f}s - {round_record.end_sec:.1f}s")
        key_frames: list[KeyFrame] = []
        background = []
        total_yolo_frames = 0
        total_vlm_calls = 0
        consecutive_unmatched = 0
        demo_round_no = _demo_round_hint(round_record)

        if dry_run:
            round_record.phase2_vision = VisionData(
                background=[],
                key_frames=[
                    KeyFrame(
                        time_sec=round(round_record.start_sec, 3),
                        gate_reason="dry_run",
                        vlm_hint="dry run",
                        vlm_response="dry run: Phase 2 did not call YOLO/OCR/VLM/demo query",
                        yolo_tags=["dry_run"],
                        has_vlm=False,
                    )
                ],
                yolo_required=yolo_enabled,
                yolo_model=str(yolo_config.get("model_path", "")),
                detector_mode="demo_driven_who_what_when_where",
                sample_interval_sec=demo_interval_sec,
                total_yolo_frames=0,
                total_vlm_calls=0,
            )
            continue

        if actual_video is None or demo is None or vlm is None:
            raise ValueError("video path and parsed demo are required")
        round_meta = demo.round_by_no(demo_round_no)
        align = RoundTimeAlign(
            round_meta,
            demo.tick_rate,
            anchor_tolerance_sec=float(demo_config.get("anchor_tolerance_sec", 2.0)),
        )
        prev_tick: int | None = None
        empty_timer_count = 0

        # ── 统一时间轴:1s 背景行 + 3s/事件 视觉行 ──
        import cv2 as _cv2
        timeline = build_timeline(
            start_sec=round_record.start_sec,
            end_sec=round_record.end_sec,
            demo=demo,
            round_meta=round_meta,
            align=align,
            demo_interval_sec=demo_interval_sec,
            vlm_interval_sec=vlm_interval_sec,
            dense_pre_sec=dense_pre_sec,
            dense_post_sec=dense_post_sec,
            dense_fps=dense_fps,
            round_no=demo_round_no,
        )

        batch_size = max(1, int(vlm_config.get("batch_size", 1)))
        vlm_prompt = load_prompt("vlm_scene")

        # ══════════════════════════════════════════════════════════════
        # PASS 1:解码 + YOLO + OCR + mask;不调 VLM,只收帧
        # ══════════════════════════════════════════════════════════════
        pending_vlm: list[dict] = []   # 每个 VLM 帧的收集数据

        cap = _cv2.VideoCapture(str(actual_video))
        try:
            for ts, is_vlm in tqdm(
                timeline,
                desc=f"  R{round_record.round_no:02d} 收帧",
                unit="f",
                leave=False,
            ):
                if not is_vlm:
                    continue   # 背景行在 Pass2 统一处理

                cap.set(_cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ok, frame = cap.read()
                if not ok:
                    pending_vlm.append({"ts": ts, "decode_failed": True})
                    continue

                if yolo is not None:
                    total_yolo_frames += 1
                    decision = yolo.decide(frame)
                    if decision.background:
                        yolo_bg = dict(decision.background)
                        yolo_bg["time_sec"] = round(ts, 3)
                        background.append(yolo_bg)
                        regions = list(decision.background.get("regions", []) or [])
                        timer_yolo_region = _first_timer_region(decision.background)
                        _yolo_bg = decision.background
                    else:
                        regions = []
                        timer_yolo_region = None
                        _yolo_bg = None
                    gate_reason = decision.reason
                    vlm_hint_str = decision.hint
                    yolo_tags = decision.tags
                    yolo_confidence = decision.confidence
                else:
                    gate_reason = "sampled_keyframe"
                    vlm_hint_str = "No YOLO. Global VLM sees full frame; POV/score OCR use fixed ROI only."
                    yolo_tags = []
                    yolo_confidence = 0.0
                    regions = []
                    timer_yolo_region = None
                    _yolo_bg = None

                timer_region, timer_crop_source = _resolve_ocr_box(
                    timer_yolo_region, timer_ocr_config, frame.shape, "yolo_timer_region"
                )
                timer_ocr = (
                    read_ocr_text(frame, timer_region, padding=crop_padding)
                    if timer_region
                    else {"raw_text": "", "engine": f"no_region:{timer_crop_source}", "region": None}
                )
                pov_ocr, pov_crop_source, pov_region = _detect_pov_ocr(
                    frame, _yolo_bg, pov_ocr_config, crop_padding
                )
                score_ocr = _detect_score_ocr(frame, _yolo_bg, score_ocr_config, crop_padding)
                timer_match = re.search(r"(\d{1,2})\s*[::]\s*(\d{2})", str(timer_ocr.get("raw_text", "")))
                timer_ocr["value"] = f"{int(timer_match.group(1))}:{timer_match.group(2)}" if timer_match else ""

                masked_frame = mask_regions(frame, _maskable_regions(regions), padding=global_mask_padding)

                # Debug crops(Pass1 保存,不需要 VLM 结果)
                if dbg.enabled:
                    fdir = dbg.frame_dir(demo_round_no)
                    stem = f"frame_{ts:.3f}"
                    dbg.save_crop(fdir, f"{stem}_pov_crop.png", dbg.crop_image(frame, pov_region, crop_padding))
                    dbg.save_crop(fdir, f"{stem}_timer_crop.png", dbg.crop_image(frame, timer_region, crop_padding))
                    dbg.save_crop(fdir, f"{stem}_masked.png", masked_frame)

                pending_vlm.append({
                    "ts": ts,
                    "decode_failed": False,
                    "masked_frame": masked_frame,
                    "gate_reason": gate_reason,
                    "vlm_hint": vlm_hint_str,
                    "yolo_tags": yolo_tags,
                    "yolo_confidence": yolo_confidence,
                    "regions": regions,
                    "pov_ocr": pov_ocr,
                    "timer_ocr": timer_ocr,
                    "score_ocr": score_ocr,
                    "pov_crop_source": pov_crop_source,
                    "timer_crop_source": timer_crop_source,
                    "pov_region": pov_region,
                    "timer_region": timer_region,
                })
        finally:
            cap.release()

        # ══════════════════════════════════════════════════════════════
        # BATCH VLM:按 batch_size 分批推理,带逐帧进度条
        # ══════════════════════════════════════════════════════════════
        valid_vlm = [item for item in pending_vlm if not item.get("decode_failed")]
        vlm_responses: dict[float, str] = {}

        with tqdm(
            total=len(valid_vlm),
            desc=f"  R{round_record.round_no:02d} VLM ×{batch_size}",
            unit="img",
            leave=False,
        ) as vlm_bar:
            for i in range(0, len(valid_vlm), batch_size):
                chunk = valid_vlm[i : i + batch_size]
                frames_batch = [item["masked_frame"] for item in chunk]

                if len(chunk) == 1:
                    responses = [vlm.describe(frames_batch[0], vlm_prompt)]
                else:
                    responses = vlm.describe_batch(frames_batch, [vlm_prompt] * len(chunk))

                for item, resp in zip(chunk, responses):
                    vlm_responses[item["ts"]] = resp

                total_vlm_calls += len(chunk)
                vlm_bar.update(len(chunk))

        # ══════════════════════════════════════════════════════════════
        # PASS 2:按时间轴顺序组装 key_frames(维护 prev_tick / consecutive_unmatched)
        # ══════════════════════════════════════════════════════════════
        pending_lookup: dict[float, dict] = {item["ts"]: item for item in pending_vlm}
        prev_tick: int | None = None
        consecutive_unmatched = 0
        empty_timer_count = 0

        def _emit_background_row(ts: float, hint: str) -> None:
            """把一帧降级为背景行:保留 demo 事实,desc 留白,避免时间轴留洞。

            用于:纯背景行、视频解码失败、VLM 退化输出。三处共用同一降级路径,
            保证任何一秒都有一行 demo 背景,不再整帧丢弃。
            """
            nonlocal prev_tick
            bg_info, tk = build_background_info(
                demo=demo,
                round_meta=round_meta,
                align=align,
                video_time=ts,
                desc="",
                vlm_response="",
                pov_ocr_result={"raw_text": "", "engine": "demo_only"},
                timer_ocr_result={"value": "", "raw_text": ""},
                score_ocr_result=None,
                prev_tick=prev_tick,
                pov_crop_source="demo_only",
                consecutive_unmatched=consecutive_unmatched,
                spectator_min_frames=spectator_min_frames,
                pov_match_min_score=pov_match_min_score,
                timer_crop_source="",
            )
            prev_tick = tk
            key_frames.append(KeyFrame(
                time_sec=round(ts, 3),
                gate_reason="demo_only",
                vlm_hint=hint,
                vlm_response="",
                has_vlm=False,
                background_info=bg_info,
            ))

        for ts, is_vlm in timeline:
            # ── 背景行:只做 demo 查询 ──
            if not is_vlm:
                _emit_background_row(ts, "background row: demo facts only, no frame decoded")
                continue

            item = pending_lookup.get(ts)
            if item is None or item.get("decode_failed"):
                _emit_background_row(ts, "decode failed: kept as background row")
                continue

            # ── 计时器补丁:bomb plant / 爆炸 / 拆弹 时间冻结 ──
            timer_ocr = item["timer_ocr"]
            if timer_ocr["value"]:
                empty_timer_count = 0
            else:
                empty_timer_count += 1
                if (
                    empty_timer_count >= plant_empty_timer_frames
                    and round_meta.get("bomb_planted_tick") is not None
                    and not align.is_frozen
                    and align.offsets
                ):
                    plant_tick = int(round_meta.get("bomb_planted_tick"))
                    plant_video_time = align.to_video_time(plant_tick)
                    if abs(plant_video_time - float(ts)) <= float(demo_config.get("anchor_tolerance_sec", 2.0)):
                        align.freeze(ts, event_tick=plant_tick)
                    else:
                        align.warnings.append(
                            f"plant freeze skipped: video={ts:.3f} demo={plant_video_time:.3f}"
                        )
                for evt_key in ("bomb_exploded_tick", "bomb_defused_tick"):
                    evt_tick = round_meta.get(evt_key)
                    if evt_tick is not None and not align.is_frozen and align.offsets:
                        evt_video_time = align.to_video_time(int(evt_tick))
                        if abs(evt_video_time - float(ts)) <= float(demo_config.get("anchor_tolerance_sec", 2.0)):
                            align.freeze(ts, event_tick=int(evt_tick))
                            break

            response = vlm_responses.get(ts, "")
            if not response.strip().strip("!").strip():
                tqdm.write(f"  [WARN] VLM degenerate output at {ts:.1f}s (all '!'), kept as background row")
                _emit_background_row(ts, "VLM degenerate output: kept as background row")
                continue

            vlm_obj = parse_vlm_json(response) or {"desc": ""}
            bg_info, tick = build_background_info(
                demo=demo,
                round_meta=round_meta,
                align=align,
                video_time=ts,
                desc=str(vlm_obj.get("desc", "")),
                vlm_response=response,
                pov_ocr_result=item["pov_ocr"],
                timer_ocr_result=timer_ocr,
                score_ocr_result=item["score_ocr"],
                prev_tick=prev_tick,
                pov_crop_source=item["pov_crop_source"],
                consecutive_unmatched=consecutive_unmatched,
                spectator_min_frames=spectator_min_frames,
                pov_match_min_score=pov_match_min_score,
                timer_crop_source=item["timer_crop_source"],
            )
            prev_tick = tick

            if bg_info["who"]["pov_source"] in ("unmatched", "spectator"):
                consecutive_unmatched += 1
            else:
                consecutive_unmatched = 0

            if dbg.enabled:
                fdir = dbg.frame_dir(demo_round_no)
                dbg.write_frame({
                    "round_no": demo_round_no,
                    "video_time": round(float(ts), 3),
                    "has_vlm": True,
                    "pov_crop_source": item["pov_crop_source"],
                    "pov_crop_box": item["pov_region"].get("box") if item["pov_region"] else None,
                    "pov_ocr_raw": item["pov_ocr"].get("raw_text", ""),
                    "pov_ocr_engine": item["pov_ocr"].get("engine", ""),
                    "timer_crop_source": item["timer_crop_source"],
                    "timer_crop_box": item["timer_region"].get("box") if item["timer_region"] else None,
                    "timer_ocr_raw": timer_ocr.get("raw_text", ""),
                    "timer_ocr_value": timer_ocr.get("value", ""),
                    "yolo_tags": item["yolo_tags"],
                    "yolo_regions": [
                        {"label": r.get("label"), "conf": round(float(r.get("confidence", 0)), 3), "box": r.get("box")}
                        for r in item["regions"]
                    ],
                    "who": bg_info.get("who", {}),
                    "when": bg_info.get("when", {}),
                    "where_pov_callout": bg_info.get("where", {}).get("pov_callout", ""),
                    "players": [
                        {
                            "name": p.get("name"), "side": p.get("side"),
                            "hp": p.get("hp"), "armor": p.get("armor"),
                            "helmet": p.get("helmet"), "weapon": p.get("weapon"),
                            "callout": p.get("callout"),
                            "money": p.get("money"),
                        }
                        for p in bg_info.get("where", {}).get("players", [])
                    ],
                    "kills_this_frame": bg_info.get("events", {}).get("kills", []),
                    "c4": bg_info.get("events", {}).get("c4", {}),
                    "vlm_desc": bg_info.get("what", {}).get("desc", ""),
                    "vlm_raw": response,
                    "align_offsets_count": len(align.offsets),
                    "align_frozen": align.is_frozen,
                    "align_warnings": list(align.warnings),
                    "images": {
                        "pov_crop": str(fdir.relative_to(debug_dir) / f"frame_{ts:.3f}_pov_crop.png"),
                        "timer_crop": str(fdir.relative_to(debug_dir) / f"frame_{ts:.3f}_timer_crop.png"),
                        "masked": str(fdir.relative_to(debug_dir) / f"frame_{ts:.3f}_masked.png"),
                    },
                })

            key_frames.append(KeyFrame(
                time_sec=round(ts, 3),
                gate_reason=item["gate_reason"],
                vlm_hint=item["vlm_hint"],
                vlm_response=response,
                yolo_tags=item["yolo_tags"],
                yolo_confidence=item["yolo_confidence"],
                global_vlm_output=response,
                ui_regions=item["regions"],
                background_info=bg_info,
                has_vlm=True,
            ))

        round_record.phase2_vision = VisionData(
            background=background,
            key_frames=key_frames,
            yolo_required=yolo_enabled,
            yolo_model=str(yolo_config.get("model_path", "")),
            detector_mode="demo_driven_who_what_when_where",
            sample_interval_sec=demo_interval_sec,
            total_yolo_frames=total_yolo_frames,
            total_vlm_calls=total_vlm_calls,
        )

    dbg.close()
    save_match(output_path, match)

    # --- 提取给 Phase 3 (LLM) 专用的纯净版 JSON ---
    semantic_output_path = output_path.with_name(output_path.stem + "_semantic.json")
    semantic_match = []
    for r in match.rounds:
        if not getattr(r, "phase2_vision", None) or not r.phase2_vision.key_frames:
            continue
        round_data = {
            "round_no": getattr(r, "round_no", 0),
            "frames": []
        }
        for kf in r.phase2_vision.key_frames:
            bg = getattr(kf, "background_info", None)
            if bg:
                round_data["frames"].append(bg)
        if round_data["frames"]:
            semantic_match.append(round_data)

    with open(semantic_output_path, "w", encoding="utf-8") as f:
        json.dump(semantic_match, f, ensure_ascii=False, indent=2)
