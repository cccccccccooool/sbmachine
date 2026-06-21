"""第二阶段（多模态视觉感知）。利用 YOLO 检测 HUD、进行 POV 以及比分 OCR 提取、结合 VLM 进行画面内容分析。"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import json
import re
import sys
from pathlib import Path
from typing import Iterator

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tqdm import tqdm

from core.prompt_loader import load_prompt
from sbmachine.common import load_config, require_path, resolve_path
from sbmachine.demo_query import DemoQuery
from sbmachine.phase2_vlm_client import VlmClient
from sbmachine.phase2_yolo_gate import YoloUiDetector
from sbmachine.schemas import KeyFrame, VisionData, load_match, save_match
from sbmachine.time_align import RoundTimeAlign, parse_timer_seconds
from vision_service.region_crops import box_from_norm, crop_frame, mask_regions


class DebugWriter:
    def __init__(self, debug_dir: Path | None) -> None:
        self.enabled = debug_dir is not None
        self.debug_dir = debug_dir
        self._jsonl_handle = None

    def open(self) -> None:
        if not self.enabled:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_handle = open(self.debug_dir / "frames.jsonl", "a", encoding="utf-8")

    def close(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None

    def save_crop(self, frame_dir: Path, stem: str, image) -> None:
        if not self.enabled or image is None:
            return
        try:
            import cv2
            frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_dir / stem), image)
        except Exception:
            pass

    def write_frame(self, record: dict) -> None:
        if not self.enabled or self._jsonl_handle is None:
            return
        try:
            self._jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_handle.flush()
        except Exception:
            pass

    def frame_dir(self, round_no: int) -> Path:
        return self.debug_dir / f"round_{round_no:02d}"

    def crop_image(self, frame, region: dict | None, padding: int = 0):
        """返回给定区域字典的裁剪图像,如果不可用则返回 None。"""
        if frame is None or region is None:
            return None
        try:
            return crop_frame(frame, region.get("box", []), padding=padding)
        except Exception:
            return None


def iter_round_frames(video_path: Path, start_sec: float, end_sec: float, interval_sec: float) -> Iterator[tuple[float, object]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        ts = start_sec
        while ts <= end_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            yield ts, frame
            ts += max(interval_sec, 0.05)
    finally:
        cap.release()


def build_timeline(
    start_sec: float,
    end_sec: float,
    demo: "DemoQuery",
    round_meta: dict,
    align: "RoundTimeAlign",
    *,
    demo_interval_sec: float = 1.0,
    vlm_interval_sec: float = 3.0,
    dense_pre_sec: float = 3.0,
    dense_post_sec: float = 1.5,
    dense_fps: float = 2.0,
    round_no: int = 0,
) -> list[tuple[float, bool]]:
    """统一时间轴构造。

    返回 [(video_time, is_vlm), ...] 严格有序。
    is_vlm=True → 视觉行(解码 + VLM);False → 背景行(仅 demo 查询,不解码)。
    """
    snap = 0.1   # 亚秒对齐网格

    def _snap(t: float) -> float:
        return round(round(t / snap) * snap, 6)

    # 1s 网格
    grid: set[float] = set()
    t = start_sec
    while t <= end_sec + 1e-6:
        grid.add(_snap(t))
        t += demo_interval_sec

    # 3s VLM 节奏(对齐到整秒)
    vlm_times: set[float] = set()
    t = start_sec
    while t <= end_sec + 1e-6:
        snapped = _snap(t)
        if snapped in grid:
            vlm_times.add(snapped)
        t += vlm_interval_sec

    # 事件窗加密:击杀 + 炸弹
    event_ticks: list[int] = []
    kills = demo.kills_in_round(round_no)
    for k in kills:
        tk = k.get("tick")
        if tk is not None:
            event_ticks.append(int(tk))
    for key in ("bomb_planted_tick", "bomb_exploded_tick", "bomb_defused_tick"):
        v = round_meta.get(key)
        if v is not None:
            event_ticks.append(int(v))

    dense_step = 1.0 / max(dense_fps, 0.1)
    for tk in event_ticks:
        center = align.to_video_time(tk)
        t = center - dense_pre_sec
        while t <= center + dense_post_sec + 1e-6:
            snapped = _snap(t)
            if start_sec - 0.05 <= snapped <= end_sec + 0.05:
                grid.add(snapped)
                vlm_times.add(snapped)
            t += dense_step

    timeline = sorted(grid)
    return [(t, t in vlm_times) for t in timeline]


def parse_vlm_json(text: str) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


_RAPID_OCR_ENGINE = None
_RAPID_OCR_IMPORT_FAILED = False


def _get_rapid_ocr():
    """返回单个共享的 RapidOCR 引擎,仅在首次调用时延迟构建。"""
    global _RAPID_OCR_ENGINE, _RAPID_OCR_IMPORT_FAILED
    if _RAPID_OCR_ENGINE is None and not _RAPID_OCR_IMPORT_FAILED:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _RAPID_OCR_ENGINE = RapidOCR()
        except Exception:
            _RAPID_OCR_IMPORT_FAILED = True
    return _RAPID_OCR_ENGINE


def read_ocr_text(frame, region: dict, *, padding: int = 0) -> dict:
    crop = crop_frame(frame, region.get("box", []), padding=padding)
    raw_text = ""
    ocr_engine = _get_rapid_ocr()
    if ocr_engine is None:
        return {"region": region, "raw_text": "", "engine": "unavailable:ImportError"}
    engine = "rapidocr_onnxruntime"
    try:
        result, _ = ocr_engine(crop)
        if result:
            raw_text = " ".join(str(item[1]) for item in result if len(item) >= 2)
    except Exception as exc:
        engine = f"unavailable:{type(exc).__name__}"
    return {"region": region, "raw_text": raw_text.strip(), "engine": engine}


def _demo_round_hint(round_record) -> int:
    hint = getattr(round_record, "demo_round_hint", None)
    if hint is not None:
        try:
            return int(hint)
        except (TypeError, ValueError):
            pass
    return int(round_record.round_no)


def _regions_by_type(background: dict, names: set[str]) -> list[dict]:
    out = []
    for region in background.get("regions", []) or []:
        label = str(region.get("label", "")).lower()
        rtype = str(region.get("type", "")).lower()
        if label in names or rtype in names:
            out.append(region)
    return out


def _first_pov_region(background: dict) -> dict | None:
    regions = _regions_by_type(background, {"pov_name", "pov_name_area", "pov_player_bar", "pov_marker_bar"})
    if regions:
        return max(regions, key=lambda item: float(item.get("confidence", 0.0)))
    return None


def _first_timer_region(background: dict) -> dict | None:
    regions = _regions_by_type(background, {"timer", "timer_area", "round_timer"})
    if regions:
        return max(regions, key=lambda item: float(item.get("confidence", 0.0)))
    return None


def _first_score_region(background: dict) -> dict | None:
    regions = _regions_by_type(background, {"score", "score_area", "top_hud_score", "top_hud"})
    if regions:
        return max(regions, key=lambda r: float(r.get("confidence", 0.0)))
    return None


def _resolve_ocr_box(
    yolo_region: dict | None,
    fixed_cfg: dict,
    frame_shape,
    yolo_source_name: str = "yolo",
) -> tuple[dict | None, str]:
    """返回 (region_dict, source_label)。

    优先级:YOLO 检测到的区域 → 配置文件中固定的归一化 ROI → (None, "no_region")。
    """
    if yolo_region is not None:
        return yolo_region, yolo_source_name
    if fixed_cfg.get("enabled", False) and fixed_cfg.get("box"):
        box_str = str(fixed_cfg.get("box", "")).strip()
        if box_str:
            try:
                parts = [float(x.strip()) for x in box_str.split(",")]
                if len(parts) == 4:
                    h, w = frame_shape[:2]
                    pixel_box = box_from_norm(parts, w, h)
                    return {"box": pixel_box, "label": "fixed_roi", "confidence": 1.0}, "fixed_roi"
            except (ValueError, IndexError):
                pass
    return None, "no_region"


def _detect_pov_ocr(
    frame,
    yolo_background: dict | None,
    pov_ocr_config: dict,
    crop_padding: int,
) -> tuple[dict, str, dict | None]:
    """POV 玩家名称 OCR。"""
    yolo_region = _first_pov_region(yolo_background or {})
    region, source = _resolve_ocr_box(yolo_region, pov_ocr_config, frame.shape, "yolo_pov_region")
    if region:
        return read_ocr_text(frame, region, padding=crop_padding), source, region
    return {"raw_text": "", "engine": f"no_region:{source}", "region": None}, source, None


def _detect_score_ocr(
    frame,
    yolo_background: dict | None,
    score_ocr_config: dict,
    crop_padding: int,
) -> dict:
    """来自顶部 HUD 的回合比分 OCR。

    首选路径:YOLO 检测 'score' / 'top_hud_score' 类别 → 裁剪图像 → OCR。
    备用路径:配置文件中固定的归一化 ROI(在 YOLO 重新训练前的过渡方案)。

    返回 {"ct": int|None, "t": int|None, "raw": str, "source": str}。
    """
    yolo_region = _first_score_region(yolo_background or {})
    region, source = _resolve_ocr_box(yolo_region, score_ocr_config, frame.shape, "yolo_score_region")
    if not region:
        return {"ct": None, "t": None, "raw": "", "source": source}
    ocr = read_ocr_text(frame, region, padding=crop_padding)
    raw = str(ocr.get("raw_text", ""))
    m = re.search(r"(\d+)\s*[:\-]\s*(\d+)", raw)
    if m:
        return {"ct": int(m.group(1)), "t": int(m.group(2)), "raw": raw, "source": source}
    return {"ct": None, "t": None, "raw": raw, "source": source}


def _maskable_regions(regions: list[dict]) -> list[dict]:
    skip = {"timer", "timer_area", "round_timer", "c4", "c4_area", "c4_status", "killfeed", "killfeed_area"}
    return [
        region
        for region in regions
        if str(region.get("label", "")).lower() not in skip
        and str(region.get("type", "")).lower() not in {"timer", "c4", "killfeed"}
    ]


def _player_state_with_callouts(demo: DemoQuery, tick: int) -> list[dict]:
    players = []
    for row in demo.state_at(tick):
        x = float(row.get("x") or 0.0)
        y = float(row.get("y") or 0.0)
        z = float(row.get("z") or 0.0)
        # Go parser 将区域名称写入 'callout' 列;旧产物无此列时降级调用 callout_of()。
        callout = str(row.get("callout") or "") or demo.callout_of(x, y, z)
        players.append(
            {
                "steamid": str(row.get("steamid", "")),
                "name": str(row.get("name", "")),
                "side": str(row.get("side", "")),
                "hp": row.get("hp"),
                "armor": row.get("armor"),
                "helmet": bool(row.get("has_helmet", False)),
                "weapon": str(row.get("active_weapon", "")),
                "callout": callout,
                "x": x,
                "y": y,
                "z": z,
                "money": row.get("money"),
            }
        )
    return players


def _pov_callout(players: list[dict], steamid: str, name: str) -> str:
    if steamid:
        for player in players:
            if str(player.get("steamid", "")) == str(steamid):
                return str(player.get("callout", ""))
    target = str(name).casefold()
    if target:
        for player in players:
            if str(player.get("name", "")).casefold() == target:
                return str(player.get("callout", ""))
    return ""


def build_background_info(
    *,
    demo: DemoQuery,
    round_meta: dict,
    align: RoundTimeAlign,
    video_time: float,
    desc: str,
    vlm_response: str,
    pov_ocr_result: dict,
    timer_ocr_result: dict,
    score_ocr_result: dict | None = None,
    prev_tick: int | None,
    pov_crop_source: str = "unknown",
    consecutive_unmatched: int = 0,
    spectator_min_frames: int = 2,
    pov_match_min_score: float = 0.6,
    timer_crop_source: str = "yolo_timer_region",
) -> tuple[dict, int]:
    timer = str(timer_ocr_result.get("value", "") or "").strip()
    if timer:
        align.add_anchor(video_time, timer)
    tick = align.to_tick(video_time)
    players = _player_state_with_callouts(demo, tick)
    match = demo.match_player(str(pov_ocr_result.get("raw_text", "")))

    # 应用最小匹配度分数守护:如果 OCR 结果与任何名单成员都不匹配,
    # 则视其为未匹配 / 可能是导播/观众视角画面。
    if match.score < pov_match_min_score:
        pov_name = ""
        steamid = ""
        # 在连续 N 帧未匹配后,判定为导播/观众镜头,而不是单纯的 OCR 识别失败。
        if consecutive_unmatched + 1 >= spectator_min_frames:
            pov_source = "spectator"
            view = "director"
        else:
            pov_source = "unmatched"
            view = "player"
    else:
        pov_name = match.name
        steamid = match.steamid
        pov_source = pov_crop_source
        view = "player"

    timer_seconds = parse_timer_seconds(timer)
    if timer_seconds is not None:
        relative_sec = 115.0 - timer_seconds
    else:
        relative_sec = align.relative_sec_for_tick(tick)

    # 标记这一帧属于回合的哪一部分。
    if relative_sec < 0:
        phase = "pre_round"
    elif relative_sec <= 115.0:
        phase = "in_round"
    else:
        phase = "post_round"

    prev = prev_tick if prev_tick is not None else tick
    kills = demo.kills_between(prev, tick)
    utilities = demo.utilities_between(prev, tick)
    damages = demo.damages_between(prev, tick) if hasattr(demo, "damages_between") else []
    flashes = demo.flashes_between(prev, tick) if hasattr(demo, "flashes_between") else []
    # active smokes / infernos at this tick (full-round scope)
    smokes_now = demo.smokes_active_at(tick) if hasattr(demo, "smokes_active_at") else []
    infernos_now = demo.infernos_active_at(tick) if hasattr(demo, "infernos_active_at") else []
    plant_tick = round_meta.get("bomb_planted_tick")
    effective_timer_source = timer_crop_source if timer else ""
    bg = {
        "when": {
            "video_time": round(float(video_time), 3),
            "timer": timer,
            "timer_source": effective_timer_source,
            "tick": tick,
            "round_no": int(round_meta.get("round_no", 0)),
            "relative_sec": round(float(relative_sec), 3),
            "phase": phase,
            "align_frozen": align.is_frozen,
            "align_warnings": list(align.warnings),
        },
        "who": {
            "pov_player": pov_name,
            "ocr_raw": str(pov_ocr_result.get("raw_text", "")),
            "match_score": match.score,
            "ocr_engine": str(pov_ocr_result.get("engine", "")),
            "pov_source": pov_source,
            "view": view,
        },
        "what": {
            "desc": str(desc or "").strip(),
            "vlm_raw": vlm_response,
        },
        "where": {
            # steamid 仅用于内部 POV 匹配(上方 _pov_callout / who 计算),不写入最终背景数据。
            "pov_callout": _pov_callout(players, steamid, pov_name),
            "players": [{k: v for k, v in p.items() if k != "steamid"} for p in players],
        },
        "events": {
            "kills":     kills,
            "utilities": utilities,
            "damages":   damages,
            "flashes":   flashes,
            "smokes_active":   smokes_now,
            "infernos_active": infernos_now,
            "c4": {
                "planted":   bool(align.is_frozen and plant_tick is not None),
                "plant_tick": plant_tick,
                "begin_defuse_tick": round_meta.get("bomb_begin_defuse_tick"),
                "defuser_has_kit":   round_meta.get("defuser_has_kit"),
            },
            "score_ocr": score_ocr_result or {},
        },
    }
    return bg, tick


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
        vlm = VlmClient.global_scene(vlm_config)

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

        # BATCH VLM:按 batch_size 分批推理,带逐帧进度条
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

        # PASS 2:按时间轴顺序组装 key_frames(维护 prev_tick / consecutive_unmatched)
        pending_lookup: dict[float, dict] = {item["ts"]: item for item in pending_vlm}
        prev_tick: int | None = None
        consecutive_unmatched = 0
        empty_timer_count = 0

        def _emit_background_row(ts: float, hint: str) -> None:
            """
            把一帧降级为背景行:保留 demo 事实,desc 留白,避免时间轴留洞。
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
