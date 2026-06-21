"""三层回合对齐器（L0 score OCR → L1 duration DP → L2 onset 互相关）。负责将切分的视频片段对齐到 demo 的具体回合，并确定精准的对局 tick 偏移量。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlignResult:
    demo_round_hint: int | str   # 绝对回合号 or "unmatched"
    align_offset: float | None   # video_time * tick_rate + offset = tick
    align_method: str            # "score_ocr" / "duration_dp" / "onset_xcorr" / "unmatched"
    confidence: float = 0.0


# ── L0: score OCR ──

def _round_no_from_score(ct: int | None, t: int | None) -> int | None:
    if ct is None or t is None:
        return None
    total = int(ct) + int(t)
    return total + 1   # 当前局号


def align_l0_score(segment: dict, score_ocr_frames: list[dict]) -> int | None:
    """L0:从 score OCR 帧列表里取众数得绝对回合号。"""
    candidates: dict[int, int] = {}
    for frame in score_ocr_frames:
        rn = _round_no_from_score(frame.get("ct"), frame.get("t"))
        if rn is not None and rn > 0:
            candidates[rn] = candidates.get(rn, 0) + 1
    if not candidates:
        return None
    return max(candidates, key=lambda k: candidates[k])


# ── L1: duration DP (Needleman-Wunsch) ──

def _nw_align(seg_durs: list[float], demo_durs: list[float], gap_penalty: float = 8.0) -> list[int | None]:
    """把 seg_durs(视频段时长,可能是 demo 子集)对齐到 demo_durs 子序列。

    返回 mapping: mapping[i] = j 表示第 i 个视频段对应 demo 第 j 局(0-indexed)。
    跳过的 demo 局(丢段)用 gap_penalty 罚分。
    """
    M, N = len(seg_durs), len(demo_durs)
    if M == 0 or N == 0:
        return [None] * M

    # dp[i][j] = 对齐 seg[0..i-1] 到 demo[0..j-1] 的最低罚分
    INF = float("inf")
    dp = [[INF] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0
    # 开头跳 demo 局免费(视频段是 demo 子集,可能从中间开始)
    for j in range(1, N + 1):
        dp[0][j] = 0.0

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            match_cost = abs(seg_durs[i - 1] - demo_durs[j - 1])
            dp[i][j] = min(
                dp[i - 1][j - 1] + match_cost,   # 匹配
                dp[i][j - 1] + gap_penalty,       # 跳过 demo 回合(丢段)
            )

    # 回溯
    mapping: list[int | None] = [None] * M
    i, j = M, N
    while i > 0:
        if j == 0:
            break
        match_cost = abs(seg_durs[i - 1] - demo_durs[j - 1])
        if dp[i][j] == dp[i - 1][j - 1] + match_cost:
            mapping[i - 1] = j - 1   # 从 0 开始索引的 demo 回合
            i -= 1
            j -= 1
        else:
            j -= 1   # 跳过 demo 回合
    return mapping



def align_l1_duration_with_tickrate(
    segments: list[dict],
    demo_rounds: list[dict],
    tick_rate: float,
    gap_penalty: float = 8.0,
) -> list[int | None]:
    seg_durs = [float(s.get("duration_sec", s.get("end_sec", 0) - s.get("start_sec", 0))) for s in segments]
    demo_durs = []
    for r in demo_rounds:
        freeze = float(r.get("freeze_end_tick", r.get("start_tick", 0)))
        end = float(r.get("end_tick", 0))
        demo_durs.append((end - freeze) / tick_rate)
    return _nw_align(seg_durs, demo_durs, gap_penalty)


# ── L2: onset 互相关(校验 / 精修) ──

def _sparse_pulse(times: list[float], duration: float, fps: float = 10.0) -> list[float]:
    """把稀疏时刻列表转成等长脉冲序列(用于互相关)。"""
    n = max(1, int(duration * fps) + 1)
    arr = [0.0] * n
    for t in times:
        idx = int(round(t * fps))
        if 0 <= idx < n:
            arr[idx] = 1.0
    return arr


def _xcorr_offset(a: list[float], b: list[float]) -> float:
    """返回使 sum(a[i] * b[i+lag]) 最大的 lag(单位:脉冲帧)。"""
    if not a or not b:
        return 0.0
    best_lag, best_val = 0, -1.0
    max_lag = len(b)
    for lag in range(-max_lag, max_lag + 1):
        val = 0.0
        for i in range(len(a)):
            j = i + lag
            if 0 <= j < len(b):
                val += a[i] * b[j]
        if val > best_val:
            best_val = val
            best_lag = lag
    return float(best_lag)


def align_l2_onset(
    video_onsets: list[float],          # 视频段内 onset 的绝对时刻(秒,相对视频起点)
    demo_event_ticks: list[int],        # 该 demo 回合爆炸/拆弹/安放 tick 列表
    candidate_offset: float,            # 候选 tick offset(来自 L0/L1)
    tick_rate: float,
    seg_start_sec: float,
    seg_end_sec: float,
    tolerance_sec: float = 0.5,
    fps: float = 10.0,
) -> tuple[float | None, float]:
    """L2:onset 互相关校验。

    返回 (精修 offset 或 None, 匹配得分 0~1)。
    offset=None 表示 L2 否决当前候选(匹配得分过低)。
    """
    duration = seg_end_sec - seg_start_sec
    if not video_onsets or not demo_event_ticks:
        return candidate_offset, 0.5   # 无数据,不否决

    # 把 demo tick → 视频时刻(用候选 offset)
    demo_video_times = [
        (float(tk) - candidate_offset) / tick_rate - seg_start_sec
        for tk in demo_event_ticks
    ]
    # 过滤出在视频段范围内的
    demo_in_range = [t for t in demo_video_times if -tolerance_sec <= t <= duration + tolerance_sec]
    if not demo_in_range:
        return candidate_offset, 0.3   # demo 事件不在范围,置低分但不否决

    a = _sparse_pulse([t for t in video_onsets if 0 <= t <= duration], duration, fps)
    b = _sparse_pulse(demo_in_range, duration, fps)

    lag = _xcorr_offset(a, b)
    refined_offset = candidate_offset - lag / fps * tick_rate
    total_b = sum(b)
    if total_b == 0:
        return refined_offset, 0.5

    # 得分:落在 onset 容差范围内的 demo 事件比例
    score = 0.0
    for dt in demo_in_range:
        for ot in video_onsets:
            if abs(ot - dt) <= tolerance_sec:
                score += 1.0
                break
    score /= len(demo_in_range)
    return refined_offset, score


# ── 主函数 ──

def align_segments(
    segments: list[dict],
    demo_rounds: list[dict],
    tick_rate: float,
    *,
    score_ocr_per_segment: list[list[dict]] | None = None,
    onset_per_segment: list[list[float]] | None = None,
    gap_penalty: float = 8.0,
    onset_tolerance_sec: float = 0.5,
    veto_threshold: float = 0.25,
) -> list[AlignResult]:
    """三层对齐入口。"""
    n = len(segments)
    results: list[AlignResult] = []

    # L1 全局 DP 先跑(快,给出子序列对齐基准)
    l1_mapping = align_l1_duration_with_tickrate(segments, demo_rounds, tick_rate, gap_penalty)

    for idx, seg in enumerate(segments):
        demo_round_idx: int | None = None
        method = "unmatched"
        offset: float | None = None
        confidence = 0.0

        # ── L0 score OCR ──
        if score_ocr_per_segment and idx < len(score_ocr_per_segment):
            rn = align_l0_score(seg, score_ocr_per_segment[idx])
            if rn is not None:
                # 找 demo_rounds 里对应 round_no
                for di, dr in enumerate(demo_rounds):
                    if int(dr.get("round_no", 0)) == rn:
                        demo_round_idx = di
                        break
                if demo_round_idx is not None:
                    method = "score_ocr"
                    confidence = 0.95

        # ── L1 duration DP ──
        if demo_round_idx is None and idx < len(l1_mapping) and l1_mapping[idx] is not None:
            demo_round_idx = l1_mapping[idx]
            method = "duration_dp"
            confidence = 0.7

        if demo_round_idx is None:
            results.append(AlignResult(
                demo_round_hint="unmatched",
                align_offset=None,
                align_method="unmatched",
                confidence=0.0,
            ))
            continue

        dr = demo_rounds[demo_round_idx]
        freeze_end = float(dr.get("freeze_end_tick", dr.get("start_tick", 0)))
        seg_start = float(seg.get("start_sec", 0))
        # 初始(朴素)偏移量:假设 seg_start 对应于 freeze_end
        naive_offset = freeze_end - seg_start * tick_rate
        offset = naive_offset

        # ── L2 onset 互相关 ──
        if onset_per_segment and idx < len(onset_per_segment):
            event_ticks: list[int] = []
            for key in ("bomb_exploded_tick", "bomb_defused_tick", "bomb_planted_tick"):
                v = dr.get(key)
                if v is not None:
                    event_ticks.append(int(v))

            if event_ticks:
                refined, score = align_l2_onset(
                    video_onsets=onset_per_segment[idx],
                    demo_event_ticks=event_ticks,
                    candidate_offset=naive_offset,
                    tick_rate=tick_rate,
                    seg_start_sec=seg_start,
                    seg_end_sec=float(seg.get("end_sec", seg_start)),
                    tolerance_sec=onset_tolerance_sec,
                )
                if score < veto_threshold:
                    # L2 否决:降级回 L1,记 warning
                    confidence = max(0.0, confidence - 0.3)
                    method += "+l2_veto"
                else:
                    offset = refined
                    confidence = min(1.0, confidence + score * 0.2)
                    method += "+onset_xcorr"

        rn_final = int(dr.get("round_no", demo_round_idx + 1))
        results.append(AlignResult(
            demo_round_hint=rn_final,
            align_offset=offset,
            align_method=method,
            confidence=confidence,
        ))
    return results


def apply_align_results(segments: list[dict], results: list[AlignResult]) -> list[dict]:
    """把对齐结果写回 segment 字典。"""
    for seg, res in zip(segments, results):
        seg["demo_round_hint"] = res.demo_round_hint
        seg["align_offset"]    = res.align_offset
        seg["align_method"]    = res.align_method
        seg["align_confidence"] = round(res.confidence, 3)
    return segments
