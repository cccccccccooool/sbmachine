"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：round_segmenter 模块的单元测试。

启动方式：pytest tests/test_round_segmenter.py（被 pytest 运行）。
输入数据流：测试内联的模拟帧分类数据。
输出数据流：pytest 断言结果（通过/失败）。
用法用途：验证 Score/VideoSegment 序列化、live gap 合并、帧类型切片逻辑的正确性。
"""
import pytest
from sbmachine.round_segmenter import (
    SegmentKind,
    Score,
    VideoSegment,
    SegmenterConfig,
    segment_observations,
    _merge_live_gaps
)

def test_score_creation():
    """测试 Score.from_values 正常/空值输入。被 pytest 调用。无参数。"""
    score = Score.from_values(10, "5")
    assert score.ct == 10
    assert score.t == 5

    score_empty = Score.from_values(None, "")
    assert score_empty.ct == 0
    assert score_empty.t == 0


def test_video_segment_serialization():
    """测试 VideoSegment.to_dict 序列化。被 pytest 调用。无参数。"""
    segment = VideoSegment(
        kind=SegmentKind.LIVE_ROUND,
        start_sec=10.0,
        end_sec=25.5,
        score=Score(5, 7),
        reason="test_segment"
    )
    data = segment.to_dict()
    assert data["kind"] == "live_round"
    assert data["start_sec"] == 10.0
    assert data["end_sec"] == 25.5
    assert data["score"] == {"ct": 5, "t": 7}
    assert data["reason"] == "test_segment"


def test_merge_live_gaps():
    """测试 _merge_live_gaps 间隔合并逻辑。被 pytest 调用。无参数。"""
    segments = [
        VideoSegment(SegmentKind.LIVE_ROUND, start_sec=10.0, end_sec=20.0, reason="part1"),
        VideoSegment(SegmentKind.LIVE_ROUND, start_sec=22.0, end_sec=30.0, reason="part2"),
        VideoSegment(SegmentKind.LIVE_ROUND, start_sec=40.0, end_sec=50.0, reason="part3"),
    ]
    # merge with gap limit 3.0 (so 22.0 - 20.0 = 2.0 <= 3.0 will merge; 40.0 - 30.0 = 10.0 > 3.0 will not merge)
    merged = _merge_live_gaps(segments, bridge_gap_sec=3.0)
    assert len(merged) == 2
    assert merged[0].start_sec == 10.0
    assert merged[0].end_sec == 30.0
    assert merged[0].reason == "part1; part2"
    assert merged[1].start_sec == 40.0
    assert merged[1].end_sec == 50.0


def test_segment_observations_from_frame_types():
    """测试从帧类型标签列表切片出 VideoSegment。被 pytest 调用。无参数。"""
    # Simulate a stream of frame_type label predictions
    observations = [
        {"time_sec": 1.0, "label": "break"},
        {"time_sec": 2.0, "label": "break"},
        # A live game segment starting at 3.0s and ending at 10.0s
        {"time_sec": 3.0, "label": "live_game"},
        {"time_sec": 4.0, "label": "live_game"},
        {"time_sec": 5.0, "label": "live_game"},
        {"time_sec": 6.0, "label": "live_game"},
        {"time_sec": 7.0, "label": "live_game"},
        {"time_sec": 8.0, "label": "live_game"},
        {"time_sec": 9.0, "label": "live_game"},
        {"time_sec": 10.0, "label": "live_game"},
        # Another break
        {"time_sec": 11.0, "label": "break"},
        # A very short live game segment (2s) which should be discarded under default min_live_segment_sec=6.0
        {"time_sec": 12.0, "label": "live_game"},
        {"time_sec": 13.0, "label": "live_game"},
        {"time_sec": 14.0, "label": "break"},
    ]

    config = SegmenterConfig(min_live_segment_sec=6.0, bridge_gap_sec=3.0, live_label="live_game")
    segments = segment_observations(observations, config)
    
    assert len(segments) == 1
    assert segments[0].kind == SegmentKind.LIVE_ROUND
    assert segments[0].start_sec == 3.0
    assert segments[0].end_sec == 10.0
