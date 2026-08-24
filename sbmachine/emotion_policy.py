"""Phase 3b 解说情绪三路融合：硬事实强度 + LLMB 强度，输出最终三档情绪标签。"""
from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass

FINAL_EMOTIONS = ("平述", "激动", "惊叹")
_ALL_EMOTION_TAGS = re.compile(r"\[(平述|平叙|激动|惊叹|紧张|惋惜)\]")
# 档序：平述(0) < 激动(1) < 惊叹(2)；非三档历史标签（平叙/紧张/惋惜）按平述档处理
_TAG_RANK = {"平述": 0, "平叙": 0, "紧张": 0, "惋惜": 0, "激动": 1, "惊叹": 2}
# capsule 情绪强度上限（§计划书阶段3门禁）：规则层 capsule 最多 0.45 档
CAPSULE_INTENSITY_CEILING = 0.45


def capsule_emotion(hard_intensity: float, *, ceiling: float = CAPSULE_INTENSITY_CEILING) -> tuple[str, float]:
    """capsule 的情绪裁决：由规则层给出，取最低硬事实档位（平述），不随 LLM-B 放大。

    返回 (标签, 强度)。强度 = min(ceiling, hard_intensity)：即使上游硬事实强度
    很高，capsule 也不超过 0.45 上限（§9.4 示例 0.45）。
    """
    score = round(min(_clamp01(ceiling), _clamp01(hard_intensity)), 3)
    return "平述", score


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def blend_intensity(hard_intensity: float, llmb_intensity: float, *, hard_weight: float = 0.7, llmb_weight: float = 0.3) -> float:
    """按权重融合硬事实与 LLMB 强度（默认硬事实占七成），结果夹到 [0, 1]。"""
    total_weight = hard_weight + llmb_weight
    if total_weight <= 0:
        return 0.0
    score = (_clamp01(hard_intensity) * hard_weight + _clamp01(llmb_intensity) * llmb_weight) / total_weight
    return round(_clamp01(score), 3)


def weighted_intensity(samples: list[tuple[float, float]]) -> float:
    """按权重（如场景时长）加权平均强度；总权重非正时回退为 0。"""
    valid_samples = [(_clamp01(value), max(0.0, float(weight))) for value, weight in samples]
    total_weight = sum(weight for _, weight in valid_samples)
    if total_weight <= 0:
        return 0.0
    return round(sum(value * weight for value, weight in valid_samples) / total_weight, 3)


def normalize_commentary_emotion(commentary: str, final_label: str) -> str:
    """把句内情绪标签钳制到最终档以内，保留 LLMB 写出的句内情绪起伏。

    规则：首段（首个标签或无标签的开头文本）对齐 final_label；其后每个句内标签
    取 min(该标签档, final_label 档)。输出保证以 [标签] 开头，空段会被清理。
    """
    if final_label not in FINAL_EMOTIONS:
        raise ValueError(f"unsupported final emotion: {final_label}")
    final_rank = FINAL_EMOTIONS.index(final_label)
    pieces = _ALL_EMOTION_TAGS.split(str(commentary))
    segments: list[tuple[int, str]] = []
    lead = pieces[0].strip()
    if lead:
        segments.append((final_rank, lead))
    for index in range(1, len(pieces), 2):
        text = pieces[index + 1].strip()
        if text:
            segments.append((min(_TAG_RANK[pieces[index]], final_rank), text))
    if not segments:
        return f"[{final_label}]"
    segments[0] = (final_rank, segments[0][1])
    return "".join(f"[{FINAL_EMOTIONS[rank]}]{text}" for rank, text in segments)


@dataclass(frozen=True)
class EmotionDecision:
    label: str
    score: float
    excited_threshold: float
    scream_threshold: float


class EmotionPolicy:
    """融合硬事实与 LLMB 强度定档；历史样本不足时安全回退到基础阈值。"""

    def __init__(self, *, hard_weight: float = 0.7, llmb_weight: float = 0.3,
                 excited_threshold: float = 0.35, scream_threshold: float = 0.72,
                 history_size: int = 200, min_history: int = 12,
                 target: dict[str, float] | None = None, max_threshold_shift: float = 0.05) -> None:
        self.hard_weight = float(hard_weight)
        self.llmb_weight = float(llmb_weight)
        self.base_excited_threshold = float(excited_threshold)
        self.base_scream_threshold = float(scream_threshold)
        self.min_history = max(0, int(min_history))
        self.max_threshold_shift = max(0.0, float(max_threshold_shift))
        self.target = target or {"平述": 0.5, "激动": 0.4, "惊叹": 0.1}
        self._history: deque[str] = deque(maxlen=max(1, int(history_size)))

    @classmethod
    def from_rules(cls, rules: dict) -> "EmotionPolicy":
        cfg = rules.get("emotion_fusion", {})
        emotions = rules.get("emotions", {})
        return cls(hard_weight=cfg.get("hard_weight", 0.7), llmb_weight=cfg.get("llmb_weight", 0.3),
                   excited_threshold=emotions.get("激动", {}).get("threshold", 0.35),
                   scream_threshold=emotions.get("尖叫", {}).get("threshold", 0.72),
                   history_size=cfg.get("history_size", 200), min_history=cfg.get("min_history", 12),
                   target=cfg.get("target"), max_threshold_shift=cfg.get("max_threshold_shift", 0.05))

    @property
    def sample_count(self) -> int:
        return len(self._history)

    @property
    def history_ready(self) -> bool:
        return len(self._history) >= self.min_history

    def _thresholds(self) -> tuple[float, float]:
        # 历史样本不足时用基础阈值，避免短场次被少量样本带偏
        if not self.history_ready or not self._history:
            return self.base_excited_threshold, self.base_scream_threshold
        counts = Counter(self._history)
        size = len(self._history)
        # 依据实际分布与目标分布的偏差微调阈值，偏移量受 max_threshold_shift 限幅
        calm_error = float(self.target.get("平述", 0.5)) - counts["平述"] / size
        scream_error = counts["惊叹"] / size - float(self.target.get("惊叹", 0.1))
        limit = self.max_threshold_shift
        excited = _clamp01(self.base_excited_threshold + max(-limit, min(limit, calm_error * 0.1)))
        scream = _clamp01(self.base_scream_threshold + max(-limit, min(limit, scream_error * 0.1)))
        return excited, max(scream, excited + 0.1)

    def decide(self, *, hard_intensity: float, llmb_intensity: float, scream_eligible: bool) -> EmotionDecision:
        score = blend_intensity(hard_intensity, llmb_intensity,
                                hard_weight=self.hard_weight, llmb_weight=self.llmb_weight)
        excited_threshold, scream_threshold = self._thresholds()
        # 最高档「惊叹」额外要求硬事实达成（scream_eligible），防止 LLMB 单独拉满
        if score >= scream_threshold and scream_eligible:
            label = "惊叹"
        elif score >= excited_threshold:
            label = "激动"
        else:
            label = "平述"
        self._history.append(label)
        return EmotionDecision(label, score, round(excited_threshold, 3), round(scream_threshold, 3))
