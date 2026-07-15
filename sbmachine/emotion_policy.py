"""Phase 3b 解说情绪三路融合：硬事实强度 + LLMB 强度，输出最终三档情绪标签。"""
from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass

FINAL_EMOTIONS = ("平述", "激动", "惊叹")
_ALL_EMOTION_TAGS = re.compile(r"\[(?:平述|平叙|激动|惊叹|紧张|惋惜)\]")


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
    """剥掉解说文本里的所有中间情绪标签，只保留融合后的最终标签作前缀。"""
    if final_label not in FINAL_EMOTIONS:
        raise ValueError(f"unsupported final emotion: {final_label}")
    body = _ALL_EMOTION_TAGS.sub("", str(commentary)).strip()
    return f"[{final_label}]{body}"


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
