"""统一语音测量：speech_units_v1 特征提取与时长估算。

设计依据：docs/plan/phase3-rule-tiny-one-way-voice-task-implementation-plan.md §11。
- `parse_features`：从文本提取可发音单位特征，不做任何时长假设。
- `measure_text`：在线测量入口；只有 validated profile 才给出安全上界。
- `load_profile` / `check_profile_match`：不可变 profile 的读取与指纹核对。
- `estimate_required_speed_factor`：只有 validated profile 且速度缩放已验证时才给建议。

本模块只使用 stdlib；不依赖 TTS 引擎或外部服务。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from sbmachine.common import count_spoken_chars

METRIC_VERSION = "speech_units_v1"
PROFILE_SCHEMA_VERSION = 1
BASE_SPEED_FACTOR = 1.0
MAX_SPEED_FACTOR = 1.5

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_PAUSE_RE = re.compile(r"[，。！？、；：…,—]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*")
_ALNUM_ONLY_RE = re.compile(r"^[0-9]+$")
_NUMERIC_RE = re.compile(r"^[0-9]+(?:[.-][0-9]+)+$")
_TAG_RE = re.compile(r"[【\[][^】\]]{1,12}[】\]]")
_MD_NOISE_RE = re.compile(r"[*_`#>|~\-]")
_SPACE_RE = re.compile(r"\s+")

FEATURE_ORDER = [
    "zh_units",
    "english_words",
    "number_groups",
    "alnum_tokens",
    "pause_units",
]


class SpeechFeatureError(ValueError):
    """特征解析失败。"""


class ProfileError(ValueError):
    """profile 结构或状态错误。"""


class ProfileUnavailableError(ProfileError):
    """profile 缺失或指纹不匹配。"""


def _normalize_text(text: str) -> str:
    """归一化：剥离情绪标签、markdown 噪音、压缩空白。返回空串表示无可发音内容。"""
    if not isinstance(text, str):
        raise SpeechFeatureError(f"text must be str, got {type(text).__name__}")
    stripped = _TAG_RE.sub("", text)
    stripped = _MD_NOISE_RE.sub("", stripped)
    stripped = _SPACE_RE.sub("", stripped)
    return stripped


def parse_features(text: str) -> dict:
    """提取 speech_units_v1 特征计数。

    规则（§11.1）：
    - 每个可发音汉字 1 个 zh_units。
    - 连续英文词/选手名按 1 个 english_words（Tauson 不算 6 个字符）。
    - 数字、小数、比分按 1 个 number_groups（3-2、1.5 都算一个组）。
    - C4、武器代号、混合字母数字串按 1 个 alnum_tokens（既含字母又含数字）。
    - 停顿标点进入 pause_units；情绪标签不计正文；空白/markdown 噪音剔除。
    """
    normalized = _normalize_text(text)
    if not normalized:
        return {
            "zh_units": 0,
            "english_words": 0,
            "number_groups": 0,
            "alnum_tokens": 0,
            "pause_units": 0,
        }

    zh_units = len(_ZH_RE.findall(normalized))
    pause_units = len(_PAUSE_RE.findall(normalized))

    english_words = 0
    number_groups = 0
    alnum_tokens = 0
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0)
        if _NUMERIC_RE.match(token) or _ALNUM_ONLY_RE.match(token):
            number_groups += 1
        elif re.search(r"[A-Za-z]", token) and re.search(r"[0-9]", token):
            alnum_tokens += 1
        else:
            english_words += 1

    return {
        "zh_units": zh_units,
        "english_words": english_words,
        "number_groups": number_groups,
        "alnum_tokens": alnum_tokens,
        "pause_units": pause_units,
    }


def spoken_units_from_features(features: dict) -> int:
    """默认单位合计：各特征 1:1 相加。有 profile 时由 profile 系数决定，此处仅诊断。"""
    return int(sum(int(features.get(k, 0)) for k in FEATURE_ORDER))


def _profile_root(profile_id: str) -> Path:
    return Path("data") / "speech_profiles" / profile_id


def load_profile(profile_id: str) -> dict | None:
    """读取不可变 profile；缺失/损坏/字段不齐返回 None。"""
    if not profile_id:
        return None
    path = _profile_root(profile_id) / "profile.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("profile_schema_version") != PROFILE_SCHEMA_VERSION:
        return None
    estimator = payload.get("duration_estimator") or {}
    safety = payload.get("safety") or {}
    required = [
        payload.get("profile_id"),
        payload.get("status"),
        estimator.get("feature_order"),
        estimator.get("coefficients_sec"),
        estimator.get("intercept_sec"),
        safety.get("upper_residual_sec"),
        safety.get("fixed_margin_sec"),
    ]
    if any(v is None for v in required):
        return None
    return payload


def validate_profile_status(profile: dict) -> str:
    """返回 status；非法状态抛 ProfileError。"""
    status = profile.get("status")
    if status not in {"validated", "exploration", "stale"}:
        raise ProfileError(f"invalid profile status: {status!r}")
    return status


def check_profile_match(
    profile: dict,
    *,
    engine_fingerprint: str,
    voice_fingerprint: str,
    preprocess_fingerprint: str,
) -> bool:
    """指纹一致性校验（Phase4 与任务单 profile 必须完全一致）。"""
    if not isinstance(profile, dict):
        return False
    return (
        profile.get("engine_fingerprint") == engine_fingerprint
        and profile.get("voice_fingerprint") == voice_fingerprint
        and profile.get("preprocess_fingerprint") == preprocess_fingerprint
    )


def _expected_duration_sec(profile: dict, features: dict) -> float | None:
    """按 profile 的线性估计器计算基准速度期望时长。"""
    estimator = profile.get("duration_estimator") or {}
    feature_order = list(estimator.get("feature_order") or [])
    coefficients = list(estimator.get("coefficients_sec") or [])
    intercept = float(estimator.get("intercept_sec") or 0.0)
    if len(feature_order) != len(coefficients):
        raise ProfileError("duration_estimator feature_order/coefficients length mismatch")
    total = intercept
    for name, coef in zip(feature_order, coefficients):
        total += float(coef) * float(features.get(name, 0))
    return max(0.0, total)


def _safe_upper_sec(profile: dict, features: dict) -> float | None:
    """safe upper = expected + calibrated upper residual + fixed margin（基准速度，与 speed 无关）。"""
    expected = _expected_duration_sec(profile, features)
    if expected is None:
        return None
    safety = profile.get("safety") or {}
    return expected + float(safety.get("upper_residual_sec") or 0.0) + float(safety.get("fixed_margin_sec") or 0.0)


def measure_text(
    text: str,
    emotion: str = "",
    profile_id: str | None = None,
    speed_factor: float = 1.0,
) -> dict:
    """在线测量：特征 + 估算时长 + 安全上界。

    - 无 profile / exploration / stale：不给出 safe_duration_upper_bound（禁止入分级）。
    - validated profile 才产生 estimated_duration_sec 与安全上界。
    """
    if not isinstance(speed_factor, (int, float)) or speed_factor <= 0:
        raise SpeechFeatureError(f"speed_factor must be positive, got {speed_factor!r}")

    features = parse_features(text)
    display_chars = count_spoken_chars(text)

    result = {
        "metric_version": METRIC_VERSION,
        "display_chars": display_chars,
        "spoken_units": spoken_units_from_features(features),
        "feature_counts": features,
        "estimated_duration_sec": None,
        "safe_duration_upper_bound_at_base_speed_sec": None,
        "profile_id": profile_id,
        "profile_status": "unavailable",
    }

    if profile_id:
        profile = load_profile(profile_id)
        if profile is None:
            return result
        status = validate_profile_status(profile)
        result["profile_status"] = status
        if status != "validated":
            return result
        expected = _expected_duration_sec(profile, features)
        if expected is not None:
            result["estimated_duration_sec"] = round(expected / float(speed_factor), 3)
        safe_upper = _safe_upper_sec(profile, features)
        if safe_upper is not None:
            result["safe_duration_upper_bound_at_base_speed_sec"] = round(safe_upper, 3)

    return result


def estimate_required_speed_factor(
    text: str,
    profile: dict,
    slot_duration_sec: float,
) -> float | None:
    """minimum_required_speed_factor = max(1.0, safe_upper / slot_duration_sec)。

    仅 validated profile 且速度缩放已验证（speed_scaling_verified=true 或缺省按 true 语义）时给出；
    否则返回 None（禁止用估算冒充安全分级）。
    """
    if not isinstance(profile, dict):
        return None
    if validate_profile_status(profile) != "validated":
        return None
    if slot_duration_sec <= 0:
        raise SpeechFeatureError(f"slot_duration_sec must be positive, got {slot_duration_sec!r}")
    if profile.get("speed_scaling_verified") is False:
        return None
    features = parse_features(text)
    safe_upper = _safe_upper_sec(profile, features)
    if safe_upper is None:
        return None
    return max(1.0, safe_upper / float(slot_duration_sec))
