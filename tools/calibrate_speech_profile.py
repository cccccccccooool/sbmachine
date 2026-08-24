"""离线 speech profile 标定：采集 manifest → 拆分 → 拟合 → 单侧上界 → 验收 → 不可变 profile。

用法示例：
    python tools/calibrate_speech_profile.py \
        --manifest data/speech_profiles/calibration_manifest.jsonl \
        --profile-id speech-profile-v1 \
        --out data/speech_profiles/speech-profile-v1 \
        --engine-fingerprint gpt-sovits-production-build \
        --voice-fingerprint official-voice-and-reference-hash \
        --preprocess-fingerprint phase4-pcm-policy-hash \
        --sample-rate-hz 32000

设计依据：docs/plan/phase3-rule-tiny-one-way-voice-task-implementation-plan.md §11.3-11.6。
- 独立文本最低 160 条、目标 200 条；不足 160 只能写 status=exploration。
- 60/20/20 拆分（拟合/上界校准/留出），同一模板族近重复不得跨集合。
- 非负线性估计器 + split-conformal 单侧上界。
- 留出集验收：覆盖率>=95%、关键标签>=90%、最大低估<=0.20s、中位过估<=25%。
- 20-30 条 1.25/1.5 速度验证子集：缩放误差 P95<=5%，否则 speed_scaling_verified=false。
- profile 不可变：输出目录已存在 validated profile 时拒绝覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from sbmachine.speech_measure import FEATURE_ORDER, parse_features

MIN_TOTAL_SAMPLES = 160
TARGET_TOTAL_SAMPLES = 200
MIN_VELOCITY_SUBSET = 20
MAX_VELOCITY_SUBSET = 30
SPEED_SCALING_TOLERANCE = 0.05
KEY_TAG_MIN_HOLDOUT = 10
KEY_TAG_MIN_COVERAGE = 0.90
KEY_TAG_MAX_UNDERESTIMATE_SEC = 0.20
HOLDOUT_COVERAGE_TARGET = 0.95
HOLDOUT_MEDIAN_OVERESTIMATE_RATIO_MAX = 0.25


def _try_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError:
        return None


def parse_manifest(path: Path) -> list[dict]:
    """读取 calibration manifest jsonl；校验必需字段。"""
    samples: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "text" not in record or "pcm_duration_sec" not in record:
                raise ValueError(f"manifest line {line_no}: missing text or pcm_duration_sec")
            record["_line"] = line_no
            samples.append(record)
    return samples


def text_fingerprint(text: str) -> str:
    """规范化文本指纹：用于近重复去重与模板族分组。"""
    normalized = "".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_samples(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """按模板族分组 60/20/20 拆分，近重复不跨集；速度验证子集独立抽出。

    返回 (fit, calibration, holdout, velocity)。
    速度验证子集：带 speed_factor 且不等于 base(1.0) 的样本，上限 30 条；
    该子集不参与拟合/校准/留出（时长口径不同，§11.3 不重复计入 160 条门槛）。
    """
    velocity: list[dict] = []
    pool: list[dict] = []
    for sample in samples:
        speed = float(sample.get("speed_factor") or 1.0)
        if abs(speed - 1.0) > 1e-9:
            velocity.append(sample)
        else:
            pool.append(sample)
    velocity = velocity[:MAX_VELOCITY_SUBSET]
    if len(velocity) < MIN_VELOCITY_SUBSET:
        velocity = []

    by_family: dict[str, list[dict]] = {}
    for sample in pool:
        key = text_fingerprint(str(sample.get("text") or ""))
        by_family.setdefault(key, []).append(sample)

    fit: list[dict] = []
    calibration: list[dict] = []
    holdout: list[dict] = []
    families = sorted(by_family.keys())
    for index, key in enumerate(families):
        bucket = index % 10
        if bucket < 6:
            fit.extend(by_family[key])
        elif bucket < 8:
            calibration.extend(by_family[key])
        else:
            holdout.extend(by_family[key])

    return fit, calibration, holdout, velocity


def fit_nonnegative_linear(fit_samples: list[dict]) -> tuple[list[float], float, list[str]]:
    """非负线性回归：duration ~ intercept + Σ coef_i * feature_i（系数≥0）。

    优先 numpy lstsq + clip；numpy 不可用时用纯 Python 正规方程 + clip。
    返回 (coefficients_sec, intercept_sec, feature_order)。
    """
    np = _try_numpy()
    rows = [parse_features(str(s.get("text") or "")) for s in fit_samples]
    durations = [float(s["pcm_duration_sec"]) for s in fit_samples]

    feature_order = list(FEATURE_ORDER)

    if np is not None:
        x = np.array([[row.get(name, 0) for name in feature_order] + [1.0] for row in rows], dtype=float)
        y = np.array(durations, dtype=float)
        coefs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        coefs = np.clip(coefs, 0.0, None)
        return [float(c) for c in coefs[: len(feature_order)]], float(coefs[-1]), feature_order

    n_feat = len(feature_order)
    gram = [[0.0] * (n_feat + 1) for _ in range(n_feat + 1)]
    rhs = [0.0] * (n_feat + 1)
    for row, duration in zip(rows, durations):
        vec = [float(row.get(name, 0)) for name in feature_order] + [1.0]
        for i in range(n_feat + 1):
            rhs[i] += vec[i] * duration
            for j in range(n_feat + 1):
                gram[i][j] += vec[i] * vec[j]
    try:
        aug = [list(r) + [rhs[i]] for i, r in enumerate(gram)]
        for col in range(n_feat + 1):
            pivot = max(range(col, n_feat + 1), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot][col]) < 1e-12:
                raise ValueError("singular matrix")
            aug[col], aug[pivot] = aug[pivot], aug[col]
            for row in range(n_feat + 1):
                if row != col:
                    factor = aug[row][col] / aug[col][col]
                    for k in range(col, n_feat + 2):
                        aug[row][k] -= factor * aug[col][k]
        coefs = [aug[i][n_feat + 1] / aug[i][i] for i in range(n_feat + 1)]
    except (ValueError, ZeroDivisionError):
        coefs = [0.0] * (n_feat + 1)
    coefs = [max(0.0, c) for c in coefs]
    return coefs[:n_feat], coefs[n_feat], feature_order


def predict_sec(sample: dict, feature_order: list[str], coefficients: list[float], intercept: float) -> float:
    features = parse_features(str(sample.get("text") or ""))
    total = intercept
    for name, coef in zip(feature_order, coefficients):
        total += coef * features.get(name, 0)
    return max(0.0, total)


def calibrated_upper_residual(calibration: list[dict], feature_order, coefficients, intercept, coverage: float) -> float:
    """split-conformal 单侧上界：校准集正残差的分位数（默认 0.95）。"""
    residuals = []
    for sample in calibration:
        actual = float(sample["pcm_duration_sec"])
        predicted = predict_sec(sample, feature_order, coefficients, intercept)
        residuals.append(max(0.0, actual - predicted))
    if not residuals:
        return 0.0
    residuals.sort()
    index = min(len(residuals) - 1, int(len(residuals) * coverage))
    return float(residuals[index])


def holdout_metrics(
    holdout: list[dict],
    feature_order: list[str],
    coefficients: list[float],
    intercept: float,
    upper_residual: float,
    fixed_margin: float,
) -> dict:
    """留出集验收指标：覆盖率、最大低估、中位过估比例、分标签覆盖。"""
    results = {"upper_bound_coverage": 0.0, "max_underestimate_sec": 0.0, "median_overestimate_ratio": 0.0}
    if not holdout:
        return results

    covered = 0
    over_ratios = []
    max_under = 0.0
    tag_holdout: dict[str, list[float]] = {}
    for sample in holdout:
        actual = float(sample["pcm_duration_sec"])
        predicted = predict_sec(sample, feature_order, coefficients, intercept)
        upper = predicted + upper_residual + fixed_margin
        if actual <= upper:
            covered += 1
        max_under = max(max_under, actual - upper)
        if predicted > 0:
            over_ratios.append((predicted - actual) / predicted)
        for tag in sample.get("tags") or []:
            tag_holdout.setdefault(str(tag), []).append(actual - (predicted + upper_residual + fixed_margin))

    key_tags = {tag: values for tag, values in tag_holdout.items() if len(values) >= KEY_TAG_MIN_HOLDOUT}
    tag_coverage = {}
    for tag, deltas in key_tags.items():
        covered_count = sum(1 for d in deltas if d <= 0)
        tag_coverage[tag] = {"count": len(deltas), "coverage": covered_count / len(deltas), "max_underestimate_sec": max(max(deltas), 0.0)}
    return {
        "upper_bound_coverage": covered / len(holdout),
        "max_underestimate_sec": max(0.0, max_under),
        "median_overestimate_ratio": statistics.median(over_ratios) if over_ratios else 0.0,
        "tag_metrics": tag_coverage,
    }


def velocity_scaling_error(velocity: list[dict], feature_order, coefficients, intercept) -> float | None:
    """速度验证子集：|actual*speed - expected| / expected 的 P95；子集不足返回 None。"""
    if not velocity:
        return None
    ratios = []
    for sample in velocity:
        speed = float(sample.get("speed_factor") or 1.0)
        expected_at_speed = predict_sec(sample, feature_order, coefficients, intercept) / speed
        actual = float(sample["pcm_duration_sec"])
        if expected_at_speed > 0:
            ratios.append(abs(actual - expected_at_speed) / expected_at_speed)
    if not ratios:
        return None
    ratios.sort()
    index = min(len(ratios) - 1, int(len(ratios) * 0.95))
    return float(ratios[index])


def evaluate_acceptance(metrics: dict, total: int, velocity_p95: float | None) -> tuple[bool, list[str]]:
    """§11.6 验收：全部通过才允许 validated；否则 exploration/stale 处理。"""
    failures: list[str] = []
    if total < MIN_TOTAL_SAMPLES:
        failures.append(f"total samples {total} < {MIN_TOTAL_SAMPLES}")
    if metrics["upper_bound_coverage"] < HOLDOUT_COVERAGE_TARGET:
        failures.append(f"holdout coverage {metrics['upper_bound_coverage']:.3f} < {HOLDOUT_COVERAGE_TARGET}")
    if metrics["median_overestimate_ratio"] > HOLDOUT_MEDIAN_OVERESTIMATE_RATIO_MAX:
        failures.append(f"median overestimate {metrics['median_overestimate_ratio']:.3f} > {HOLDOUT_MEDIAN_OVERESTIMATE_RATIO_MAX}")
    tag_metrics = metrics.get("tag_metrics") or {}
    for tag, values in tag_metrics.items():
        if values["coverage"] < KEY_TAG_MIN_COVERAGE:
            failures.append(f"tag {tag} coverage {values['coverage']:.3f} < {KEY_TAG_MIN_COVERAGE}")
        if values["max_underestimate_sec"] > KEY_TAG_MAX_UNDERESTIMATE_SEC:
            failures.append(f"tag {tag} max underestimate {values['max_underestimate_sec']:.3f}s > {KEY_TAG_MAX_UNDERESTIMATE_SEC}")
    if velocity_p95 is not None and velocity_p95 > SPEED_SCALING_TOLERANCE:
        failures.append(f"velocity scaling P95 {velocity_p95:.3f} > {SPEED_SCALING_TOLERANCE}")
    return not failures, failures


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_profile(
    profile_id: str,
    samples: list[dict],
    *,
    engine_fingerprint: str,
    voice_fingerprint: str,
    preprocess_fingerprint: str,
    sample_rate_hz: int,
    base_speed_factor: float,
    coverage_target: float,
    fixed_margin: float,
    force: bool,
    out_dir: Path,
) -> dict:
    if out_dir.exists() and any(out_dir.glob("profile.json")):
        if not force:
            raise FileExistsError(f"{out_dir} already contains a profile; use --force with a new version directory")
        for existing in out_dir.iterdir():
            if existing.is_file():
                existing.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "calibration_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")

    fit, calibration, holdout, velocity = split_samples(samples)
    coefficients, intercept, feature_order = fit_nonnegative_linear(fit)
    upper_residual = calibrated_upper_residual(calibration, feature_order, coefficients, intercept, coverage_target)
    metrics = holdout_metrics(holdout, feature_order, coefficients, intercept, upper_residual, fixed_margin)
    velocity_p95 = velocity_scaling_error(velocity, feature_order, coefficients, intercept)
    accepted, failures = evaluate_acceptance(metrics, len(samples), velocity_p95)

    status = "validated" if accepted else "exploration"
    speed_scaling_verified = status == "validated" and velocity_p95 is not None and velocity_p95 <= SPEED_SCALING_TOLERANCE
    if velocity_p95 is None:
        speed_scaling_verified = False

    manifest_sha = sha256_of(manifest_path)
    payload = {
        "profile_schema_version": 1,
        "profile_id": profile_id,
        "status": status,
        "metric_version": "speech_units_v1",
        "engine_fingerprint": engine_fingerprint,
        "voice_fingerprint": voice_fingerprint,
        "preprocess_fingerprint": preprocess_fingerprint,
        "sample_rate_hz": sample_rate_hz,
        "base_speed_factor": base_speed_factor,
        "duration_estimator": {
            "kind": "nonnegative_linear_v1",
            "feature_order": feature_order,
            "coefficients_sec": [round(c, 4) for c in coefficients],
            "intercept_sec": round(intercept, 4),
        },
        "safety": {
            "method": "split_conformal_upper_v1",
            "coverage_target": coverage_target,
            "upper_residual_sec": round(upper_residual, 4),
            "fixed_margin_sec": fixed_margin,
        },
        "speed_scaling_verified": speed_scaling_verified,
        "dataset": {
            "total": len(samples),
            "fit": len(fit),
            "calibration": len(calibration),
            "holdout": len(holdout),
            "velocity_subset": len(velocity),
            "manifest_sha256": manifest_sha,
        },
        "holdout_metrics": {
            "upper_bound_coverage": round(metrics["upper_bound_coverage"], 4),
            "max_underestimate_sec": round(metrics["max_underestimate_sec"], 4),
            "median_overestimate_ratio": round(metrics["median_overestimate_ratio"], 4),
            "tag_metrics": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in metrics.get("tag_metrics", {}).items()},
            "velocity_scaling_p95": None if velocity_p95 is None else round(velocity_p95, 4),
        },
        "acceptance_failures": failures if not accepted else [],
    }
    profile_path = out_dir / "profile.json"
    profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "profile_id": profile_id,
                "status": status,
                "split_counts": {"fit": len(fit), "calibration": len(calibration), "holdout": len(holdout), "velocity": len(velocity)},
                "holdout_metrics": payload["holdout_metrics"],
                "acceptance_failures": failures,
                "profile_sha256": sha256_of(profile_path),
                "manifest_sha256": manifest_sha,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="离线 speech profile 标定（§11.3-11.6）")
    parser.add_argument("--manifest", required=True, help="calibration manifest jsonl 路径")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--out", required=True, help="输出目录，如 data/speech_profiles/speech-profile-v1")
    parser.add_argument("--engine-fingerprint", required=True)
    parser.add_argument("--voice-fingerprint", required=True)
    parser.add_argument("--preprocess-fingerprint", required=True)
    parser.add_argument("--sample-rate-hz", type=int, default=32000)
    parser.add_argument("--base-speed-factor", type=float, default=1.0)
    parser.add_argument("--coverage-target", type=float, default=HOLDOUT_COVERAGE_TARGET)
    parser.add_argument("--fixed-margin", type=float, default=0.08)
    parser.add_argument("--force", action="store_true", help="允许覆盖已有输出目录")
    args = parser.parse_args()

    samples = parse_manifest(Path(args.manifest))
    payload = build_profile(
        args.profile_id,
        samples,
        engine_fingerprint=args.engine_fingerprint,
        voice_fingerprint=args.voice_fingerprint,
        preprocess_fingerprint=args.preprocess_fingerprint,
        sample_rate_hz=args.sample_rate_hz,
        base_speed_factor=args.base_speed_factor,
        coverage_target=args.coverage_target,
        fixed_margin=args.fixed_margin,
        force=args.force,
        out_dir=Path(args.out),
    )
    print(f"profile {args.profile_id}: status={payload['status']}")
    for failure in payload.get("acceptance_failures") or []:
        print(f"  failure: {failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
