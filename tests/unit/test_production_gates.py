"""从真实 Phase3 产物中采样、喂给当前门禁的黑盒测试。"""
import collections
import json
import os
import random
import sys

import pytest

from sbmachine.phase3b_prompt import build_fact_anchors, build_delivery, validate_style_commentary


BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "fixtures", "production_samples")
os.makedirs(DATA_DIR, exist_ok=True)

# 若数据文件不存在，从 729 成功产物中提取并缓存
_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_NEUTRAL_PATH = os.path.join(_PROJECT, "data", "729", "sbmachine", "rounds_with_neutral.json")
_COMMENTARY_PATH = os.path.join(_PROJECT, "data", "729", "sbmachine", "commentary.json")
_ALIASES_PATH = os.path.join(_PROJECT, "database", "player_aliases.json")

NEUTRAL_SAMPLES = os.path.join(DATA_DIR, "neutral_samples.jsonl")
STYLE_SAMPLES = os.path.join(DATA_DIR, "style_samples.jsonl")


def _load_aliases():
    if not os.path.exists(_ALIASES_PATH):
        return {}
    try:
        return json.loads(open(_ALIASES_PATH, "r", encoding="utf-8").read())
    except Exception:
        return {}


def _extract_production_samples():
    """从 729 成功产物中提取 real-world 数据，全量缓存供门禁黑盒断言。

    不使用随机采样：断言基于全集，避免随机抽样造成 flaky。
    """
    aliases = _load_aliases()
    neutrals, styles = [], []

    if os.path.exists(_NEUTRAL_PATH) and os.path.exists(_COMMENTARY_PATH):
        ndata = json.loads(open(_NEUTRAL_PATH, "r", encoding="utf-8").read())
        cdata = json.loads(open(_COMMENTARY_PATH, "r", encoding="utf-8").read())

        for rdn in ndata.get("rounds", []):
            rn = rdn.get("round_no")
            rdc = next((r for r in cdata.get("rounds", []) if r.get("round_no") == rn), None)
            if rdc is None:
                continue
            # 按 t_start 匹配 neutral scene 和 commentary scene
            commentary_map = {}
            for sc in rdc.get("scenes", []):
                if isinstance(sc, dict) and sc.get("text", "").strip():
                    commentary_map[(sc["t_start"], sc["t_end"])] = sc

            for sc in rdn.get("scenes", []):
                key = (sc.get("t_start"), sc.get("t_end"))
                comm_scene = commentary_map.get(key)
                if comm_scene is None:
                    continue
                neutral_text = str(sc.get("neutral") or "")
                if not neutral_text.strip():
                    continue
                # fact_anchors 从 neutral 推导（旧产物无此字段）
                anchors = build_fact_anchors(sc, aliases)
                delivery = build_delivery(sc)
                style_text = f"[{comm_scene.get('emotion', '平述')}]{comm_scene.get('text', '')}"

                styles.append({
                    "window_id": sc.get("window_id", ""),
                    "neutral": neutral_text,
                    "style_text": style_text,
                    "anchors": anchors,
                    "hard_char_limit": delivery.get("hard_char_limit", 100),
                    "char_budget": sc.get("char_budget", 100),
                })
                neutrals.append({
                    "window_id": sc.get("window_id", ""),
                    "neutral": neutral_text,
                    "anchors": anchors,
                    "hard_char_limit": delivery.get("hard_char_limit", 100),
                })

    # 缓存到 fixture 文件
    if neutrals:
        with open(NEUTRAL_SAMPLES, "w", encoding="utf-8") as f:
            for item in neutrals:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if styles:
        with open(STYLE_SAMPLES, "w", encoding="utf-8") as f:
            for item in styles:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return neutrals, styles


def _load_cached_or_extract(kind):
    path = NEUTRAL_SAMPLES if kind == "neutral" else STYLE_SAMPLES
    if os.path.exists(path):
        items = []
        lines = open(path, "r", encoding="utf-8").readlines()
        for line in lines:
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
        if items:
            return items
    all_n, all_s = _extract_production_samples()
    return all_n if kind == "neutral" else all_s


@pytest.fixture(scope="module")
def production_neutrals():
    items = _load_cached_or_extract("neutral")
    assert items, "no production neutral samples available"
    return items


@pytest.fixture(scope="module")
def production_styles():
    items = _load_cached_or_extract("style")
    assert items, "no production style samples available"
    return items


class TestProductionGateCompliance:
    """喂 real-world 产出给当前 5 道门禁，验证门禁行为符合预期。"""

    def test_production_gates_detect_real_budget_violations(self, production_styles):
        """旧成功产物的风格稿中，门禁应检出「防错」违规（unexpected_fact），
        不再拦截「防漏」（missing_anchor 已于 2026-08-16 移除）。"""
        reasons = collections.Counter()
        for s in production_styles:
            result = validate_style_commentary(s["style_text"], s["neutral"], s["anchors"], {}, [],
                                               hard_char_limit=s["hard_char_limit"], char_tolerance=0.5)
            if not result["ok"]:
                reasons[result.get("reason", "unknown")] += 1
        total = len(production_styles)
        print(f"\n  total={total}, failures by reason: {dict(reasons)}")
        assert reasons.get("unexpected_fact", 0) >= 2, "gates should catch unexpected facts in old data"
        assert reasons.get("missing_anchor", 0) == 0, "missing_anchor gate has been removed (防错不防漏)"

    def test_production_gates_report_budget_overage(self, production_styles):
        """超软上限但未超硬上限时应返回 budget_overage > 1.0。"""
        overage_count = 0
        for s in production_styles:
            result = validate_style_commentary(s["style_text"], s["neutral"], s["anchors"], {}, [],
                                               hard_char_limit=s["hard_char_limit"], char_tolerance=0.5)
            if result.get("budget_overage", 1.0) > 1.0:
                overage_count += 1
        print(f"\n  budget_overage > 1.0: {overage_count}/{len(production_styles)}")
        assert overage_count >= 1, "should detect at least one overage"

    def test_empty_commentary_rejected(self, production_neutrals):
        for s in production_neutrals:
            assert isinstance(s["neutral"], str) and s["neutral"].strip()
