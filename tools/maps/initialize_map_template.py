"""从解析器的 tick 产物交互式生成一份经人工审核、带层级的点位（callout）图。

解析器的 ``Player.LastPlaceName()`` 值被视为唯一的点位 ID。
本脚本从不臆造英文点位名：它只展示在 ``ticks.jsonl`` 中观测到的 ID，
再请人工补充中文标签、逻辑层级与合法连边。
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPS_DIR = PROJECT_ROOT / "database" / "maps"

# 抄自 demoinfocs-golang v5.2.0 examples/_assets/metadata/*.txt。
# 这些只是可选的雷达参考，从不用于指定逻辑地图层级。
RADAR_SECTIONS = {
    "de_nuke": {
        "default": {"altitude_min": -495.0, "altitude_max": 10000.0},
        "lower": {"altitude_min": -10000.0, "altitude_max": -495.0},
    },
    "de_train": {
        "default": {"altitude_min": -50.0, "altitude_max": 20000.0},
        "lower": {"altitude_min": -5000.0, "altitude_max": -50.0},
    },
}

EDGE_KINDS = ("walk", "stairs", "ladder", "ramp", "vent", "drop", "lift")

# 第三轮关系输入时展示的数字编号映射
_KIND_NUMBERS: dict[str, str] = {
    "1": "walk", "2": "stairs", "3": "ladder",
    "4": "ramp", "5": "vent", "6": "drop", "7": "lift",
}
_KIND_DISPLAY = "1=walk  2=stairs  3=ladder  4=ramp  5=vent  6=drop  7=lift"


def _rows(path: Path) -> Iterable[dict]:
    """逐行读取 ticks.jsonl（人工初始化只支持该格式）。"""
    if path.suffix.casefold() != ".jsonl":
        raise ValueError(f"only ticks.jsonl is supported for manual initialization: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def collect_callout_stats(paths: Iterable[Path]) -> dict[str, dict[str, float | int | None]]:
    """仅返回解析器观测到的点位 ID 及其 z 坐标分布。"""
    z_values: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for path in paths:
        for row in _rows(path):
            callout = str(row.get("callout") or "").strip()
            if not callout:
                continue
            counts[callout] += 1
            try:
                z_values[callout].append(float(row.get("z")))
            except (TypeError, ValueError):
                pass
    return {
        name: {
            "samples": counts[name],
            "median_z": round(statistics.median(z_values[name]), 2) if z_values[name] else None,
        }
        for name in sorted(counts, key=str.casefold)
    }


def radar_layer(map_name: str, z: float | int | None) -> str | None:
    """从官方随附的元数据推荐一个雷达分区；绝不据此推断行走路线。"""
    if z is None:
        return None
    for name, section in RADAR_SECTIONS.get(map_name, {}).items():
        if section["altitude_min"] <= float(z) <= section["altitude_max"]:
            return name
    return None


def new_manual_template(
    map_name: str,
    callouts: Iterable[str],
    *,
    labels: dict[str, str] | None = None,
    observations: dict[str, dict[str, float | int | None]] | None = None,
) -> dict:
    """依据解析器观测到的点位 ID 生成一份未经审核的 v2 草稿。"""
    labels = labels or {}
    observations = observations or {}
    nodes: dict[str, dict] = {}
    for callout in sorted({str(value).strip() for value in callouts if str(value).strip()}, key=str.casefold):
        observed = observations.get(callout, {})
        median_z = observed.get("median_z")
        nodes[callout] = {
            "zh": labels.get(callout, callout),
            # 逻辑层级是人工掌控的离散取值：L1 为默认层，
            # L2/L3+ 在其上方，L0/-1... 在其下方。
            "layer": "L1",
            "level": 1,
            "radar_layer": radar_layer(map_name, median_z),
            "samples": int(observed.get("samples") or 0),
            "median_z": median_z,
            "centroid": None,
            "bounds_p05_p95": None,
        }
    return {
        "version": 2,
        "map_name": map_name,
        "source": {
            "method": "manual_bootstrap_from_demoinfocs_ticks",
            "parser_field": "Player.LastPlaceName / m_szLastPlaceName",
            "manual_review_required": True,
            "manual_reviewed": False,
        },
        "radar_sections": RADAR_SECTIONS.get(map_name, {}),
        "callouts": nodes,
        "directed_transitions": [],
    }


def validate_template(template: dict) -> list[str]:
    """校验人工模板的完整性，返回错误信息列表（空列表表示通过）。"""
    errors: list[str] = []
    source = template.get("source") or {}
    if not isinstance(source, dict) or source.get("manual_review_required") is not True:
        errors.append("source.manual_review_required must be true")
    if not isinstance(source, dict) or source.get("manual_reviewed") is not True:
        errors.append("source.manual_reviewed must be true after human review")
    callouts = template.get("callouts")
    if not isinstance(callouts, dict) or not callouts:
        return ["callouts must contain at least one parser-observed node"]
    for name, node in callouts.items():
        if not str(name).strip() or not isinstance(node, dict):
            errors.append(f"invalid callout node: {name!r}")
            continue
        if not str(node.get("zh") or "").strip():
            errors.append(f"{name}: Chinese label is required")
        elif not any("\u4e00" <= char <= "\u9fff" for char in str(node.get("zh"))):
            errors.append(f"{name}: Chinese label must contain a Chinese character")
        level = node.get("level")
        if not isinstance(level, int):
            errors.append(f"{name}: level must be an integer")
        if not str(node.get("layer") or "").strip():
            errors.append(f"{name}: layer is required")
    for index, edge in enumerate(template.get("directed_transitions") or []):
        source, target = str(edge.get("from") or ""), str(edge.get("to") or "")
        if source not in callouts or target not in callouts:
            errors.append(f"edge {index}: endpoints must be existing callouts")
        if source == target:
            errors.append(f"edge {index}: self-loop is not allowed")
        if edge.get("kind") not in EDGE_KINDS:
            errors.append(f"edge {index}: kind must be one of {', '.join(EDGE_KINDS)}")
        if not isinstance(edge.get("one_way"), bool):
            errors.append(f"edge {index}: one_way must be boolean")
        if not isinstance(edge.get("level_delta"), int):
            errors.append(f"edge {index}: level_delta must be integer")
    return errors


def _make_node_index(callouts: dict[str, dict]) -> list[tuple[int, str, str]]:
    """生成点位编号列表：[(编号, 英文ID, 中文名)]，按英文 ID 不区分大小写排序。"""
    return [
        (i + 1, cid, str(callouts[cid].get("zh") or cid))
        for i, cid in enumerate(sorted(callouts, key=str.casefold))
    ]


def _add_edge(template: dict, source: str, target: str, kind: str, one_way: bool, level_delta: int) -> None:
    """向模板追加一条边；非单向时自动补反向边（level_delta 取反）。"""
    edge: dict = {"from": source, "to": target, "kind": kind, "level_delta": level_delta,
                  "one_way": one_way, "samples": 0, "source": "manual"}
    template["directed_transitions"].append(edge)
    if not one_way:
        template["directed_transitions"].append({**edge, "from": target, "to": source, "level_delta": -level_delta})


def _choose_map(default: str | None) -> str:
    """确定地图 ID：优先用命令行参数，否则交互式询问人工。"""
    if default:
        return default.strip().casefold()
    maps = sorted(RADAR_SECTIONS)
    print("常见地图：", ", ".join(maps), "；也可输入其他 de_xxx")
    while True:
        value = input("地图 ID（例如 de_nuke）：").strip().casefold()
        if value:
            return value


def _print_catalog(callouts: list[str]) -> None:
    """打印解析出的英文点位 ID，供人工确认翻译。"""
    print("\n当前 demo 解析 ID -> 中文（请人工确认翻译）：")
    for name in callouts:
        print(f"  {name:<24} -> 待人工翻译")


def _collect_chinese_labels(template: dict) -> None:
    """第一轮：为所有点位填写中文名（回车保留默认值）。"""
    print("\n【第一轮】填写中文名（英文 ID 来自 demo，回车保留默认值）：")
    for callout, node in template["callouts"].items():
        print("  英文默认值必须填写中文名；已有中文名才可回车保留。")
        zh = input(f"  {callout} 中文 [{node['zh']}]：").strip()
        if zh:
            node["zh"] = zh

        while not any("\u4e00" <= char <= "\u9fff" for char in str(node["zh"])):
            print("  请填写中文名，不能保留英文 ID。")
            zh = input(f"  {callout} 中文 [{node['zh']}]：").strip()
            if zh:
                node["zh"] = zh

def _collect_levels(template: dict) -> None:
    """第二轮：填写逻辑层级整数（回车接受当前值，默认 1）。"""
    print("\n【第二轮】填写层级整数（任意整数；回车接受当前值）：")
    for callout, node in template["callouts"].items():
        while True:
            raw = input(f"  {callout}（{node['zh']}）层级 [{node['level']}]：").strip()
            try:
                node["level"] = int(raw) if raw else node["level"]
                node["layer"] = f"L{node['level']}"
                break
            except ValueError:
                print("  请输入整数。")


def _configure_edges(template: dict) -> None:
    """第三轮：用编号选择起终点和关系类型，添加真实可走连接。"""
    callouts = template["callouts"]
    existing: set[tuple[str, str]] = {(e["from"], e["to"]) for e in template["directed_transitions"]}

    def show_index() -> dict[int, str]:
        idx = _make_node_index(callouts)
        print("\n点位编号：")
        for num, cid, zh in idx:
            outgoing = [
                f"{callouts.get(e['to'], {}).get('zh', e['to'])}({e['kind']})"
                for e in template["directed_transitions"] if e["from"] == cid
            ]
            suffix = "、".join(outgoing) if outgoing else "无"
            print(f"  {num}. {zh}（{cid}）→ {suffix}")
        return {num: cid for num, cid, _ in idx}

    print(f"\n【第三轮】添加可走通路。只填真实可经过的连接；Nuke/Train 跨层必须显式添加。")
    print(f"关系类型：{_KIND_DISPLAY}")

    while True:
        num_to_cid = show_index()
        if input("\n添加一条连接？[y/N]：").strip().casefold() not in {"y", "yes", "是"}:
            break

        # 起点
        try:
            src_num = int(input("  起点编号：").strip())
        except ValueError:
            print("  请输入编号数字。")
            continue
        if src_num not in num_to_cid:
            print("  编号不在列表中。")
            continue

        # 终点
        try:
            dst_num = int(input("  终点编号：").strip())
        except ValueError:
            print("  请输入编号数字。")
            continue
        if dst_num not in num_to_cid:
            print("  编号不在列表中。")
            continue

        source, target = num_to_cid[src_num], num_to_cid[dst_num]
        if source == target:
            print("  起终点不能相同。")
            continue
        if (source, target) in existing:
            print("  该连接已存在。")
            continue

        # 关系类型
        kind_raw = input(f"  关系类型编号 [1=walk]：").strip() or "1"
        kind = _KIND_NUMBERS.get(kind_raw)
        if kind is None:
            print("  无效编号，请输入 1–7。")
            continue

        # 单向（drop 默认单向）
        default_prompt = "Y/n" if kind == "drop" else "y/N"
        while True:
            raw_ow = input(f"  单向？[{default_prompt}]：").strip().casefold()
            if not raw_ow:
                one_way = kind == "drop"
                break
            if raw_ow in {"y", "yes", "是"}:
                one_way = True
                break
            if raw_ow in {"n", "no", "否"}:
                one_way = False
                break
            print("  请输入 y 或 n，回车使用默认值。")

        # 层级变化
        src_lv = callouts[source].get("level")
        dst_lv = callouts[target].get("level")
        suggested = dst_lv - src_lv if isinstance(src_lv, int) and isinstance(dst_lv, int) else 0
        raw_delta = input(f"  层级变化（起→终）[{suggested}]：").strip()
        try:
            delta = int(raw_delta) if raw_delta else suggested
        except ValueError:
            print("  层级变化必须是整数。")
            continue

        _add_edge(template, source, target, kind, one_way, delta)
        existing.add((source, target))
        if not one_way:
            existing.add((target, source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", help="地图 ID，例如 de_nuke")
    parser.add_argument("--ticks", type=Path, nargs="+", required=True, help="当前解析器产出的 ticks.jsonl（唯一英文点位来源，可多个）")
    parser.add_argument("--output", type=Path, help="默认 database/maps/<map>.json")
    parser.add_argument("--force", action="store_true", help="允许覆盖已有模板")
    parser.add_argument("--show-catalog", action="store_true", help="仅打印点位中英对照，不写文件")
    args = parser.parse_args()

    map_name = _choose_map(args.map_name)
    observations = collect_callout_stats(args.ticks)
    callouts = list(observations)
    if not callouts:
        raise SystemExit("没有可展示的点位：当前 ticks.jsonl 未包含任何 callout。")
    labels: dict[str, str] = {}
    _print_catalog(callouts)
    if args.show_catalog:
        return

    template = new_manual_template(map_name, callouts, labels=labels, observations=observations)
    _collect_chinese_labels(template)
    _collect_levels(template)
    _configure_edges(template)
    template["source"]["manual_reviewed"] = True
    errors = validate_template(template)
    if errors:
        raise SystemExit("模板校验失败：\n- " + "\n- ".join(errors))
    output = args.output or MAPS_DIR / f"{map_name}.json"
    if output.exists() and not args.force:
        raise SystemExit(f"拒绝覆盖已有文件：{output}（确认后加 --force）")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入：{output}")


if __name__ == "__main__":
    main()
