"""从解析出的 tick 文件构建带坐标标定的 CS2 点位（callout）图。"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def _rows(path: Path) -> Iterable[dict]:
    """逐行读取 tick 文件，支持 jsonl 与 parquet 两种格式。"""
    if path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if path.suffix.casefold() == ".parquet":
        import pandas as pd
        for row in pd.read_parquet(path).to_dict(orient="records"):
            yield row
        return
    raise ValueError(f"unsupported tick format: {path}")


def _percentile(values: list[float], ratio: float) -> float:
    """按线性插值计算给定分位数，空序列返回 0.0。"""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * ratio
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _load_base_template(path: Path | None, map_name: str) -> dict:
    """读取人工基础模板，并校验其 map_name 与 --map 一致。"""
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read base template: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("base template must be a JSON object")
    existing_map = str(payload.get("map_name") or "")
    if existing_map and existing_map != map_name:
        raise ValueError(f"base template map_name={existing_map!r} does not match --map={map_name!r}")
    return payload


def _merge_edges(observed: Counter[tuple[str, str]], base: dict, callouts: dict) -> list[dict]:
    """仅保留人工编写的连边，并附上对应的观测计数。"""
    merged: list[dict] = []
    for item in base.get("directed_transitions") or []:
        if not isinstance(item, dict):
            continue
        source, target = str(item.get("from") or ""), str(item.get("to") or "")
        if source not in callouts or target not in callouts or source == target:
            continue
        edge = dict(item)
        samples = observed.get((source, target), 0)
        edge["samples"] = samples
        edge["source"] = "manual+observed" if samples else "manual"
        merged.append(edge)
    return merged


def build_template(paths: list[Path], map_name: str, *, base_template: dict) -> dict:
    """在一份已完成人工审核的 v2 模板上叠加观测数据。"""
    source = base_template.get("source") or {}
    if int(base_template.get("version") or 0) < 2 or source.get("manual_reviewed") is not True:
        raise ValueError("base_template must be a completed manual v2 template")
    points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    transitions: Counter[tuple[str, str]] = Counter()
    previous: dict[str, tuple[int, str]] = {}
    total = 0
    for path in paths:
        for row in _rows(path):
            callout = str(row.get("callout") or "").strip()
            if not callout:
                continue
            try:
                point = (float(row["x"]), float(row["y"]), float(row.get("z", 0.0)))
                tick = int(row.get("tick", 0))
            except (KeyError, TypeError, ValueError):
                continue
            points[callout].append(point)
            total += 1
            identity = str(row.get("steamid") or row.get("name") or "")
            if identity and identity in previous:
                previous_tick, previous_callout = previous[identity]
                if callout != previous_callout and tick >= previous_tick:
                    transitions[(previous_callout, callout)] += 1
            if identity:
                previous[identity] = (tick, callout)
    base_callouts = base_template.get("callouts") if isinstance(base_template.get("callouts"), dict) else {}
    callouts = {name: dict(node) for name, node in base_callouts.items() if isinstance(node, dict)}
    for name, samples in sorted(points.items()):
        xs, ys, zs = ([point[index] for point in samples] for index in range(3))
        node = dict(callouts.get(name) or {})
        node.update({
            "samples": len(samples),
            "centroid": [round(sum(xs) / len(xs), 2), round(sum(ys) / len(ys), 2), round(sum(zs) / len(zs), 2)],
            "bounds_p05_p95": {
                "x": [round(_percentile(xs, 0.05), 2), round(_percentile(xs, 0.95), 2)],
                "y": [round(_percentile(ys, 0.05), 2), round(_percentile(ys, 0.95), 2)],
                "z": [round(_percentile(zs, 0.05), 2), round(_percentile(zs, 0.95), 2)],
            },
        })
        callouts[name] = node
    edges = _merge_edges(transitions, base_template, callouts)
    result = {
        "version": 2, "map_name": map_name,
        "source_files": [str(path) for path in paths],
        "sample_count": total, "sample_files": len(paths),
        "callouts": callouts, "directed_transitions": edges,
    }
    for key in ("source", "radar_sections"):
        if key in base_template:
            result[key] = base_template[key]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticks", nargs="+", type=Path, help="ticks.jsonl or ticks.parquet files")
    parser.add_argument("--map", required=True, dest="map_name")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-template", type=Path, required=True, help="completed manual v2 template to enrich")
    args = parser.parse_args()
    base_template = _load_base_template(args.base_template, args.map_name)
    payload = build_template(args.ticks, args.map_name, base_template=base_template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
