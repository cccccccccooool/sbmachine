"""战术规则书的严格加载与编译。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SUPPORTED_VERSION = 1
_SIDES = {"T", "CT"}
_KINDS = {"zone_count", "alive_count", "event_count", "bomb_planted"}
_EVENTS = {"utility_throw", "kill", "flash", "bomb_planted", "defuse_started"}


@dataclass(frozen=True)
class CompiledTactic:
    rule_id: str
    label: str
    hint: str
    side: str
    scene: tuple[str, ...] | None
    time_window_sec: tuple[float, float] | None
    when: tuple[dict[str, Any], ...]
    priority: int


@dataclass(frozen=True)
class CompiledTacticBook:
    map_name: str
    tactics: tuple[CompiledTactic, ...] = ()


def _empty(map_name: str) -> CompiledTacticBook:
    return CompiledTacticBook(map_name=map_name)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_count(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    lower, upper = value
    if not _is_number(lower) or lower < 0:
        return False
    return upper is None or (_is_number(upper) and upper >= lower)


def _valid_bbox(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"x", "y", "z"}:
        return False
    for axis in ("x", "y", "z"):
        bounds = value.get(axis)
        if not isinstance(bounds, list) or len(bounds) != 2:
            return False
        if not _is_number(bounds[0]) or not _is_number(bounds[1]) or bounds[0] > bounds[1]:
            return False
    return True


def _valid_zone(value: object) -> bool:
    if not isinstance(value, dict) or len(value) != 1:
        return False
    if "callouts_any" in value:
        callouts = value["callouts_any"]
        return isinstance(callouts, list) and bool(callouts) and all(isinstance(item, str) and item for item in callouts)
    return "bbox" in value and _valid_bbox(value["bbox"])


def _valid_condition(condition: object) -> bool:
    if not isinstance(condition, dict) or condition.get("kind") not in _KINDS:
        return False
    kind = condition["kind"]
    if kind == "zone_count":
        return (
            set(condition) == {"kind", "side", "zone", "count"}
            and condition.get("side") in _SIDES
            and _valid_zone(condition.get("zone"))
            and _valid_count(condition.get("count"))
        )
    if kind == "alive_count":
        return (
            set(condition) == {"kind", "side", "count"}
            and condition.get("side") in _SIDES
            and _valid_count(condition.get("count"))
        )
    if kind == "bomb_planted":
        return set(condition) == {"kind", "value"} and isinstance(condition.get("value"), bool)

    allowed = {"kind", "event", "actor_side", "actor_zone", "types_any", "destination_bbox", "window_sec", "count"}
    if not set(condition).issubset(allowed) or set(condition) != {"kind", "event", "count"} | (set(condition) & {"actor_side", "actor_zone", "types_any", "destination_bbox", "window_sec"}):
        return False
    if condition.get("event") not in _EVENTS or not _valid_count(condition.get("count")):
        return False
    if "actor_side" in condition and condition["actor_side"] not in _SIDES:
        return False
    if "actor_zone" in condition and not _valid_zone(condition["actor_zone"]):
        return False
    if "types_any" in condition:
        values = condition["types_any"]
        if not isinstance(values, list) or not values or not all(isinstance(item, str) and item for item in values):
            return False
    if "destination_bbox" in condition and not _valid_bbox(condition["destination_bbox"]):
        return False
    if "window_sec" in condition and (not _is_number(condition["window_sec"]) or condition["window_sec"] < 0):
        return False
    return True


def _compile_tactic(raw: object) -> CompiledTactic | None:
    if not isinstance(raw, dict):
        return None
    allowed = {"id", "label", "hint", "side", "scene", "time_window_sec", "when", "priority"}
    if not set(raw).issubset(allowed):
        return None
    rule_id = raw.get("id")
    label = raw.get("label")
    side = raw.get("side")
    priority = raw.get("priority")
    when = raw.get("when")
    if not all(isinstance(value, str) and value for value in (rule_id, label)):
        return None
    if side not in _SIDES or not isinstance(priority, int) or isinstance(priority, bool):
        return None
    if not isinstance(when, list) or not when or not all(_valid_condition(condition) for condition in when):
        return None
    hint = raw.get("hint", label)
    if not isinstance(hint, str) or not hint:
        return None
    scene_value = raw.get("scene")
    if scene_value is not None and (
        not isinstance(scene_value, list)
        or not scene_value
        or not all(isinstance(item, str) and item for item in scene_value)
    ):
        return None
    time_value = raw.get("time_window_sec")
    if time_value is not None and (
        not isinstance(time_value, list)
        or len(time_value) != 2
        or not all(_is_number(item) for item in time_value)
        or time_value[0] > time_value[1]
    ):
        return None
    return CompiledTactic(
        rule_id=rule_id,
        label=label,
        hint=hint,
        side=side,
        scene=tuple(scene_value) if scene_value is not None else None,
        time_window_sec=(float(time_value[0]), float(time_value[1])) if time_value is not None else None,
        when=tuple(dict(condition) for condition in when),
        priority=priority,
    )


def compile_tactic_book(map_name: str, source: object) -> CompiledTacticBook:
    """将单图 JSON 编译为只读规则；任一格式错误均使整本书失效。"""
    if not isinstance(map_name, str) or not map_name:
        return _empty(str(map_name))
    if (
        not isinstance(source, dict)
        or set(source) != {"version", "map", "tactics"}
        or source.get("version") != _SUPPORTED_VERSION
        or source.get("map") != map_name
    ):
        return _empty(map_name)
    raw_tactics = source.get("tactics")
    if not isinstance(raw_tactics, list):
        return _empty(map_name)
    tactics = [_compile_tactic(raw) for raw in raw_tactics]
    if any(tactic is None for tactic in tactics):
        return _empty(map_name)
    compiled = tuple(tactic for tactic in tactics if tactic is not None)
    if len({tactic.rule_id for tactic in compiled}) != len(compiled):
        return _empty(map_name)
    return CompiledTacticBook(map_name=map_name, tactics=compiled)


def load_tactic_book(map_name: str, *, database_root: Path | None = None) -> CompiledTacticBook:
    """只从 ``database/tactics/<map>.json`` 读取；任何失败均静默回退空集。"""
    if not isinstance(map_name, str) or not map_name or Path(map_name).name != map_name:
        return _empty(str(map_name))
    root = database_root or Path(__file__).resolve().parents[1] / "database"
    path = root / "tactics" / f"{map_name}.json"
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty(map_name)
    return compile_tactic_book(map_name, source)
