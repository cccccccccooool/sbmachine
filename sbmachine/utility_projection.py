"""把解析器道具记录压成规则层可消费的稳定最小事实。"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


_KIND_ALIASES = {
    "smoke": "smoke",
    "smoke grenade": "smoke",
    "flash": "flash",
    "flashbang": "flash",
    "he": "he",
    "he grenade": "he",
    "high explosive grenade": "he",
    "molotov": "molotov",
    "incendiary": "molotov",
    "incendiary grenade": "molotov",
    "decoy": "decoy",
    "decoy grenade": "decoy",
}


def normalize_utility_kind(value: object) -> str:
    """返回稳定的道具 kind；未知值保留规范化文本，证据不足时不猜类型。"""
    text = str(value or "").replace("_", " ").strip().casefold()
    return _KIND_ALIASES.get(text, text or "unknown")


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _number_or_none(value: object) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _xyz(record: dict, prefix: str) -> list[int | float | None] | None:
    values = [
        _number_or_none(record.get(f"{prefix}_{axis}"))
        for axis in ("x", "y", "z")
    ]
    return values if any(value is not None for value in values) else None


def _fallback_stable_id(record: dict, ordinal: int) -> str:
    """旧产物缺 stable_event_id 时生成仅用于本次输入的确定性兼容键。"""
    return "|".join(
        (
            str(_int_or_none(record.get("round_no")) or 0),
            str(_int_or_none(record.get("throw_tick")) or 0),
            str(
                record.get("thrower_steamid")
                or record.get("thrower_steam")
                or record.get("thrower")
                or "unknown"
            ),
            str(record.get("type") or record.get("grenade_type") or "unknown"),
            str(max(1, int(ordinal))),
        )
    )


def project_grenade(record: dict, *, ordinal: int = 1) -> dict:
    """将一条 grenades.json 记录投影为一条、且只一条 throw 事件。"""
    raw_type = str(record.get("type") or record.get("grenade_type") or "")
    stable_id = str(record.get("stable_event_id") or "").strip()
    if not stable_id:
        stable_id = _fallback_stable_id(record, ordinal)
    throw_tick = _int_or_none(record.get("throw_tick"))
    effect_tick = _int_or_none(
        record.get("effect_tick", record.get("det_tick"))
    )
    projected = {
        "_event": "throw",
        "event_id": f"utility_throw:{stable_id}",
        "stable_event_id": stable_id,
        "dedup_key": stable_id,
        "round_no": _int_or_none(record.get("round_no")),
        "kind": normalize_utility_kind(raw_type),
        # type 保留解析器原名，兼容 tactic_matcher 与既有规则配置。
        "type": raw_type,
        "thrower": record.get("thrower"),
        "thrower_steamid": (
            record.get("thrower_steamid") or record.get("thrower_steam")
        ),
        "throw_tick": throw_tick,
        "effect_tick": effect_tick,
        "throw_xyz": _xyz(record, "throw_pos"),
        "landing_xyz": _xyz(record, "dest"),
        # 随机 entity_id 只作诊断字段，不再参与稳定身份。
        "entity_id": record.get("entity_id"),
    }
    return {key: value for key, value in projected.items() if value is not None}


def _effect_position(record: dict, kind: str) -> tuple[float, float] | None:
    keys = ("pos_x", "pos_y") if kind == "smoke" else ("centroid_x", "centroid_y")
    values = [_number_or_none(record.get(key)) for key in keys]
    if all(isinstance(value, (int, float)) for value in values):
        return float(values[0]), float(values[1])
    return None


def _match_effect(
    grenade: dict,
    effects: list[dict],
    *,
    kind: str,
    used: set[int],
) -> tuple[int | None, str | None]:
    throw_tick = _int_or_none(grenade.get("throw_tick"))
    destination = _xyz(grenade, "dest")
    target_xy = (
        (float(destination[0]), float(destination[1]))
        if destination and destination[0] is not None and destination[1] is not None
        else None
    )
    candidates: list[tuple[float, int, int, int]] = []
    for index, effect in enumerate(effects):
        if index in used:
            continue
        if _int_or_none(effect.get("round_no")) != _int_or_none(grenade.get("round_no")):
            continue
        if str(effect.get("thrower") or "") != str(grenade.get("thrower") or ""):
            continue
        start_tick = _int_or_none(effect.get("start_tick"))
        if start_tick is None or (throw_tick is not None and start_tick < throw_tick):
            continue
        effect_xy = _effect_position(effect, kind)
        distance = (
            math.hypot(effect_xy[0] - target_xy[0], effect_xy[1] - target_xy[1])
            if effect_xy is not None and target_xy is not None
            else float("inf")
        )
        tick_gap = start_tick - (throw_tick or start_tick)
        candidates.append((distance, tick_gap, index, start_tick))
    if not candidates:
        return None, None
    distance, _, index, start_tick = min(candidates)
    # 有坐标时要求落点接近；无坐标才按时间序 fail-soft 配对。
    if math.isfinite(distance) and distance > 256.0:
        return None, None
    used.add(index)
    return start_tick, "smokes.json" if kind == "smoke" else "infernos.json"


def project_grenades(
    records: Iterable[dict],
    *,
    smokes: Iterable[dict] = (),
    infernos: Iterable[dict] = (),
) -> list[dict]:
    """按输入顺序批量投影，并为旧记录的同 tick 冲突补稳定序号。"""
    ordinals: defaultdict[tuple, int] = defaultdict(int)
    smoke_rows = [row for row in smokes if isinstance(row, dict)]
    inferno_rows = [row for row in infernos if isinstance(row, dict)]
    used_smokes: set[int] = set()
    used_infernos: set[int] = set()
    out: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        signature = (
            _int_or_none(record.get("round_no")),
            _int_or_none(record.get("throw_tick")),
            str(
                record.get("thrower_steamid")
                or record.get("thrower_steam")
                or record.get("thrower")
                or ""
            ),
            str(record.get("type") or record.get("grenade_type") or ""),
        )
        ordinals[signature] += 1
        projected = project_grenade(record, ordinal=ordinals[signature])
        kind = str(projected.get("kind") or "")
        effect_tick: int | None = None
        effect_source: str | None = None
        if kind == "smoke":
            effect_tick, effect_source = _match_effect(
                record, smoke_rows, kind=kind, used=used_smokes
            )
        elif kind == "molotov":
            effect_tick, effect_source = _match_effect(
                record, inferno_rows, kind=kind, used=used_infernos
            )
        if effect_tick is not None:
            projected["effect_tick"] = effect_tick
            projected["effect_source"] = effect_source
        out.append(projected)
    return sorted(
        out,
        key=lambda event: (
            _int_or_none(event.get("round_no")) or 0,
            _int_or_none(event.get("throw_tick")) or 0,
            str(event.get("stable_event_id") or ""),
        ),
    )
