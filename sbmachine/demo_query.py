"""Demo 数据查询器。负责读取、解析和查询 CS2 demo 中各回合、各 tick 的玩家状态、击杀、道具等事件数据。"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sbmachine.common import read_json
from sbmachine.utility_projection import project_grenades
from tools.demo.demo_manifest import validate_demo_manifest


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _timer_to_seconds(timer: str) -> float | None:
    """将计时器字符串 (M:SS) 转换为秒数。"""
    text = str(timer or "").strip()
    if not text:
        return None
    text = text.replace(":", ":")
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        return None
    if seconds < 0 or seconds >= 60:
        return None
    return float(minutes * 60 + seconds)


class _IntervalTree:
    """只读区间树，用于道具活跃查询（interval stabbing）。

    每个区间是 [start, end]（闭区间），end 为 None 表示右侧无限延伸（如未消散的
    烟雾/火焰）。给定一个 tick，返回所有覆盖它的区间；命中结果按区间录入顺序
    （payload_index）返回，以复刻原线性扫描的输出顺序。

    实现为按区间中点划分的平衡二叉树，每个节点缓存子树内所有区间的最大右端点
    （max_end），从而把点覆盖查询从 O(n) 降到 O(log n + k)。
    """

    __slots__ = ("_starts", "_ends", "_indices", "_left", "_right", "_center", "_max_end")

    def __init__(self, intervals: list[tuple[float, float | None, int]]) -> None:
        # intervals: (start, end, payload_index)；end 为 None 视为 +∞。
        self._starts: list[float] = []
        self._ends: list[float | None] = []
        self._indices: list[int] = []
        self._left: _IntervalTree | None = None
        self._right: _IntervalTree | None = None
        self._center: float = 0.0
        self._max_end: float = float("-inf")
        self._build(intervals)

    def _build(self, intervals: list[tuple[float, float | None, int]]) -> None:
        if not intervals:
            return
        midpoints = sorted(
            interval[0] if interval[1] is None else (interval[0] + interval[1]) / 2.0
            for interval in intervals
        )
        self._center = midpoints[len(midpoints) // 2]
        left_side: list[tuple[float, float | None, int]] = []
        right_side: list[tuple[float, float | None, int]] = []
        here: list[tuple[float, float | None, int]] = []
        for start, end, payload_index in intervals:
            if end is not None and end < self._center:
                left_side.append((start, end, payload_index))
            elif start > self._center:
                right_side.append((start, end, payload_index))
            else:
                here.append((start, end, payload_index))
        # 落在中心的区间按录入顺序保留，用于后续按 payload_index 复原输出顺序。
        for start, end, payload_index in here:
            self._starts.append(start)
            self._ends.append(end)
            self._indices.append(payload_index)
        self._max_end = max(
            (float("inf") if end is None else end) for _, end, _ in here
        ) if here else float("-inf")
        if left_side:
            self._left = _IntervalTree(left_side)
            self._max_end = max(self._max_end, self._left._max_end)
        if right_side:
            self._right = _IntervalTree(right_side)
            self._max_end = max(self._max_end, self._right._max_end)

    def stab(self, point: float) -> list[int]:
        """返回覆盖 point 的所有区间的 payload_index，已按录入顺序升序排列。"""
        hits: list[int] = []
        self._collect(point, hits)
        hits.sort()
        return hits

    def _collect(self, point: float, hits: list[int]) -> None:
        # point 超过整棵子树的最大右端点时不可能命中，直接剪枝。
        if point > self._max_end:
            return
        for start, end, payload_index in zip(self._starts, self._ends, self._indices):
            if start <= point and (end is None or point <= end):
                hits.append(payload_index)
        # 左子树全部区间的右端点 < center，右子树全部区间的左端点 > center，
        # 因此 point 只需按它与 center 的关系向一侧下降。
        if point < self._center:
            if self._left is not None:
                self._left._collect(point, hits)
        elif point > self._center:
            if self._right is not None:
                self._right._collect(point, hits)


@dataclass(frozen=True)
class PlayerMatch:
    steamid: str
    name: str
    score: float

    def as_dict(self) -> dict:
        return {"steamid": self.steamid, "name": self.name, "score": self.score}


class DemoQuery:
    """基于 tools/demo/parse_demo.py 生成的文件的小型查询层。

    Demo 的 tick 是从录像开始算起的绝对值。回合计时器锚点的转换方式如下:
    relative_sec = 115 - timer_seconds
    absolute_tick = freeze_end_tick + relative_sec * tick_rate

    tick_rate 必须来自 demo_meta.json/header。故意不硬编码,因为不同的 CS demo 可能使用不同的 tick rate。
    """

    def __init__(self, parsed_dir: Path) -> None:
        self.parsed_dir = parsed_dir
        self.manifest: dict = {}
        self.meta: dict = {}
        self.rounds: list[dict] = []
        self.roster: list[dict] = []
        self.kills: list[dict] = []
        self.grenades: list[dict] = []
        self.damages: list[dict] = []
        self.smokes: list[dict] = []
        self.infernos: list[dict] = []
        self.flashes: list[dict] = []
        self.fired: list[dict] = []
        self.equips: list[dict] = []
        self.event_snapshots: list[dict] = []
        self._ticks_df: Any | None = None
        self._tick_values: list[int] = []
        self._tick_values_by_round: dict[int, list[int]] = {}
        # 道具活跃查询的区间树索引，惰性构建。值为 (源列表对象, 区间树)，
        # 一旦底层列表被替换（如测试直接赋值）就凭对象身份检测并重建，避免陈旧索引。
        self._interval_trees: dict[str, tuple[list[dict], _IntervalTree]] = {}
        # round_no → 回合字典 的惰性索引。存 (源列表对象, 映射)，一旦 self.rounds
        # 被替换就凭对象身份检测并重建。重复 round_no 保留第一个，复刻线性扫描语义。
        self._round_index: tuple[list[dict], dict[int, dict]] | None = None
        self._utility_projection_cache: tuple[
            list[dict], list[dict], list[dict], list[dict]
        ] | None = None
        # roster 名字归一化结果的惰性缓存。存 (源列表对象, [(steamid, name, 归一化名)])，
        # match_player 每帧调用却对不变的 roster 反复归一化，按对象身份缓存避免重算。
        self._roster_normalized: tuple[list[dict], list[tuple[str, str, str]]] | None = None
        # roster 归一化名的惰性缓存。match_player 在逐帧循环里被反复调用，而 roster
        # 不变，故预归一化一次。存 (源列表对象, [(steamid, name, 归一化名)])，
        # self.roster 被替换时凭对象身份检测并重建。
        self._roster_normalized: tuple[list[dict], list[tuple[str, str, str]]] | None = None

    @classmethod
    def load(cls, parsed_dir: str | Path) -> "DemoQuery":
        query = cls(Path(parsed_dir))
        query._load()
        return query

    @property
    def tick_rate(self) -> float:
        value = self.meta.get("tick_rate") or self.meta.get("tickrate")
        if value is None:
            raise ValueError("demo_meta.json is missing tick_rate")
        return float(value)

    @property
    def map_name(self) -> str:
        return str(self.meta.get("map_name", ""))

    @property
    def capabilities(self) -> dict[str, bool]:
        raw = self.meta.get("capabilities")
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): bool(value)
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, bool)
        }

    def roster_names(self) -> list[tuple[str, str]]:
        return [(str(p.get("steamid", "")), str(p.get("name", ""))) for p in self.roster]

    def _normalized_roster(self) -> list[tuple[str, str, str]]:
        """返回 [(steamid, name, 归一化名)]，按 self.roster 对象身份缓存。"""
        cached = self._roster_normalized
        if cached is None or cached[0] is not self.roster:
            normalized = [
                (str(player.get("steamid", "")), str(player.get("name", "")), _normalize_name(str(player.get("name", ""))))
                for player in self.roster
            ]
            self._roster_normalized = (self.roster, normalized)
            cached = self._roster_normalized
        return cached[1]

    def match_player(self, ocr_name: str) -> PlayerMatch:
        target = _normalize_name(ocr_name)
        if len(target) < 2:
            return PlayerMatch("", "", 0.0)
        best = PlayerMatch("", "", 0.0)
        for steamid, name, candidate in self._normalized_roster():
            if not candidate:
                continue
            score = SequenceMatcher(None, target, candidate).ratio()
            if target in candidate or candidate in target:
                score = max(score, 0.9)
            if score > best.score:
                best = PlayerMatch(steamid, name, float(score))
        return best

    def round_by_no(self, round_no: int) -> dict:
        cached = self._round_index
        if cached is None or cached[0] is not self.rounds:
            index: dict[int, dict] = {}
            for item in self.rounds:
                key = int(item.get("round_no", 0))
                if key not in index:   # 重复 round_no 保留第一个，复刻原线性扫描语义
                    index[key] = item
            self._round_index = (self.rounds, index)
            cached = self._round_index
        try:
            return cached[1][int(round_no)]
        except KeyError:
            raise IndexError(f"round_no not found in parsed demo: {round_no}")

    def tick_at(self, round_no: int, relative_sec: float) -> int:
        round_meta = self.round_by_no(round_no)
        freeze_end_tick = int(round_meta.get("freeze_end_tick", round_meta.get("start_tick", 0)))
        return int(round(freeze_end_tick + float(relative_sec) * self.tick_rate))

    def tick_from_timer(self, round_no: int, timer: str) -> int | None:
        timer_sec = _timer_to_seconds(timer)
        if timer_sec is None:
            return None
        return self.tick_at(round_no, 115.0 - timer_sec)

    def state_at(
        self,
        tick: int,
        round_no: int,
        max_distance_ticks: int | None = None,
    ) -> list[dict]:
        self._ensure_ticks()
        if self._ticks_df is None or not self._tick_values:
            return []
        target = int(tick)
        current_round = int(round_no)
        self.round_by_no(current_round)
        round_ticks = self._tick_values_by_round.get(current_round, [])
        if not round_ticks:
            return []
        if max_distance_ticks is None:
            max_distance_ticks = max(1, int(round(self.tick_rate)))
        max_distance = int(max_distance_ticks)
        if max_distance < 0:
            raise ValueError("max_distance_ticks must be non-negative")
        pos = bisect.bisect_left(round_ticks, target)
        candidates = []
        if pos > 0:
            candidates.append(round_ticks[pos - 1])
        if pos < len(round_ticks):
            candidates.append(round_ticks[pos])
        nearest = min(candidates, key=lambda value: abs(value - target)) if candidates else target
        if abs(nearest - target) > max_distance:
            return []
        rows = self._ticks_df[
            (self._ticks_df["round_no"] == current_round)
            & (self._ticks_df["tick"] == nearest)
        ]
        return [self._row_to_dict(row) for _, row in rows.iterrows()]

    def kills_between(self, tick_a: int, tick_b: int) -> list[dict]:
        lo, hi = sorted((int(tick_a), int(tick_b)))
        return [kill for kill in self.kills if lo < int(kill.get("tick", -1)) <= hi]

    def kills_in_round(self, round_no: int) -> list[dict]:
        return [k for k in self.kills if int(k.get("round_no", 0)) == int(round_no)]

    def utilities_between(self, tick_a: int, tick_b: int) -> list[dict]:
        """返回 [tick_a, tick_b] 区间内投掷或爆开的道具事件(来自 grenades.json)。"""
        lo, hi = sorted((int(tick_a), int(tick_b)))
        result = []
        for g in self.grenades:
            throw = g.get("throw_tick")
            det = g.get("det_tick")
            if throw is not None and lo <= int(throw) <= hi:
                result.append({**g, "_event": "throw"})
            elif det is not None and lo <= int(det) <= hi:
                result.append({**g, "_event": "detonate"})
        return result

    def utility_throws_between(self, tick_a: int, tick_b: int) -> list[dict]:
        """返回稳定最小投影；同一 grenades.json 记录永远只产生一条 throw。"""
        cached = self._utility_projection_cache
        if (
            cached is None
            or cached[0] is not self.grenades
            or cached[1] is not self.smokes
            or cached[2] is not self.infernos
        ):
            cached = (
                self.grenades,
                self.smokes,
                self.infernos,
                project_grenades(
                    self.grenades, smokes=self.smokes, infernos=self.infernos
                ),
            )
            self._utility_projection_cache = cached
        lo, hi = sorted((int(tick_a), int(tick_b)))
        return [
            event
            for event in cached[3]
            if event.get("throw_tick") is not None
            and lo <= int(event["throw_tick"]) <= hi
        ]

    # ── 新增事件查询方法（demoinfocs-golang 输出）──

    def damages_between(self, tick_a: int, tick_b: int) -> list[dict]:
        lo, hi = sorted((int(tick_a), int(tick_b)))
        return [d for d in self.damages if lo < int(d.get("tick", -1)) <= hi]

    def smokes_in_round(self, round_no: int) -> list[dict]:
        return [s for s in self.smokes if int(s.get("round_no", 0)) == int(round_no)]

    def smokes_active_at(self, tick: int) -> list[dict]:
        """返回在给定 tick 处仍处于活动状态（已生成、尚未消散）的烟雾。"""
        return self._active_at(self.smokes, "_smoke_tree", tick)

    def infernos_in_round(self, round_no: int) -> list[dict]:
        return [i for i in self.infernos if int(i.get("round_no", 0)) == int(round_no)]

    def infernos_active_at(self, tick: int) -> list[dict]:
        """返回在给定 tick 处仍在燃烧的火焰。"""
        return self._active_at(self.infernos, "_inferno_tree", tick)

    def _active_at(self, events: list[dict], cache_attr: str, tick: int) -> list[dict]:
        """区间点覆盖查询：返回覆盖 tick 的所有区间事件，顺序与录入顺序一致。

        首次查询时惰性构建区间树并缓存；若底层事件列表对象发生替换（例如测试
        直接改写 demo.smokes），则重建索引。start_tick 缺失的记录被跳过，
        end_tick 缺失表示持续生效。
        """
        cached = self._interval_trees.get(cache_attr)
        if cached is None or cached[0] is not events:
            intervals: list[tuple[float, float | None, int]] = []
            for payload_index, event in enumerate(events):
                start = event.get("start_tick")
                if start is None:
                    continue
                end = event.get("end_tick")
                intervals.append((int(start), None if end is None else int(end), payload_index))
            self._interval_trees[cache_attr] = (events, _IntervalTree(intervals))
            cached = self._interval_trees[cache_attr]
        tree = cached[1]
        point = int(tick)
        return [events[index] for index in tree.stab(point)]

    def flashes_between(self, tick_a: int, tick_b: int) -> list[dict]:
        lo, hi = sorted((int(tick_a), int(tick_b)))
        return [f for f in self.flashes if lo < int(f.get("tick", -1)) <= hi]

    def weapon_fires_between(self, tick_a: int, tick_b: int) -> list[dict]:
        lo, hi = sorted((int(tick_a), int(tick_b)))
        return [row for row in self.fired if lo < int(row.get("tick", -1)) <= hi]

    def item_equips_between(self, tick_a: int, tick_b: int) -> list[dict]:
        lo, hi = sorted((int(tick_a), int(tick_b)))
        return [row for row in self.equips if lo < int(row.get("tick", -1)) <= hi]

    def event_snapshots_for_kills(self, kills: list[dict]) -> list[dict]:
        event_ticks = {
            int(kill["tick"])
            for kill in kills
            if isinstance(kill, dict) and kill.get("tick") is not None
        }
        if not event_ticks:
            return []
        return [
            row
            for row in self.event_snapshots
            if str(row.get("event_kind") or "") == "kill"
            and int(row.get("event_tick", -1)) in event_ticks
        ]

    def _load(self) -> None:
        self.manifest = validate_demo_manifest(self.parsed_dir)
        self.meta = self._read_required_json("demo_meta.json", dict)
        self.rounds = self._read_required_json("rounds.json", list)
        self.roster = self._read_required_json("roster.json", list)
        self.kills = self._read_required_json("kills.json", list)
        self.grenades = self._read_required_json("grenades.json", list)
        self.damages = self._read_required_json("damages.json", list)
        self.smokes = self._read_required_json("smokes.json", list)
        self.infernos = self._read_required_json("infernos.json", list)
        self.flashes = self._read_required_json("flashes.json", list)
        self.fired = self._read_optional_json("fired.json", "weapon_fire")
        self.equips = self._read_optional_json("equips.json", "item_equip")
        self.event_snapshots = self._read_optional_json(
            "event_snapshots.json", "event_snapshots"
        )

    def _read_required_json(self, name: str, expected_type: type) -> Any:
        path = self.parsed_dir / name
        val = read_json(path)
        if not isinstance(val, expected_type):
            raise ValueError(f"{name} must contain a {expected_type.__name__}")
        return val

    def _read_optional_json(self, name: str, capability: str) -> list[dict]:
        """只读取 manifest 已校验的增量文件；能力已声明但文件缺失时拒绝降级。"""
        manifest_files = self.manifest.get("files") or {}
        declared = isinstance(manifest_files, dict) and name in manifest_files
        if not declared:
            if self.capabilities.get(capability) is True:
                raise ValueError(
                    f"demo_meta.json declares capability {capability!r}, "
                    f"but demo_manifest.json does not contain {name}"
                )
            return []
        return self._read_required_json(name, list)

    def _ensure_ticks(self) -> None:
        if self._ticks_df is not None:
            return
        manifest_files = self.manifest.get("files", {})
        parquet_path = self.parsed_dir / "ticks.parquet"
        jsonl_path = self.parsed_dir / "ticks.jsonl"
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("DemoQuery.state_at requires pandas") from exc
        if "ticks.parquet" in manifest_files:
            df = pd.read_parquet(parquet_path)
        elif "ticks.jsonl" in manifest_files:
            rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            df = pd.DataFrame(rows)
        else:
            raise ValueError("validated demo manifest has no tick artifact")
        if "tick" in df.columns and not df.empty:
            if "round_no" not in df.columns:
                raise ValueError("tick artifact is missing round_no")
            df = df.copy()
            df["tick"] = df["tick"].astype(int)
            df["round_no"] = df["round_no"].astype(int)
            self._tick_values = sorted(int(v) for v in df["tick"].drop_duplicates().tolist())
            for round_no, rows in df.groupby("round_no"):
                self._tick_values_by_round[int(round_no)] = sorted(
                    int(value) for value in rows["tick"].drop_duplicates().tolist()
                )
        self._ticks_df = df

    @staticmethod
    def _row_to_dict(row: Any) -> dict:
        out = {}
        for key, value in row.to_dict().items():
            if hasattr(value, "item"):
                value = value.item()
            out[key] = value
        return out
