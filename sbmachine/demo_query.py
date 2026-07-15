"""Demo 数据查询器。负责读取、解析和查询 CS2 demo 中各回合、各 tick 的玩家状态、击杀、道具等事件数据。"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sbmachine.common import read_json
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
        self._ticks_df: Any | None = None
        self._tick_values: list[int] = []
        self._tick_values_by_round: dict[int, list[int]] = {}

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

    def roster_names(self) -> list[tuple[str, str]]:
        return [(str(p.get("steamid", "")), str(p.get("name", ""))) for p in self.roster]

    def match_player(self, ocr_name: str) -> PlayerMatch:
        target = _normalize_name(ocr_name)
        if len(target) < 2:
            return PlayerMatch("", "", 0.0)
        best = PlayerMatch("", "", 0.0)
        for player in self.roster:
            name = str(player.get("name", ""))
            candidate = _normalize_name(name)
            if not candidate:
                continue
            score = SequenceMatcher(None, target, candidate).ratio()
            if target in candidate or candidate in target:
                score = max(score, 0.9)
            if score > best.score:
                best = PlayerMatch(str(player.get("steamid", "")), name, float(score))
        return best

    def round_by_no(self, round_no: int) -> dict:
        for item in self.rounds:
            if int(item.get("round_no", 0)) == int(round_no):
                return item
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

    # ── 新增事件查询方法（demoinfocs-golang 输出）──

    def damages_between(self, tick_a: int, tick_b: int) -> list[dict]:
        lo, hi = sorted((int(tick_a), int(tick_b)))
        return [d for d in self.damages if lo < int(d.get("tick", -1)) <= hi]

    def smokes_in_round(self, round_no: int) -> list[dict]:
        return [s for s in self.smokes if int(s.get("round_no", 0)) == int(round_no)]

    def smokes_active_at(self, tick: int) -> list[dict]:
        """返回在给定 tick 处仍处于活动状态（已生成、尚未消散）的烟雾。"""
        result = []
        for s in self.smokes:
            start = s.get("start_tick")
            end = s.get("end_tick")
            if start is None:
                continue
            if int(start) <= int(tick) and (end is None or int(tick) <= int(end)):
                result.append(s)
        return result

    def infernos_in_round(self, round_no: int) -> list[dict]:
        return [i for i in self.infernos if int(i.get("round_no", 0)) == int(round_no)]

    def infernos_active_at(self, tick: int) -> list[dict]:
        """返回在给定 tick 处仍在燃烧的火焰。"""
        result = []
        for inf in self.infernos:
            start = inf.get("start_tick")
            end = inf.get("end_tick")
            if start is None:
                continue
            if int(start) <= int(tick) and (end is None or int(tick) <= int(end)):
                result.append(inf)
        return result

    def flashes_between(self, tick_a: int, tick_b: int) -> list[dict]:
        lo, hi = sorted((int(tick_a), int(tick_b)))
        return [f for f in self.flashes if lo < int(f.get("tick", -1)) <= hi]

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

    def _read_required_json(self, name: str, expected_type: type) -> Any:
        path = self.parsed_dir / name
        val = read_json(path)
        if not isinstance(val, expected_type):
            raise ValueError(f"{name} must contain a {expected_type.__name__}")
        return val

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
