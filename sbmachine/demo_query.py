"""Demo 数据查询器。负责读取、解析和查询 CS2 demo 中各回合、各 tick 的玩家状态、击杀、道具等事件数据。"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sbmachine.common import read_json


def _norm_name(value: str) -> str:
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
    """基于 tools/parse_demo.py 生成的文件的小型查询层。

    Demo 的 tick 是从录像开始算起的绝对值。回合计时器锚点的转换方式如下:
    relative_sec = 115 - timer_seconds
    absolute_tick = freeze_end_tick + relative_sec * tick_rate

    tick_rate 必须来自 demo_meta.json/header。故意不硬编码,因为不同的 CS demo 可能使用不同的 tick rate。
    """

    def __init__(self, parsed_dir: Path) -> None:
        self.parsed_dir = parsed_dir
        self.meta: dict = {}
        self.rounds: list[dict] = []
        self.roster: list[dict] = []
        self.kills: list[dict] = []
        self.grenades: list[dict] = []
        self.damages: list[dict] = []
        self.smokes: list[dict] = []
        self.infernos: list[dict] = []
        self.flashes: list[dict] = []
        self.callouts: dict[str, str] = {}
        self._ticks_df: Any | None = None
        self._tick_values: list[int] = []

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
        target = _norm_name(ocr_name)
        if not target:
            return PlayerMatch("", "", 0.0)
        best = PlayerMatch("", "", 0.0)
        for player in self.roster:
            name = str(player.get("name", ""))
            candidate = _norm_name(name)
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
        if 1 <= int(round_no) <= len(self.rounds):
            return self.rounds[int(round_no) - 1]
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

    def state_at(self, tick: int) -> list[dict]:
        self._ensure_ticks()
        if self._ticks_df is None or not self._tick_values:
            return []
        target = int(tick)
        pos = bisect.bisect_left(self._tick_values, target)
        candidates = []
        if pos < len(self._tick_values):
            candidates.append(self._tick_values[pos])
        if pos > 0:
            candidates.append(self._tick_values[pos - 1])
        nearest = min(candidates, key=lambda value: abs(value - target)) if candidates else target
        rows = self._ticks_df[self._ticks_df["tick"] == nearest]
        return [self._row_to_dict(row) for _, row in rows.iterrows()]

    def callout_of(self, x: float, y: float, z: float = 0.0) -> str:
        # Go parser 已将区域名称写入每个 tick 的 'callout' 列;
        # 保留此方法兼容 API,state_at() 返回的行已包含 'callout'。
        return ""

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

    # ── lineup KB lookup ──

    def lookup_lineup(
        self,
        nade_type: str,
        from_callout: str,
        to_callout: str,
        *,
        lineups_dir: "Path | None" = None,
    ) -> dict | None:
        """Match a grenade throw to a named lineup entry.

        Matching is exact on (nade_type, from_callout, to_callout) with
        case-insensitive substring tolerance:
          - any entry whose from_callout is a substring of from_callout (or vice-versa)
          - and same for to_callout
        Returns first match or None.
        """
        if lineups_dir is None:
            # default: database/lineups/ next to repo root
            lineups_dir = Path(__file__).resolve().parents[1] / "database" / "lineups"
        map_name = self.map_name or "unknown"
        lineup_file = Path(lineups_dir) / f"{map_name}.json"
        if not lineup_file.exists():
            return None
        try:
            entries: list[dict] = json.loads(lineup_file.read_text(encoding="utf-8"))
        except Exception:
            return None
        nt = nade_type.lower().replace("grenade", "he").replace("incendiary", "molotov")
        fc = from_callout.lower()
        tc = to_callout.lower()
        for entry in entries:
            et = entry.get("type", "").lower()
            ef = entry.get("from_callout", "").lower()
            eto = entry.get("to_callout", "").lower()
            if et != nt:
                continue
            if (ef in fc or fc in ef) and (eto in tc or tc in eto):
                return entry
        return None

    # ── new event query methods (demoinfocs-golang outputs) ──

    def damages_between(self, tick_a: int, tick_b: int) -> list[dict]:
        lo, hi = sorted((int(tick_a), int(tick_b)))
        return [d for d in self.damages if lo < int(d.get("tick", -1)) <= hi]

    def smokes_in_round(self, round_no: int) -> list[dict]:
        return [s for s in self.smokes if int(s.get("round_no", 0)) == int(round_no)]

    def smokes_active_at(self, tick: int) -> list[dict]:
        """Smokes active (started, not yet expired) at the given tick."""
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
        """Infernos still burning at the given tick."""
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
        self.meta = self._read_optional_json("demo_meta.json", {})
        self.rounds = list(self._read_optional_json("rounds.json", []))
        self.roster = list(self._read_optional_json("roster.json", []))
        self.kills = list(self._read_optional_json("kills.json", []))
        self.grenades = list(self._read_optional_json("grenades.json", []))
        self.damages = list(self._read_optional_json("damages.json", []))
        self.smokes = list(self._read_optional_json("smokes.json", []))
        self.infernos = list(self._read_optional_json("infernos.json", []))
        self.flashes = list(self._read_optional_json("flashes.json", []))
        callouts = self._read_optional_json("callouts.json", {})
        self.callouts = callouts if isinstance(callouts, dict) else {}

    def _read_optional_json(self, name: str, default: Any) -> Any:
        path = self.parsed_dir / name
        if not path.exists():
            return default
        val = read_json(path)
        return default if val is None else val

    def _ensure_ticks(self) -> None:
        if self._ticks_df is not None:
            return
        parquet_path = self.parsed_dir / "ticks.parquet"
        jsonl_path = self.parsed_dir / "ticks.jsonl"
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("DemoQuery.state_at requires pandas") from exc
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif jsonl_path.exists():
            rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(columns=["tick"])
        if "tick" in df.columns and not df.empty:
            df = df.copy()
            df["tick"] = df["tick"].astype(int)
            self._tick_values = sorted(int(v) for v in df["tick"].drop_duplicates().tolist())
        self._ticks_df = df

    @staticmethod
    def _row_to_dict(row: Any) -> dict:
        out = {}
        for key, value in row.to_dict().items():
            if hasattr(value, "item"):
                value = value.item()
            out[key] = value
        return out
