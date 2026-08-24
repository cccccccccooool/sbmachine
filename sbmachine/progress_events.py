"""运行期进度 JSONL 契约。

事件仅供终端 UI 与 diagnostics 使用，绝不写入正式阶段产物。这个模块不裁决阶段
是否发布；权威的 done/canceled 仍然属于父编排器。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_EVENTS = {"stage_start", "stage_progress", "stage_work_complete", "stage_error"}
_STAGES = {"demo_parse", "video_marking", "phase1", "phase2", "phase3a", "phase3b", "phase3c", "phase4"}


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    sequence: int
    event: str
    stage: str
    completed: int | None = None
    total: int | None = None
    unit: str | None = None
    detail: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event": self.event,
            "stage": self.stage,
            "completed": self.completed,
            "total": self.total,
            "unit": self.unit,
            "detail": self.detail,
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> "ProgressEvent":
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported progress event schema")
        run_id = payload.get("run_id")
        sequence = payload.get("sequence")
        event = payload.get("event")
        stage = payload.get("stage")
        if not isinstance(run_id, str) or not run_id or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("invalid progress event identity")
        if event not in _EVENTS or stage not in _STAGES:
            raise ValueError("invalid progress event kind")
        completed, total = payload.get("completed"), payload.get("total")
        for value in (completed, total):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError("progress count must be a non-negative integer or null")
        if completed is not None and total is not None and completed > total:
            raise ValueError("completed exceeds total")
        if total == 0 and completed not in (None, 0):
            raise ValueError("zero total requires zero completed")
        unit, detail = payload.get("unit"), payload.get("detail")
        if unit is not None and (not isinstance(unit, str) or not unit):
            raise ValueError("invalid progress unit")
        if detail is not None and (not isinstance(detail, str) or len(detail) > 240):
            raise ValueError("invalid progress detail")
        return cls(run_id, sequence, event, stage, completed, total, unit, detail)


class ProgressEventWriter:
    """Best-effort 单 writer；写入失败只返回 False，绝不影响业务函数。"""

    def __init__(self, path: Path, *, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._sequence = 0

    def emit(
        self,
        *,
        event: str,
        stage: str,
        completed: int | None = None,
        total: int | None = None,
        unit: str | None = None,
        detail: str | None = None,
    ) -> bool:
        try:
            candidate = ProgressEvent(
                self.run_id, self._sequence + 1, event, stage, completed, total, unit, detail
            )
            line = json.dumps(candidate.to_mapping(), ensure_ascii=False, separators=(",", ":"))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")
                stream.flush()
            self._sequence = candidate.sequence
            return True
        except Exception:
            return False


class ProgressEventReader:
    """增量读取一个 JSONL writer；保留未换行尾部以容忍写入中的半行。"""

    def __init__(self, path: Path, *, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._offset = 0
        self._tail = ""
        self._last_sequence = 0
        self.summary = {
            "events_received": 0,
            "events_accepted": 0,
            "invalid_json": 0,
            "wrong_run_id": 0,
            "out_of_order": 0,
            "channel_failed": False,
        }

    def read_available(self) -> list[ProgressEvent]:
        try:
            if not self.path.exists():
                return []
            with self.path.open("rb") as stream:
                stream.seek(self._offset)
                chunk = stream.read()
                self._offset = stream.tell()
        except OSError:
            self.summary["channel_failed"] = True
            return []
        try:
            text = self._tail + chunk.decode("utf-8")
        except UnicodeDecodeError:
            self.summary["invalid_json"] += 1
            return []
        lines = text.split("\n")
        self._tail = lines.pop()
        accepted: list[ProgressEvent] = []
        for line in lines:
            if not line.strip():
                continue
            self.summary["events_received"] += 1
            try:
                event = ProgressEvent.from_mapping(json.loads(line))
            except (ValueError, TypeError, json.JSONDecodeError):
                self.summary["invalid_json"] += 1
                continue
            if event.run_id != self.run_id:
                self.summary["wrong_run_id"] += 1
                continue
            if event.sequence <= self._last_sequence:
                self.summary["out_of_order"] += 1
                continue
            self._last_sequence = event.sequence
            self.summary["events_accepted"] += 1
            accepted.append(event)
        return accepted
