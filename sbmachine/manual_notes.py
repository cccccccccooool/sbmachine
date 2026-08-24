"""人工逐局笔记的 fail-silent 加载器。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_manual_notes(demo_id: str | None) -> dict[int, str]:
    """返回按 DEM 回合号索引的笔记；文件或 schema 不可信时不注入。"""
    if not isinstance(demo_id, str) or not demo_id.strip():
        return {}
    path = _PROJECT_ROOT / "database" / "match_notes" / f"{demo_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        rounds = value["rounds"]
        if not isinstance(rounds, dict):
            return {}
        notes: dict[int, str] = {}
        for raw_round, entry in rounds.items():
            round_no = int(raw_round)
            if round_no <= 0 or not isinstance(entry, dict):
                return {}
            note = entry.get("note")
            tactic_id = entry.get("tactic_id")
            if not isinstance(note, str) or not note.strip() or tactic_id is not None and not isinstance(tactic_id, str):
                return {}
            notes[round_no] = note.strip()
        return notes
    except Exception:
        return {}


def lookup_manual_note(
    notes: dict[int, str],
    *,
    round_no: int,
    demo_round_hint: int | str | None,
) -> str | None:
    try:
        demo_round_no = int(demo_round_hint)
    except (TypeError, ValueError):
        return None
    if demo_round_no <= 0:
        return None

    selected = notes.get(demo_round_no)
    video_key_note = notes.get(round_no)
    if round_no != demo_round_no and video_key_note is not None and video_key_note != selected:
        print(
            f"[manual_notes] video round {round_no} has a conflicting note; "
            f"using demo round hint {demo_round_no}",
            file=sys.stderr,
        )
    return selected
