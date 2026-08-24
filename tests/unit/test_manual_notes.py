import json

from sbmachine import manual_notes


def test_load_manual_notes_is_fail_silent_for_missing_invalid_and_schema_invalid_files(tmp_path, monkeypatch):
    monkeypatch.setattr(manual_notes, "_PROJECT_ROOT", tmp_path)
    assert manual_notes.load_manual_notes("missing-demo") == {}

    notes_path = tmp_path / "database" / "match_notes" / "demo.json"
    notes_path.parent.mkdir(parents=True)
    notes_path.write_text("not json", encoding="utf-8")
    assert manual_notes.load_manual_notes("demo") == {}

    notes_path.write_text(json.dumps({"rounds": {"14": {"note": 1}}}), encoding="utf-8")
    assert manual_notes.load_manual_notes("demo") == {}


def test_load_manual_notes_returns_only_schema_valid_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(manual_notes, "_PROJECT_ROOT", tmp_path)
    notes_path = tmp_path / "database" / "match_notes" / "demo.json"
    notes_path.parent.mkdir(parents=True)
    notes_path.write_text(
        json.dumps({"rounds": {"14": {"note": "B 点佯攻", "tactic_id": None}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert manual_notes.load_manual_notes("demo") == {14: "B 点佯攻"}
