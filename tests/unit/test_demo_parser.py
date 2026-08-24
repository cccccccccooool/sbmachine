from types import SimpleNamespace

import pytest

from tools.demo import parse_demo
from tools.demo.parse_demo import DemoEntityStateError, _commit_output, _validated_tick_row_count


def test_empty_tick_artifact_is_reported_as_entity_state_failure(tmp_path):
    (tmp_path / "ticks.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(DemoEntityStateError, match="no participant snapshots"):
        parse_demo.jsonl_to_parquet(tmp_path)


def test_empty_roster_is_reported_as_entity_state_failure(tmp_path):
    (tmp_path / "ticks.jsonl").write_text('{"tick": 1}\n', encoding="utf-8")
    (tmp_path / "roster.json").write_text("[]", encoding="utf-8")

    with pytest.raises(DemoEntityStateError, match="roster.json has no players"):
        _validated_tick_row_count(tmp_path)


def test_commit_output_replaces_only_after_staging_is_complete(tmp_path):
    output = tmp_path / "demo"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".demo.parse-test"
    staging.mkdir()
    (staging / "demo_manifest.json").write_text('{"status":"complete"}', encoding="utf-8")

    _commit_output(staging, output)

    assert not staging.exists()
    assert not (output / "old.txt").exists()
    assert (output / "demo_manifest.json").is_file()


def test_failed_parser_does_not_replace_previous_output(tmp_path, monkeypatch):
    output = tmp_path / "demo"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    binary = tmp_path / "parse_demo_go.exe"
    binary.write_bytes(b"fixture")
    source = tmp_path / "match.dem"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(parse_demo, "_go_binary", lambda: binary)
    monkeypatch.setattr(parse_demo.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=7))
    monkeypatch.setattr(
        parse_demo.sys,
        "argv",
        ["parse_demo.py", "--demo", str(source), "--output-dir", str(output)],
    )

    assert parse_demo.main() == 7
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"


def test_failed_parquet_conversion_does_not_replace_previous_output(tmp_path, monkeypatch):
    output = tmp_path / "demo"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    binary = tmp_path / "parse_demo_go.exe"
    binary.write_bytes(b"fixture")
    source = tmp_path / "match.dem"
    source.write_bytes(b"fixture")

    def fail_conversion(output_dir):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(parse_demo, "_go_binary", lambda: binary)
    monkeypatch.setattr(parse_demo.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(parse_demo, "jsonl_to_parquet", fail_conversion)
    monkeypatch.setattr(
        parse_demo.sys,
        "argv",
        ["parse_demo.py", "--demo", str(source), "--output-dir", str(output)],
    )

    assert parse_demo.main() == 1
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"


def test_backup_cleanup_failure_does_not_report_committed_output_as_failed(tmp_path, monkeypatch):
    output = tmp_path / "demo"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".demo.parse-test"
    staging.mkdir()
    (staging / "demo_manifest.json").write_text('{"status":"complete"}', encoding="utf-8")
    real_rmtree = parse_demo.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if ".backup-" in path.name:
            raise OSError("cleanup failed")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(parse_demo.shutil, "rmtree", fail_backup_cleanup)

    _commit_output(staging, output)

    assert (output / "demo_manifest.json").is_file()
    assert not (output / "old.txt").exists()
