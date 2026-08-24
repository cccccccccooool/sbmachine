import hashlib
import json
import shutil
from pathlib import Path

import pytest

from core.config_loader import ConfigError, load_config
from sbmachine.common import require_debug_output
from sbmachine.file_lock import FileLock, FileLockUnavailable
from sbmachine.preflight import (
    PublishContractError,
    preflight_config,
    validate_commentary_publishable,
)
from sbmachine import run_all
from sbmachine.neutral_contract import new_manifest_metadata, rounds_sha256, validate_neutral_manifest
from sbmachine.run_context import RunContext


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_directory_config_recursively_merges_and_isolates_train_profile(tmp_path):
    config_dir = tmp_path / "config"
    _write(
        config_dir / "pipeline.yaml",
        "profile: lite\npaths:\n  demo_output_dir: output/demo\n  rounds_json: output/sbmachine/rounds.json\ndemo:\n  parsed_dir: output/demo\n",
    )
    _write(config_dir / "llm.yaml", "paths:\n  style_skill: Prompt/skill/style_skill.md\n")
    _write(config_dir / "train.yaml", "profile: analyst\ndataset_dir: data/sft\n")

    config = load_config(config_dir)

    assert config["profile"] == "lite"
    assert config["train"]["profile"] == "analyst"
    assert config["paths"]["rounds_json"] == "output/sbmachine/rounds.json"
    assert config["paths"]["style_skill"] == "Prompt/skill/style_skill.md"
    assert config["demo"]["parsed_dir"] == config["paths"]["demo_output_dir"]


def test_directory_config_rejects_nested_scalar_conflict(tmp_path):
    config_dir = tmp_path / "config"
    _write(config_dir / "a.yaml", "runtime:\n  timeout_sec: 10\n")
    _write(config_dir / "b.yaml", "runtime:\n  timeout_sec: 20\n")

    with pytest.raises(ConfigError, match="runtime.timeout_sec"):
        load_config(config_dir)


def test_config_rejects_conflicting_demo_paths(tmp_path):
    path = _write(
        tmp_path / "pipeline.yaml",
        "paths:\n  demo_output_dir: output/demo-a\ndemo:\n  parsed_dir: output/demo-b\n",
    )

    with pytest.raises(ConfigError, match="demo path conflict"):
        load_config(path)


def test_missing_config_path_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="config path does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml_returns_truthful_dry_run_status(tmp_path):
    config_path = _write(tmp_path / "bad.yaml", "paths: [\n")

    report = run_all.run_all(config_path, dry_run=True)

    assert report["config_valid"] is False
    assert report["writes_performed"] is False
    assert report["errors"]


def test_preflight_is_read_only_and_validates_inputs_and_positive_values(tmp_path):
    config = {
        "phases": {"phase2_yolo": False, "phase3a_semantic": True, "phase3b_semantic": False, "phase4_assemble": False},
        "paths": {"rounds_with_yolo_json": "missing.json"},
        "semantic": {"window_min_sec": 0, "window_max_sec": 10},
    }

    report = preflight_config(config, root=tmp_path)

    assert report["config_valid"] is False
    assert report["writes_performed"] is False
    assert report["services_started"] == []
    assert any("semantic.window_min_sec" in error for error in report["errors"])
    assert not (tmp_path / "output").exists()


def test_phase_scoped_preflight_reports_the_requested_phase(tmp_path, fixtures_dir):
    rounds = _write(tmp_path / "rounds.json", "{}")
    demo = tmp_path / "demo"
    shutil.copytree(fixtures_dir / "demo", demo)
    config = {
        "phases": {"phase2_yolo": False},
        "paths": {"rounds_json": str(rounds)},
        "demo": {"parsed_dir": str(demo)},
        "yolo": {"yolo": {"enabled": False}},
    }

    report = preflight_config(config, root=tmp_path, only={"phase2"})

    assert report["config_valid"] is True
    assert report["enabled_phases"] == ["phase2"]


def test_video_marking_preflight_does_not_require_phase2_yolo_model(tmp_path):
    video = _write(tmp_path / "match.mp4", "")
    slicer_model = _write(tmp_path / "frame_type.pt", "")
    hud_model = _write(tmp_path / "best3.pt", "")
    config = {
        "paths": {"video": str(video)},
        "slicer": {"model": str(slicer_model)},
        "yolo": {"yolo": {"model_path": str(hud_model)}},
    }

    report = preflight_config(config, root=tmp_path, only={"video_marking"})

    assert report["config_valid"] is True
    assert not any(item["name"] == "vision_yolo_model" for item in report["required_inputs"])

    phase2_report = preflight_config(config, root=tmp_path, only={"phase2"})

    assert any(item["name"] == "vision_yolo_model" for item in phase2_report["required_inputs"])


def test_file_lock_rejects_concurrent_holder(tmp_path):
    first = FileLock(tmp_path / "pipeline.lock").acquire()
    try:
        with pytest.raises(FileLockUnavailable):
            FileLock(tmp_path / "pipeline.lock").acquire()
    finally:
        first.release()
    with FileLock(tmp_path / "pipeline.lock"):
        pass


def test_standalone_phase_cannot_target_published_output():
    with pytest.raises(ValueError, match="published output"):
        require_debug_output(Path("output/sbmachine/debug.json"), "output")


def test_commentary_contract_accepts_silent_and_style_failure_as_empty(tmp_path):
    manifest = tmp_path / "commentary.json"
    base = {
        "commentary_schema_version": 2,
        "source_neutral_run_id": "test-run",
        "source_neutral_sha256": "0" * 64,
        "source_window_count": 1,
    }
    silent_window = {
        "window_id": "r001_w01",
        "t_start": 0.0,
        "t_end": 1.0,
        "neutral_source": "intentional_empty",
        "neutral_nonempty": False,
        "style_status": "skipped_intentional_empty",
        "retry_count": 0,
        "failure_reason": None,
        "char_budget": 8,
        "output_chars": None,
        "published_scene_index": None,
    }
    manifest.write_text(json.dumps({
        **base,
        "rounds": [{
            "status": "silent",
            "window_results": [silent_window],
            "scenes": [],
            "commentary_text": "",
            "emotion_segments": [],
        }],
    }), encoding="utf-8")
    validate_commentary_publishable(manifest)

    failed_window = {
        **silent_window,
        "neutral_source": "llm",
        "neutral_nonempty": True,
        "style_status": "style_failed",
    }
    # 单窗口 100% 失败 → 每回合判为 empty，可发布（p4 跳过留空）。
    manifest.write_text(json.dumps({
        **base,
        "source_window_count": 3,
        "rounds": [
            {"status": "empty", "window_results": [{**failed_window, "window_id": "r001_w01", "failure_reason": "missing_anchor"}], "scenes": [], "commentary_text": "", "emotion_segments": []},
            {"status": "empty", "window_results": [{**failed_window, "window_id": "r002_w01", "failure_reason": "missing_anchor"}], "scenes": [], "commentary_text": "", "emotion_segments": []},
            {"status": "empty", "window_results": [{**failed_window, "window_id": "r003_w01", "failure_reason": "over_budget"}], "scenes": [], "commentary_text": "", "emotion_segments": []},
        ],
    }), encoding="utf-8")
    validate_commentary_publishable(manifest)


def test_dry_run_does_not_create_output_or_invoke_pipeline(tmp_path, monkeypatch):
    config_path = _write(
        tmp_path / "config.yaml",
        "phases:\n  demo_parse: false\n  video_marking: false\n  preprocess_slice: false\n  phase2_yolo: false\n  phase3a_semantic: false\n  phase3b_semantic: false\n  phase4_assemble: false\n",
    )
    monkeypatch.setattr(run_all, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(run_all, "_execute_pipeline", lambda *args: pytest.fail("pipeline invoked"))

    report = run_all.run_all(config_path, dry_run=True)

    assert report["config_valid"] is True
    assert report["writes_performed"] is False
    assert not (tmp_path / "output").exists()


def test_failed_run_archives_staging_and_preserves_previous_success(tmp_path, monkeypatch):
    output = tmp_path / "output"
    previous = _write(output / "sbmachine" / "rounds.json", '{"old": true}')
    before = previous.read_bytes()
    video = _write(tmp_path / "video.mp4", "video")
    segments = _write(tmp_path / "segments.json", '{"segments": []}')
    config_path = _write(
        tmp_path / "config.yaml",
        f"""
runtime:
  manage_services: true
phases:
  demo_parse: false
  video_marking: false
  preprocess_slice: true
  phase2_yolo: false
  phase3a_semantic: false
  phase3b_semantic: false
  phase4_assemble: false
paths:
  video: {video.as_posix()}
  segments_json: {segments.as_posix()}
""",
    )
    monkeypatch.setattr(run_all, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(run_all, "run_preprocess_slice", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    result = run_all.run_all(config_path)

    assert result["status"] == "failed"
    assert result["failed_stage"] == "phase1"
    assert previous.read_bytes() == before
    failure = Path(result["error_dir"]) / "failure.json"
    failure_payload = json.loads(failure.read_text(encoding="utf-8"))
    assert failure_payload["previous_success_preserved"] is True
    assert failure_payload["exception_type"] == "RuntimeError"
    assert "boom" in failure_payload["traceback"]
    assert not (output / ".staging" / result["run_id"]).exists()


def test_upstream_checkpoint_survives_later_phase_failure(tmp_path, monkeypatch, fixtures_dir):
    output = tmp_path / "output"
    video = _write(tmp_path / "video.mp4", "video")
    segments = _write(tmp_path / "segments.json", '{"segments": []}')
    demo = tmp_path / "demo"
    shutil.copytree(fixtures_dir / "demo", demo)
    config_path = _write(
        tmp_path / "config.yaml",
        f"""
runtime:
  manage_services: true
phases:
  demo_parse: false
  video_marking: false
  preprocess_slice: true
  phase2_yolo: true
  phase3a_semantic: false
  phase3b_semantic: false
  phase4_assemble: false
paths:
  video: {video.as_posix()}
  segments_json: {segments.as_posix()}
demo:
  parsed_dir: {demo.as_posix()}
yolo:
  yolo:
    enabled: false
""",
    )
    round_data = {"round_no": 1, "start_sec": 0.0, "end_sec": 10.0}

    def fake_preprocess(**kwargs):
        kwargs["output_rounds_path"].write_text(
            json.dumps({"video_path": str(video), "rounds": [round_data]}), encoding="utf-8"
        )
        kwargs["output_list_path"].write_text(json.dumps({"rounds": [round_data]}), encoding="utf-8")
        kwargs["output_segments_path"].write_text(
            json.dumps([{"start_sec": 0.0, "end_sec": 10.0}]), encoding="utf-8"
        )

    def fail_phase2(*args, **kwargs):
        args[3].current_stage = "phase2"
        raise RuntimeError("phase2 boom")

    monkeypatch.setattr(run_all, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(run_all, "run_preprocess_slice", fake_preprocess)
    monkeypatch.setattr(run_all, "_run_phases_subprocess", fail_phase2)

    result = run_all.run_all(config_path)

    assert result["status"] == "failed"
    assert result["failed_stage"] == "phase2"
    assert result["checkpointed_stages"] == ["phase1"]
    assert result["previous_success_preserved"] is False
    published = json.loads((output / "sbmachine" / "rounds.json").read_text(encoding="utf-8"))
    assert published["rounds"] == [round_data]
    assert not (output / "sbmachine" / "run_manifest.json").exists()


def test_successful_run_promotes_staged_outputs(tmp_path, monkeypatch):
    output = tmp_path / "output"
    _write(output / "sbmachine" / "rounds.json", '{"old": true}')
    video = _write(tmp_path / "video.mp4", "video")
    segments = _write(tmp_path / "segments.json", '{"segments": []}')
    config_path = _write(
        tmp_path / "config.yaml",
        f"""
runtime:
  manage_services: true
phases:
  demo_parse: false
  video_marking: false
  preprocess_slice: true
  phase2_yolo: false
  phase3a_semantic: false
  phase3b_semantic: false
  phase4_assemble: false
paths:
  video: {video.as_posix()}
  segments_json: {segments.as_posix()}
""",
    )

    round_data = {"round_no": 1, "start_sec": 0.0, "end_sec": 10.0}

    def fake_preprocess(**kwargs):
        kwargs["output_rounds_path"].write_text(
            json.dumps({"video_path": str(video), "rounds": [round_data]}), encoding="utf-8"
        )
        kwargs["output_list_path"].write_text(json.dumps({"rounds": [round_data]}), encoding="utf-8")
        kwargs["output_segments_path"].write_text(
            json.dumps([{"start_sec": 0.0, "end_sec": 10.0}]), encoding="utf-8"
        )

    monkeypatch.setattr(run_all, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(run_all, "run_preprocess_slice", fake_preprocess)
    monkeypatch.setattr(run_all, "_run_phases_subprocess", lambda *args, **kwargs: None)

    result = run_all.run_all(config_path)

    assert result["status"] == "complete"
    assert json.loads((output / "sbmachine" / "rounds.json").read_text(encoding="utf-8"))["rounds"] == [round_data]
    manifest = json.loads((output / "sbmachine" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["publishable"] is True
    assert not (output / ".staging" / result["run_id"]).exists()


def test_promotion_failure_restores_previous_success(tmp_path, monkeypatch):
    import sbmachine.run_context as run_context_module

    output = tmp_path / "output"
    previous = _write(output / "sbmachine" / "rounds.json", '{"old": true}')
    config_path = _write(
        tmp_path / "config.yaml",
        """
runtime:
  manage_services: true
phases:
  demo_parse: false
  video_marking: false
  preprocess_slice: false
  phase2_yolo: false
  phase3a_semantic: false
  phase3b_semantic: false
  phase4_assemble: false
""",
    )
    real_replace = run_context_module.os.replace
    failed_once = False

    def fail_new_sbmachine(source, target):
        nonlocal failed_once
        source_path = Path(source)
        if not failed_once and source_path.name == "sbmachine" and source_path.parent.name == "publish":
            failed_once = True
            raise OSError("promotion failed")
        return real_replace(source, target)

    monkeypatch.setattr(run_all, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(run_all, "_run_phases_subprocess", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_context_module.os, "replace", fail_new_sbmachine)

    result = run_all.run_all(config_path)

    assert result["status"] == "failed"
    assert result["failed_stage"] == "publish"
    assert previous.read_text(encoding="utf-8") == '{"old": true}'


def test_second_pipeline_run_fails_while_run_lock_is_held(tmp_path, monkeypatch):
    config_path = _write(
        tmp_path / "config.yaml",
        """
phases:
  demo_parse: false
  video_marking: false
  preprocess_slice: false
  phase2_yolo: false
  phase3a_semantic: false
  phase3b_semantic: false
  phase4_assemble: false
""",
    )
    monkeypatch.setattr(run_all, "PACKAGE_ROOT", tmp_path)

    with FileLock(tmp_path / "output" / ".run.lock"):
        result = run_all.run_all(config_path)

    assert result["status"] == "failed"
    assert result["failed_stage"] == "lock"
    assert result["exit_code"] == 3


def test_staging_removes_outputs_from_earliest_enabled_stage(tmp_path):
    output = tmp_path / "output"
    for relative in (
        "demo/keep.json",
        "sbmachine/rounds.json",
        "sbmachine/rounds_with_yolo.json",
        "sbmachine/rounds_with_neutral.json",
        "sbmachine/rounds_final.json",
    ):
        _write(output / relative, "{}")
    config = {
        "phases": {
            "demo_parse": False,
            "video_marking": False,
            "preprocess_slice": False,
            "phase2_yolo": True,
            "phase3a_semantic": False,
            "phase3b_semantic": False,
            "phase4_assemble": False,
        }
    }

    context = RunContext(output, run_id="stale-cleanup")
    context.prepare(config)

    assert (context.publish_dir / "demo" / "keep.json").is_file()
    assert (context.publish_dir / "sbmachine" / "rounds.json").is_file()
    assert not (context.publish_dir / "sbmachine" / "rounds_with_yolo.json").exists()
    assert not (context.publish_dir / "sbmachine" / "rounds_with_neutral.json").exists()
    assert not (context.publish_dir / "sbmachine" / "rounds_final.json").exists()


def test_reference_rewrite_refreshes_and_revalidates_neutral_hash(tmp_path):
    output = tmp_path / "output"
    (output / "sbmachine").mkdir(parents=True)
    context = RunContext(output, run_id="rewrite-hash")
    context.prepare({"phases": {key: False for key in (
        "demo_parse", "video_marking", "preprocess_slice", "phase2_yolo",
        "phase3a_semantic", "phase3b_semantic", "phase4_assemble",
    )}})
    rounds_path = context.publish_dir / "sbmachine" / "rounds_with_yolo.json"
    rounds_path.write_text(
        json.dumps({"rounds": [], "source": str(context.publish_dir / "sbmachine" / "clip.mp4")}),
        encoding="utf-8",
    )
    neutral_path = context.publish_dir / "sbmachine" / "rounds_with_neutral.json"
    neutral_path.write_text(
        json.dumps({**new_manifest_metadata(rounds_path), "rounds": []}),
        encoding="utf-8",
    )

    context.rewrite_published_references()

    rounds_payload = json.loads(rounds_path.read_text(encoding="utf-8"))
    neutral_payload = json.loads(neutral_path.read_text(encoding="utf-8"))
    assert rounds_payload["source"].startswith(str(output))
    assert neutral_payload["source_rounds_sha256"] == rounds_sha256(rounds_path)
    validate_neutral_manifest(neutral_payload, rounds_path)


def test_post_promotion_cleanup_failure_does_not_change_success(tmp_path, monkeypatch):
    import sbmachine.run_context as run_context_module

    output = tmp_path / "output"
    context = RunContext(output, run_id="cleanup-failure")
    context.prepare({"phases": {key: False for key in (
        "demo_parse", "video_marking", "preprocess_slice", "phase2_yolo",
        "phase3a_semantic", "phase3b_semantic", "phase4_assemble",
    )}})
    real_rmtree = run_context_module.shutil.rmtree

    def fail_staging_cleanup(path, *args, **kwargs):
        if Path(path) == context.staging_dir:
            raise OSError("cleanup failed")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(run_context_module.shutil, "rmtree", fail_staging_cleanup)

    manifest = context.complete({"status": "complete", "publishable": True})

    assert manifest["status"] == "complete"
    assert (output / "sbmachine" / "run_manifest.json").is_file()


def test_rewrite_value_handles_win_and_posix_slashes(tmp_path):
    output = tmp_path / "output"
    context = RunContext(output, run_id="slash-test")
    
    # 模拟 Publish 目录与 Output 目录混用斜杠
    pub_win = str(context.publish_dir).replace("/", "\\") + "\\data.json"
    pub_posix = str(context.publish_dir).replace("\\", "/") + "/data.json"
    
    data = {
        "win_path": pub_win,
        "posix_path": pub_posix,
        "nested": [pub_posix, {"item": pub_win}],
    }
    
    rewritten = context._rewrite_value(data)
    
    expected_win = str(context.output_root).replace("/", "\\") + "\\data.json"
    expected_posix = str(context.output_root).replace("\\", "/") + "/data.json"
    
    assert rewritten["win_path"] == expected_win
    assert rewritten["posix_path"] == expected_posix
    assert rewritten["nested"][0] == expected_posix
    assert rewritten["nested"][1]["item"] == expected_win


def test_service_manager_handles_runtime_null():
    from sbmachine.service_manager import ServiceManager
    config = {"runtime": None}
    mgr = ServiceManager(config)
    assert mgr._svc_cfg == {}


def test_select_preprocess_segments_validates_file_existence(tmp_path):
    from sbmachine.run_all import _select_preprocess_segments
    non_existent = tmp_path / "does_not_exist.json"
    result = _select_preprocess_segments(None, non_existent)
    assert result is None



def test_run_context_rewrites_llma_input_into_the_staging_publish_tree(tmp_path):
    context = RunContext(tmp_path / "output", run_id="llma-staging")

    effective = context._effective_config({
        "paths": {"llma_input_json": "output/sbmachine/llma_input.json"},
    })

    assert effective["paths"]["llma_input_json"] == str(
        context.publish_dir / "sbmachine" / "llma_input.json"
    )


def test_phase3a_checkpoint_promotes_only_declared_phase3a_artifacts(tmp_path):
    output_root = tmp_path / "output"
    old_commentary = _write(output_root / "sbmachine" / "commentary.json", '{"rounds":[{"status":"ok","marker":"old"}]}')
    _write(output_root / "sbmachine" / "rounds_with_commentary.json", '{"rounds":[{"marker":"old"}]}')
    context = RunContext(output_root, run_id="phase3a-artifacts")
    context.prepare({
        "phases": {
            "demo_parse": False,
            "video_marking": False,
            "preprocess_slice": False,
            "phase2_yolo": False,
            "phase3a_semantic": True,
            "phase3b_semantic": True,
            "phase4_assemble": False,
        },
    })
    staged_sb = context.publish_dir / "sbmachine"
    rounds_path = _write(staged_sb / "rounds_with_yolo.json", '{"rounds":[]}')
    _write(
        staged_sb / "rounds_with_neutral.json",
        json.dumps({**new_manifest_metadata(rounds_path), "rounds": []}),
    )
    _write(staged_sb / "llma_input.json", '{"artifact_kind":"phase3a_llm_input"}')
    _write(staged_sb / "rounds_with_commentary.json", '{"rounds":[{"marker":"new"}]}')
    _write(staged_sb / "commentary.json", '{"rounds":[{"status":"ok","marker":"new"}]}')

    context.checkpoint("phase3a")

    published_sb = output_root / "sbmachine"
    assert (published_sb / "rounds_with_neutral.json").is_file()
    assert (published_sb / "llma_input.json").is_file()
    assert json.loads((published_sb / "commentary.json").read_text(encoding="utf-8"))["rounds"][0]["marker"] == "old"
    assert old_commentary.is_file()
    assert json.loads((published_sb / "rounds_with_commentary.json").read_text(encoding="utf-8"))["rounds"][0]["marker"] == "old"

def test_managed_phase3_parent_validates_commentary_manifest_not_rounds_file(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    config = {
        "runtime": {"one_model_at_a_time": False, "manage_services": True},
        "phases": {
            "demo_parse": False,
            "video_marking": False,
            "preprocess_slice": False,
            "phase2_yolo": False,
            "phase3a_semantic": True,
            "phase3b_semantic": True,
            "phase4_assemble": False,
        },
    }
    context = RunContext(output_root, run_id="managed-commentary-manifest")
    effective, effective_config_path = context.prepare(config)
    paths = effective["paths"]
    rounds_p2 = Path(paths["rounds_with_yolo_json"])
    _write(rounds_p2, '{"rounds":[]}')

    def fake_spawn(*args, **kwargs):
        neutral_path = Path(paths["rounds_with_neutral_json"])
        neutral_payload = {**new_manifest_metadata(rounds_p2), "rounds": []}
        _write(neutral_path, json.dumps(neutral_payload))
        _write(Path(paths["llma_input_json"]), '{"artifact_kind":"phase3a_llm_input"}')
        _write(Path(paths["rounds_with_commentary_json"]), '{"rounds":[{"round_no":1}]}')
        commentary = {
            "commentary_schema_version": 2,
            "source_neutral_run_id": neutral_payload["run_id"],
            "source_neutral_sha256": hashlib.sha256(neutral_path.read_bytes()).hexdigest(),
            "source_window_count": 0,
            "rounds": [],
        }
        _write(Path(paths["commentary_json"]), json.dumps(commentary))
        # Phase3b 出口封存：checkpoint("phase3b") 声明了 llmb_draft_package.json 产物。
        _write(Path(paths["llmb_draft_package_json"]), json.dumps({
            "contract": "llmb_draft_package_v1",
            "producer": "phase3b",
            "run_id": "managed-test",
            "source": {
                "neutral_run_id": neutral_payload["run_id"],
                "neutral_sha256": "0" * 64,
                "timeline_id": "tl:managed-test:030",
                "source_video_sha256": "",
            },
            "rounds": [],
            "artifact_identity": "managed-test",
        }))

    monkeypatch.setattr(run_all, "_spawn", fake_spawn)
    run_all._run_phases_subprocess(
        effective_config_path,
        effective["phases"],
        effective,
        context,
        use_gpu_guard=False,
    )

    assert context.checkpointed_stages == ["phase3a", "phase3b"]

def test_subprocess_phase2_checkpoints_before_displaying_done(tmp_path, monkeypatch):
    """Catches the managed Phase2 path showing green before its checkpoint succeeds."""
    events: list[str] = []

    class FakeManager:
        def __init__(self, _config):
            pass
        def start(self, _name):
            pass
        def stop(self, _name):
            pass
        def stop_all(self):
            pass

    class FakeContext:
        current_stage = ""
        diagnostics_dir = tmp_path / "diagnostics"
        run_id = "run-1"
        def checkpoint(self, stage):
            events.append(f"checkpoint:{stage}")
        def write_diagnostic(self, _name, _payload):
            pass

    import sbmachine.service_manager as service_manager
    monkeypatch.setattr(service_manager, "ServiceManager", FakeManager)
    monkeypatch.setattr(run_all, "_call_gpu_guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_all, "_spawn", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_all, "validate_phase2_publishable", lambda _path: None)

    callbacks = {"on_stage_done": lambda stage, _artifacts: events.append(f"done:{stage}")}
    run_all._run_phases_subprocess(
        tmp_path / "config.yaml",
        {"phase2_yolo": True, "phase3a_semantic": False, "phase3b_semantic": False, "phase4_assemble": False},
        {"runtime": {"one_model_at_a_time": True}, "paths": {"rounds_with_yolo_json": str(tmp_path / "rounds.json")}},
        FakeContext(),
        use_gpu_guard=False,
        callbacks=callbacks,
    )

    assert events == ["checkpoint:phase2", "done:phase2"]


def test_subprocess_phase3_starts_phase3b_only_after_phase3a_work_complete(tmp_path, monkeypatch):
    """Catches the managed path pre-starting Phase3b before Phase3a's actual boundary."""
    events: list[tuple] = []

    class FakeManager:
        def __init__(self, _config):
            pass
        def start(self, _name):
            pass
        def stop(self, _name):
            pass
        def stop_all(self):
            pass

    class FakeContext:
        current_stage = ""
        diagnostics_dir = tmp_path / "diagnostics"
        run_id = "run-1"
        def checkpoint(self, stage):
            events.append(("checkpoint", stage))
        def write_diagnostic(self, _name, _payload):
            pass

    import sbmachine.service_manager as service_manager
    from sbmachine.progress_events import ProgressEvent
    monkeypatch.setattr(service_manager, "ServiceManager", FakeManager)
    monkeypatch.setattr(run_all, "_call_gpu_guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_all, "validate_neutral_publishable", lambda _path: None)
    monkeypatch.setattr(run_all, "validate_commentary_publishable", lambda _path: None)

    def child_boundary(*_args, **kwargs):
        callbacks = kwargs["callbacks"]
        run_all._forward_progress_event(callbacks, ProgressEvent("run-1", 1, "stage_progress", "phase3a", 1, 1, "round"))
        run_all._forward_progress_event(callbacks, ProgressEvent("run-1", 2, "stage_work_complete", "phase3a", 1, 1, "round"))
        run_all._forward_progress_event(callbacks, ProgressEvent("run-1", 3, "stage_start", "phase3b"))
        return {"events_received": 3}

    monkeypatch.setattr(run_all, "_spawn", child_boundary)
    callbacks = {
        "on_stage_start": lambda stage: events.append(("start", stage)),
        "on_stage_progress": lambda *args: events.append(("progress", *args)),
        "on_stage_done": lambda stage, _artifacts: events.append(("done", stage)),
    }
    run_all._run_phases_subprocess(
        tmp_path / "config.yaml",
        {"phase2_yolo": False, "phase3a_semantic": True, "phase3b_semantic": True, "phase4_assemble": False},
        {"runtime": {"one_model_at_a_time": True}, "paths": {"rounds_with_neutral_json": str(tmp_path / "neutral.json"), "rounds_with_commentary_json": str(tmp_path / "commentary-rounds.json"), "commentary_json": str(tmp_path / "commentary.json")}},
        FakeContext(),
        use_gpu_guard=False,
        callbacks=callbacks,
    )

    assert events.index(("progress", "phase3a", 1, 1, "round", "处理完成，等待门禁")) < events.index(("start", "phase3b"))
    assert events.index(("start", "phase3b")) < events.index(("done", "phase3a"))
