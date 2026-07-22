from pathlib import Path

from sbmachine.runtime_backend import create_runtime_backend, resolve_runtime_backend
from sbmachine.pipeline_interface import select_pipeline_interface
from sbmachine.run_context import RunContext


def _config(backend: str = "local") -> dict:
    return {
        "runtime": {"backend": backend, "one_model_at_a_time": True},
        "phases": {
            "demo_parse": True,
            "video_marking": True,
            "preprocess_slice": True,
            "phase2_yolo": True,
            "phase3a_semantic": True,
            "phase3b_semantic": True,
            "phase4_assemble": True,
        },
    }


def test_legacy_runtime_mapping_is_preserved():
    assert resolve_runtime_backend({"runtime": {"manage_services": True}}) == "local"
    assert resolve_runtime_backend({"runtime": {"manage_services": False}}) == "container"
    assert resolve_runtime_backend({"runtime": {"backend": "container", "manage_services": True}}) == "container"


def test_mock_local_and_container_share_one_business_stage_plan():
    local_runtime = create_runtime_backend(_config("local"), mock=True)
    local = local_runtime.simulate_pipeline()
    container = create_runtime_backend(_config("container"), mock=True).simulate_pipeline()

    assert local["stages"] == container["stages"]
    assert local["downloads_performed"] is False
    assert local["writes_performed"] is False
    assert container["downloads_performed"] is False
    assert container["writes_performed"] is False
    workspace = Path.cwd() / "tests" / ".runtime-mock-no-write"
    assert not workspace.exists()
    assert not local_runtime.run_stage("phase2", ("never-runs",), workspace).exists()
    assert not workspace.exists()

    assert local["events"] == [
        {"action": "start", "component": "core"},
        {"action": "run", "component": "core", "phases": ["demo_parse", "video_marking", "phase1", "phase2"]},
        {"action": "stop", "component": "core"},
        {"action": "start", "component": "talk"},
        {"action": "run", "component": "talk", "phases": ["phase3a", "phase3b"]},
        {"action": "stop", "component": "talk"},
        {"action": "start", "component": "voice"},
        {"action": "run", "component": "voice", "phases": ["phase4"]},
        {"action": "stop", "component": "voice"},
    ]


def test_selected_runtime_reaches_the_one_pipeline_interface():
    paths = {
        "rounds_json": "tests/.runtime-rounds.json",
        "rounds_with_yolo_json": "tests/.runtime-yolo.json",
        "rounds_with_neutral_json": "tests/.runtime-neutral.json",
        "rounds_with_commentary_json": "tests/.runtime-commentary.json",
        "rounds_final_json": "tests/.runtime-final.json",
    }
    output_root = Path.cwd() / "tests" / ".runtime-interface-output"
    assert not output_root.exists()
    for backend in ("local", "container"):
        config = _config(backend)
        config["paths"] = paths
        pipeline = select_pipeline_interface(Path("config"), config, RunContext(output_root))
        assert pipeline.executor.services.runtime.name == backend
    assert not output_root.exists()

def test_cleanup_remembers_talk_service_identity():
    class RecordingManager:
        def __init__(self):
            self.calls = []

        def start(self, name):
            self.calls.append(("start", name))

        def stop(self, name):
            self.calls.append(("stop", name))

    runtime = create_runtime_backend({"runtime": {}}, "local")
    manager = RecordingManager()
    runtime._manager = manager
    runtime.start_component("talk", phase3_service="vllm")
    runtime.cleanup()

    assert manager.calls == [("start", "vllm"), ("stop", "vllm")]