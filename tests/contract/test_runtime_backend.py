from pathlib import Path

from sbmachine.common import talk_component_requirement
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

def _api_phase3_config(backend: str = "container") -> dict:
    config = _config(backend)
    config["llm"] = {"backend": "api"}
    config["semantic"] = {"analyst_backend": "api", "style_backend": "api"}
    return config


def test_api_only_phase3_does_not_require_or_simulate_talk(monkeypatch):
    monkeypatch.delenv("AI6657_ANALYST_BACKEND", raising=False)
    monkeypatch.delenv("AI6657_STYLE_BACKEND", raising=False)
    config = _api_phase3_config()

    requirement = talk_component_requirement(config)
    assert requirement["required"] is False
    assert requirement["roles"] == []

    runtime = create_runtime_backend(config, mock=True)
    report = runtime.doctor()
    plan = runtime.simulate_pipeline()
    setup = runtime.setup(install=True)

    assert report["talk_addon"]["required"] is False
    assert all(event["component"] != "talk" for event in plan["events"])
    assert plan["stages"][0] == {
        "component": "core",
        "phases": ["demo_parse", "video_marking", "phase1", "phase2", "phase3a", "phase3b"],
    }
    assert setup["downloads_performed"] is False
    assert setup["setup_actions"] == ["simulated backend selection only"]


def test_mixed_phase3_backends_only_assign_vllm_role_to_talk(monkeypatch):
    monkeypatch.delenv("AI6657_ANALYST_BACKEND", raising=False)
    monkeypatch.delenv("AI6657_STYLE_BACKEND", raising=False)
    config = _api_phase3_config()
    config["semantic"]["style_backend"] = "vllm"

    requirement = talk_component_requirement(config)
    plan = create_runtime_backend(config, mock=True).simulate_pipeline()

    assert requirement["required"] is True
    assert requirement["roles"] == ["style"]
    assert {item["component"]: item["phases"] for item in plan["stages"]}["talk"] == ["phase3b"]
    assert "phase3a" in {item["component"]: item["phases"] for item in plan["stages"]}["core"]


def test_api_only_doctors_do_not_check_local_vllm_or_docker(monkeypatch):
    monkeypatch.delenv("AI6657_ANALYST_BACKEND", raising=False)
    monkeypatch.delenv("AI6657_STYLE_BACKEND", raising=False)
    config = _api_phase3_config("local")
    config["phases"]["phase4_assemble"] = False

    local_report = create_runtime_backend(config, "local").doctor()
    container_report = create_runtime_backend(config, "container").doctor()
    local_check_names = {str(check["name"]) for check in local_report["checks"]}
    container_check_names = {str(check["name"]) for check in container_report["checks"]}

    assert local_report["ready"] is True
    assert "vllm_start_command" not in local_check_names
    assert "talk_addon" in local_check_names
    assert container_report["ready"] is True
    assert "docker" not in container_check_names
    assert container_report["talk_addon"]["required"] is False


def test_mock_vllm_install_is_explicit_but_never_downloads(monkeypatch):
    monkeypatch.delenv("AI6657_ANALYST_BACKEND", raising=False)
    monkeypatch.delenv("AI6657_STYLE_BACKEND", raising=False)
    report = create_runtime_backend(_config("container"), mock=True).setup(install=True)

    assert report["talk_requirement"]["required"] is True
    assert report["setup_actions"] == ["simulated optional talk installation"]
    assert report["downloads_performed"] is False