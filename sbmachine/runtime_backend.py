"""Small runtime boundary for the release pipeline.

The business pipeline owns stage order.  A runtime backend only owns how the
talk/voice services are prepared, started, stopped, and how the current host
core process is launched.  The current container backend is intentionally a
hybrid: core stays on the host while talk and voice use Compose services.
"""
from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence


_BACKENDS = frozenset({"local", "container"})
_STATE_DIR = ".sbmachine"
_STATE_FILE = "runtime.json"


class RuntimeBackendError(RuntimeError):
    """Raised when a selected runtime cannot be used safely."""


def _runtime_config(config: dict) -> dict:
    runtime = config.get("runtime", {})
    return runtime if isinstance(runtime, dict) else {}


def _normalize_backend(value: object) -> str:
    backend = str(value).strip().lower()
    if backend not in _BACKENDS:
        choices = ", ".join(sorted(_BACKENDS))
        raise RuntimeBackendError(f"unsupported runtime backend {value!r}; choose one of: {choices}")
    return backend


def resolve_runtime_backend(config: dict, override: str | None = None) -> str:
    """Resolve explicit configuration first, then preserve the legacy mapping."""
    if override is not None:
        return _normalize_backend(override)
    runtime = _runtime_config(config)
    configured = runtime.get("backend")
    if configured not in (None, ""):
        return _normalize_backend(configured)
    # Compatibility with release configurations written before runtime.backend.
    return "local" if bool(runtime.get("manage_services", False)) else "container"


def runtime_state_path(workspace: Path | str) -> Path:
    return Path(workspace).resolve() / _STATE_DIR / _STATE_FILE


def load_runtime_selection(workspace: Path | str) -> str | None:
    """Return the backend selected by setup, without falling back implicitly."""
    path = runtime_state_path(workspace)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeBackendError(f"invalid runtime selection file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeBackendError(f"invalid runtime selection file: {path}")
    if payload.get("schema_version") != 1:
        raise RuntimeBackendError(f"unsupported runtime selection schema in {path}")
    return _normalize_backend(payload.get("backend"))


def save_runtime_selection(workspace: Path | str, backend: str) -> Path:
    """Persist the user's setup choice outside the published configuration."""
    path = runtime_state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "backend": _normalize_backend(backend),
        "selected_by": "sbmachine setup",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _phase_enabled(phases: dict, key: str, legacy_key: str | None = None, default: bool = True) -> bool:
    if key in phases:
        return bool(phases[key])
    if legacy_key is not None and legacy_key in phases:
        return bool(phases[legacy_key])
    return default


def pipeline_stage_plan(config: dict) -> list[dict[str, object]]:
    """Describe the shared business sequence without touching a runtime."""
    phases = config.get("phases", {})
    phases = phases if isinstance(phases, dict) else {}
    core: list[str] = []
    if bool(phases.get("demo_parse", False)):
        core.append("demo_parse")
    if bool(phases.get("video_marking", False)):
        core.append("video_marking")
    if _phase_enabled(phases, "preprocess_slice", "phase1_slice"):
        core.append("phase1")
    if _phase_enabled(phases, "phase2_yolo"):
        core.append("phase2")

    talk: list[str] = []
    if _phase_enabled(phases, "phase3a_semantic", "phase3_semantic"):
        talk.append("phase3a")
    if _phase_enabled(phases, "phase3b_semantic", "phase3_semantic"):
        talk.append("phase3b")

    voice: list[str] = []
    if _phase_enabled(phases, "phase4_assemble", None):
        voice.append("phase4")

    return [
        {"component": component, "phases": stage_names}
        for component, stage_names in (("core", core), ("talk", talk), ("voice", voice))
        if stage_names
    ]


def verify_bundled_models() -> dict[str, object]:
    """Validate release-bundled core weights without contacting any registry."""
    from sbmachine.common import PROJECT_ROOT

    manifest_path = PROJECT_ROOT / "models" / "manifest.yaml"
    if not manifest_path.is_file():
        return {"name": "model_manifest", "ok": False, "detail": f"missing {manifest_path}"}
    try:
        import yaml

        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as exc:
        return {"name": "model_manifest", "ok": False, "detail": str(exc)}
    bundled = payload.get("bundled", []) if isinstance(payload, dict) else []
    if not isinstance(bundled, list):
        return {"name": "model_manifest", "ok": False, "detail": "bundled must be a list"}
    failures: list[str] = []
    for item in bundled:
        if not isinstance(item, dict):
            failures.append("invalid bundled item")
            continue
        path = PROJECT_ROOT / "models" / str(item.get("path", ""))
        expected_size = item.get("size_bytes")
        expected_hash = str(item.get("sha256", "")).lower()
        if not path.is_file():
            failures.append(f"missing {path.name}")
            continue
        if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
            failures.append(f"size mismatch {path.name}")
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            failures.append(f"sha256 mismatch {path.name}")
    detail = "all bundled core models verified" if not failures else "; ".join(failures)
    return {"name": "bundled_models", "ok": not failures, "detail": detail}

class RuntimeBackend(ABC):
    """Runtime abstraction shared by the one business pipeline.

    It deliberately has no business-rule imports.  Future core-in-container
    support can override ``run_stage`` without changing stage ordering.
    """

    name: str

    def __init__(self, config: dict) -> None:
        self.config = config
        self.runtime = _runtime_config(config)
        self.one_model_at_a_time = bool(self.runtime.get("one_model_at_a_time", True))
        self._started: list[str] = []
        self._component_options: dict[str, str | None] = {}

    def _mark_started(self, component: str, phase3_service: str | None = None) -> None:
        if component not in self._started:
            self._started.append(component)
        self._component_options[component] = phase3_service

    def _mark_stopped(self, component: str) -> None:
        if component in self._started:
            self._started.remove(component)
        self._component_options.pop(component, None)

    @abstractmethod
    def doctor(self) -> dict[str, object]:
        """Inspect only; do not download, start services, or alter state."""

    def setup(self) -> dict[str, object]:
        """Validate the selected backend.  Dependency downloads stay explicit."""
        report = dict(self.doctor())
        report.update(
            {
                "action": "setup",
                "downloads_performed": False,
                "setup_actions": ["validated existing runtime prerequisites only"],
            }
        )
        return report

    def prepare_runtime(self, *, enable_talk: bool, enable_voice: bool, phase3_service: str | None) -> None:
        if self.one_model_at_a_time:
            return
        if enable_talk and phase3_service:
            self.start_component("talk", phase3_service=phase3_service)
        if enable_voice:
            self.start_component("voice")

    def start_group(self, group: str, *, phase3_service: str | None) -> None:
        if not self.one_model_at_a_time:
            return
        if group == "semantic" and phase3_service:
            self.start_component("talk", phase3_service=phase3_service)
            return
        if group == "audio":
            self.start_component("voice")
            return
        if group != "semantic":
            raise RuntimeBackendError(f"unknown service group: {group}")

    def stop_group(self, group: str, *, phase3_service: str | None) -> None:
        if not self.one_model_at_a_time:
            return
        if group == "semantic" and phase3_service:
            self.stop_component("talk", phase3_service=phase3_service)
            return
        if group == "audio":
            self.stop_component("voice")
            return
        if group != "semantic":
            raise RuntimeBackendError(f"unknown service group: {group}")

    @abstractmethod
    def start_component(self, component: str, *, phase3_service: str | None = None) -> None:
        """Start one logical component and verify its existing health contract."""

    @abstractmethod
    def stop_component(self, component: str, *, phase3_service: str | None = None) -> None:
        """Stop a component started by this backend."""

    def run_stage(self, stage: str, args: Sequence[str], workspace: Path) -> Path:
        """Run the current host core stage and retain its diagnostic log."""
        log_path = workspace / f"{stage}.log"
        command = [sys.executable, *[str(arg) for arg in args]]
        print(f"[runtime:{self.name}] spawn {' '.join(command)}", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeBackendError(f"{stage} exited with code {result.returncode}; see {log_path}")
        return log_path

    def cleanup(self) -> None:
        """Best-effort reverse-order teardown that must not hide a stage failure."""
        for component in reversed(tuple(self._started)):
            try:
                self.stop_component(component, phase3_service=self._component_options.get(component))
            except Exception as exc:  # pragma: no cover - emergency cleanup path
                print(f"[runtime:{self.name}] cleanup failed for {component}: {exc}", flush=True)


class LocalRuntimeBackend(RuntimeBackend):
    """Host core plus independently configured local talk/voice processes."""

    name = "local"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._manager = None

    def _manager_or_create(self):
        if self._manager is None:
            from sbmachine.service_manager import ServiceManager

            self._manager = ServiceManager(self.config)
        return self._manager

    @staticmethod
    def _service_name(component: str, phase3_service: str | None = None) -> str | None:
        if component == "talk":
            return phase3_service
        if component == "voice":
            return "sovits"
        if component == "core":
            return None
        raise RuntimeBackendError(f"unknown local component: {component}")

    def doctor(self) -> dict[str, object]:
        services = self.runtime.get("services", {})
        services = services if isinstance(services, dict) else {}
        checks: list[dict[str, object]] = [
            {"name": "python", "ok": bool(sys.executable), "detail": sys.executable},
        ]
        checks.append(verify_bundled_models())
        ready = bool(checks[-1]["ok"] )
        for name in ("vllm", "sovits"):
            service = services.get(name, {})
            service = service if isinstance(service, dict) else {}
            if not service.get("enabled", True):
                continue
            command = str(service.get("start", "")).strip()
            command_ok = bool(command)
            checks.append({"name": f"{name}_start_command", "ok": command_ok, "detail": command})
            ready = ready and command_ok
            if command.startswith("bash"):
                bash_ok = shutil.which("bash") is not None
                checks.append({"name": "bash", "ok": bash_ok, "detail": "required by configured start command"})
                ready = ready and bash_ok
            if "/opt/GPT-SoVITS" in command:
                sovits_root = Path("/opt/GPT-SoVITS")
                root_ok = sovits_root.is_dir()
                checks.append({"name": "gpt_sovits_root", "ok": root_ok, "detail": str(sovits_root)})
                ready = ready and root_ok
        return {
            "action": "doctor",
            "backend": self.name,
            "ready": ready,
            "checks": checks,
            "downloads_performed": False,
            "limitations": [
                "local uses the configured service commands; this release does not create Python environments",
            ],
        }

    def start_component(self, component: str, *, phase3_service: str | None = None) -> None:
        service = self._service_name(component, phase3_service)
        if service is None or component in self._started:
            return
        self._manager_or_create().start(service)
        self._mark_started(component, phase3_service)

    def stop_component(self, component: str, *, phase3_service: str | None = None) -> None:
        service = self._service_name(component, phase3_service)
        if service is None or component not in self._started:
            return
        self._manager_or_create().stop(service)
        self._mark_stopped(component)


class ContainerRuntimeBackend(RuntimeBackend):
    """Host core plus Compose-managed talk/voice service containers."""

    name = "container"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._manager = None
        self._compose_file = str(self.runtime.get("compose_file", "docker-compose.yml"))

    def _manager_or_create(self):
        if self._manager is None:
            from sbmachine.compose_manager import ComposeManager

            self._manager = ComposeManager(self.config, compose_file=self._compose_file)
        return self._manager

    @staticmethod
    def _service_name(component: str, phase3_service: str | None = None) -> str | None:
        if component == "talk":
            return "talk_service" if phase3_service else None
        if component == "voice":
            return "audio_service"
        if component == "core":
            return None
        raise RuntimeBackendError(f"unknown container component: {component}")

    def doctor(self) -> dict[str, object]:
        from sbmachine.common import PROJECT_ROOT

        compose_path = Path(self._compose_file)
        if not compose_path.is_absolute():
            compose_path = PROJECT_ROOT / compose_path
        docker = shutil.which("docker")
        checks = [
            {"name": "docker", "ok": docker is not None, "detail": docker or "not found"},
            {"name": "compose_file", "ok": compose_path.is_file(), "detail": str(compose_path)},
        ]
        checks.append(verify_bundled_models())
        return {
            "action": "doctor",
            "backend": self.name,
            "ready": all(bool(check["ok"]) for check in checks),
            "checks": checks,
            "downloads_performed": False,
            "limitations": [
                "current container backend keeps core on the host and containers only talk/voice",
                "Docker daemon, NVIDIA runtime, image availability, and model availability are verified when a real run starts",
            ],
        }

    def start_component(self, component: str, *, phase3_service: str | None = None) -> None:
        service = self._service_name(component, phase3_service)
        if service is None or component in self._started:
            return
        self._manager_or_create().up_one(service)
        self._mark_started(component, phase3_service)

    def stop_component(self, component: str, *, phase3_service: str | None = None) -> None:
        service = self._service_name(component, phase3_service)
        if service is None or component not in self._started:
            return
        self._manager_or_create().down_one(service)
        self._mark_stopped(component)


class MockRuntimeBackend(RuntimeBackend):
    """No-I/O backend used to verify CLI selection and stage dispatch."""

    def __init__(self, config: dict, name: str) -> None:
        super().__init__(config)
        self.name = _normalize_backend(name)
        self.events: list[dict[str, object]] = []

    def doctor(self) -> dict[str, object]:
        return {
            "action": "doctor",
            "backend": self.name,
            "ready": True,
            "simulated": True,
            "checks": [{"name": "mock_runtime", "ok": True, "detail": "no process, network, Docker, GPU, or model access"}],
            "downloads_performed": False,
            "writes_performed": False,
        }

    def setup(self) -> dict[str, object]:
        report = self.doctor()
        report.update(
            {
                "action": "setup",
                "setup_actions": ["simulated backend selection only"],
            }
        )
        return report

    def start_component(self, component: str, *, phase3_service: str | None = None) -> None:
        if component not in self._started:
            self._mark_started(component, phase3_service)
            self.events.append({"action": "start", "component": component})

    def stop_component(self, component: str, *, phase3_service: str | None = None) -> None:
        if component in self._started:
            self._mark_stopped(component)
            self.events.append({"action": "stop", "component": component})

    def run_stage(self, stage: str, args: Sequence[str], workspace: Path) -> Path:
        self.events.append({"action": "run", "stage": stage})
        return workspace / f"{stage}.log"

    def simulate_pipeline(self) -> dict[str, object]:
        stages = pipeline_stage_plan(self.config)
        events: list[dict[str, object]] = []
        for item in stages:
            component = str(item["component"])
            events.append({"action": "start", "component": component})
            events.append({"action": "run", "component": component, "phases": item["phases"]})
            events.append({"action": "stop", "component": component})
        return {
            "action": "run",
            "backend": self.name,
            "simulated": True,
            "downloads_performed": False,
            "writes_performed": False,
            "stages": stages,
            "events": events,
        }


def create_runtime_backend(config: dict, backend: str | None = None, *, mock: bool = False) -> RuntimeBackend:
    """Create exactly one selected runtime backend for a pipeline invocation."""
    selected = resolve_runtime_backend(config, backend)
    if mock:
        return MockRuntimeBackend(config, selected)
    if selected == "local":
        return LocalRuntimeBackend(config)
    return ContainerRuntimeBackend(config)
