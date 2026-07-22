"""Compose lifecycle management for optional inference services."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from sbmachine.service_manager import ServiceManager


class ComposeManager:
    """Start one Compose service, verify it, and stop it after its stage."""

    _HEALTH_NAME = {
        "talk_service": "talk",
        "audio_service": "sovits",
    }

    def __init__(self, config: dict, compose_file: str = "docker-compose.yml") -> None:
        self.config = config
        self.compose_file = compose_file
        self._health = ServiceManager(config)
        self._running: set[str] = set()

    def _compose_path(self) -> Path:
        from sbmachine.common import PROJECT_ROOT

        path = Path(self.compose_file)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _compose(self, *args: str) -> subprocess.CompletedProcess:
        from sbmachine.common import PROJECT_ROOT

        cmd = ["docker", "compose", "-f", self.compose_file, *args]
        print(f"[compose] {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    def _service_image(self, service: str) -> str | None:
        path = self._compose_path()
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(f"cannot read compose service metadata from {path}: {exc}") from exc
        services = payload.get("services", {}) if isinstance(payload, dict) else {}
        item = services.get(service, {}) if isinstance(services, dict) else {}
        image = item.get("image") if isinstance(item, dict) else None
        return str(image).strip() if image else None

    @staticmethod
    def docker_daemon_available() -> bool:
        """Return whether the Docker CLI can reach a server without pulling anything."""
        if not shutil.which("docker"):
            return False
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    @staticmethod
    def _image_is_available(image: str) -> bool:
        if not shutil.which("docker"):
            return False
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def talk_addon_status(self, *, verify: bool = False) -> dict[str, object]:
        """Inspect the optional talk image and, when requested, its model cache.

        ``verify`` runs the entrypoint's short ``verify`` command only after
        confirming the image exists.  It never builds, pulls, or downloads a
        model; this is what keeps ordinary ``run`` calls deterministic.
        """
        image = self._service_image("talk_service")
        if not image:
            return {
                "required": True,
                "ready": False,
                "detail": "talk_service has no image reference",
            }
        if not self._image_is_available(image):
            return {
                "required": True,
                "ready": False,
                "image": image,
                "detail": "talk runtime image is absent; run python run.py setup --backend container --install",
            }
        if not verify:
            return {
                "required": True,
                "ready": True,
                "image": image,
                "detail": "talk runtime image is present; model cache was not verified",
            }
        result = self._compose("run", "--rm", "--no-deps", "talk_service", "verify")
        if result.returncode != 0:
            return {
                "required": True,
                "ready": False,
                "image": image,
                "detail": "talk model cache is not prepared; run python run.py setup --backend container --install",
            }
        return {
            "required": True,
            "ready": True,
            "image": image,
            "detail": "talk runtime image and configured model cache are ready",
        }

    def install_talk(self) -> dict[str, object]:
        """Explicitly build the optional runtime and download its configured model."""
        image = self._service_image("talk_service")
        if not image:
            raise RuntimeError("talk_service has no image reference")
        built = False
        if not self._image_is_available(image):
            result = self._compose("build", "talk_service")
            if result.returncode != 0:
                raise RuntimeError(f"docker compose build talk_service failed (exit {result.returncode})")
            built = True
        result = self._compose("run", "--rm", "--no-deps", "talk_service", "prepare")
        if result.returncode != 0:
            raise RuntimeError(f"talk model preparation failed (exit {result.returncode})")
        status = self.talk_addon_status(verify=True)
        if not bool(status.get("ready")):
            raise RuntimeError(str(status.get("detail") or "talk add-on verification failed"))
        status["built_runtime"] = built
        status["model_preparation_requested"] = True
        return status

    def require_talk_addon(self) -> None:
        status = self.talk_addon_status(verify=True)
        if not bool(status.get("ready")):
            raise RuntimeError(str(status.get("detail") or "talk add-on is not ready"))

    def _talk_health_name(self) -> str:
        return "vllm"

    def _startup_timeout(self, health_name: str) -> int:
        return int(
            self.config.get("runtime", {})
            .get("services", {})
            .get(health_name, {})
            .get("startup_timeout_sec", 90)
        )

    def up_one(self, service: str) -> None:
        if service == "talk_service":
            self.require_talk_addon()
        result = self._compose("up", "-d", service)
        if result.returncode != 0:
            raise RuntimeError(f"docker compose up {service} failed (exit {result.returncode})")
        self._running.add(service)

        health_name = self._HEALTH_NAME.get(service)
        if health_name == "talk":
            health_name = self._talk_health_name()
        if health_name is None:
            self.down_one(service)
            raise RuntimeError(f"[compose] unknown service '{service}'")

        url = self._health._health_url(health_name)
        identity = self._health._health_identity(health_name)
        timeout = self._startup_timeout(health_name)
        print(f"[compose] waiting {service} health: {url} (<={timeout}s)", flush=True)
        if not ServiceManager._poll_health(url, timeout, identity=identity):
            self.down_one(service)
            raise RuntimeError(f"[compose] {service} not healthy within {timeout}s; container stopped")
        print(f"[compose] {service} healthy", flush=True)

    def down_one(self, service: str) -> None:
        if service not in self._running:
            return
        self._compose("stop", service)
        self._running.discard(service)

    def down_all(self) -> None:
        for service in list(self._running):
            self.down_one(service)