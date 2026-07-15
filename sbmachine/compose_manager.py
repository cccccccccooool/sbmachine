"""Docker Compose 生命周期管理：逐个启动后端服务（同一时刻只跑一个）。"""
from __future__ import annotations

import subprocess

from sbmachine.service_manager import ServiceManager


class ComposeManager:
    """启动单个 compose 服务，等待健康，阶段结束后停掉它。"""

    _HEALTH_NAME = {
        "vision_service": "vlm",
        "talk_service": "talk",
        "audio_service": "sovits",
    }

    def __init__(self, config: dict, compose_file: str = "docker-compose.yml") -> None:
        self.config = config
        self.compose_file = compose_file
        self._health = ServiceManager(config)
        self._running: set[str] = set()

    def _compose(self, *args: str) -> subprocess.CompletedProcess:
        from sbmachine.common import PROJECT_ROOT
        cmd = ["docker", "compose", "-f", self.compose_file, *args]
        print(f"[compose] {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    def _talk_health_name(self) -> str:
        return "vllm"  # 迁移后只保留 vLLM 后端

    def _startup_timeout(self, health_name: str) -> int:
        return int(
            self.config.get("runtime", {})
            .get("services", {})
            .get(health_name, {})
            .get("startup_timeout_sec", 90)
        )

    def up_one(self, service: str) -> None:
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
