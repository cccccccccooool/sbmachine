"""多容器编排生命周期管理（docker compose 按阶段错峰拉起/挂掉单个后端容器）。

仅在 runtime.manage_services = false（多容器模式）时激活，由 run.py 调用。
对称于 service_manager.ServiceManager（单容器内多进程错峰）：这里错峰的粒度是 docker 容器。

★ 核心约束：目标 8-12G 显存部署，任意时刻只能有一个模型在卡上。
  所以严格单容器错峰——用到哪个阶段才 up 对应容器，阶段跑完立刻 stop 释放显存，再 up 下一个。
  绝不一次性拉起三容器。

health URL 复用 ServiceManager 的派生逻辑，避免重复硬编码端口。
"""
from __future__ import annotations

import subprocess

from sbmachine.service_manager import ServiceManager


class ComposeManager:
    """用 docker compose 按阶段错峰管理单个后端容器的拉起与销毁。"""

    # compose 服务名 → 该服务健康检查所用的 ServiceManager 名
    _HEALTH_NAME = {
        "vision_service": "vlm",
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

    def _timeout(self, health_name: str) -> int:
        return int(
            self.config.get("runtime", {})
            .get("services", {})
            .get(health_name, {})
            .get("startup_timeout_sec", 90)
        )

    def up_one(self, service: str) -> None:
        """拉起单个容器并轮询其健康端点；失败即 stop 并抛出。"""
        result = self._compose("up", "-d", service)
        if result.returncode != 0:
            raise RuntimeError(f"docker compose up {service} 失败 (exit {result.returncode})")
        self._running.add(service)

        health_name = self._HEALTH_NAME[service]
        url = self._health._health_url(health_name)
        timeout = self._timeout(health_name)
        print(f"[compose] 等待 {service} 健康: {url} (≤{timeout}s)", flush=True)
        if not ServiceManager._poll_health(url, timeout):
            self.down_one(service)
            raise RuntimeError(f"[compose] {service} 在 {timeout}s 内未就绪,已 stop。检查容器日志。")
        print(f"[compose] {service} healthy", flush=True)

    def down_one(self, service: str) -> None:
        if service not in self._running:
            return
        self._compose("stop", service)
        self._running.discard(service)

    def down_all(self) -> None:
        self._compose("down")
        self._running.clear()
