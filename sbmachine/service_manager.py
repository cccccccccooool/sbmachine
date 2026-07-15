"""外部推理服务生命周期管理（VLM / vLLM / SoVITS）。"""
from __future__ import annotations

import subprocess
import time

import requests


class ServiceManager:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._procs: dict[str, subprocess.Popen] = {}
        self._log_fhs: dict[str, object] = {}
        self._svc_cfg: dict = config.get("runtime", {}).get("services", {})


    # ── 健康轮询 ──

    @staticmethod
    def _matches_identity(response: requests.Response, identity: dict) -> bool:
        if response.status_code != 200:
            return False
        try:
            payload = response.json()
        except (requests.JSONDecodeError, ValueError):
            return False
        service = identity.get("service")
        if service == "vlm":
            return isinstance(payload, dict) and payload.get("service") == "ai-6657-vlm" and payload.get("status") == "ok"
        if service == "vllm":
            models = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(payload, dict) or payload.get("object") != "list" or not isinstance(models, list):
                return False
            ids = {
                str(item.get("id"))
                for item in models
                if isinstance(item, dict) and item.get("id")
            }
            expected = identity.get("models")
            if isinstance(expected, list):
                expected_models = {str(model) for model in expected if model}
            else:
                expected_model = str(identity.get("model") or "")
                expected_models = {expected_model} if expected_model else set()
            return bool(ids) and expected_models.issubset(ids)
        if service == "sovits":
            paths = payload.get("paths") if isinstance(payload, dict) else None
            return isinstance(paths, dict) and "/tts" in paths
        return False

    @staticmethod
    def _poll_health(
        url: str,
        timeout_sec: int,
        interval: float = 2.0,
        identity: dict | None = None,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                response = requests.get(url, timeout=3)
                if identity is not None and ServiceManager._matches_identity(response, identity):
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        return False

    # ── 健康检查 URL 推导 ──

    def _health_url(self, name: str) -> str:
        """返回给定服务名的健康检查 URL。"""
        if name == "vllm":
            base = self.config.get("llm", {}).get(
                "base_url", "http://127.0.0.1:8000/v1"
            )
            return f"{str(base).rstrip('/')}/models"

        if name == "vlm":
            return "http://127.0.0.1:23333/health"

        if name == "sovits":
            return "http://127.0.0.1:9880/openapi.json"

        raise ValueError(f"unknown service identity: {name}")

    def _health_identity(self, name: str) -> dict:
        if name == "vllm":
            from sbmachine.common import resolve_backend

            semantic = self.config.get("semantic", {}) if isinstance(self.config.get("semantic", {}), dict) else {}
            llm = self.config.get("llm", {}) if isinstance(self.config.get("llm", {}), dict) else {}
            phases = self.config.get("phases", {}) if isinstance(self.config.get("phases", {}), dict) else {}
            legacy_active = bool(phases.get("phase3_semantic", True))
            active_roles = (
                ("analyst", bool(phases.get("phase3a_semantic", legacy_active))),
                ("style", bool(phases.get("phase3b_semantic", legacy_active))),
            )
            models: list[str] = []
            for role, active in active_roles:
                if not active or resolve_backend(self.config, role) != "vllm":
                    continue
                model = str(semantic.get(f"{role}_model") or semantic.get("model") or llm.get("model") or "")
                if model and model not in models:
                    models.append(model)
            return {"service": "vllm", "models": models}
        if name in {"vlm", "sovits"}:
            return {"service": name}
        raise ValueError(f"unknown service identity: {name}")

    # ── 启动 ──

    def start(self, name: str) -> None:
        if name in self._procs:
            return
        svc = self._svc_cfg.get(name, {})
        if not svc.get("enabled", True):
            print(f"[services] {name} disabled in config, skip", flush=True)
            return
        cmd = svc.get("start", "")
        if not cmd:
            print(f"[services] {name} has no start command, skip", flush=True)
            return

        timeout = int(svc.get("startup_timeout_sec", 60))
        print(f"[services] start {name}: {cmd}", flush=True)

        # 若服务已在运行（用户手动启动过），直接 health 确认即可
        health_url = self._health_url(name)
        identity = self._health_identity(name)
        already_up = self._poll_health(health_url, timeout_sec=3, interval=1.0, identity=identity)
        if already_up:
            print(f"[services] {name} already up (skipping spawn)", flush=True)
            # 记为 None 标记"已就绪但非我们启动"，stop 时不 kill
            self._procs[name] = None  # type: ignore[assignment]
        else:
            from sbmachine.common import PROJECT_ROOT
            tmp_dir = PROJECT_ROOT / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            log_fh = open(tmp_dir / f"{name}.log", "w")
            self._log_fhs[name] = log_fh
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(PROJECT_ROOT),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            self._procs[name] = proc
            if not self._poll_health(health_url, timeout, identity=identity):
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                self._procs.pop(name, None)
                log_fh.close()
                self._log_fhs.pop(name, None)
                raise RuntimeError(
                    f"[services] {name} did not become healthy within {timeout}s. "
                    f"Check tmp/{name}.log for details."
                )

        print(f"[services] {name} healthy", flush=True)


    # ── 停止 ──

    def stop(self, name: str) -> None:
        proc = self._procs.pop(name, None)
        log_fh = self._log_fhs.pop(name, None)
        if proc is None:
            return  # 已就绪但非我们启动 / 已停
        print(f"[services] stop {name}", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass

    def stop_all(self) -> None:
        for name in list(self._procs):
            self.stop(name)
