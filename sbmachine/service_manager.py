"""外部推理服务生命周期管理（VLM / SoVITS）。"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


class ServiceManager:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._procs: dict[str, subprocess.Popen] = {}
        self._log_fhs: dict[str, object] = {}
        self._svc_cfg: dict = config.get("runtime", {}).get("services", {})

    # ── health URL helpers（从现有 config 派生，无硬编码） ──

    def _health_url(self, name: str) -> str:
        if name == "vlm":
            ep = self.config.get("vision", {}).get("vlm", {}).get(
                "endpoint", "http://127.0.0.1:23333/v1/chat/completions"
            )
            p = urlparse(ep)
            return f"{p.scheme}://{p.netloc}/health"

        if name == "sovits":
            host, port = "127.0.0.1", 9880
            try:
                import yaml
                from sbmachine.common import resolve_path
                tts_rel = self.config.get("tts", {}).get(
                    "config", "audio_service/gpt_sovits_runtime.yaml"
                )
                p = resolve_path(tts_rel)
                if p and p.exists():
                    rt = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    host = rt.get("api", {}).get("host", host)
                    port = int(rt.get("api", {}).get("port", port))
            except Exception:
                pass
            return f"http://{host}:{port}/"

        raise ValueError(f"Unknown service name: {name}")

    # ── 健康轮询 ──

    @staticmethod
    def _poll_health(url: str, timeout_sec: int, interval: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                r = requests.get(url, timeout=3)
                if r.status_code < 500:
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False

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
        already_up = self._poll_health(health_url, timeout_sec=3, interval=1.0)
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
            if not self._poll_health(health_url, timeout):
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
