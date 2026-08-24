"""轻量系统资源监控与节流。

供 video_marking 等重负载阶段的 worker 进程调用：系统 CPU/内存超过配置上限时
主动 sleep 放缓节奏，避免把整机打满（默认模式的安全阀；--turbo 可关闭）。
"""

from __future__ import annotations

import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover - 可选依赖
    psutil = None

_lock = threading.Lock()
_last_cpu_sample: float | None = None


def psutil_available() -> bool:
    return psutil is not None


def current_utilization() -> tuple[float | None, float | None]:
    """返回 (cpu_percent, mem_percent)，取值 0-100；psutil 缺失时返回 (None, None)。"""
    if psutil is None:
        return None, None
    with _lock:
        global _last_cpu_sample
        # interval=None：返回自上次采样以来的均值；内部用两次单调时钟差分计算，
        # 进程内首个采样点可能为 0，对节流判断是保守方向（不会误触发）。
        cpu = psutil.cpu_percent(interval=None)
        _last_cpu_sample = float(cpu)
    mem = float(psutil.virtual_memory().percent)
    return _last_cpu_sample, mem


def throttled(cpu_ceiling: float = 0.8, mem_ceiling: float = 0.8, throttle_sec: float = 1.0) -> bool:
    """资源超限时 sleep(throttle_sec) 并返回 True；未超限或 psutil 缺失返回 False。

    上限参数是 0-1 的比例（0.8 = 80%）。
    """
    if psutil is None or throttle_sec is None or float(throttle_sec) <= 0:
        return False
    cpu, mem = current_utilization()
    over_cpu = cpu is not None and cpu >= float(cpu_ceiling) * 100.0
    over_mem = mem is not None and mem >= float(mem_ceiling) * 100.0
    if over_cpu or over_mem:
        time.sleep(float(throttle_sec))
        return True
    return False
