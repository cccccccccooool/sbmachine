"""资源守卫（CPU/内存节流）单测。"""

from __future__ import annotations

import time

from sbmachine import resource_guard


def test_psutil_available_reflects_backend(monkeypatch):
    monkeypatch.setattr(resource_guard, "psutil", None)
    assert resource_guard.psutil_available() is False


def test_current_utilization_without_psutil(monkeypatch):
    monkeypatch.setattr(resource_guard, "psutil", None)
    assert resource_guard.current_utilization() == (None, None)


class _FakePsutil:
    def __init__(self, cpu: float, mem: float):
        self._cpu = cpu
        self._mem = mem

    def cpu_percent(self, interval=None):  # noqa: ARG002
        return self._cpu

    def virtual_memory(self):
        class _Mem:
            percent = self._mem

        return _Mem()


def test_throttled_sleeps_when_cpu_over_ceiling(monkeypatch):
    monkeypatch.setattr(resource_guard, "psutil", _FakePsutil(cpu=95.0, mem=30.0))
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(sec))
    assert resource_guard.throttled(cpu_ceiling=0.8, mem_ceiling=0.8, throttle_sec=2.0) is True
    assert sleeps == [2.0]


def test_throttled_sleeps_when_mem_over_ceiling(monkeypatch):
    monkeypatch.setattr(resource_guard, "psutil", _FakePsutil(cpu=40.0, mem=92.0))
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(sec))
    assert resource_guard.throttled(cpu_ceiling=0.8, mem_ceiling=0.8, throttle_sec=1.0) is True
    assert sleeps == [1.0]


def test_throttled_skips_when_under_ceiling(monkeypatch):
    monkeypatch.setattr(resource_guard, "psutil", _FakePsutil(cpu=30.0, mem=40.0))
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(sec))
    assert resource_guard.throttled(cpu_ceiling=0.8, mem_ceiling=0.8, throttle_sec=1.0) is False
    assert sleeps == []


def test_throttled_skips_without_psutil(monkeypatch):
    monkeypatch.setattr(resource_guard, "psutil", None)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(sec))
    assert resource_guard.throttled(throttle_sec=1.0) is False
    assert sleeps == []


def test_throttled_disabled_when_throttle_sec_zero(monkeypatch):
    monkeypatch.setattr(resource_guard, "psutil", _FakePsutil(cpu=99.0, mem=99.0))
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(sec))
    assert resource_guard.throttled(cpu_ceiling=0.8, mem_ceiling=0.8, throttle_sec=0.0) is False
    assert sleeps == []
