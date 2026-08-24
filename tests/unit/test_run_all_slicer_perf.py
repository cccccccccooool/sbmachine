"""run_all._slicer_perf_args：slicer workers/batch/节流决策矩阵。"""
from __future__ import annotations

import os

from sbmachine.run_all import _slicer_perf_args


def test_default_cpu_non_turbo_converges_workers_and_throttles(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    args = _slicer_perf_args({"workers": 8, "device": "cpu"}, {}, turbo=False)
    assert args["workers"] == 4  # min(8, 4, 16//4)
    assert args["batch_size"] == 1
    assert args["throttle"] is True
    assert args["cpu_ceiling"] == 0.8 and args["mem_ceiling"] == 0.8


def test_cpu_turbo_releases_workers_and_disables_throttle(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    args = _slicer_perf_args({"workers": 8, "device": "cpu"}, {"enabled": True}, turbo=True)
    assert args["workers"] == 8  # 配置全量
    assert args["throttle"] is False


def _fake_torch_gpu():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda()

    return FakeTorch()


def test_gpu_force_single_worker_and_gpu_batch(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch_gpu())
    args = _slicer_perf_args({"workers": 16, "device": "auto", "gpu_batch_size": 64}, {}, turbo=False)
    assert args["workers"] == 1
    assert args["batch_size"] == 64
    assert args["throttle"] is True  # 非 turbo 仍节流


def test_gpu_turbo_disables_throttle(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch_gpu())
    args = _slicer_perf_args({"workers": 16, "device": "cuda:0", "gpu_batch_size": 64}, {"enabled": True}, turbo=True)
    assert args["workers"] == 1
    assert args["throttle"] is False


def test_cpu_small_machine_keeps_at_least_one_worker(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    args = _slicer_perf_args({"workers": 8, "device": "cpu"}, {}, turbo=False)
    assert args["workers"] == 1  # 2//4=0 -> max(1,0)
    assert args["batch_size"] == 1


def test_custom_perf_ceiling_passthrough():
    args = _slicer_perf_args(
        {"workers": 2, "device": "cpu"},
        {"cpu_ceiling": 0.9, "mem_ceiling": 0.7, "throttle_sec": 2.0, "enabled": True},
        turbo=False,
    )
    assert args["cpu_ceiling"] == 0.9
    assert args["mem_ceiling"] == 0.7
    assert args["throttle_sec"] == 2.0


def test_perf_disabled_never_throttles(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    args = _slicer_perf_args({"workers": 8, "device": "cpu"}, {"enabled": False}, turbo=False)
    assert args["throttle"] is False
