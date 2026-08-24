#!/usr/bin/env python3
"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：GPU Guard - 共享 GPU 显存守护/资源管理器。

在多人共享 GPU 的开发机环境下，防止其他人的后台程序抢走空闲显存：
按安全边际与占用比例（reserve_ratio）霸占可用显存，并通过本地 Socket 接收客户端的租约申请；
己方启动训练任务时释放对应显存，任务结束后自动收回并再次占满。

输入数据流：无（通过 nvidia-smi 查询 GPU 状态，通过 Socket 接收客户端请求）。
输出数据流：日志文件 (/tmp/.gpu_guard_sandbox_<uid>/gpu_guard.log) 与状态信息到 stdout。

使用方法：
    1. 启动守护进程：
        python tools/gpu_guard.py start [--device <id>] [--chunk-size <mb>] [--reserve-ratio <float>] [--poll-interval <sec>]
    2. 托管运行己方训练任务（按需临时释放并挂起哨兵）：
        python tools/gpu_guard.py run "python train.py --epochs 10"
    3. 查询显存分配与霸占状态：
        python tools/gpu_guard.py status
    4. 手动让出并暂停霸占 / 恢复霸占：
        python tools/gpu_guard.py release  |  python tools/gpu_guard.py resume
    5. 停止守护进程：
        python tools/gpu_guard.py stop
"""

import os
import sys
import json
import time
import signal
import socket
import struct
import subprocess
import threading
import argparse
import logging
import gc
import atexit
from typing import List, Optional, Dict

# ============================================================================
# 配置与状态隔离
# ============================================================================

_uid = os.getuid()
GUARD_DIR = f"/tmp/.gpu_guard_sandbox_{_uid}"

SOCKET_PATH = f"{GUARD_DIR}/gpu_guard.sock"
PID_FILE = f"{GUARD_DIR}/gpu_guard.pid"
LOG_FILE = f"{GUARD_DIR}/gpu_guard.log"

# 显存预留参数
SAFETY_MARGIN_MB = 300              # 给CUDA驱动/运行时保留的基础空间
CUDA_CONTEXT_OVERHEAD_MB = 500      # 新进程建CUDA context的开销(300-600MB，取中上)

# 抢占策略参数
SENTINEL_INTERVAL = 0.5             # 哨兵线程巡检间隔(秒)
SENTINEL_TRIGGER_MB = 64            # 哨兵触发阈值：空闲超过此值才抢占
ADAPTIVE_MIN_INTERVAL = 0.5         # 自适应巡逻最小间隔(秒)
ADAPTIVE_MAX_INTERVAL = 5.0         # 自适应巡逻最大间隔(秒)
CONTEST_DECAY_SECONDS = 30          # 竞争状态衰减时间(秒)

# ============================================================================
# 日志设置
# ============================================================================

def setup_logging(daemon_mode: bool = False) -> logging.Logger:
    """配置日志：守护进程写入日志文件，前台模式打印到 stdout。"""
    handlers = []
    if daemon_mode:
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers
    )
    return logging.getLogger("gpu_guard")


# ============================================================================
# GPU 监控工具
# ============================================================================

class GPUMonitor:
    """封装 nvidia-smi 查询，提供带短 TTL 缓存的 GPU 显存/进程信息。"""

    _cache = None
    _cache_time = 0
    _cache_ttl = 0.3

    @classmethod
    def query(cls, force=False) -> Dict:
        """查询各 GPU 的显存与利用率；带短 TTL 缓存，force=True 强制刷新。"""
        now = time.time()
        if not force and cls._cache and (now - cls._cache_time) < cls._cache_ttl:
            return cls._cache
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return {"error": f"nvidia-smi failed: {result.stderr.strip()}"}
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_mb": int(parts[2]),
                        "used_mb": int(parts[3]),
                        "free_mb": int(parts[4]),
                        "gpu_util": int(parts[5]),
                    })
            data = {"gpus": gpus}
            cls._cache = data
            cls._cache_time = now
            return data
        except FileNotFoundError:
            return {"error": "nvidia-smi not found"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_gpu_processes() -> List[Dict]:
        """列出当前占用 GPU 的计算进程（pid、显存、进程名）。"""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-compute-apps=pid,used_gpu_memory,name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            processes = []
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        processes.append({
                            "pid": int(parts[0]),
                            "memory_mb": int(parts[1]),
                            "name": parts[2],
                        })
            return processes
        except Exception:
            return []

    @staticmethod
    def is_our_process(pid: int) -> bool:
        """判断给定 pid 是否属于当前用户（比对 /proc/<pid>/status 的 Uid）。"""
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                for line in f:
                    if line.startswith("Uid:"):
                        proc_uid = int(line.split()[1])
                        return proc_uid == os.getuid()
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        return False


# ============================================================================
# GPU 显存霸占器 v3
#
# 修复:
#   - reserve_ratio 实际生效，控制最大占用上限
#   - Phase1 OOM 时自动折半重试
#   - task_active 标志阻止巡逻抢占任务显存
# ============================================================================

class GPUOccupier:
    """
    三级抢占策略:
      Phase 1 - 大块扫荡（OOM时自动折半: 64→32→16→...）
      Phase 2 - 碎片填充（8→4 MB）
      Phase 3 - 极限填充（2 MB 微块）
    """

    def __init__(self, device_id: int = 0, chunk_size_mb: int = 64,
                 reserve_ratio: float = 0.95, shutdown_event: threading.Event = None, logger=None):
        self.device_id = device_id
        self.chunk_size_mb = chunk_size_mb
        self.reserve_ratio = reserve_ratio
        self.logger = logger or logging.getLogger("gpu_guard")
        self._shutdown_event = shutdown_event or threading.Event()
        self.chunks: List = []
        self.lock = threading.Lock()          # 仅保护 chunks 列表（短临界区）
        self.occupy_lock = threading.RLock()  # 串行化 occupy/quick_grab/reoccupy/release_all
        self._torch = None
        self._device = None
        self._total_vram_mb = 0

        # ★ 核心修复: 任务执行互斥标志
        # 当此标志为 True 时，巡逻/哨兵线程必须停止抢占
        # 生命周期: 在 run_task 释放显存之前 set → 重新霸占之后 clear
        self.task_active = False
        # ★ 手动暂停标志：release 命令置位、resume 清除。
        #   置位期间巡逻/哨兵停止抢占（与 task_active 等效），用于 WebUI 等交互式训练。
        self.paused = False
        self.task_active_lock = threading.Lock()

        self._init_torch()

    def _init_torch(self):
        try:
            import torch
            self._torch = torch
            if not torch.cuda.is_available():
                self.logger.error("CUDA 不可用！")
                sys.exit(1)
            self._device = torch.device(f"cuda:{self.device_id}")
            torch.cuda.set_device(self._device)

            self._total_vram_mb = torch.cuda.get_device_properties(
                self.device_id
            ).total_memory // (1024 * 1024)

            # 预热CUDA分配器
            warmup = torch.zeros(1024, dtype=torch.float32, device=self._device)
            del warmup
            torch.cuda.empty_cache()

            try:
                os.environ.setdefault(
                    "PYTORCH_CUDA_ALLOC_CONF",
                    "expandable_segments:True"
                )
            except Exception:
                pass

            self.logger.info(
                f"PyTorch CUDA 初始化 | "
                f"设备: {torch.cuda.get_device_name(self.device_id)} | "
                f"总显存: {self._total_vram_mb}MB | "
                f"占用上限: {int(self._total_vram_mb * self.reserve_ratio)}MB "
                f"({self.reserve_ratio*100:.0f}%)"
            )
        except ImportError:
            self.logger.error("未找到 PyTorch！请安装: pip install torch")
            sys.exit(1)

    @property
    def max_occupy_mb(self) -> int:
        """★ reserve_ratio 实际控制的最大占用上限"""
        return int(self._total_vram_mb * self.reserve_ratio)

    @property
    def busy(self) -> bool:
        """任务运行中(task_active) 或 手动暂停(paused) 或 正在关闭 时为 True —— 此时不抢占。"""
        return self.task_active or self.paused or self._shutdown_event.is_set()

    @property
    def occupied_mb(self) -> int:
        with self.lock:
            return sum(size for _, size in self.chunks)

    @property
    def chunk_count(self) -> int:
        with self.lock:
            return len(self.chunks)

    def _alloc_chunk(self, size_mb: int) -> bool:
        """
        尝试分配一个块。
        ★ 修复: 昂贵的 torch.zeros 在锁外执行，仅 chunks.append 持 self.lock，
          使得占用进行时 status/occupied_mb 等读取不会被长时间阻塞。
          并发霸占由更外层的 occupy_lock 串行化。
        """
        torch = self._torch
        try:
            num_elements = (size_mb * 1024 * 1024) // 4
            tensor = torch.zeros(num_elements, dtype=torch.float32, device=self._device)
        except RuntimeError:
            torch.cuda.empty_cache()
            return False
        with self.lock:
            self.chunks.append((tensor, size_mb))
        return True

    def occupy(self, target_mb: Optional[int] = None) -> int:
        """
        三级抢占。

        ★ 修复1: 受 reserve_ratio 上限约束
        ★ 修复2: Phase1 OOM 自动折半重试（64→32→16→8）
        ★ 修复3: 检查 task_active 标志，任务运行期间不抢占
        """
        # ★ 检查任务互斥
        with self.task_active_lock:
            if self.busy:
                return 0

        gpu_info = GPUMonitor.query(force=True)
        if "error" in gpu_info:
            self.logger.warning(f"GPU查询失败: {gpu_info['error']}")
            return 0

        if self.device_id >= len(gpu_info["gpus"]):
            self.logger.warning(f"设备 {self.device_id} 不存在")
            return 0

        gpu = gpu_info["gpus"][self.device_id]
        free_mb = gpu["free_mb"]

        if target_mb is None:
            target_mb = max(0, free_mb - SAFETY_MARGIN_MB)

        if target_mb <= 0:
            return 0

        allocated = 0

        with self.occupy_lock:
            # ★ reserve_ratio 上限约束：headroom 在 occupy_lock 内计算，
            #   避免与哨兵/巡逻并发时用到过期快照而一起越过上限
            headroom = self.max_occupy_mb - self.occupied_mb
            target_mb = min(target_mb, headroom) if headroom > 0 else 0

            # === Phase 1: 大块扫荡 + OOM 自动折半 ===
            current_chunk = self.chunk_size_mb
            min_phase1_chunk = 8  # 折半下限

            while current_chunk >= min_phase1_chunk and allocated < target_mb:
                consecutive_fails = 0
                # ★ 修复: 用 allocated+chunk<=target 而非 allocated<target，
                #   避免最后一块跨过上限（之前会越界一个 chunk）
                while allocated + current_chunk <= target_mb and consecutive_fails < 2:
                    # ★ 每次分配前再次检查互斥（避免长循环中被卡住）
                    if self.busy:
                        break
                    if self._alloc_chunk(current_chunk):
                        allocated += current_chunk
                        consecutive_fails = 0
                    else:
                        consecutive_fails += 1

                # ★ 当前粒度失败/装不下，折半重试
                current_chunk //= 2

            # === Phase 2: 碎片填充（4 MB） ===
            # ★ 修复: 加入 allocated+frag<=target，使 reserve_ratio 上限真正生效且不越界
            for frag_size in [4]:
                consecutive_fails = 0
                while allocated + frag_size <= target_mb and consecutive_fails < 2:
                    if self.busy:
                        break
                    if self._alloc_chunk(frag_size):
                        allocated += frag_size
                        consecutive_fails = 0
                    else:
                        consecutive_fails += 1

            # === Phase 3: 微块极限填充（2 MB） ===
            # ★ 修复: 同样 allocated+2<=target，最终误差 ≤2MB 且不越上限
            consecutive_fails = 0
            while allocated + 2 <= target_mb and consecutive_fails < 3:
                if self.busy:
                    break
                if self._alloc_chunk(2):
                    allocated += 2
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1

        if allocated > 0:
            self.logger.info(
                f"🔒 新占用 {allocated}MB | "
                f"总占用: {self.occupied_mb}MB / {self.max_occupy_mb}MB上限 | "
                f"块数: {self.chunk_count}"
            )
        return allocated

    def quick_grab(self) -> int:
        """
        快速抢占 — 仅大块，用于哨兵线程。
        同样受 task_active 和 reserve_ratio 约束。
        """
        if self.busy:
            return 0

        gpu_info = GPUMonitor.query(force=True)
        if "error" in gpu_info:
            return 0
        if self.device_id >= len(gpu_info["gpus"]):
            return 0

        gpu = gpu_info["gpus"][self.device_id]
        free_mb = gpu["free_mb"]
        target = max(0, free_mb - SAFETY_MARGIN_MB)

        if target < self.chunk_size_mb:
            return 0

        allocated = 0
        with self.occupy_lock:
            # reserve_ratio 约束：headroom 在锁内计算，避免并发过期快照越界
            headroom = self.max_occupy_mb - self.occupied_mb
            target = min(target, headroom)
            while target - allocated >= self.chunk_size_mb:
                if self.busy:
                    break
                if self._alloc_chunk(self.chunk_size_mb):
                    allocated += self.chunk_size_mb
                else:
                    break

        if allocated > 0:
            self.logger.info(f"⚡ 快速抢占 {allocated}MB")
        return allocated

    def release(self, amount_mb: int) -> int:
        """LIFO释放指定量的显存"""
        torch = self._torch
        released = 0
        with self.lock:
            while released < amount_mb and self.chunks:
                tensor, size = self.chunks.pop()
                del tensor
                released += size
        gc.collect()
        torch.cuda.empty_cache()
        if released > 0:
            self.logger.info(
                f"🔓 释放 {released}MB | "
                f"剩余占用: {self.occupied_mb}MB | "
                f"块数: {self.chunk_count}"
            )
        return released

    def release_all(self):
        torch = self._torch
        with self.occupy_lock:
            total = self.occupied_mb
            with self.lock:
                for tensor, _ in self.chunks:
                    del tensor
                self.chunks.clear()
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info(f"🔓 全部释放 {total}MB")


# ============================================================================
# 任务执行器 v4 — 租约模式
#
# ★ v4 改动：任务不再在守护进程内执行，而是由客户端在“你自己的终端”本地原生
#   执行（输出/进度条/颜色零转发，和直接运行一模一样）。守护进程只负责
#   “释放显存 + 暂停抢占”和“恢复霸占”，并通过一条常驻的租约连接监测客户端：
#   客户端正常结束 / Ctrl+C / 崩溃断连，守护进程都会立即自动收回显存。
#   TaskRunner 退化为纯状态持有者，供 status 展示与并发串行化。
# ============================================================================

class TaskRunner:
    def __init__(self, occupier: GPUOccupier, logger=None):
        self.occupier = occupier
        self.logger = logger or logging.getLogger("gpu_guard")
        self.running_tasks: Dict[int, dict] = {}
        self.task_lock = threading.Lock()
        # 串行化租约：并发 run 排队，保证同一时刻只有一个任务在用让出的显存
        self.exec_lock = threading.Lock()


# ============================================================================
# Socket 通信协议
# ============================================================================

def send_msg(sock: socket.socket, data: dict):
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack("!I", len(raw)) + raw)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    try:
        header = b""
        while len(header) < 4:
            chunk = sock.recv(4 - len(header))
            if not chunk:
                return None
            header += chunk
        length = struct.unpack("!I", header)[0]
        if length > 50 * 1024 * 1024:
            return None
        data = b""
        while len(data) < length:
            chunk = sock.recv(min(65536, length - len(data)))
            if not chunk:
                return None
            data += chunk
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# ============================================================================
# 守护进程 v3
# ============================================================================

class GPUGuardDaemon:
    def __init__(self, device_id=0, chunk_size_mb=64,
                 reserve_ratio=0.95, poll_interval=3):
        self.device_id = device_id
        self.chunk_size_mb = chunk_size_mb
        self.reserve_ratio = reserve_ratio
        self.base_poll_interval = poll_interval
        self.running = False
        self._cleaned = False
        self._shutdown_event = threading.Event()
        self._lease_conns = []          # 跟踪活跃租约连接
        self._lease_conns_lock = threading.Lock()   # ★ 防止 _cleanup 被 atexit 与主循环重复执行
        self.logger = setup_logging(daemon_mode=True)

        # ★ reserve_ratio 传入 occupier，实际生效
        self.occupier = GPUOccupier(
            device_id, chunk_size_mb,
            reserve_ratio=reserve_ratio,
            shutdown_event=self._shutdown_event,
            logger=self.logger
        )
        self.task_runner = TaskRunner(self.occupier, self.logger)
        self.start_time = None
        self._cleanup_lock = threading.Lock()

        # 自适应巡逻状态
        self._last_contest_time = 0
        self._contest_count = 0
        self._total_grabbed_mb = 0
        self._patrol_stats = {"patrols": 0, "grabs": 0, "grabbed_mb": 0}

    def start(self):
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        self.running = True
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("GPU Guard v3 守护进程启动")
        self.logger.info(f"PID: {os.getpid()}")
        self.logger.info(f"设备: cuda:{self.device_id}")
        self.logger.info(f"块大小: {self.chunk_size_mb}MB")
        self.logger.info(f"占用上限: {self.reserve_ratio*100:.0f}% "
                         f"({self.occupier.max_occupy_mb}MB)")
        self.logger.info(f"安全边际: {SAFETY_MARGIN_MB}MB")
        self.logger.info(f"Context预留: {CUDA_CONTEXT_OVERHEAD_MB}MB")
        self.logger.info(f"基准巡逻: {self.base_poll_interval}s | "
                         f"哨兵: {SENTINEL_INTERVAL}s")
        self.logger.info(f"Socket: {SOCKET_PATH}")
        self.logger.info("=" * 60)

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        atexit.register(self._cleanup)

        # 初始霸占
        self.logger.info("🚀 初始显存霸占（三级策略）...")
        grabbed = self.occupier.occupy()
        self.logger.info(
            f"🚀 初始霸占完成: {grabbed}MB | "
            f"总占用: {self.occupier.occupied_mb}MB / "
            f"{self.occupier.max_occupy_mb}MB上限"
        )

        # 启动哨兵线程
        threading.Thread(
            target=self._sentinel_loop, daemon=True, name="sentinel"
        ).start()

        # 启动巡逻线程
        threading.Thread(
            target=self._patrol_loop, daemon=True, name="patrol"
        ).start()

        # 主线程: socket服务
        self._run_server()

    def _handle_signal(self, signum, frame):
        self.logger.info(f"收到信号 {signum}，正在停止...")
        self.running = False
        self._shutdown_event.set()
        self._close_all_lease_conns()

    def _close_all_lease_conns(self):
        """关闭所有活跃租约连接，使阻塞在 recv() 的租约线程立即返回。"""
        with self._lease_conns_lock:
            for conn in self._lease_conns:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    conn.close()
                except OSError:
                    pass

    def _cleanup(self):
        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True
        self.logger.info("🧹 清理资源...")
        # ★ 先关闭所有租约连接，触发租约线程的 finally 块
        self._close_all_lease_conns()
        self.occupier.release_all()
        for path in [SOCKET_PATH, PID_FILE]:
            try:
                os.unlink(path)
            except OSError:
                pass
        self.logger.info("GPU Guard 已停止")

    # ------------------------------------------------------------------
    # 哨兵线程
    # ------------------------------------------------------------------

    def _sentinel_loop(self):
        self.logger.info(
            f"🛡️ 哨兵启动 | 间隔: {SENTINEL_INTERVAL}s | "
            f"触发阈值: {SENTINEL_TRIGGER_MB}MB"
        )
        while not self._shutdown_event.is_set():
            try:
                if self._shutdown_event.wait(SENTINEL_INTERVAL):
                    break  # 事件触发 = 正在关闭

                # ★ 检查 task_active（在 occupier 上），不抢
                with self.occupier.task_active_lock:
                    if self.occupier.busy:
                        continue

                gpu_info = GPUMonitor.query()
                if "error" in gpu_info:
                    continue
                if self.device_id >= len(gpu_info["gpus"]):
                    continue

                gpu = gpu_info["gpus"][self.device_id]
                if gpu["free_mb"] > SENTINEL_TRIGGER_MB + SAFETY_MARGIN_MB:
                    grabbed = self.occupier.quick_grab()
                    if grabbed > 0:
                        self._on_contest(grabbed)

            except Exception as e:
                self.logger.error(f"哨兵异常: {e}")

    # ------------------------------------------------------------------
    # 自适应巡逻线程
    # ------------------------------------------------------------------

    def _get_adaptive_interval(self) -> float:
        now = time.time()
        since = now - self._last_contest_time
        if since < CONTEST_DECAY_SECONDS:
            ratio = since / CONTEST_DECAY_SECONDS
            return max(ADAPTIVE_MIN_INTERVAL,
                       ADAPTIVE_MIN_INTERVAL + ratio * (
                           self.base_poll_interval - ADAPTIVE_MIN_INTERVAL))
        else:
            extra = since - CONTEST_DECAY_SECONDS
            slowdown = min(extra / 60.0, 1.0)
            return min(ADAPTIVE_MAX_INTERVAL,
                       self.base_poll_interval + slowdown * (
                           ADAPTIVE_MAX_INTERVAL - self.base_poll_interval))

    def _on_contest(self, grabbed_mb: int):
        self._last_contest_time = time.time()
        self._contest_count += 1
        self._total_grabbed_mb += grabbed_mb
        self._patrol_stats["grabs"] += 1
        self._patrol_stats["grabbed_mb"] += grabbed_mb

    def _patrol_loop(self):
        self.logger.info("🔄 自适应巡逻启动")
        last_status_log = 0

        while not self._shutdown_event.is_set():
            try:
                interval = self._get_adaptive_interval()
                if self._shutdown_event.wait(interval):
                    break  # 事件触发 = 正在关闭

                self._patrol_stats["patrols"] += 1

                # ★ 检查 task_active
                with self.occupier.task_active_lock:
                    if self.occupier.busy:
                        continue

                gpu_info = GPUMonitor.query(force=True)
                if "error" in gpu_info:
                    continue
                if self.device_id >= len(gpu_info["gpus"]):
                    continue

                gpu = gpu_info["gpus"][self.device_id]
                free_mb = gpu["free_mb"]

                if free_mb > self.chunk_size_mb + SAFETY_MARGIN_MB:
                    self.logger.info(
                        f"🔍 巡逻发现空闲 {free_mb}MB | "
                        f"间隔: {interval:.1f}s | 三级抢占..."
                    )
                    grabbed = self.occupier.occupy()
                    if grabbed > 0:
                        self._on_contest(grabbed)

                # 状态报告（每60秒）
                now = time.time()
                if now - last_status_log >= 60:
                    last_status_log = now
                    procs = GPUMonitor.get_gpu_processes()
                    our = [p for p in procs if GPUMonitor.is_our_process(p["pid"])]
                    other = [p for p in procs if not GPUMonitor.is_our_process(p["pid"])]
                    uptime = now - self.start_time
                    self.logger.info(
                        f"📊 报告 | 运行: {_format_duration(uptime)} | "
                        f"占用: {self.occupier.occupied_mb}MB/"
                        f"{self.occupier.max_occupy_mb}MB | "
                        f"空闲: {gpu['free_mb']}MB | 利用率: {gpu['gpu_util']}% | "
                        f"己方: {len(our)} | 他方: {len(other)} | "
                        f"间隔: {interval:.1f}s | "
                        f"抢占: {self._patrol_stats['grabs']}次/"
                        f"{self._patrol_stats['grabbed_mb']}MB"
                    )

            except Exception as e:
                self.logger.error(f"巡逻异常: {e}")
                self._shutdown_event.wait(5)

    # ------------------------------------------------------------------
    # Socket 服务
    # ------------------------------------------------------------------

    def _run_server(self):
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o600)
        server.listen(5)
        server.settimeout(1.0)
        self.logger.info("📡 Socket服务已启动")

        while not self._shutdown_event.is_set():
            try:
                conn, _ = server.accept()
                threading.Thread(
                    target=self._handle_client, args=(conn,), daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"Socket异常: {e}")
                break

        server.close()
        self._cleanup()

    def _handle_client(self, conn: socket.socket):
        try:
            msg = recv_msg(conn)
            if not msg:
                return

            action = msg.get("action", "")
            self.logger.info(f"📨 收到请求: {action}")

            if action == "status":
                response = self._get_status()
                send_msg(conn, response)
            elif action == "run":
                # ★ v4 租约模式：在此函数内完成 ack + 持连监测 + 收回，
                #   不再走统一的 send_msg(response)
                self._handle_run_lease(conn, msg)
                return
            elif action == "release":
                response = self._handle_release(msg.get("memory_mb", 0))
                send_msg(conn, response)
            elif action == "resume":
                response = self._handle_resume()
                send_msg(conn, response)
            elif action == "stop":
                response = {"status": "stopping"}
                self.running = False
                self._shutdown_event.set()
                self._close_all_lease_conns()
                send_msg(conn, response)
            elif action == "ping":
                response = {"status": "alive", "pid": os.getpid()}
                send_msg(conn, response)
            else:
                response = {"error": f"未知操作: {action}"}
                send_msg(conn, response)
        except Exception as e:
            self.logger.error(f"处理请求异常: {e}")
            try:
                send_msg(conn, {"error": str(e)})
            except Exception:
                pass
        finally:
            conn.close()

    def _handle_run_lease(self, conn: socket.socket, msg: dict):
        """
        ★ v4 租约模式核心。
        任务在客户端本地执行，守护进程只负责让出/收回显存：
          1. 串行化(exec_lock)：同一时刻只有一个租约
          2. 置 task_active=True（停止抢占）并释放显存
          3. 回 ack（告知客户端显存已就绪，可以开跑）
          4. 保持连接：阻塞读，直到客户端断开（任务结束/Ctrl+C/崩溃）
          5. 收回显存、清 task_active —— 无论客户端如何退出都会执行
        """
        memory_mb = msg.get("memory_mb", 0)
        command = msg.get("command", "")   # 仅用于 status 展示，可为空
        task_id = int(time.time() * 1000)
        self.logger.info(
            f"📋 租约请求 [{task_id}] | 释放: "
            f"{'全部' if memory_mb <= 0 else f'{memory_mb}MB'}"
        )

        # 串行化：同一时刻只允许一个租约持有让出的显存
        with self.task_runner.exec_lock:
            with self.occupier.occupy_lock:
                with self.occupier.task_active_lock:
                    self.occupier.task_active = True

                if memory_mb and memory_mb > 0:
                    released = self.occupier.release(memory_mb + CUDA_CONTEXT_OVERHEAD_MB)
                else:
                    released = self.occupier.occupied_mb
                    self.occupier.release_all()

            lease = {
                "task_id": task_id,
                "command": command,
                "released_mb": released,
                "start_time": time.time(),
                "memory_mb": memory_mb,
            }
            with self.task_lock_or_runner():
                self.task_runner.running_tasks[task_id] = lease

            # 注册租约连接，使关闭时可以主动断开以解除 recv() 阻塞
            with self._lease_conns_lock:
                self._lease_conns.append(conn)

            self.logger.info(f"📋 租约 [{task_id}] 已释放 {released}MB，巡逻暂停，等待客户端...")

            try:
                # 回 ack：显存已就绪
                send_msg(conn, {
                    "status": "leased",
                    "task_id": task_id,
                    "released_mb": released,
                })

                # 保持连接，阻塞直到客户端断开。客户端跑任务期间不会发任何数据；
                # recv 返回空 = 对端关闭（正常结束/Ctrl+C/崩溃都会触发）。
                conn.settimeout(None)
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break  # 客户端断开 → 租约结束
                    # 预留：客户端可发送显式 "done" 提前结束（当前不依赖）
            except (OSError, socket.error):
                pass
            finally:
                # 注销租约连接
                with self._lease_conns_lock:
                    try:
                        self._lease_conns.remove(conn)
                    except ValueError:
                        pass
                with self.task_lock_or_runner():
                    self.task_runner.running_tasks.pop(task_id, None)
                with self.occupier.task_active_lock:
                    self.occupier.task_active = False
                # ★ 核心修复：守护进程关闭中时跳过重新霸占，
                #   否则 _cleanup().release_all() 之后又 occupy() 会导致
                #   孤立的 CUDA tensor 锁死整张卡的显存
                if not self._shutdown_event.is_set():
                    grabbed = self.occupier.occupy()
                    self.logger.info(
                        f"📋 租约 [{task_id}] 结束，已收回 +{grabbed}MB，"
                        f"当前 {self.occupier.occupied_mb}MB，巡逻恢复"
                    )
                else:
                    self.logger.info(
                        f"📋 租约 [{task_id}] 结束"
                        f"（守护进程关闭中，跳过重新霸占）"
                    )

    def task_lock_or_runner(self):
        """running_tasks 的写锁（复用 task_runner.task_lock）。"""
        return self.task_runner.task_lock

    def _handle_release(self, memory_mb: int = 0) -> dict:
        """手动让出显存并暂停霸占（巡逻/哨兵停止抢占），等待 resume。"""
        with self.occupier.occupy_lock:
            with self.occupier.task_active_lock:
                self.occupier.paused = True
            if memory_mb and memory_mb > 0:
                released = self.occupier.release(memory_mb + CUDA_CONTEXT_OVERHEAD_MB)
            else:
                released = self.occupier.occupied_mb
                self.occupier.release_all()
        self.logger.info(
            f"⏸️  手动释放 {released}MB，已暂停霸占（等待 resume）"
        )
        return {
            "status": "released",
            "released_mb": released,
            "occupied_mb": self.occupier.occupied_mb,
            "paused": True,
        }

    def _handle_resume(self) -> dict:
        """解除暂停并重新霸占显存。"""
        with self.occupier.task_active_lock:
            self.occupier.paused = False
        if not self._shutdown_event.is_set():
            grabbed = self.occupier.occupy()
        else:
            grabbed = 0
        self.logger.info(
            f"▶️  恢复霸占 +{grabbed}MB，当前 {self.occupier.occupied_mb}MB"
        )
        return {
            "status": "resumed",
            "grabbed_mb": grabbed,
            "occupied_mb": self.occupier.occupied_mb,
            "paused": False,
        }

    def _get_status(self) -> dict:
        gpu_info = GPUMonitor.query(force=True)
        procs = GPUMonitor.get_gpu_processes()
        our = [p for p in procs if GPUMonitor.is_our_process(p["pid"])]
        other = [p for p in procs if not GPUMonitor.is_our_process(p["pid"])]
        uptime = time.time() - self.start_time if self.start_time else 0

        return {
            "status": "running",
            "pid": os.getpid(),
            "uptime_seconds": round(uptime, 1),
            "uptime_human": _format_duration(uptime),
            "device_id": self.device_id,
            "occupied_mb": self.occupier.occupied_mb,
            "max_occupy_mb": self.occupier.max_occupy_mb,
            "chunk_count": self.occupier.chunk_count,
            "chunk_size_mb": self.chunk_size_mb,
            "reserve_ratio": self.reserve_ratio,
            "task_active": self.occupier.task_active,
            "paused": self.occupier.paused,
            "base_poll_interval": self.base_poll_interval,
            "current_poll_interval": round(self._get_adaptive_interval(), 2),
            "gpu_info": gpu_info,
            "our_processes": our,
            "other_processes": other,
            "running_tasks": list(self.task_runner.running_tasks.values()),
            "contest_count": self._contest_count,
            "total_grabbed_mb": self._total_grabbed_mb,
            "patrol_stats": self._patrol_stats,
        }


# ============================================================================
# 客户端 — 环境转发
# ============================================================================

class GPUGuardClient:
    @staticmethod
    def _send_request(request: dict, timeout: float = 300) -> dict:
        if not os.path.exists(SOCKET_PATH):
            return {"error": "守护进程未运行。请先执行: python gpu_guard.py start"}
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(SOCKET_PATH)
            send_msg(sock, request)
            response = recv_msg(sock)
            sock.close()
            return response or {"error": "未收到响应"}
        except ConnectionRefusedError:
            return {"error": "无法连接到守护进程。请检查是否已启动。"}
        except socket.timeout:
            return {"error": f"请求超时 ({timeout}s)"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def status():
        result = GPUGuardClient._send_request({"action": "status"}, timeout=10)
        if "error" in result:
            print(f"\n❌ {result['error']}")
            return

        print("\n" + "=" * 60)
        print("  GPU Guard v3 状态报告")
        print("=" * 60)
        print(f"  状态: 🟢 运行中")
        print(f"  PID:  {result['pid']}")
        print(f"  运行时间: {result['uptime_human']}")
        print(f"  设备: cuda:{result['device_id']}")
        print()

        # 显存
        max_occ = result.get('max_occupy_mb', '?')
        ratio = result.get('reserve_ratio', 0)
        print(f"  📦 占用: {result['occupied_mb']}MB / {max_occ}MB上限 "
              f"({result['chunk_count']}块) | ratio={ratio*100:.0f}%")

        if result.get('task_active'):
            print(f"  ⏸️  巡逻已暂停（任务运行中）")

        # GPU 状态
        gpu_info = result.get("gpu_info", {})
        gpus = gpu_info.get("gpus", [])
        device_id = result.get('device_id', 0)
        if device_id < len(gpus):
            gpu = gpus[device_id]
            bar_len = 30
            used_ratio = gpu["used_mb"] / gpu["total_mb"] if gpu["total_mb"] > 0 else 0
            filled = int(bar_len * used_ratio)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\n  🖥️  GPU: {gpu['name']}")
            print(f"  显存: [{bar}] {gpu['used_mb']}/{gpu['total_mb']}MB "
                  f"({used_ratio*100:.1f}%)")
            print(f"  GPU利用率: {gpu['gpu_util']}%")

        # 巡逻
        stats = result.get("patrol_stats", {})
        print(f"\n  🔄 巡逻:")
        print(f"     当前间隔: {result.get('current_poll_interval', '?')}s")
        print(f"     巡逻/抢占: {stats.get('patrols', 0)}次 / "
              f"{stats.get('grabs', 0)}次 ({stats.get('grabbed_mb', 0)}MB)")
        print(f"     竞争检测: {result.get('contest_count', 0)}次")

        # 进程
        our = result.get("our_processes", [])
        other = result.get("other_processes", [])
        print(f"\n  👤 己方: {len(our)}")
        for p in our:
            print(f"     PID {p['pid']:>7} | {p['memory_mb']:>6}MB | {p['name']}")
        print(f"  👥 他方: {len(other)}")
        for p in other:
            print(f"     PID {p['pid']:>7} | {p['memory_mb']:>6}MB | {p['name']}")

        tasks = result.get("running_tasks", [])
        if tasks:
            print(f"\n  📋 任务: {len(tasks)}")
            for t in tasks:
                elapsed = time.time() - t.get("start_time", time.time())
                cmd = (t.get("command") or "(本地任务)")[:50]
                print(f"     [{t.get('task_id', '?')}] 让出 {t.get('released_mb', '?')}MB | "
                      f"{_format_duration(elapsed)} | {cmd}")

        print("\n" + "=" * 60)

    @staticmethod
    def run(command: str, memory_mb: int = 0):
        """
        ★ v4 租约模式：守护进程释放显存 → 任务在本地终端原生执行 → 结束自动收回。

        与直接运行完全一致（实时输出、颜色、进度条、stdin 交互都正常），
        没有任何输出转发或环境序列化。断开/Ctrl+C 时守护进程自动收回显存。
        """
        if not os.path.exists(SOCKET_PATH):
            print("❌ 守护进程未运行。请先执行: python gpu_guard.py start")
            return

        print(f"\n🚀 提交任务: {command}")
        if memory_mb > 0:
            print(f"   让出显存: {memory_mb}MB (+ {CUDA_CONTEXT_OVERHEAD_MB}MB context预留)")
        else:
            print(f"   让出显存: 全部")

        # 1) 建立租约连接：通知守护进程释放显存并暂停抢占
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(120)
            sock.connect(SOCKET_PATH)
            send_msg(sock, {"action": "run", "memory_mb": memory_mb, "command": command})
            ack = recv_msg(sock)
        except Exception as e:
            print(f"❌ 无法建立租约: {e}")
            return

        if not ack or ack.get("status") != "leased":
            print(f"❌ 租约失败: {ack.get('error') if ack else '无响应'}")
            try:
                sock.close()
            except Exception:
                pass
            return

        released = ack.get("released_mb", "?")
        print(f"   ✅ 已让出 {released}MB，开始本地执行（输出如下）\n")

        # 2) 在本地终端原生执行任务（继承 stdin/stdout/stderr，零转发）
        sock.settimeout(None)   # 任务期间保持租约连接，不超时
        start = time.time()
        returncode = 1
        try:
            returncode = subprocess.call(command, shell=True)
        except KeyboardInterrupt:
            print("\n⚠️  已中断 (Ctrl+C)")
            returncode = 130
        finally:
            # 3) 关闭租约连接 → 守护进程检测到断开后自动收回显存
            try:
                sock.close()
            except Exception:
                pass

        elapsed = time.time() - start
        if returncode == 0:
            print(f"\n✅ 任务成功 | 耗时: {elapsed:.1f}s → 显存后台收回中")
        else:
            print(f"\n❌ 任务结束 (exit code: {returncode}) | 耗时: {elapsed:.1f}s → 显存后台收回中")

    @staticmethod
    def mine():
        """一眼看清：己方霸占了多少显存"""
        result = GPUGuardClient._send_request({"action": "status"}, timeout=10)
        if "error" in result:
            print(f"\n❌ {result['error']}")
            return

        occupied = result.get("occupied_mb", 0)
        max_occ = result.get("max_occupy_mb", 0)
        chunks = result.get("chunk_count", 0)
        task_active = result.get("task_active", False)
        paused = result.get("paused", False)

        # 从 GPU 信息获取总量和他方占用
        gpu_info = result.get("gpu_info", {})
        gpus = gpu_info.get("gpus", [])
        device_id = result.get("device_id", 0)

        if device_id < len(gpus):
            gpu = gpus[device_id]
            total = gpu["total_mb"]
            used = gpu["used_mb"]
            free = gpu["free_mb"]
        else:
            total = max_occ
            used = occupied
            free = 0

        other_used = max(0, used - occupied)

        # 构建可视化进度条
        bar_len = 40
        if total > 0:
            ours_ratio = occupied / total
            other_ratio = other_used / total
            free_ratio = free / total
        else:
            ours_ratio = other_ratio = free_ratio = 0

        ours_blocks = int(bar_len * ours_ratio)
        other_blocks = int(bar_len * other_ratio)
        free_blocks = bar_len - ours_blocks - other_blocks

        bar = "\u2588" * ours_blocks + "\u2593" * other_blocks + "\u2591" * free_blocks

        # 输出
        print()
        print(f"  \U0001f512 己方霸占: {occupied}MB / {total}MB ({ours_ratio*100:.1f}%)")
        print(f"  [{bar}]")
        print(f"  \u2588 己方 {occupied}MB  \u2593 他方 {other_used}MB  \u2591 空闲 {free}MB")
        if task_active:
            print(f"  \u23f8\ufe0f  当前有任务运行中，显存已临时释放")
        print()

    @staticmethod
    def release(memory_mb: int = 0):
        result = GPUGuardClient._send_request(
            {"action": "release", "memory_mb": memory_mb}, timeout=120)
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        print(f"⏸️  已让出 {result.get('released_mb', '?')}MB，当前占用 "
              f"{result.get('occupied_mb', '?')}MB —— 巡逻已暂停")
        print(f"   用完后执行: python gpu_guard.py resume  恢复霸占")

    @staticmethod
    def resume():
        result = GPUGuardClient._send_request({"action": "resume"}, timeout=300)
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        print(f"▶️  已恢复霸占 (+{result.get('grabbed_mb', '?')}MB)，当前占用 "
              f"{result.get('occupied_mb', '?')}MB")

    @staticmethod
    def stop():
        result = GPUGuardClient._send_request({"action": "stop"}, timeout=10)
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print("🛑 GPU Guard 正在停止...")
            time.sleep(2)
            print("✅ 已停止")


# ============================================================================
# 辅助
# ============================================================================

def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def is_daemon_running() -> bool:
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, PermissionError):
        try:
            os.unlink(PID_FILE)
        except OSError:
            pass
        return False


def force_cleanup_environment():
    """暴力清理机制：一刀斩断旧进程，物理抹除所有状态文件，确保启动环境绝对干净"""
    if not os.path.exists(GUARD_DIR):
        os.makedirs(GUARD_DIR, mode=0o700, exist_ok=True)
        
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                old_pid = int(f.read().strip())
            # 极度暴力：直接发送 SIGKILL (-9) 瞬间超度，强制回收显卡资源
            os.kill(old_pid, signal.SIGKILL)
            print(f"🗡️ 已强制清理旧的守护进程 (PID: {old_pid})")
            time.sleep(0.5)  # 稍微等一下让驱动彻底回收显存
        except Exception:
            pass

    for f_path in [SOCKET_PATH, PID_FILE]:
        try:
            os.unlink(f_path)
        except OSError:
            pass

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU Guard v3 - 自私的GPU资源管理器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s start                              # 启动（默认: 64MB块, 95%%占用上限）
  %(prog)s start --chunk-size 32              # 32MB块（更精细控制）
  %(prog)s start --reserve-ratio 0.90         # 占用上限90%%（多留空间给己方任务）
  %(prog)s start --poll-interval 1            # 基准巡逻1s（竞争时自动加速到0.5s）
  %(prog)s start -f                           # 前台运行（调试）
  %(prog)s run "python train.py"              # 运行任务（自动继承当前环境）
  %(prog)s run -m 4096 "python train.py"      # 指定4GB + 自动预留500MB context
  %(prog)s status                             # 查看状态
  %(prog)s logs                               # 查看日志
  %(prog)s stop                               # 停止

环境说明:
  'run' 提交的任务自动继承你当前终端的 conda/venv/PATH/工作目录，
  与直接运行效果完全一致。
        """
    )

    sub = parser.add_subparsers(dest="action", help="操作")

    sp = sub.add_parser("start", help="启动守护进程")
    sp.add_argument("--device", type=int, default=0, help="GPU设备ID (默认: 0)")
    sp.add_argument("--chunk-size", type=int, default=64,
                    help="显存块大小MB (默认: 64)")
    sp.add_argument("--reserve-ratio", type=float, default=0.95,
                    help="最大占用比例 (默认: 0.95，即占总显存的95%%)")
    sp.add_argument("--poll-interval", type=float, default=3,
                    help="基准巡逻间隔秒 (默认: 3，竞争时自动加速)")
    sp.add_argument("--foreground", "-f", action="store_true",
                    help="前台运行（调试用）")

    rp = sub.add_parser("run", help="通过中介执行GPU任务")
    rp.add_argument("command", help="要执行的命令")
    rp.add_argument("--memory", "-m", type=int, default=0,
                    help="需要的显存MB (0=释放全部)")

    sub.add_parser("mine", help="一眼看清己方霸占了多少显存")
    sub.add_parser("status", help="查看完整状态")
    relp = sub.add_parser("release", help="手动让出显存并暂停霸占（WebUI 等交互训练前用）")
    relp.add_argument("--memory", "-m", type=int, default=0,
                      help="只释放的显存MB (0=全部释放)")
    sub.add_parser("resume", help="恢复霸占（与 release 配对，训练完用）")
    sub.add_parser("stop", help="停止守护进程")
    sub.add_parser("logs", help="查看日志")

    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        sys.exit(1)

    if args.action == "start":
        # ★ 不管三七二十一，启动前先暴力清理（先杀后建），彻底告别假死/卡死
        force_cleanup_environment()

        if args.foreground:
            print("🚀 GPU Guard v3 前台模式...")
            daemon = GPUGuardDaemon(
                device_id=args.device,
                chunk_size_mb=args.chunk_size,
                reserve_ratio=args.reserve_ratio,
                poll_interval=args.poll_interval,
            )
            daemon.start()
        else:
            # ★ 后台启动 + 自动重试机制
            # GPU Guard 需要约 500MB VRAM 初始化 CUDA context，
            # 如果当前显存不足会启动失败。自动重试可以等待显存释放后再启动。
            MAX_START_RETRIES = 3
            for attempt in range(1, MAX_START_RETRIES + 1):
                if attempt > 1:
                    wait_time = 3 * attempt
                    print(f"   等待 {wait_time} 秒后重试（等待 VRAM 释放）...")
                    time.sleep(wait_time)

                print(f"\n🚀 GPU Guard v3 启动中... (尝试 {attempt}/{MAX_START_RETRIES})")

                # 清理可能的残留文件
                for path in [PID_FILE, SOCKET_PATH]:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

                pid = os.fork()
                if pid == 0:
                    # 子进程：成为守护进程
                    os.setsid()
                    sys.stdin = open(os.devnull, 'r')
                    sys.stdout = open(os.devnull, 'w')
                    sys.stderr = open(os.devnull, 'w')
                    try:
                        GPUGuardDaemon(
                            device_id=args.device,
                            chunk_size_mb=args.chunk_size,
                            reserve_ratio=args.reserve_ratio,
                            poll_interval=args.poll_interval,
                        ).start()
                    except SystemExit as e:
                        os._exit(e.code if isinstance(e.code, int) else 1)
                    except Exception:
                        os._exit(1)
                    os._exit(0)

                # 父进程：通过 status 命令验证守护进程是否真正启动成功
                # ★ 不依赖进程退出码（有时不知道自己启动成功了），
                #   而是用 socket ping 确认守护进程已完全就绪
                started = False
                for _wait in range(10):  # 最多等 10 秒（CUDA 初始化 + 初始霸占可能较慢）
                    time.sleep(1)
                    # 先检查子进程是否已崩溃退出
                    try:
                        wpid, wstatus = os.waitpid(pid, os.WNOHANG)
                        if wpid != 0:
                            exit_code = wstatus >> 8 if os.WIFEXITED(wstatus) else -1
                            print(f"   ⚠️  守护子进程已退出 (exit code={exit_code})")
                            break
                    except ChildProcessError:
                        break

                    # 子进程仍存活，尝试 socket ping
                    result = GPUGuardClient._send_request(
                        {"action": "ping"}, timeout=2)
                    if result and result.get("status") == "alive":
                        started = True
                        break

                if started:
                    print(f"✅ 已启动 (PID: {pid})")
                    print(f"   日志: {LOG_FILE}")
                    print(f"   状态: python gpu_guard.py status")
                    print(f"   停止: python gpu_guard.py stop")
                    sys.exit(0)

                # 本次启动失败 — 清理
                print(f"   ❌ 第 {attempt} 次启动未成功")
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
                for path in [PID_FILE, SOCKET_PATH]:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

            # for 循环耗尽，所有重试均失败
            else:
                print(f"\n❌ 启动失败 ({MAX_START_RETRIES} 次尝试均未成功)")
                print(f"   查看日志: {LOG_FILE}")
                print(f"   可能原因: GPU 显存不足（至少需要 ~500MB 供 CUDA 初始化）")
                sys.exit(1)

    elif args.action == "run":
        GPUGuardClient.run(args.command, args.memory)

    elif args.action == "mine":
        GPUGuardClient.mine()

    elif args.action == "status":
        GPUGuardClient.status()

    elif args.action == "release":
        GPUGuardClient.release(args.memory)

    elif args.action == "resume":
        GPUGuardClient.resume()

    elif args.action == "stop":
        GPUGuardClient.stop()

    elif args.action == "logs":
        if os.path.exists(LOG_FILE):
            try:
                result = subprocess.run(
                    ["tail", "-n", "80", LOG_FILE],
                    capture_output=True, text=True
                )
                print(result.stdout)
            except FileNotFoundError:
                with open(LOG_FILE, "r") as f:
                    lines = f.readlines()
                    print("".join(lines[-80:]))
        else:
            print("📄 暂无日志文件")


if __name__ == "__main__":
    main()
