"""文件传输服务的公共工具：文件锁、原子写入、哈希计算、鉴权与状态修复。

本模块被传输服务端、客户端及 tools/start/transfer.py 共同引用，
其中的公共函数与配置常量属于对外契约，请勿改名。
"""

import os
import time
import hashlib
import json
import re
import hmac
import threading
from pathlib import Path
from typing import Optional

from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ── 默认配置项 ──
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7459
DEFAULT_DATA_DIR = "./data/file-transfer"
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB
DEFAULT_MAX_CONCURRENCY = 4

# ── 全局锁 ──
_global_state_lock = threading.Lock()

# ── 简易跨平台文件锁实现 ──
try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None


class FileLock:
    """基于文件的跨平台互斥锁，用于串行化对共享状态文件的读写。"""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.fd = None

    def acquire(self, timeout: float = 60.0):
        """尝试获取文件锁，超时则抛出 TimeoutError。"""
        start = time.time()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self.fd = os.open(self.filepath, os.O_RDWR | os.O_CREAT)
                if fcntl:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt:
                    msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
                return
            except (IOError, OSError):
                if self.fd is not None:
                    try:
                        os.close(self.fd)
                    except OSError:
                        pass
                    self.fd = None
                if time.time() - start > timeout:
                    raise TimeoutError(f"无法获取文件锁: {self.filepath}")
                time.sleep(0.05)

    def release(self):
        """释放文件锁并清理锁文件描述符。"""
        if self.fd is not None:
            try:
                if fcntl:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                elif msvcrt:
                    os.lseek(self.fd, 0, os.SEEK_SET)
                    msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
            try:
                self.filepath.unlink(missing_ok=True)
            except Exception:
                pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def sanitize_filename(filename: str) -> str:
    """净化文件名，过滤路径穿越、特殊字符及绝对路径"""
    name = os.path.basename(filename)
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    # 仅保留字母、数字、点、减号和下划线
    name = "".join(c for c in name if c.isalnum() or c in (".", "-", "_")).strip()
    if not name:
        name = "unnamed_transfer"
    return name


def atomic_write_json(filepath: Path, data: dict) -> None:
    """原子性地写入 JSON 文件，写入临时文件 -> fsync -> 重命名替换"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def discover_public_url(port: int) -> Optional[str]:
    """发现并提取当前 CNB 环境提供的公网代理 URI"""
    proxy_uri = os.environ.get("CNB_VSCODE_PROXY_URI")
    if proxy_uri:
        return proxy_uri.replace("{{port}}", str(port))
    return None


def get_file_hash(filepath: Path) -> str:
    """流式计算大文件的 SHA-256 哈希值，防止内存溢出"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def repair_state(incoming_dir: Path, metadata: dict) -> dict:
    """当服务端状态 state.json 被损坏或不一致时，扫描 chunks/ 并原子修复"""
    file_hash = metadata["file_hash"]
    chunk_size = metadata["chunk_size"]
    total_chunks = metadata["total_chunks"]
    file_size = metadata["file_size"]

    chunks_dir = incoming_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    completed_chunks = []
    chunk_hashes = {}
    received_bytes = 0

    for i in range(total_chunks):
        chunk_file = chunks_dir / f"{i:08d}.part"
        if chunk_file.exists():
            expected_size = chunk_size
            if i == total_chunks - 1:
                expected_size = file_size - (chunk_size * (total_chunks - 1))

            if chunk_file.stat().st_size == expected_size:
                h = hashlib.sha256()
                with open(chunk_file, "rb") as f:
                    while chunk := f.read(8192):
                        h.update(chunk)
                completed_chunks.append(i)
                chunk_hashes[str(i)] = h.hexdigest()
                received_bytes += expected_size
            else:
                try:
                    chunk_file.unlink()
                except OSError:
                    pass

    missing_chunks = [i for i in range(total_chunks) if i not in completed_chunks]

    state = {
        "completed_chunks": completed_chunks,
        "missing_chunks": missing_chunks,
        "chunk_hashes": chunk_hashes,
        "received_bytes": received_bytes,
        "retries": 0,
        "last_error": "",
        "last_active_at": time.time()
    }

    state_file = incoming_dir / "state.json"
    atomic_write_json(state_file, state)
    return state


security = HTTPBearer(auto_error=False)


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security), request: Request = None):
    """校验请求携带的预共享令牌，作为 FastAPI 路由的鉴权依赖使用。"""
    expected_token = os.environ.get("FILE_TRANSFER_TOKEN")
    if not expected_token:
        raise HTTPException(status_code=401, detail="FILE_TRANSFER_TOKEN is not configured on the server")

    token = None
    if credentials:
        token = credentials.credentials
    elif request and "x-transfer-token" in request.headers:
        token = request.headers["x-transfer-token"]

    if not token:
        time.sleep(1.0)  # 防暴力破解限速
        raise HTTPException(status_code=401, detail="Missing authorization token")

    if not hmac.compare_digest(token, expected_token):
        time.sleep(1.0)  # 防暴力破解限速
        raise HTTPException(status_code=403, detail="Invalid authorization token")


def get_data_dir() -> Path:
    """返回传输服务的数据根目录，优先取环境变量，否则用默认路径。"""
    return Path(os.environ.get("FILE_TRANSFER_DATA_DIR", DEFAULT_DATA_DIR)).resolve()


def parse_duration(duration_str: str) -> float:
    """将 "7d"、"24h"、"60m"、"30s" 形式的时长字符串解析为秒数。"""
    match = re.match(r"^(\d+)([dhms])$", duration_str.lower())
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}")
    val, unit = match.groups()
    val = int(val)
    seconds_map = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return val * seconds_map[unit]
