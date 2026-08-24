"""Web 文件上传器的公共工具：路径安全拼接、异步写入、文件信息与临时文件清理。"""

import os
import glob
import asyncio
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException

shutdown_event = threading.Event()
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

_assembly_locks: dict = {}
_io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="uploader-io")


async def _write_bytes(path: Path, data: bytes, mode: str = "wb") -> None:
    """将数据写入指定路径，通过线程池执行以避免阻塞事件循环。"""
    loop = asyncio.get_event_loop()
    def _sync():
        with open(path, mode) as fh:
            fh.write(data)
    await loop.run_in_executor(_io_executor, _sync)


def safe_join(base: Path, relative: str) -> Path:
    """安全地拼接相对路径与基准路径，防止路径穿越攻击。"""
    clean_rel = relative.replace("\\", "/").strip("/")
    parts = clean_rel.split("/")

    safe_parts = []
    for part in parts:
        if not part or part == ".":
            continue
        if part == "..":
            if safe_parts:
                safe_parts.pop()
            continue
        safe_parts.append(part)

    target_path = Path(os.path.abspath(base / "/".join(safe_parts)))
    base_path = Path(os.path.abspath(base))

    try:
        target_path.relative_to(base_path)
    except ValueError:
        raise HTTPException(status_code=400, detail="检测到路径穿越行为，已被拒绝")

    return target_path


def get_file_info(path: Path, base: Path):
    """返回文件或目录的元信息字典（名称、相对路径、类型、大小、修改时间）。"""
    stat = path.stat()
    rel_path = path.relative_to(base).as_posix()
    return {
        "name": path.name,
        "path": rel_path,
        "is_dir": path.is_dir(),
        "size": stat.st_size if path.is_file() else 0,
        "mtime": stat.st_mtime
    }


def _cleanup_temp_files():
    """清理工作区中可能残留的 .upload.tmp / .uploading / .chunkXXXXX. 临时文件。"""
    patterns = [
        str(WORKSPACE_ROOT / "**" / "*.upload.tmp"),
        str(WORKSPACE_ROOT / "**" / "*.uploading"),
        str(WORKSPACE_ROOT / "**" / ".*.chunk*"),
    ]
    cleaned = 0
    for pattern in patterns:
        for tmp_file in glob.glob(pattern, recursive=True):
            try:
                os.unlink(tmp_file)
                cleaned += 1
            except OSError:
                pass
    if cleaned:
        print(f"[cleanup] 已清理 {cleaned} 个残留临时文件")
