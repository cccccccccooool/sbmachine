"""文件传输客户端：负责分片上传、断点续传、状态查询与本地清理。"""

import os
import sys
import time
import hashlib
import json
import shutil
import random
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from tools.start.transfer.utils import (
    get_file_hash, FileLock, parse_duration, get_data_dir
)


def send_file(target_url: str, filepath: Path, token: str, concurrency: int, chunk_size: int, max_retries: int):
    """客户端分片与断点续传核心实现"""
    if not filepath.exists() or not filepath.is_file():
        print(f"Error: Target file {filepath} does not exist.")
        sys.exit(1)

    # 1. 计算文件基本信息
    print(f"Calculating SHA-256 for {filepath.name}...")
    file_hash = get_file_hash(filepath)
    file_size = filepath.stat().st_size
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    if total_chunks == 0:
        total_chunks = 1

    print(f"File Hash: {file_hash}")
    print(f"File Size: {file_size} bytes ({total_chunks} chunks, chunk size: {chunk_size} bytes)")

    # 2. 查询服务端断点状态
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    status_url = f"{target_url.rstrip('/')}/api/v1/transfers/{file_hash}"

    try:
        response = requests.get(status_url, headers=headers, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to target server: {e}")
        sys.exit(1)

    completed_chunks = set()
    if response.status_code == 200:
        res_data = response.json()
        if res_data.get("status") == "completed":
            print("\nFile has already been fully uploaded and verified on the server.")
            print(f"Remote Path: {res_data.get('completed_path')}")
            return
        elif res_data.get("status") == "in_progress":
            print("Resuming existing transfer...")
            completed_chunks = set(res_data.get("state", {}).get("completed_chunks", []))
            print(f"Already uploaded chunks: {len(completed_chunks)} / {total_chunks}")
    elif response.status_code == 404:
        # 未注册任务，进行初始化
        init_url = f"{target_url.rstrip('/')}/api/v1/transfers/init"
        init_data = {
            "file_name": filepath.name,
            "file_size": file_size,
            "file_hash": file_hash,
            "chunk_size": chunk_size,
            "total_chunks": total_chunks
        }
        try:
            init_res = requests.post(init_url, json=init_data, headers=headers, timeout=15)
            if init_res.status_code != 200:
                print(f"Failed to initialize transfer: {init_res.status_code} - {init_res.text}")
                sys.exit(1)
            print("Initialized new transfer on the server.")
        except Exception as e:
            print(f"Failed to initialize transfer on server: {e}")
            sys.exit(1)
    else:
        print(f"Error checking status: HTTP {response.status_code} - {response.text}")
        sys.exit(1)

    # 3. 多线程上传分片
    chunks_to_upload = [i for i in range(total_chunks) if i not in completed_chunks]

    interrupted = False
    uploaded_count = len(completed_chunks)
    lock = threading.Lock()

    def upload_worker(chunk_index: int) -> bool:
        nonlocal interrupted
        if interrupted:
            return False

        offset = chunk_index * chunk_size
        with open(filepath, "rb") as f:
            f.seek(offset)
            data = f.read(chunk_size)

        chunk_hash = hashlib.sha256(data).hexdigest()
        chunk_url = f"{target_url.rstrip('/')}/api/v1/transfers/{file_hash}/chunks/{chunk_index}"
        chunk_headers = {
            "Authorization": f"Bearer {token}",
            "X-Chunk-Hash": chunk_hash,
            "Content-Type": "application/octet-stream"
        }

        # 指数退避重试逻辑
        backoff = 1.0
        for attempt in range(max_retries):
            if interrupted:
                return False
            try:
                res = requests.put(chunk_url, data=data, headers=chunk_headers, timeout=45)
                if res.status_code == 200:
                    with lock:
                        nonlocal uploaded_count
                        uploaded_count += 1
                        print(f"Uploaded chunk {chunk_index+1}/{total_chunks} (Attempt {attempt+1})")
                    return True
                elif res.status_code in (400, 401, 403):
                    # 不可恢复错误
                    print(f"\nUnrecoverable error on chunk {chunk_index} (HTTP {res.status_code}): {res.text}")
                    interrupted = True
                    return False
                else:
                    raise Exception(f"HTTP {res.status_code}")
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\nFailed to upload chunk {chunk_index} after {max_retries} attempts: {e}")
                    interrupted = True
                    return False
                sleep_time = backoff * (2 ** attempt) + random.uniform(0.1, 0.5)
                time.sleep(min(sleep_time, 15.0))
        return False

    # 信号拦截
    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\nInterrupt signal received. Gracefully stopping...")
        interrupted = True

    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

    try:
        if chunks_to_upload:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(upload_worker, idx): idx for idx in chunks_to_upload}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        success = future.result()
                        if not success:
                            interrupted = True
                    except Exception as e:
                        print(f"Exception during upload of chunk {idx}: {e}")
                        interrupted = True
                    if interrupted:
                        break
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

    if interrupted:
        print("\nTransfer interrupted safely.")
        print(f"File hash:\n{file_hash}")
        print(f"Completed chunks:\n{uploaded_count} / {total_chunks}")
        print(f"Resume with:\npython tools/start/transfer.py send --target {target_url} --file {filepath}")
        sys.exit(1)

    # 4. 请求合并与最终校验
    print("All chunks uploaded successfully. Requesting server merge and verification...")
    complete_url = f"{target_url.rstrip('/')}/api/v1/transfers/{file_hash}/complete"
    try:
        complete_res = requests.post(complete_url, headers=headers, timeout=60)
        if complete_res.status_code == 200:
            res_data = complete_res.json()
            print("\nTransfer completed successfully!")
            print(f"Remote Path: {res_data.get('path')}")
            print(f"File Name: {res_data.get('file_name')}")
            print(f"Size: {res_data.get('size')} bytes")
        else:
            print(f"\nFailed to complete transfer on server: {complete_res.status_code} - {complete_res.text}")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to request server file merge: {e}")
        sys.exit(1)


def run_send(args):
    """send 子命令入口：从命令行参数解析目标与令牌后发起分片上传。"""
    target = args.target or os.environ.get("FILE_TRANSFER_TARGET")
    if not target:
        print("Error: Target server URL must be specified via --target or FILE_TRANSFER_TARGET environment variable.")
        sys.exit(1)

    token = args.token or os.environ.get("FILE_TRANSFER_TOKEN")
    if not token:
        print("Error: Pre-shared token must be specified via --token or FILE_TRANSFER_TOKEN environment variable.")
        sys.exit(1)

    filepath = Path(args.file).resolve()
    send_file(
        target_url=target,
        filepath=filepath,
        token=token,
        concurrency=args.concurrency,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries
    )


def run_status(args):
    """status 子命令入口：查询指定文件哈希在服务端的传输状态。"""
    target = args.target or os.environ.get("FILE_TRANSFER_TARGET")
    if not target:
        print("Error: Target server URL must be specified.")
        sys.exit(1)

    token = args.token or os.environ.get("FILE_TRANSFER_TOKEN")
    if not token:
        print("Error: Pre-shared token must be specified.")
        sys.exit(1)

    check_status(target, args.hash, token)


def check_status(target_url: str, file_hash: str, token: str):
    """向服务端查询指定哈希的传输状态并打印结果。"""
    headers = {"Authorization": f"Bearer {token}"}
    status_url = f"{target_url.rstrip('/')}/api/v1/transfers/{file_hash}"
    try:
        res = requests.get(status_url, headers=headers, timeout=10)
        if res.status_code == 200:
            print(json.dumps(res.json(), indent=2, ensure_ascii=False))
        elif res.status_code == 404:
            print(f"Transfer task not found for hash {file_hash}")
        else:
            print(f"HTTP Error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"Connection failure: {e}")


def run_receive_list(args):
    """列出本地存储中所有已完成的传输记录（receive-list 子命令入口）。"""
    data_dir = Path(args.data_dir).resolve()
    index_file = data_dir / "completed_index.json"
    if not index_file.exists():
        print("No completed transfers found.")
        return

    lock_file = data_dir / "locks" / "completed_index.lock"
    with FileLock(lock_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = {}

    if not index:
        print("No completed transfers found.")
        return

    print(f"{'Safe File Name':<40} {'Size (Bytes)':<15} {'Completed Time':<25} {'File Hash':<64}")
    print("-" * 148)
    for file_hash, info in index.items():
        comp_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info["completed_time"]))
        print(f"{info['file_name']:<40} {info['file_size']:<15} {comp_time:<25} {file_hash:<64}")


def run_cleanup(args):
    """清理长时间不活跃的分片目录与过期锁文件（cleanup 子命令入口）。"""
    data_dir = Path(args.data_dir).resolve()
    incoming_dir = data_dir / "incoming"
    if not incoming_dir.exists():
        return

    older_than_seconds = parse_duration(args.older_than)
    now = time.time()
    cleaned_count = 0

    print(f"Scanning for transfers inactive for more than {args.older_than} ({older_than_seconds}s)...")
    for item in incoming_dir.iterdir():
        if item.is_dir():
            state_file = item / "state.json"
            last_active = None
            if state_file.exists():
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    last_active = state.get("last_active_at")
                except Exception:
                    pass
            if last_active is None:
                last_active = item.stat().st_mtime

            if now - last_active > older_than_seconds:
                if args.dry_run:
                    print(f"[Dry-run] Would delete inactive transfer: {item.name} (last active {now - last_active:.0f}s ago)")
                else:
                    print(f"Deleting inactive transfer: {item.name}")
                    shutil.rmtree(item, ignore_errors=True)
                    cleaned_count += 1

    # 清理锁文件
    locks_dir = data_dir / "locks"
    if locks_dir.exists():
        for lock_file in locks_dir.iterdir():
            if lock_file.is_file() and now - lock_file.stat().st_mtime > 3600:
                if args.dry_run:
                    print(f"[Dry-run] Would delete old lock file: {lock_file.name}")
                else:
                    try:
                        lock_file.unlink(missing_ok=True)
                        cleaned_count += 1
                    except OSError:
                        pass

    if not args.dry_run:
        print(f"Cleanup finished. Removed {cleaned_count} items.")
