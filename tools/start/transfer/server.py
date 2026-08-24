"""文件传输服务端：基于 FastAPI 提供分片上传、断点续传与合并校验的 HTTP 接口。"""

import os
import time
import shutil
import tempfile
import re
import hashlib
import json
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Header, Depends

from tools.start.transfer.utils import (
    verify_token, get_data_dir, repair_state, sanitize_filename,
    atomic_write_json, FileLock, discover_public_url,
    DEFAULT_CHUNK_SIZE, DEFAULT_MAX_FILE_SIZE
)

app = FastAPI(title="CNB Secure File Transfer Service")


@app.get("/api/v1/health")
def health_check():
    """无需鉴权的健康检查接口"""
    return {"status": "ok"}


@app.post("/api/v1/transfers/init", dependencies=[Depends(verify_token)])
def init_transfer(meta_input: dict):
    """初始化一个大文件传输任务，注册元数据并返回状态"""
    file_name = meta_input.get("file_name")
    file_size = meta_input.get("file_size")
    file_hash = meta_input.get("file_hash")
    chunk_size = meta_input.get("chunk_size", DEFAULT_CHUNK_SIZE)
    total_chunks = meta_input.get("total_chunks")

    if not all([file_name, file_size is not None, file_hash, total_chunks]):
        raise HTTPException(status_code=400, detail="Missing required metadata parameters")

    # 1. 安全验证与规范化
    if not re.match(r"^[a-fA-F0-9]{64}$", file_hash):
        raise HTTPException(status_code=400, detail="Invalid SHA-256 hash format")

    # 限制单个文件最大大小
    max_file_size = int(os.environ.get("FILE_TRANSFER_MAX_FILE_SIZE", DEFAULT_MAX_FILE_SIZE))
    if file_size > max_file_size:
        raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {max_file_size} bytes")

    safe_name = sanitize_filename(file_name)
    data_dir = get_data_dir()

    # 2. 检查索引，看是否已完成过相同哈希的文件
    index_file = data_dir / "completed_index.json"
    if index_file.exists():
        with FileLock(data_dir / "locks" / "completed_index.lock"):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
                    if file_hash in index:
                        info = index[file_hash]
                        if Path(info["completed_path"]).exists():
                            return {
                                "status": "completed",
                                "file_name": info["file_name"],
                                "file_size": info["file_size"],
                                "completed_path": info["completed_path"]
                            }
            except Exception:
                pass

    incoming_dir = data_dir / "incoming" / file_hash
    incoming_dir.mkdir(parents=True, exist_ok=True)
    (incoming_dir / "chunks").mkdir(parents=True, exist_ok=True)

    metadata_file = incoming_dir / "metadata.json"
    state_file = incoming_dir / "state.json"

    # 如果元数据文件已经存在，进行比较
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
            if existing_meta["file_hash"] != file_hash or existing_meta["file_size"] != file_size:
                raise HTTPException(status_code=400, detail="Metadata conflict for existing file hash")
        except json.JSONDecodeError:
            pass

    # 3. 写入/更新元数据和初始化状态
    meta = {
        "file_name": safe_name,
        "file_size": file_size,
        "file_hash": file_hash,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "created_at": time.time(),
        "updated_at": time.time(),
        "status": "in_progress"
    }
    atomic_write_json(metadata_file, meta)

    # 读取或重建 state
    state = None
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            pass

    if not state:
        state = repair_state(incoming_dir, meta)
    else:
        # 进行一次自检以同步真实磁盘分片
        state = repair_state(incoming_dir, meta)

    return {
        "status": "in_progress",
        "file_hash": file_hash,
        "metadata": meta,
        "state": state
    }


@app.get("/api/v1/transfers/{file_hash}", dependencies=[Depends(verify_token)])
def get_transfer_status(file_hash: str):
    """查询指定哈希的文件上传状态"""
    if not re.match(r"^[a-fA-F0-9]{64}$", file_hash):
        raise HTTPException(status_code=400, detail="Invalid SHA-256 hash format")

    data_dir = get_data_dir()

    # 1. 优先查已完成索引
    index_file = data_dir / "completed_index.json"
    if index_file.exists():
        with FileLock(data_dir / "locks" / "completed_index.lock"):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
                    if file_hash in index:
                        info = index[file_hash]
                        if Path(info["completed_path"]).exists():
                            return {
                                "status": "completed",
                                "file_name": info["file_name"],
                                "file_size": info["file_size"],
                                "completed_path": info["completed_path"]
                            }
            except Exception:
                pass

    # 2. 查询进行中临时文件夹
    incoming_dir = data_dir / "incoming" / file_hash
    metadata_file = incoming_dir / "metadata.json"
    state_file = incoming_dir / "state.json"

    if not incoming_dir.exists() or not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Transfer task not found")

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 获取或自检修复状态
        state = repair_state(incoming_dir, meta)
        return {
            "status": "in_progress",
            "file_hash": file_hash,
            "metadata": meta,
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read transfer task state: {e}")


@app.put("/api/v1/transfers/{file_hash}/chunks/{chunk_index}", dependencies=[Depends(verify_token)])
async def upload_chunk(
    file_hash: str,
    chunk_index: int,
    request: Request,
    x_chunk_hash: Optional[str] = Header(None)
):
    """接收文件切片（Binary Body），进行分片校验、原子写入和状态更新"""
    if not re.match(r"^[a-fA-F0-9]{64}$", file_hash):
        raise HTTPException(status_code=400, detail="Invalid SHA-256 hash format")

    data_dir = get_data_dir()
    incoming_dir = data_dir / "incoming" / file_hash
    metadata_file = incoming_dir / "metadata.json"
    state_file = incoming_dir / "state.json"

    if not incoming_dir.exists() or not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Transfer task not initialized")

    # 1. 读取元数据并校验参数
    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_chunks = meta["total_chunks"]
    chunk_size = meta["chunk_size"]
    file_size = meta["file_size"]

    if chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(status_code=400, detail="Invalid chunk index")

    # 计算预期的分片大小
    expected_chunk_size = chunk_size
    if chunk_index == total_chunks - 1:
        expected_chunk_size = file_size - (chunk_size * (total_chunks - 1))

    # 限制单个请求分片上限为配置项大小的 1.5 倍
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > chunk_size * 1.5:
        raise HTTPException(status_code=400, detail="Request chunk body too large")

    # 读取请求流
    try:
        body = await request.body()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading chunk stream: {e}")

    if len(body) != expected_chunk_size:
        raise HTTPException(status_code=400, detail=f"Chunk size mismatch. Expected {expected_chunk_size}, got {len(body)}")

    # 2. 校验分片哈希
    sha = hashlib.sha256(body).hexdigest()
    if x_chunk_hash and sha.lower() != x_chunk_hash.lower():
        raise HTTPException(status_code=400, detail="Chunk integrity verification failed (hash mismatch)")

    # 3. 原子性写入分片到 chunks 目录
    chunks_dir = incoming_dir / "chunks"
    chunk_path = chunks_dir / f"{chunk_index:08d}.part"
    tmp_chunk_path = chunk_path.with_suffix(".tmp")

    try:
        with open(tmp_chunk_path, "wb") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_chunk_path, chunk_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write chunk to disk: {e}")
    finally:
        try:
            tmp_chunk_path.unlink(missing_ok=True)
        except OSError:
            pass

    # 4. 安全锁操作更新 state.json
    state_lock_path = data_dir / "locks" / f"{file_hash}_state.lock"
    with FileLock(state_lock_path):
        try:
            if state_file.exists():
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
            else:
                state = {}
        except Exception:
            state = {}

        completed = set(state.get("completed_chunks", []))
        completed.add(chunk_index)

        chunk_hashes = state.get("chunk_hashes", {})
        chunk_hashes[str(chunk_index)] = sha

        completed_list = sorted(list(completed))
        missing_list = [i for i in range(total_chunks) if i not in completed_list]

        state.update({
            "completed_chunks": completed_list,
            "missing_chunks": missing_list,
            "chunk_hashes": chunk_hashes,
            "received_bytes": sum(
                (chunk_size if i < total_chunks - 1 else file_size - (chunk_size * (total_chunks - 1)))
                for i in completed_list
            ),
            "last_active_at": time.time()
        })
        atomic_write_json(state_file, state)

    return {"status": "success", "chunk_index": chunk_index}


@app.post("/api/v1/transfers/{file_hash}/complete", dependencies=[Depends(verify_token)])
def complete_transfer(file_hash: str):
    """请求服务端合并并验证完整文件，返回存储路径"""
    if not re.match(r"^[a-fA-F0-9]{64}$", file_hash):
        raise HTTPException(status_code=400, detail="Invalid SHA-256 hash format")

    data_dir = get_data_dir()
    incoming_dir = data_dir / "incoming" / file_hash
    metadata_file = incoming_dir / "metadata.json"
    state_file = incoming_dir / "state.json"

    if not incoming_dir.exists() or not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Transfer task not found")

    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_chunks = meta["total_chunks"]
    file_size = meta["file_size"]
    file_name = meta["file_name"]

    # 1. 确认所有分片都已到位
    chunks_dir = incoming_dir / "chunks"
    for i in range(total_chunks):
        if not (chunks_dir / f"{i:08d}.part").exists():
            raise HTTPException(status_code=400, detail=f"Missing chunk index: {i}")

    # 2. 独占锁合并
    merge_lock_path = data_dir / "locks" / f"{file_hash}_merge.lock"
    completed_dir = data_dir / "completed"
    completed_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(merge_lock_path):
        tmp_fd, tmp_str = tempfile.mkstemp(suffix=".merge.tmp", dir=str(completed_dir))
        os.close(tmp_fd)
        tmp_path = Path(tmp_str)
        try:
            # 3. 流式合并大文件并计算 SHA-256
            h = hashlib.sha256()
            size = 0
            with open(tmp_path, "wb") as out_f:
                for i in range(total_chunks):
                    chunk_file = chunks_dir / f"{i:08d}.part"
                    with open(chunk_file, "rb") as in_f:
                        while block := in_f.read(65536):
                            out_f.write(block)
                            h.update(block)
                            size += len(block)

            calculated_hash = h.hexdigest()
            if calculated_hash.lower() != file_hash.lower():
                raise HTTPException(status_code=400, detail="Completed file hash mismatch")
            if size != file_size:
                raise HTTPException(status_code=400, detail="Completed file size mismatch")

            # 4. 冲突解决：防目录穿越与防文件覆写
            safe_name = sanitize_filename(file_name)
            final_path = completed_dir / safe_name
            if final_path.exists():
                # 重名防覆写规则
                final_path = completed_dir / f"{file_hash[:12]}-{safe_name}"

            os.replace(tmp_path, final_path)

            # 5. 更新已完成文件索引
            index_file = data_dir / "completed_index.json"
            index = {}
            with FileLock(data_dir / "locks" / "completed_index.lock"):
                if index_file.exists():
                    try:
                        with open(index_file, "r", encoding="utf-8") as f:
                            index = json.load(f)
                    except Exception:
                        pass
                index[file_hash] = {
                    "file_name": final_path.name,
                    "file_size": file_size,
                    "completed_path": str(final_path),
                    "completed_time": time.time()
                }
                atomic_write_json(index_file, index)

            # 6. 删除临时分片和进行中记录目录
            shutil.rmtree(incoming_dir, ignore_errors=True)

            return {
                "status": "success",
                "file_name": final_path.name,
                "size": file_size,
                "path": str(final_path)
            }
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to merge chunks: {e}")
        finally:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


@app.delete("/api/v1/transfers/{file_hash}", dependencies=[Depends(verify_token)])
def cancel_transfer(file_hash: str):
    """取消或清除指定哈希文件的分片及记录目录"""
    if not re.match(r"^[a-fA-F0-9]{64}$", file_hash):
        raise HTTPException(status_code=400, detail="Invalid SHA-256 hash format")

    data_dir = get_data_dir()
    incoming_dir = data_dir / "incoming" / file_hash
    if incoming_dir.exists():
        shutil.rmtree(incoming_dir, ignore_errors=True)
        return {"status": "success", "detail": "Transfer tasks and chunks deleted"}
    return {"status": "success", "detail": "Task was not active"}


def run_serve(args):
    """初始化数据目录与鉴权令牌，并启动 uvicorn 服务。"""
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "locks").mkdir(parents=True, exist_ok=True)
    (data_dir / "incoming").mkdir(parents=True, exist_ok=True)
    (data_dir / "completed").mkdir(parents=True, exist_ok=True)

    os.environ["FILE_TRANSFER_DATA_DIR"] = str(data_dir)
    if args.token:
        os.environ["FILE_TRANSFER_TOKEN"] = args.token

    if not os.environ.get("FILE_TRANSFER_TOKEN"):
        # 生成随机安全令牌作为警告和备用
        fallback_token = hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]
        os.environ["FILE_TRANSFER_TOKEN"] = fallback_token
        print(f"[Warning] FILE_TRANSFER_TOKEN is not set in environment or args.")
        print(f"[Warning] Temporary Fallback token generated: {fallback_token}")

    public_url = discover_public_url(args.port)
    print(f"Starting CNB File Transfer Service...")
    print(f"Host: {args.host} | Port: {args.port}")
    print(f"Local Storage Data Directory: {data_dir}")
    print(f"Auth Token: {os.environ.get('FILE_TRANSFER_TOKEN')[:4]}***{os.environ.get('FILE_TRANSFER_TOKEN')[-4:] if len(os.environ.get('FILE_TRANSFER_TOKEN', '')) > 8 else ''}")
    if public_url:
        print(f"Auto-Detected Public Access Domain: {public_url}")

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)
    server.run()
