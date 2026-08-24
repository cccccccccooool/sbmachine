"""Web 文件上传器的 FastAPI 服务端：提供文件浏览、分片上传、建目录与删除接口。"""

import os
import shutil
import tempfile
import asyncio
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse

from tools.start.uploader.utils import (
    WORKSPACE_ROOT, safe_join, get_file_info, _write_bytes, _io_executor,
    shutdown_event, _assembly_locks
)
from tools.start.uploader.html import html_content

app = FastAPI(title="CNB Workspace Uploader")


@app.get("/api/files")
def list_files(path: str = Query("")):
    """列出指定目录下的文件与子目录，过滤隐藏及缓存目录。"""
    try:
        target_dir = safe_join(WORKSPACE_ROOT, path)
        if not target_dir.exists() or not target_dir.is_dir():
            raise HTTPException(status_code=404, detail="目录不存在")

        items = []
        for p in target_dir.iterdir():
            if p.name in (".git", ".idea", "__pycache__", ".cache", ".claude"):
                continue
            items.append(get_file_info(p, WORKSPACE_ROOT))

        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

        return {
            "current_path": target_dir.relative_to(WORKSPACE_ROOT).as_posix(),
            "items": items
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/upload/check")
def check_upload_status(upload_id: str, relative_path: str):
    """查询指定上传ID的已上传分片和目标文件状态"""
    if not upload_id or not relative_path:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    dest_path = safe_join(WORKSPACE_ROOT, relative_path)

    chunks = []
    if dest_path.parent.exists():
        for p in dest_path.parent.glob(f".{dest_path.name}.chunk*.{upload_id}"):
            try:
                idx = int(p.name.split(".chunk")[1].split(".")[0])
                chunks.append(idx)
            except Exception:
                pass

    exists = dest_path.exists()
    return {
        "exists": exists,
        "size": dest_path.stat().st_size if exists else 0,
        "completed_chunks": chunks
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    relative_path: str = Form(""),
    chunk_index: int = Form(0),
    total_chunks: int = Form(1),
    upload_id: str = Form(""),
):
    """接收整文件或分片上传，分片齐全后原子合并为目标文件。"""
    if shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="服务器正在关闭，请稍后重试")

    dest_rel_path = relative_path if relative_path else file.filename
    if not dest_rel_path:
        raise HTTPException(status_code=400, detail="未提供文件名")

    dest_path = safe_join(WORKSPACE_ROOT, dest_rel_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = await file.read()
    except asyncio.CancelledError:
        raise HTTPException(status_code=503, detail="上传被中断")

    if shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="服务器关闭，上传中止")

    try:
        if total_chunks <= 1:
            tmp_fd, tmp_str = tempfile.mkstemp(suffix=".upload.tmp", dir=str(dest_path.parent))
            os.close(tmp_fd)
            tmp_path = Path(tmp_str)
            try:
                await _write_bytes(tmp_path, data, "wb")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(_io_executor, tmp_path.replace, dest_path)
            except BaseException:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        else:
            shard_path = dest_path.parent / f".{dest_path.name}.chunk{chunk_index:05d}.{upload_id}"

            if not (shard_path.exists() and shard_path.stat().st_size == len(data)):
                tmp_shard_path = shard_path.with_suffix(".tmp")
                try:
                    await _write_bytes(tmp_shard_path, data, "wb")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(_io_executor, tmp_shard_path.replace, shard_path)
                except BaseException:
                    try:
                        tmp_shard_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise

            shard_glob = list(dest_path.parent.glob(f".{dest_path.name}.chunk*.{upload_id}"))
            if len(shard_glob) == total_chunks:
                lock_key = f"{upload_id}:{dest_rel_path}"
                if lock_key not in _assembly_locks:
                    _assembly_locks[lock_key] = asyncio.Lock()
                async with _assembly_locks[lock_key]:
                    shard_glob = list(dest_path.parent.glob(f".{dest_path.name}.chunk*.{upload_id}"))
                    shard_glob.sort(key=lambda p: int(p.name.split(".chunk")[1].split(".")[0]))
                    if len(shard_glob) == total_chunks:
                        tmp_fd2, tmp_str2 = tempfile.mkstemp(suffix=".upload.tmp", dir=str(dest_path.parent))
                        os.close(tmp_fd2)
                        tmp_path2 = Path(tmp_str2)
                        try:
                            loop = asyncio.get_event_loop()
                            def _assemble():
                                with open(tmp_path2, "wb") as out:
                                    for sp in shard_glob:
                                        with open(sp, "rb") as inp:
                                            shutil.copyfileobj(inp, out)
                                tmp_path2.replace(dest_path)
                                for sp in shard_glob:
                                    try:
                                        sp.unlink()
                                    except OSError:
                                        pass
                            await loop.run_in_executor(_io_executor, _assemble)
                        except BaseException:
                            try:
                                tmp_path2.unlink(missing_ok=True)
                            except OSError:
                                pass
                            raise
                    _assembly_locks.pop(lock_key, None)

        return {"status": "success", "file": dest_rel_path, "chunk": chunk_index}
    except (InterruptedError, asyncio.CancelledError):
        raise HTTPException(status_code=503, detail="上传被中断")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/mkdir")
def make_directory(path: str = Form(...)):
    """在工作区内创建目录（支持多级），返回创建的相对路径。"""
    try:
        target_dir = safe_join(WORKSPACE_ROOT, path)
        target_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "path": path}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/delete")
def delete_item(path: str = Query(...)):
    """删除工作区内指定的文件或目录，禁止删除工作区根目录。"""
    try:
        target_item = safe_join(WORKSPACE_ROOT, path)
        if not target_item.exists():
            raise HTTPException(status_code=404, detail="文件或目录不存在")

        if target_item == WORKSPACE_ROOT:
            raise HTTPException(status_code=400, detail="不能删除工作区根目录")

        if target_item.is_dir():
            shutil.rmtree(target_item)
        else:
            target_item.unlink()

        return {"status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def get_index():
    """返回上传器前端页面的 HTML 内容。"""
    return html_content
