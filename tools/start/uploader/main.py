"""Web 文件上传器的命令行入口：清理残留临时文件并启动 uvicorn 服务。"""

import argparse
import atexit

import uvicorn

from tools.start.uploader.utils import WORKSPACE_ROOT, _cleanup_temp_files, shutdown_event
from tools.start.uploader.server import app


def main():
    parser = argparse.ArgumentParser(description="CNB Workspace Web File Uploader")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server (default: 8000)")
    args = parser.parse_args()

    _cleanup_temp_files()
    atexit.register(_cleanup_temp_files)

    print(f"Starting CNB Workspace File Uploader...")
    print(f"Workspace Root: {WORKSPACE_ROOT}")
    print(f"Server is starting on http://{args.host}:{args.port}")
    print(f"Please use CNB Port Forwarding to access this interface.")

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)

    _original_handle_exit = server.handle_exit

    def _patched_handle_exit(sig, frame):
        print("\n[shutdown] 收到中断信号，正在优雅关闭...")
        shutdown_event.set()
        _original_handle_exit(sig, frame)

    server.handle_exit = _patched_handle_exit
    server.run()

    print("[shutdown] 服务器已安全退出")


if __name__ == "__main__":
    main()
