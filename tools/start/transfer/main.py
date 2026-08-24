"""文件传输工具的命令行入口：解析子命令并分发到服务端或客户端逻辑。"""

import os
import sys
import argparse

from tools.start.transfer.utils import (
    DEFAULT_HOST, DEFAULT_PORT, DEFAULT_DATA_DIR,
    DEFAULT_CHUNK_SIZE, DEFAULT_MAX_CONCURRENCY
)
from tools.start.transfer.server import run_serve
from tools.start.transfer.client import (
    run_send, run_status, run_receive_list, run_cleanup
)


def main():
    parser = argparse.ArgumentParser(description="CNB Secure File Transfer Facility")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the file receiving server")
    serve_parser.add_argument("--host", default=os.environ.get("FILE_TRANSFER_HOST", DEFAULT_HOST), help="Host binding IP")
    serve_parser.add_argument("--port", type=int, default=int(os.environ.get("FILE_TRANSFER_PORT", DEFAULT_PORT)), help="Port binding")
    serve_parser.add_argument("--data-dir", default=os.environ.get("FILE_TRANSFER_DATA_DIR", DEFAULT_DATA_DIR), help="Data storage root")
    serve_parser.add_argument("--token", default=os.environ.get("FILE_TRANSFER_TOKEN"), help="Pre-shared authentication token")
    serve_parser.add_argument("--log-level", default="INFO", help="Server log level (debug, info, warning)")

    # send command
    send_parser = subparsers.add_parser("send", help="Send file to remote receiver")
    send_parser.add_argument("--target", help="Remote server URL (e.g. https://xxxx-7459.cnb.run)")
    send_parser.add_argument("--file", required=True, help="Local file path to transfer")
    send_parser.add_argument("--token", help="Pre-shared authentication token")
    send_parser.add_argument("--concurrency", type=int, default=int(os.environ.get("FILE_TRANSFER_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)), help="Chunk upload concurrency")
    send_parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("FILE_TRANSFER_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)), help="Chunk size in bytes")
    send_parser.add_argument("--max-retries", type=int, default=5, help="Max upload retry attempts per chunk")

    # status command
    status_parser = subparsers.add_parser("status", help="Get status of a file hash transfer")
    status_parser.add_argument("--target", help="Remote server URL")
    status_parser.add_argument("--hash", required=True, help="SHA-256 file hash to query")
    status_parser.add_argument("--token", help="Pre-shared authentication token")

    # receive-list command
    list_parser = subparsers.add_parser("receive-list", help="List completed transfers in local storage")
    list_parser.add_argument("--data-dir", default=os.environ.get("FILE_TRANSFER_DATA_DIR", DEFAULT_DATA_DIR), help="Data storage root")

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up stale chunk folders and locks")
    cleanup_parser.add_argument("--data-dir", default=os.environ.get("FILE_TRANSFER_DATA_DIR", DEFAULT_DATA_DIR), help="Data storage root")
    cleanup_parser.add_argument("--older-than", default="7d", help="Clean up folders older than this duration (e.g. 7d, 24h, 60m)")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without executing")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "serve":
        run_serve(args)
    elif args.command == "send":
        run_send(args)
    elif args.command == "status":
        run_status(args)
    elif args.command == "receive-list":
        run_receive_list(args)
    elif args.command == "cleanup":
        run_cleanup(args)


if __name__ == "__main__":
    main()
