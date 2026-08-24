#!/usr/bin/env python3
"""
CNB 安全文件传输设施 (CNB File Transfer Utility) - 启动入口
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.start.transfer.utils import (
    FileLock, sanitize_filename, get_file_hash, discover_public_url,
    get_data_dir, atomic_write_json
)
from tools.start.transfer.client import run_cleanup
from tools.start.transfer.server import app
from tools.start.transfer.main import main

if __name__ == "__main__":
    main()
