#!/usr/bin/env python3
"""
CNB 工作区文件上传器 - 启动入口
"""
import sys
from pathlib import Path

# 确保项目根目录在模块搜索路径中
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.start.uploader.server import app
from tools.start.uploader.main import main

if __name__ == "__main__":
    main()
