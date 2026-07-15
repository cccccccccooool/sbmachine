#!/usr/bin/env bash
# 6657 风格离线录像解说 AI 项目
# 本文件功能：编译 parse_demo_go 二进制文件。
# 启动方式：bash tools/parse_demo_go/build.sh
# 输入数据流：Go 源码文件 (main.go) 和 go.mod。
# 输出数据流：编译后的 parse_demo_go 可执行文件。
# 用法用途：首次使用前或更新 main.go 后运行一次，生成 parse_demo_go 二进制文件。
#
# Build parse_demo_go binary.
# Run once before first use, or after updating main.go.
set -e
cd "$(dirname "$0")"
echo "Fetching dependencies..."
go mod tidy
echo "Building..."
go build -o parse_demo_go .
echo "Done: $(pwd)/parse_demo_go"
