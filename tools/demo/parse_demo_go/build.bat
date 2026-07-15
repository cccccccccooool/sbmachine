@echo off
cd /d "%~dp0"
echo Fetching dependencies...
go mod tidy
echo Building...
go build -o parse_demo_go.exe .
echo Done.
