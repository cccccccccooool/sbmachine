"""
本文件功能：解析 CS2 demo 文件为 DemoQuery 可消费的 JSON/parquet 产物。

启动方式：被 sbmachine/run_all.py 的 _run_demo_parse() 通过子进程调用；也可独立运行 python tools/parse_demo.py --demo ... --output-dir ...
输入数据流：CS2 .dem 文件。
输出数据流：output/demo/ 目录下的 demo_meta.json/rounds.json/kills.json/ticks.parquet 等。
用法用途：委托 Go 二进制（tools/parse_demo_go/parse_demo_go）使用 demoinfocs-golang 解析 demo，
然后将 ticks.jsonl 转为 ticks.parquet。支持 --build 重新编译 Go 二进制。
"""
import argparse
import subprocess
import sys
from pathlib import Path


def _go_binary() -> Path:
    script_dir = Path(__file__).parent
    go_dir     = script_dir / "parse_demo_go"
    name = "parse_demo_go.exe" if sys.platform == "win32" else "parse_demo_go"
    return go_dir / name


def _check_go() -> None:
    result = subprocess.run(["go", "version"], capture_output=True)
    if result.returncode != 0:
        print("[error] 'go' not found. Install Go 1.21+ from https://go.dev/dl/", file=sys.stderr)
        sys.exit(1)


def build_go(go_dir: Path) -> None:
    _check_go()
    print("Downloading Go dependencies (go mod tidy)...")
    r = subprocess.run(["go", "mod", "tidy"], cwd=str(go_dir), capture_output=False)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("Building Go binary...")
    result = subprocess.run(
        ["go", "build", "-o", "parse_demo_go.exe" if sys.platform == "win32" else "parse_demo_go", "."],
        cwd=str(go_dir),
        capture_output=False,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
    print("Build OK.")


def jsonl_to_parquet(out_dir: Path) -> None:
    """Convert ticks.jsonl → ticks.parquet using polars."""
    jsonl = out_dir / "ticks.jsonl"
    parquet = out_dir / "ticks.parquet"
    if not jsonl.exists():
        return
    try:
        import polars as pl
        df = pl.read_ndjson(str(jsonl), infer_schema_length=10000)
        df.write_parquet(str(parquet))
        print(f"  ticks.parquet: {len(df)} rows")
    except Exception as e:
        print(f"  [warn] ticks.jsonl → parquet failed: {e}. jsonl kept.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse CS2 demo → DemoQuery artefacts (via Go)")
    parser.add_argument("--demo",       required=True, help=".dem file path")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--build", action="store_true", help="rebuild Go binary first")
    args = parser.parse_args()

    go_dir = Path(__file__).parent / "parse_demo_go"
    binary = _go_binary()

    if args.build or not binary.exists():
        build_go(go_dir)

    if not binary.exists():
        print(f"[error] Go binary not found at {binary}. Run with --build first.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform != "win32":
        try:
            # Ensure the binary has execute permissions
            binary.chmod(binary.stat().st_mode | 0o111)
        except Exception as e:
            print(f"[warn] Failed to set execute permission on {binary}: {e}", file=sys.stderr)

    result = subprocess.run(
        [str(binary), "--demo", args.demo, "--output-dir", args.output_dir],
        capture_output=False,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)

    jsonl_to_parquet(out_dir)


if __name__ == "__main__":
    main()
