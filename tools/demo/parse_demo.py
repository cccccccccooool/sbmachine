"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：解析 CS2 demo 文件为 DemoQuery 可消费的 JSON/parquet 产物。

启动方式：被 sbmachine/run_all.py 的 _run_demo_parse() 通过子进程调用；也可独立运行 python tools/demo/parse_demo.py --demo ... --output-dir ...
输入数据流：CS2 .dem 文件。
输出数据流：output/demo/ 目录下的 demo_meta.json/rounds.json/kills.json/ticks.parquet 等。
用法用途：委托 Go 二进制（tools/demo/parse_demo_go/parse_demo_go）使用 demoinfocs-golang 解析 demo，
然后将 ticks.jsonl 转为 ticks.parquet。支持 --build 重新编译 Go 二进制。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

try:
    from tools.demo.demo_manifest import write_demo_manifest
except ModuleNotFoundError:  # Direct script execution adds tools/demo to sys.path.
    from demo_manifest import write_demo_manifest


class DemoEntityStateError(RuntimeError):
    """DEM 暴露了事件，但没有任何可用的参与者实体快照。"""


def _validated_tick_row_count(out_dir: Path) -> int:
    """在交给任何 dataframe 库之前，拒绝残缺的解析器产物。"""
    jsonl = out_dir / "ticks.jsonl"
    roster_path = out_dir / "roster.json"
    if not jsonl.is_file():
        raise DemoEntityStateError(f"Go parser did not produce {jsonl.name}")
    with jsonl.open("r", encoding="utf-8") as handle:
        tick_rows = sum(1 for line in handle if line.strip())
    if tick_rows == 0:
        raise DemoEntityStateError(
            "DEM entity-state recovery failed: ticks.jsonl has no participant snapshots; "
            "PacketEntities may be incompatible with this demo. No output was published."
        )
    try:
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoEntityStateError(
            "DEM entity-state recovery failed: roster.json is unavailable; no output was published."
        ) from exc
    if not isinstance(roster, list) or not roster:
        raise DemoEntityStateError(
            "DEM entity-state recovery failed: roster.json has no players; "
            "PacketEntities may be incompatible with this demo. No output was published."
        )
    return tick_rows


def _go_binary() -> Path:
    """返回当前平台对应的 Go 解析器二进制路径。"""
    script_dir = Path(__file__).parent
    go_dir = script_dir / "parse_demo_go"
    name = "parse_demo_go.exe" if sys.platform == "win32" else "parse_demo_go"
    return go_dir / name


def _check_go() -> None:
    """确认本机可用 go 命令，缺失则报错退出。"""
    result = subprocess.run(["go", "version"], capture_output=True)
    if result.returncode != 0:
        print("[error] 'go' not found. Install Go 1.21+ from https://go.dev/dl/", file=sys.stderr)
        sys.exit(1)


def build_go(go_dir: Path) -> None:
    """执行 go mod tidy 并编译 Go 解析器二进制。"""
    _check_go()
    print("Downloading Go dependencies (go mod tidy)...")
    tidy_result = subprocess.run(["go", "mod", "tidy"], cwd=str(go_dir), capture_output=False)
    if tidy_result.returncode != 0:
        sys.exit(tidy_result.returncode)
    print("Building Go binary...")
    result = subprocess.run(
        ["go", "build", "-o", "parse_demo_go.exe" if sys.platform == "win32" else "parse_demo_go", "."],
        cwd=str(go_dir),
        capture_output=False,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
    print("Build OK.")


def jsonl_to_parquet(out_dir: Path) -> int:
    """用 polars 将 ticks.jsonl 转换为 ticks.parquet，返回行数。"""
    jsonl = out_dir / "ticks.jsonl"
    parquet = out_dir / "ticks.parquet"
    tick_rows = _validated_tick_row_count(out_dir)
    try:
        import polars as pl
        df = pl.read_ndjson(str(jsonl), infer_schema_length=10000)
        if len(df) != tick_rows:
            raise RuntimeError(f"ticks.jsonl row count changed during conversion: {tick_rows} -> {len(df)}")
        df.write_parquet(str(parquet))
        print(f"  ticks.parquet: {tick_rows} rows")
        return tick_rows
    except Exception as exc:
        raise RuntimeError(f"ticks.jsonl → parquet failed: {exc}") from exc


def _commit_output(staging_dir: Path, out_dir: Path) -> None:
    """将完成的临时目录原子替换为正式输出目录，失败时保留旧输出。"""
    backup_dir: Path | None = None
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {out_dir}")
        backup_dir = out_dir.with_name(f".{out_dir.name}.backup-{uuid.uuid4().hex}")
        os.replace(out_dir, backup_dir)
    try:
        os.replace(staging_dir, out_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not out_dir.exists():
            os.replace(backup_dir, out_dir)
        raise
    if backup_dir is not None:
        try:
            shutil.rmtree(backup_dir, ignore_errors=True)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse CS2 demo → DemoQuery artefacts (via Go)")
    parser.add_argument("--demo", required=True, help=".dem file path")
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

    demo_path = Path(args.demo).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.parse-", dir=out_dir.parent))

    if sys.platform != "win32":
        try:
            # 确保二进制具备可执行权限
            binary.chmod(binary.stat().st_mode | 0o111)
        except Exception as exc:
            print(f"[warn] Failed to set execute permission on {binary}: {exc}", file=sys.stderr)

    try:
        result = subprocess.run(
            [str(binary), "--demo", str(demo_path), "--output-dir", str(staging_dir)],
            capture_output=False,
        )
        if result.returncode != 0:
            return result.returncode

        tick_rows = jsonl_to_parquet(staging_dir)
        write_demo_manifest(
            staging_dir,
            demo_path,
            row_counts={"ticks.jsonl": tick_rows, "ticks.parquet": tick_rows},
        )
        _commit_output(staging_dir, out_dir)
        return 0
    except Exception as exc:
        print(f"[error] demo parse failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
