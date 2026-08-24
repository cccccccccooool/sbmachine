"""为解析出的 demo 产物生成与校验完整性清单（manifest）的辅助函数。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MANIFEST_NAME = "demo_manifest.json"
REQUIRED_JSON_FILES = (
    "demo_meta.json",
    "rounds.json",
    "roster.json",
    "kills.json",
    "grenades.json",
    "damages.json",
    "smokes.json",
    "infernos.json",
    "flashes.json",
)
TICK_FILES = ("ticks.parquet", "ticks.jsonl")
OPTIONAL_CAPABILITY_FILES = {
    "weapon_fire": "fired.json",
    "item_equip": "equips.json",
    "event_snapshots": "event_snapshots.json",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DemoManifestError(ValueError):
    """当解析出的 demo 目录不完整或被篡改时抛出。"""


def sha256_file(path: Path) -> str:
    """流式计算文件的 sha256 摘要，避免一次性读入大文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    """统计产物的行数：jsonl 按非空行计，json 按顶层元素个数计。"""
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, (list, dict)):
            return len(payload)
        return 1
    raise DemoManifestError(f"row count is required for {path.name}")


def write_demo_manifest(
    parsed_dir: Path,
    source_demo: Path,
    *,
    row_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """最后写清单：在所有必需产物都已落盘后再写，保证清单指向完整数据。"""
    parsed_dir = Path(parsed_dir)
    source_demo = Path(source_demo)
    row_counts = dict(row_counts or {})
    meta = json.loads((parsed_dir / "demo_meta.json").read_text(encoding="utf-8"))
    capabilities = meta.get("capabilities") if isinstance(meta, dict) else {}
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    names = list(REQUIRED_JSON_FILES)
    for capability, name in OPTIONAL_CAPABILITY_FILES.items():
        path = parsed_dir / name
        if capabilities.get(capability) is True and not path.is_file():
            raise DemoManifestError(
                f"demo_meta.json declares capability {capability!r}, but {name} is missing"
            )
        if path.is_file():
            names.append(name)
    names.extend(name for name in TICK_FILES if (parsed_dir / name).is_file())
    if not any(name in names for name in TICK_FILES):
        raise DemoManifestError("parsed demo has no tick artifact")

    files: dict[str, dict[str, Any]] = {}
    for name in names:
        path = parsed_dir / name
        if not path.is_file():
            raise DemoManifestError(f"parsed demo is missing required artifact: {name}")
        rows = row_counts.get(name)
        if rows is None:
            rows = _row_count(path)
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise DemoManifestError(f"invalid row count for {name}: {rows!r}")
        files[name] = {"rows": rows, "sha256": sha256_file(path)}

    try:
        tick_rate = float(meta["tick_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DemoManifestError("demo_meta.json has no valid tick_rate") from exc
    if tick_rate <= 0:
        raise DemoManifestError("demo_meta.json tick_rate must be positive")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source_demo_sha256": sha256_file(source_demo),
        "tick_rate": tick_rate,
        "files": files,
    }
    manifest_path = parsed_dir / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def validate_demo_manifest(parsed_dir: Path) -> dict[str, Any]:
    """校验完成状态、必需文件、行数与哈希是否与清单一致。"""
    parsed_dir = Path(parsed_dir)
    manifest_path = parsed_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise DemoManifestError(f"parsed demo is missing {MANIFEST_NAME}; reparse the demo")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoManifestError(f"cannot read {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DemoManifestError(f"{MANIFEST_NAME} must contain an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DemoManifestError("unsupported demo manifest schema_version")
    if manifest.get("status") != "complete":
        raise DemoManifestError("demo manifest status is not complete")
    source_hash = manifest.get("source_demo_sha256")
    if not isinstance(source_hash, str) or not _SHA256_RE.fullmatch(source_hash):
        raise DemoManifestError("demo manifest has an invalid source_demo_sha256")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DemoManifestError("demo manifest files must be an object")
    for name in REQUIRED_JSON_FILES:
        if name not in files:
            raise DemoManifestError(f"demo manifest is missing required artifact: {name}")
    if not any(name in files for name in TICK_FILES):
        raise DemoManifestError("demo manifest is missing a tick artifact")

    try:
        meta = json.loads((parsed_dir / "demo_meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoManifestError("cannot read demo_meta.json") from exc
    capabilities = meta.get("capabilities") if isinstance(meta, dict) else {}
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    for capability, name in OPTIONAL_CAPABILITY_FILES.items():
        if capabilities.get(capability) is True and name not in files:
            raise DemoManifestError(
                f"demo capability {capability!r} requires manifest artifact: {name}"
            )

    for name, info in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise DemoManifestError(f"invalid artifact name in demo manifest: {name!r}")
        if not isinstance(info, dict):
            raise DemoManifestError(f"invalid manifest entry for {name}")
        expected_hash = info.get("sha256")
        rows = info.get("rows")
        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
            raise DemoManifestError(f"invalid sha256 for {name}")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise DemoManifestError(f"invalid row count for {name}")
        path = parsed_dir / name
        if not path.is_file():
            raise DemoManifestError(f"manifest artifact is missing: {name}")
        if sha256_file(path) != expected_hash:
            raise DemoManifestError(f"manifest hash mismatch for {name}")
        if path.suffix in {".json", ".jsonl"} and _row_count(path) != rows:
            raise DemoManifestError(f"manifest row count mismatch for {name}")

    try:
        manifest_tick_rate = float(manifest["tick_rate"])
        meta_tick_rate = float(meta["tick_rate"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DemoManifestError("demo manifest or metadata has no valid tick_rate") from exc
    if manifest_tick_rate <= 0 or manifest_tick_rate != meta_tick_rate:
        raise DemoManifestError("demo manifest tick_rate does not match demo_meta.json")
    return manifest
