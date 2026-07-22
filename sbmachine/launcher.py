"""User-facing runtime commands, kept separate from the legacy pipeline entry.

``doctor`` only inspects prerequisites.  ``setup`` records one explicit
backend choice.  ``run`` uses that choice and never falls back silently to a
different backend.  ``--mock`` exercises the same selection and dispatch path
without starting services, invoking Docker, downloading, or writing state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from core.config_loader import ConfigError
from sbmachine.common import load_config, require_path
from sbmachine.run_all import run_all
from sbmachine.runtime_backend import (
    RuntimeBackendError,
    create_runtime_backend,
    load_runtime_selection,
    resolve_runtime_backend,
    save_runtime_selection,
)


def _print_report(report: dict[str, object]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def _error(action: str, error: Exception) -> dict[str, object]:
    return {
        "action": action,
        "status": "failed",
        "error": str(error),
        "downloads_performed": False,
    }


def _load(args: argparse.Namespace) -> tuple[Path, dict]:
    config_path = require_path(args.config, "--config")
    return config_path, load_config(config_path)


def _backend_for(
    command: str,
    args: argparse.Namespace,
    config: dict,
) -> tuple[str, str]:
    if args.backend:
        return resolve_runtime_backend(config, args.backend), "command line"
    if command == "run" and not args.mock:
        selected = load_runtime_selection(args.workspace)
        if selected is None:
            state_path = Path(args.workspace).resolve() / ".sbmachine" / "runtime.json"
            raise RuntimeBackendError(
                f"no runtime backend has been selected; run setup first ({state_path})"
            )
        return selected, "setup state"
    return resolve_runtime_backend(config), "configuration"


def _doctor(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    _, config = _load(args)
    selected, selected_by = _backend_for("doctor", args, config)
    report = create_runtime_backend(config, selected, mock=args.mock).doctor()
    report["selected_by"] = selected_by
    return (0 if bool(report.get("ready")) else 2), report


def _setup(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    _, config = _load(args)
    selected, selected_by = _backend_for("setup", args, config)
    runtime = create_runtime_backend(config, selected, mock=args.mock)
    report = runtime.setup()
    report["selected_by"] = selected_by
    if args.mock:
        report["state_written"] = False
        report["state_reason"] = "mock mode never writes runtime state"
        return 0, report

    state_path = save_runtime_selection(args.workspace, selected)
    report["state_written"] = True
    report["state_path"] = str(state_path)
    # A selection is persistent even if the doctor truthfully reports missing
    # prerequisites.  A later run will use this exact backend and fail clearly
    # instead of switching to another one.
    return (0 if bool(report.get("ready")) else 2), report


def _run(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    config_path, config = _load(args)
    selected, selected_by = _backend_for("run", args, config)
    runtime = create_runtime_backend(config, selected, mock=args.mock)
    if args.mock:
        report = runtime.simulate_pipeline()  # type: ignore[attr-defined]
        report["selected_by"] = selected_by
        return 0, report

    doctor = runtime.doctor()
    if not bool(doctor.get("ready")):
        return 2, {
            "action": "run",
            "status": "failed",
            "backend": selected,
            "selected_by": selected_by,
            "error": "selected backend is not ready; run doctor for details",
            "doctor": doctor,
            "downloads_performed": False,
        }
    result = run_all(config_path, runtime_backend=selected)
    result["backend"] = selected
    result["selected_by"] = selected_by
    return int(result.get("exit_code", 1)), result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="sbmachine runtime launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "setup", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default="config/", help="configuration directory or YAML file")
        child.add_argument("--workspace", default=".", help="workspace containing .sbmachine/runtime.json")
        child.add_argument("--backend", choices=("local", "container"), help="explicit runtime selection")
        child.add_argument("--mock", action="store_true", help="simulate without Docker, models, GPU, network, or state writes")
        if command == "setup":
            child.add_argument("--no-download", action="store_true", help="accepted for explicit no-download setup scripts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            code, report = _doctor(args)
        elif args.command == "setup":
            code, report = _setup(args)
        else:
            code, report = _run(args)
    except (ConfigError, OSError, ValueError, RuntimeBackendError) as exc:
        code, report = 2, _error(args.command, exc)
    _print_report(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
