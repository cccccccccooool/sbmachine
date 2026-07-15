"""单次流水线运行的事务化输出：先暂存（staging）再提升（promote）到正式目录。"""
from __future__ import annotations

import copy
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


_PATH_OUTPUTS = {
    "rounds_json": "rounds.json",
    "round_list_json": "round_list.json",
    "segments_out_json": "segments.json",
    "rounds_with_yolo_json": "rounds_with_yolo.json",
    "rounds_with_yolo_semantic_json": "rounds_with_yolo_semantic.json",
    "rounds_with_neutral_json": "rounds_with_neutral.json",
    "rounds_with_commentary_json": "rounds_with_commentary.json",
    "rounds_final_json": "rounds_final.json",
    "commentary_json": "commentary.json",
    "assemble_manifest_json": "assemble_manifest.json",
}

_STAGE_ARTIFACTS = (
    ("demo_parse", ("demo",)),
    ("video_marking", ("sbmachine/detector_rows.jsonl", "sbmachine/segments.json")),
    ("phase1", ("sbmachine/rounds.json", "sbmachine/round_list.json", "sbmachine/segments.json", "sbmachine/clips")),
    ("phase2", ("sbmachine/rounds_with_yolo.json", "sbmachine/rounds_with_yolo_semantic.json")),
    ("phase3a", ("sbmachine/rounds_with_neutral.json",)),
    ("phase3b", ("sbmachine/rounds_with_commentary.json", "sbmachine/commentary.json")),
    (
        "phase4",
        (
            "sbmachine/rounds_final.json",
            "sbmachine/assemble_manifest.json",
            "sbmachine/rounds",
            "sbmachine/audio",
            "sbmachine/video_clips",
            "sbmachine/final_commentary.wav",
            "sbmachine/final_video.mp4",
            "sbmachine/.cache",
        ),
    ),
)


class RunContext:
    def __init__(self, output_root: Path, run_id: str | None = None) -> None:
        self.output_root = output_root
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.run_id = run_id or f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.staging_dir = output_root / ".staging" / self.run_id
        self.publish_dir = self.staging_dir / "publish"
        self.diagnostics_dir = self.staging_dir / "diagnostics"
        self.error_dir = output_root / "error" / self.run_id
        self.effective_config_path = self.diagnostics_dir / "effective_config.yaml"
        self.current_stage = "setup"
        self.started = False
        self.checkpointed_stages: list[str] = []

    def prepare(self, config: dict) -> tuple[dict, Path]:
        self.publish_dir.mkdir(parents=True, exist_ok=False)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=False)
        for name in ("demo", "sbmachine"):
            source = self.output_root / name
            target = self.publish_dir / name
            if source.exists():
                shutil.copytree(source, target)
        self._invalidate_stale_outputs(config)
        effective = self._effective_config(config)
        self.effective_config_path.write_text(
            yaml.safe_dump(effective, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.started = True
        return effective, self.effective_config_path

    def _invalidate_stale_outputs(self, config: dict) -> None:
        from sbmachine.preflight import enabled_phases

        active = set(enabled_phases(config))
        first = next((index for index, (stage, _) in enumerate(_STAGE_ARTIFACTS) if stage in active), None)
        if first is None:
            return
        for _, relative_paths in _STAGE_ARTIFACTS[first:]:
            for relative in relative_paths:
                path = self.publish_dir / relative
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                elif path.exists() or path.is_symlink():
                    path.unlink()
        # 旧的整轮运行 manifest 无法描述这次部分 checkpoint 的运行，必须清掉。
        manifest = self.publish_dir / "sbmachine" / "run_manifest.json"
        if manifest.exists() or manifest.is_symlink():
            manifest.unlink()

    def _effective_config(self, config: dict) -> dict:
        effective = copy.deepcopy(config)
        demo_dir = self.publish_dir / "demo"
        sb_dir = self.publish_dir / "sbmachine"
        # 各阶段写入方拿到的是文件路径，并非都负责在首次运行时创建共享的父目录。
        sb_dir.mkdir(parents=True, exist_ok=True)

        paths = effective.setdefault("paths", {})
        demo = effective.setdefault("demo", {})
        paths["demo_output_dir"] = str(demo_dir)
        demo["parsed_dir"] = str(demo_dir)
        for key, relative in _PATH_OUTPUTS.items():
            paths[key] = str(sb_dir / relative)
        if paths.get("clip_dir"):
            paths["clip_dir"] = str(sb_dir / "clips")
        phases = effective.get("phases", {}) if isinstance(effective.get("phases", {}), dict) else {}
        if phases.get("video_marking", False):
            paths["hud_detections_jsonl"] = str(sb_dir / "detector_rows.jsonl")

        phase4 = effective.setdefault("phase4", {})
        phase4["output_dir"] = str(sb_dir / "rounds")
        phase4["tts_cache_dir"] = str(sb_dir / ".cache" / "tts")
        phase4["clip_cache_dir"] = str(sb_dir / ".cache" / "clips")

        tts = effective.setdefault("tts", {})
        tts["output_dir"] = str(sb_dir / "audio")
        tts["final_audio"] = str(sb_dir / "final_commentary.wav")
        video = effective.setdefault("video", {})
        video["clip_dir"] = str(sb_dir / "video_clips")
        video["final_video"] = str(sb_dir / "final_video.mp4")

        # 现有的 debug 写入方用的是仓库全局路径，不属于发布契约，
        # 因此事务化运行时禁用它们。
        debug = effective.setdefault("debug", {})
        debug["phase3"] = False
        return effective

    def write_diagnostic(self, name: str, payload: Any) -> Path:
        path = self.diagnostics_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _rewrite_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._rewrite_value(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self._rewrite_value(child) for child in value]
        if isinstance(value, str):
            return value.replace(str(self.publish_dir), str(self.output_root))
        return value

    def rewrite_published_references(self) -> None:
        self._rewrite_references_in(self.publish_dir)

    def _refresh_neutral_source_hash(self, sb_dir: Path | None = None) -> None:
        from sbmachine.neutral_contract import rounds_sha256, validate_neutral_manifest

        sb_dir = self.publish_dir / "sbmachine" if sb_dir is None else sb_dir
        neutral_path = sb_dir / "rounds_with_neutral.json"
        if not neutral_path.is_file():
            return
        rounds_path = sb_dir / "rounds_with_yolo.json"
        if not rounds_path.is_file():
            raise ValueError("cannot publish neutral artifact without rounds_with_yolo.json")
        payload = json.loads(neutral_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("neutral artifact must be a JSON object")
        payload["source_rounds_sha256"] = rounds_sha256(rounds_path)
        neutral_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        validate_neutral_manifest(payload, rounds_path)

    def checkpoint(self, stage: str) -> None:
        """发布一个已校验的独立阶段，但不结束本次运行。

        checkpoint 刻意保持暂存目录树完整：后续阶段仍消费各自暂存的输入，
        而一旦之后失败，已校验的上游工件仍可用于按阶段选择性重跑。
        """
        names = ("demo",) if stage == "demo_parse" else ("sbmachine",)
        checkpoint_root = self.staging_dir / "checkpoints" / stage
        if checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)
        for name in names:
            source = self.publish_dir / name
            if source.exists():
                shutil.copytree(source, checkpoint_root / name)
        if not any((checkpoint_root / name).exists() for name in names):
            raise ValueError(f"checkpoint {stage} has no publishable outputs")
        self._rewrite_references_in(checkpoint_root)
        self._promote_from(checkpoint_root, names)
        self.checkpointed_stages.append(stage)

    def _rewrite_references_in(self, root: Path) -> None:
        for path in root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rewritten = self._rewrite_value(payload)
            if rewritten != payload:
                path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8")

        neutral_path = root / "sbmachine" / "rounds_with_neutral.json"
        if neutral_path.is_file():
            self._refresh_neutral_source_hash(root / "sbmachine")

    def complete(self, manifest: dict) -> dict:
        sb_dir = self.publish_dir / "sbmachine"
        sb_dir.mkdir(parents=True, exist_ok=True)
        (sb_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._rewrite_references_in(self.publish_dir)
        self._promote_from(self.publish_dir, ("demo", "sbmachine"))
        if self.staging_dir.exists():
            try:
                shutil.rmtree(self.staging_dir, ignore_errors=True)
            except OSError:
                pass
        return manifest

    def _promote_from(self, source_root: Path, requested_names: tuple[str, ...]) -> None:
        backups = self.diagnostics_dir / "previous_success"
        moved_old: list[str] = []
        moved_new: list[str] = []
        names = [name for name in requested_names if (source_root / name).exists()]
        try:
            for name in names:
                target = self.output_root / name
                if target.exists():
                    backups.mkdir(parents=True, exist_ok=True)
                    os.replace(target, backups / name)
                    moved_old.append(name)
            for name in names:
                os.replace(source_root / name, self.output_root / name)
                moved_new.append(name)
        except Exception:
            for name in reversed(moved_new):
                target = self.output_root / name
                if target.exists():
                    os.replace(target, source_root / name)
            for name in reversed(moved_old):
                backup = backups / name
                if backup.exists():
                    os.replace(backup, self.output_root / name)
            raise
        if backups.exists():
            try:
                shutil.rmtree(backups, ignore_errors=True)
            except OSError:
                pass

    def fail(self, failed_stage: str, error: str, *, extra: dict | None = None) -> dict:
        failure = {
            "run_id": self.run_id,
            "status": "failed",
            "publishable": False,
            "failed_stage": failed_stage,
            "error": error,
            "previous_success_preserved": not self.checkpointed_stages,
            "checkpointed_stages": list(self.checkpointed_stages),
            "partial_outputs_published": bool(self.checkpointed_stages),
        }
        if extra:
            failure.update(extra)
        if not self.staging_dir.exists():
            self.staging_dir.mkdir(parents=True, exist_ok=True)
        (self.staging_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.error_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.staging_dir, self.error_dir)
        failure["error_dir"] = str(self.error_dir)
        return failure
