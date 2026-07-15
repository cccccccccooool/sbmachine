"""Phase 2 diagnostic output helpers."""
from __future__ import annotations

import json
from pathlib import Path

from vision_service.region_crops import crop_frame


class DebugWriter:
    """Optional phase-2 debug sink: writes per-frame JSONL and crop images when enabled."""

    def __init__(self, debug_dir: Path | None) -> None:
        self.enabled = debug_dir is not None
        self.debug_dir = debug_dir
        self._jsonl_handle = None

    def open(self) -> None:
        """Open the frames.jsonl handle for appending; no-op when debugging is disabled."""
        if not self.enabled:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_handle = open(self.debug_dir / "frames.jsonl", "a", encoding="utf-8")

    def close(self) -> None:
        """Close the frames.jsonl handle if it is open."""
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None

    def save_crop(self, frame_dir: Path, stem: str, image) -> None:
        """Write a crop image to disk, swallowing errors so debugging never breaks the pipeline."""
        if not self.enabled or image is None:
            return
        try:
            import cv2
            frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_dir / stem), image)
        except Exception:
            pass

    def write_frame(self, record: dict) -> None:
        """Append one frame record as a JSONL line, swallowing write errors."""
        if not self.enabled or self._jsonl_handle is None:
            return
        try:
            self._jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_handle.flush()
        except Exception:
            pass

    def frame_dir(self, round_no: int) -> Path:
        """Return the per-round debug subdirectory path."""
        return self.debug_dir / f"round_{round_no:02d}"

    def crop_image(self, frame, region: dict | None, padding: int = 0):
        """Crop a region box out of a frame, returning None on any failure."""
        if frame is None or region is None:
            return None
        try:
            return crop_frame(frame, region.get("box", []), padding=padding)
        except Exception:
            return None
