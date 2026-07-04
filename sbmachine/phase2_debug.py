"""Phase 2 debug output helpers."""
from __future__ import annotations

import json
from pathlib import Path

from vision_service.region_crops import crop_frame


class DebugWriter:
    """当设置了 --debug-dir 时,写入逐帧的调试产物。

    在 debug_dir/ 下的输出布局:
      round_<NN>/
        frame_<ts>_pov_crop.png      - 喂给第一人称(POV)OCR 的感兴趣区域(ROI)
        frame_<ts>_timer_crop.png    - 喂给计时器 OCR 的感兴趣区域(ROI)
        frame_<ts>_masked.png        - 发送给 VLM 的遮罩了 UI 的画面
      frames.jsonl                   - 包含所有字段的逐帧 JSONL 文件,每帧一行
    """

    def __init__(self, debug_dir: Path | None) -> None:
        self.enabled = debug_dir is not None
        self.debug_dir = debug_dir
        self._jsonl_handle = None

    def open(self) -> None:
        if not self.enabled:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_handle = open(self.debug_dir / "frames.jsonl", "a", encoding="utf-8")

    def close(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None

    def save_crop(self, frame_dir: Path, stem: str, image) -> None:
        if not self.enabled or image is None:
            return
        try:
            import cv2
            frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_dir / stem), image)
        except Exception:
            pass

    def write_frame(self, record: dict) -> None:
        if not self.enabled or self._jsonl_handle is None:
            return
        try:
            self._jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_handle.flush()
        except Exception:
            pass

    def frame_dir(self, round_no: int) -> Path:
        return self.debug_dir / f"round_{round_no:02d}"

    def crop_image(self, frame, region: dict | None, padding: int = 0):
        """返回给定区域字典的裁剪图像,如果不可用则返回 None。"""
        if frame is None or region is None:
            return None
        try:
            return crop_frame(frame, region.get("box", []), padding=padding)
        except Exception:
            return None

