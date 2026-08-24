"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：视频随机抽帧操作界面与命令行工具。

启动方式：
  - GUI 操作页面：python tools/utils/random_frame_review.py
  - 命令行静默运行：python tools/utils/random_frame_review.py --video <video_path> [选项]
输入数据流：输入视频文件路径。
输出数据流：抽取的视频帧图片 (JPEG) 和 metadata JSON。
用法用途：从视频中随机抽取帧用于后续标注或审查，支持 GUI 配置和命令行批量运行。

行为说明：
  1. 启动 GUI 操作页面，配置并一键抽取视频帧；
  2. 如果通过命令行传入参数，可在后台无 GUI 批量运行抽帧；
  3. 剔除了任何人工审核（保留/拒绝/不确定）交互模块。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

# PyQt5/PyTorch WinError DLL load workaround
try:
    import torch
except ImportError:
    pass

try:
    import cv2
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    sys.exit(f"缺少依赖项: {exc}。请运行: pip install PyQt5 opencv-python pyyaml")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXT = ".jpg"


def resolve_path(value: str | Path) -> Path:
    """将相对路径解析为基于项目根目录的绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def to_relative_path(path: Path | str | None) -> str:
    """把路径转成相对项目根的 POSIX 字符串，无法相对化则返回原始 POSIX 路径。"""
    if not path:
        return ""
    target = Path(path)
    try:
        return target.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return target.as_posix()


def video_duration_sec(video_path: Path) -> float:
    """读取视频总时长（秒），无法解析 fps 或帧数时返回 0。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frames <= 0:
            return 0.0
        return frames / fps
    finally:
        cap.release()


def sample_times(
    *,
    duration: float,
    count: int,
    start_sec: float,
    end_sec: float,
    min_gap_sec: float,
    seed: int | None,
) -> list[float]:
    """在 [start_sec, end_sec] 区间内随机采样 count 个时间点，尽量满足最小间隔约束。"""
    stop = min(end_sec if end_sec > 0 else duration, duration)
    start = max(0.0, min(start_sec, stop))
    if stop <= start:
        raise ValueError("终止时间必须大于起始时间。")

    rng = random.Random(seed)
    count = max(1, int(count))
    min_gap_sec = max(0.0, float(min_gap_sec))
    selected: list[float] = []
    attempts = 0
    max_attempts = count * 200
    while len(selected) < count and attempts < max_attempts:
        attempts += 1
        ts = rng.uniform(start, stop)
        if min_gap_sec and any(abs(ts - old) < min_gap_sec for old in selected):
            continue
        selected.append(ts)

    if len(selected) < count:
        remaining = count - len(selected)
        selected.extend(rng.uniform(start, stop) for _ in range(remaining))
    return sorted(selected)


def extract_random_frames(
    video_path: Path,
    output_dir: Path,
    *,
    count: int,
    start_sec: float,
    end_sec: float,
    min_gap_sec: float,
    seed: int | None,
    jpeg_quality: int,
    progress_callback: Callable[[int, int, float], None] | None = None,
    check_cancel: Callable[[], bool] | None = None,
) -> dict:
    """按随机采样的时间点用 FFmpeg 抽帧写盘，并生成 review.json 索引。

    优先使用 GPU 硬件解码，失败时自动降级为 CPU 解码。
    """
    duration = video_duration_sec(video_path)
    if duration <= 0:
        raise RuntimeError("无法读取视频时长。")
    times = sample_times(
        duration=duration,
        count=count,
        start_sec=start_sec,
        end_sec=end_sec,
        min_gap_sec=min_gap_sec,
        seed=seed,
    )

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 将 quality (50-100) 转换为 FFmpeg 的 qscale 参数 (2-31，2质量最好，31最差)
    q_scale = max(2, min(31, int(2 + (100 - jpeg_quality) * 29 / 50)))

    samples = []
    total = len(times)
    start_time = time.time()

    gpu_success_count = [0]
    gpu_lock = threading.Lock()

    def extract_single(idx, ts):
        if check_cancel and check_cancel():
            return None

        name = f"frame_{idx:05d}_{ts:09.3f}s{IMAGE_EXT}"
        image_path = frames_dir / name

        # 1. 优先使用 GPU 显卡硬件加速解码 (CUDA/NVDEC)
        cmd_gpu = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", str(q_scale),
            str(image_path)
        ]

        ok = False
        try:
            subprocess.run(cmd_gpu, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            ok = True
            with gpu_lock:
                gpu_success_count[0] += 1
        except Exception:
            # 2. 如果 GPU 解码失败，自动降级为 CPU 解码提取
            cmd_cpu = [
                "ffmpeg", "-y",
                "-ss", f"{ts:.3f}",
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", str(q_scale),
                str(image_path)
            ]
            try:
                subprocess.run(cmd_cpu, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                ok = True
            except Exception:
                pass

        if ok:
            return {
                "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                "time_sec": round(ts, 3),
                "status": "unsure",
                "note": "",
                "tags": [],
            }
        return None

    completed_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(extract_single, idx, ts): (idx, ts) for idx, ts in enumerate(times, start=1)}
        for future in as_completed(futures):
            if check_cancel and check_cancel():
                for f in futures:
                    f.cancel()
                break

            res = future.result()
            if res:
                samples.append(res)

            completed_count += 1
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(completed_count, total, elapsed)

    samples.sort(key=lambda x: x["time_sec"])
    print(f"\n[硬件加速统计] GPU 加速成功率: {gpu_success_count[0]}/{total} 帧")

    payload = {
        "video_path": to_relative_path(video_path),
        "output_dir": to_relative_path(output_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": round(duration, 3),
        "sample_count": len(samples),
        "samples": samples,
    }

    # 写入 JSON 描述索引
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "review.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


class ExtractorThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, float)  # current, total, elapsed_seconds
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, extractor_fn, *args, **kwargs) -> None:
        super().__init__()
        self.extractor_fn = extractor_fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            self.kwargs["progress_callback"] = self.on_progress
            self.kwargs["check_cancel"] = self.isInterruptionRequested
            payload = self.extractor_fn(*self.args, **self.kwargs)
            if self.isInterruptionRequested():
                self.error.emit("已取消抽帧")
            else:
                self.finished.emit(payload)
        except Exception as exc:
            self.error.emit(str(exc))

    def on_progress(self, current: int, total: int, elapsed: float) -> None:
        self.progress.emit(current, total, elapsed)


class ReviewWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("视频随机抽帧配置工具 (Random Frame Extractor)")
        self.resize(720, 360)
        self.output_dir = PROJECT_ROOT / "data" / "vision" / "random_review"

        self.video_edit = QtWidgets.QLineEdit()
        self.out_edit = QtWidgets.QLineEdit(str(self.output_dir))
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(1, 20000)
        self.count_spin.setValue(300)
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setRange(0, 999999)
        self.start_spin.setDecimals(1)
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setRange(0, 999999)
        self.end_spin.setDecimals(1)
        self.end_spin.setSpecialValueText("视频结尾")
        self.gap_spin = QtWidgets.QDoubleSpinBox()
        self.gap_spin.setRange(0, 3600)
        self.gap_spin.setDecimals(1)
        self.gap_spin.setValue(1.0)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(-1, 2147483647)
        self.seed_spin.setValue(6657)
        self.quality_spin = QtWidgets.QSpinBox()
        self.quality_spin.setRange(50, 100)
        self.quality_spin.setValue(92)

        # 进度展示组件
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setVisible(False)
        self.cancel_btn = QtWidgets.QPushButton("取消抽帧")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_extraction)

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        form = QtWidgets.QGridLayout()
        root.addLayout(form)
        form.addWidget(QtWidgets.QLabel("视频文件"), 0, 0)
        form.addWidget(self.video_edit, 0, 1)
        self.browse_video_btn = QtWidgets.QPushButton("浏览")
        form.addWidget(self.browse_video_btn, 0, 2)
        self.browse_video_btn.clicked.connect(self.browse_video)

        form.addWidget(QtWidgets.QLabel("输出目录"), 1, 0)
        form.addWidget(self.out_edit, 1, 1)
        self.browse_out_btn = QtWidgets.QPushButton("浏览")
        form.addWidget(self.browse_out_btn, 1, 2)
        self.browse_out_btn.clicked.connect(self.browse_output)

        # 配置参数区
        settings_grid = QtWidgets.QGridLayout()
        root.addLayout(settings_grid)
        params = [
            ("抽取数量", self.count_spin, 0, 0),
            ("起始时间", self.start_spin, 0, 2),
            ("终止时间", self.end_spin, 0, 4),
            ("最小间隔(秒)", self.gap_spin, 1, 0),
            ("随机种子", self.seed_spin, 1, 2),
            ("JPEG画质", self.quality_spin, 1, 4),
        ]
        for label, widget, row, col in params:
            settings_grid.addWidget(QtWidgets.QLabel(label), row, col)
            settings_grid.addWidget(widget, row, col + 1)

        root.addStretch(1)

        # 进度展示条
        progress_layout = QtWidgets.QHBoxLayout()
        progress_layout.addWidget(self.progress_bar, 3)
        progress_layout.addWidget(self.progress_label, 1)
        progress_layout.addWidget(self.cancel_btn, 0)
        root.addLayout(progress_layout)

        # 操作按钮区
        buttons = QtWidgets.QHBoxLayout()
        root.addLayout(buttons)
        self.extract_btn = QtWidgets.QPushButton("开始随机抽帧")
        self.extract_btn.setStyleSheet("font-weight: bold; min-height: 35px;")
        buttons.addWidget(self.extract_btn)
        self.extract_btn.clicked.connect(self.extract_frames)

    def browse_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", str(PROJECT_ROOT), "Videos (*.mp4 *.mkv *.mov *.avi);;All files (*.*)")
        if path:
            self.video_edit.setText(path)

    def browse_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录", self.out_edit.text() or str(PROJECT_ROOT))
        if path:
            self.out_edit.setText(path)

    def extract_frames(self) -> None:
        try:
            video = resolve_path(self.video_edit.text().strip())
            output = resolve_path(self.out_edit.text().strip())
            seed_value = self.seed_spin.value()
            seed = None if seed_value < 0 else seed_value

            self.progress_bar.setMaximum(self.count_spin.value())
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.progress_label.setText("正在初始化...")
            self.progress_label.setVisible(True)
            self.cancel_btn.setEnabled(True)
            self.cancel_btn.setVisible(True)

            self.set_controls_enabled(False)

            self.extractor_thread = ExtractorThread(
                extract_random_frames,
                video,
                output,
                count=self.count_spin.value(),
                start_sec=self.start_spin.value(),
                end_sec=self.end_spin.value(),
                min_gap_sec=self.gap_spin.value(),
                seed=seed,
                jpeg_quality=self.quality_spin.value(),
            )
            self.extractor_thread.progress.connect(self.on_extraction_progress)
            self.extractor_thread.finished.connect(self.on_extraction_finished)
            self.extractor_thread.error.connect(self.on_extraction_error)
            self.extractor_thread.start()

        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "提取失败", str(exc))
            self.reset_progress_ui()

    def on_extraction_progress(self, current: int, total: int, elapsed: float) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        if current > 0:
            speed = current / elapsed
            remaining = total - current
            eta_sec = remaining / speed

            if eta_sec < 60:
                eta_str = f"{eta_sec:.1f} 秒"
            else:
                m, s = divmod(int(eta_sec), 60)
                eta_str = f"{m} 分 {s} 秒"

            self.progress_label.setText(
                f"进度: {current}/{total} | 速度: {speed:.1f} 帧/秒 | 剩余时间: {eta_str}"
            )

    def on_extraction_finished(self, payload: dict) -> None:
        self.reset_progress_ui()
        QtWidgets.QMessageBox.information(
            self, "提取完成", 
            f"成功抽取了 {len(payload.get('samples', []))} 帧！\n图片已写入 'frames/' 子目录。\n索引已保存至: {Path(payload['output_dir'])/'review.json'}"
        )

    def on_extraction_error(self, error_msg: str) -> None:
        if error_msg == "已取消抽帧":
            QtWidgets.QMessageBox.information(self, "任务取消", "抽帧已成功取消。")
        else:
            QtWidgets.QMessageBox.critical(self, "提取失败", error_msg)
        self.reset_progress_ui()

    def cancel_extraction(self) -> None:
        if hasattr(self, "extractor_thread") and self.extractor_thread.isRunning():
            self.extractor_thread.requestInterruption()
            self.progress_label.setText("正在取消...")
            self.cancel_btn.setEnabled(False)

    def reset_progress_ui(self) -> None:
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.set_controls_enabled(True)

    def set_controls_enabled(self, enabled: bool) -> None:
        self.video_edit.setEnabled(enabled)
        self.out_edit.setEnabled(enabled)
        self.browse_video_btn.setEnabled(enabled)
        self.browse_out_btn.setEnabled(enabled)
        self.count_spin.setEnabled(enabled)
        self.start_spin.setEnabled(enabled)
        self.end_spin.setEnabled(enabled)
        self.gap_spin.setEnabled(enabled)
        self.seed_spin.setEnabled(enabled)
        self.quality_spin.setEnabled(enabled)
        self.extract_btn.setEnabled(enabled)


def main() -> int:
    """命令行入口：传入 --video 走静默批处理，否则启动 PyQt5 操作界面。"""
    parser = argparse.ArgumentParser(description="视频随机抽帧工具")
    parser.add_argument("--video", default="", help="命令行静默运行：视频路径")
    parser.add_argument("--output-dir", default="", help="命令行静默运行：输出文件夹")
    parser.add_argument("--count", type=int, default=300, help="命令行静默运行：抽帧数量")
    parser.add_argument("--start-sec", type=float, default=0.0, help="命令行静默运行：起始时间")
    parser.add_argument("--end-sec", type=float, default=0.0, help="命令行静默运行：终止时间")
    parser.add_argument("--min-gap", type=float, default=1.0, help="命令行静默运行：相邻间隔")
    parser.add_argument("--seed", type=int, default=6657, help="命令行静默运行：随机种子")
    parser.add_argument("--quality", type=int, default=92, help="命令行静默运行：画质")

    args = parser.parse_args()

    # 如果传入了视频路径，则直接命令行模式运行，不显示 GUI
    if args.video:
        video_path = resolve_path(args.video)
        output_dir = resolve_path(args.output_dir or "data/vision/random_review")
        seed = None if args.seed < 0 else args.seed
        if not video_path.exists():
            print(f"Error: 视频文件不存在: {video_path}")
            return 1
        try:
            extract_random_frames(
                video_path,
                output_dir,
                count=args.count,
                start_sec=args.start_sec,
                end_sec=args.end_sec,
                min_gap_sec=args.min_gap,
                seed=seed,
                jpeg_quality=args.quality,
            )
            print("随机抽帧命令行静默任务顺利完成！")
            return 0
        except Exception as exc:
            print(f"抽帧失败: {exc}")
            return 1

    # 如果未传参数，则启动 PyQt5 操作界面
    app = QtWidgets.QApplication(sys.argv)
    window = ReviewWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
