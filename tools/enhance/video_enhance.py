"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：使用 Real-ESRGAN 将视频超分辨率到 4K。

启动方式：python tools/enhance/video_enhance.py -i <video_path> [选项]
输入数据流：输入视频文件 (mp4/mkv/avi 等)。
输出数据流：超分辨率后的 4K H.265 视频文件 (3840×2160, 音频拷贝)。
用法用途：提取视频帧并用 Real-ESRGAN 在 GPU 上批量超分，再重组为 4K 视频；推荐用于恢复模糊的 CS2 录像。

Upscale video to 4K using Real-ESRGAN.

This tool extracts video frames, upscales them in batches on GPU using Real-ESRGAN,
and re-assembles them back into an H.265/HEVC video preserving the original audio.
Highly recommended for restoring blurry CS2 match recordings before slicing or labeling.

Pipeline:
  1. ffprobe  → Extract video FPS, resolution, and audio stream info.
  2. ffmpeg   → Extract all frames as PNG to a temporary directory.
  3. Real-ESRGAN → Upscale frames in batches using CUDA tensor cores.
  4. ffmpeg   → Reassemble SR frames, apply Lanczos scale to 4K, copy original audio.

Model weights are downloaded automatically to models/realesrgan/ on first run.

GPU Acceleration & VRAM Configuration Guide:
  - L40 / L40S (48 GB VRAM):
    Use fp16, no tiling (tile=0), and batch-size=4. Processes 4 full 1080p frames in parallel.
    Command: python tools/enhance/video_enhance.py -i data/raw/match.mp4 --half --batch-size 4 --tile 0
  
  - H20 (96 GB VRAM, Capped Compute, High Bandwidth):
    Use the built-in --h20 preset. Enables fp16, batch-size=8, and tile=0.
    Maximizes memory bandwidth saturation to achieve optimal throughput.
    Command: python tools/enhance/video_enhance.py -i data/raw/match.mp4 --h20

  - RTX 3090 / 4090 (24 GB VRAM):
    Use fp16, no tiling (tile=0), and batch-size=2 (or batch-size=1 if out-of-memory).
    Command: python tools/enhance/video_enhance.py -i data/raw/match.mp4 --half --batch-size 2 --tile 0
    If OOM persists, use tiled inference:
    Command: python tools/enhance/video_enhance.py -i data/raw/match.mp4 --half --tile 256

Usage Examples:
    # Default: RealESRGAN_x4plus, single-frame mode (safe for any GPU)
    python tools/enhance/video_enhance.py -i data/raw/match.mp4

    # Anime / Cartoon / UI-focused model (sharper edges on hud elements)
    python tools/enhance/video_enhance.py -i data/raw/match.mp4 --model RealESRGAN_x4plus_anime_6B

    # Dry-run: Probe video info and print parameters without upscaling
    python tools/enhance/video_enhance.py -i data/raw/match.mp4 --dry-run

Output:
    {input_dir}/{input_stem}_4k.mp4  (H.265, 3840×2160, audio copied)
"""
from __future__ import annotations

# Fix AttributeError: module 'collections' has no attribute 'MutableMapping' for Python 3.10+
import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping

import argparse
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "realesrgan"

# ── model registry ────────────────────────────────────────────────────────────
MODEL_INFO: dict[str, dict] = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
        "netscale": 4,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
        "netscale": 4,
    },
    "RealESRNet_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "scale": 4,
        "num_block": 23,
        "netscale": 4,
    },
}

TARGET_4K = (3840, 2160)


# ── path helpers ─────────────────────────────────────────────────────────────

def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ── ffprobe ───────────────────────────────────────────────────────────────────

def ffprobe_info(video: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(video),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    data = json.loads(out)
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), {}
    )
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    # fps may be a fraction string like "60000/1001"
    fps_raw = video_stream.get("r_frame_rate", "25/1")
    num, den = [int(x) for x in fps_raw.split("/")]
    fps = num / den
    return {
        "fps": fps,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "duration": float(data.get("format", {}).get("duration", 0)),
        "has_audio": has_audio,
    }


# ── model download ────────────────────────────────────────────────────────────

def download_weights(model_name: str) -> Path:
    info = MODEL_INFO[model_name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / f"{model_name}.pth"
    if dest.exists():
        return dest
    print(f"Downloading {model_name} weights → {dest} …")
    import requests
    resp = requests.get(info["url"], stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct:3d}%", end="", flush=True)
    print()
    return dest


# ── model loader ──────────────────────────────────────────────────────────────

def load_upsampler(model_name: str, device: str, half: bool, tile: int):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    info = MODEL_INFO[model_name]
    weights = download_weights(model_name)

    net = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=info["num_block"],
        num_grow_ch=32, scale=info["netscale"],
    )
    upsampler = RealESRGANer(
        scale=info["scale"],
        model_path=str(weights),
        model=net,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device,
    )
    return upsampler


# ── frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video: Path, frames_dir: Path) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", str(video),
        "-vsync", "0",          # preserve every frame, no duplicates
        "-f", "image2",
        str(frames_dir / "%08d.png"),
        "-y",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return len(list(frames_dir.glob("*.png")))


# ── batch SR core ─────────────────────────────────────────────────────────────

def _batch_sr_no_tile(
    upsampler,
    frames_bgr: list[np.ndarray],
    half: bool,
) -> list[np.ndarray]:
    """
    Batch-process frames by calling net_g directly (no tiling).
    Requires all frames to have identical spatial dimensions.
    Suitable for H20 / large-VRAM setups where an entire frame fits in VRAM.
    """
    import torch

    tensors = []
    for frame in frames_bgr:
        rgb = frame[:, :, ::-1].copy()
        t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors).to(upsampler.device)
    if half:
        batch = batch.half()

    with torch.no_grad():
        # net_g is the RRDB network; processes [B, C, H, W] natively
        outputs = upsampler.model.net_g(batch).float().clamp(0, 1)

    results: list[np.ndarray] = []
    for i in range(outputs.shape[0]):
        out = outputs[i].cpu().numpy().transpose(1, 2, 0)
        out = (out * 255.0).round().astype(np.uint8)
        results.append(out[:, :, ::-1].copy())   # RGB → BGR
    return results


def _frames_same_shape(frames: list[np.ndarray]) -> bool:
    if not frames:
        return True
    h0, w0 = frames[0].shape[:2]
    return all(f.shape[:2] == (h0, w0) for f in frames)


# ── frame processing loop ─────────────────────────────────────────────────────

def _load_frame_batch(paths: list[Path]) -> list[np.ndarray]:
    return [cv2.imread(str(p)) for p in paths]


def process_frames(
    upsampler,
    frames_dir: Path,
    sr_dir: Path,
    batch_size: int,
    half: bool,
    use_batch_mode: bool,
) -> int:
    sr_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("*.png"))
    total = len(frame_paths)

    try:
        from tqdm import tqdm
        bar = tqdm(total=total, desc="SR frames", unit="frame")
    except ImportError:
        bar = None

    bs = max(1, batch_size)
    batches = [frame_paths[i : i + bs] for i in range(0, total, bs)]
    done = 0

    with ThreadPoolExecutor(max_workers=2) as loader:
        # Prefetch next batch while GPU processes current
        prefetch: Future | None = None
        current_frames: list[np.ndarray] = []

        for bi, batch_paths in enumerate(batches):
            next_paths = batches[bi + 1] if bi + 1 < len(batches) else None
            if next_paths:
                prefetch = loader.submit(_load_frame_batch, next_paths)

            # First batch loads synchronously; subsequent from prefetch
            if bi == 0:
                current_frames = _load_frame_batch(batch_paths)
            # else: current_frames already set from previous iteration's prefetch

            if use_batch_mode and _frames_same_shape(current_frames):
                sr_frames = _batch_sr_no_tile(upsampler, current_frames, half)
            else:
                # Fall back to single-frame with tiling (any VRAM)
                sr_frames = []
                for frame in current_frames:
                    out, _ = upsampler.enhance(frame, outscale=4)
                    sr_frames.append(out)

            for path, sr in zip(batch_paths, sr_frames):
                out_path = sr_dir / path.name
                cv2.imwrite(str(out_path), sr)

            done += len(batch_paths)
            if bar:
                bar.update(len(batch_paths))

            # Resolve prefetch for next iteration
            current_frames = prefetch.result() if prefetch else []

    if bar:
        bar.close()
    return done


# ── video assembly ────────────────────────────────────────────────────────────

def assemble_video(
    sr_dir: Path,
    original_video: Path,
    output: Path,
    fps: float,
    crf: int,
    has_audio: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # vf: lanczos resize to exact 4K, preserve SAR
    vf = f"scale={TARGET_4K[0]}:{TARGET_4K[1]}:flags=lanczos,setsar=1"
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", str(sr_dir / "%08d.png"),
    ]
    if has_audio:
        cmd += ["-i", str(original_video)]
    cmd += [
        "-map", "0:v",
    ]
    if has_audio:
        cmd += ["-map", "1:a?"]
    cmd += [
        "-c:v", "libx265",
        "-crf", str(crf),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-vf", vf,
        "-tag:v", "hvc1",       # broad player compatibility
    ]
    if has_audio:
        cmd += ["-c:a", "copy"]
    cmd += ["-y", str(output)]

    print(f"Assembling → {output}")
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    # H20 preset
    if args.h20:
        if not args.half:
            args.half = True
        if args.batch_size == 1:
            args.batch_size = 8
        if args.tile == 128:        # user didn't override tile
            args.tile = 0
        print("H20 preset: half=True  batch=8  tile=0 (full-frame, no tiling)")

    video = resolve_path(args.input)
    if not video.exists():
        print(f"Input not found: {video}")
        return 1

    if args.output:
        output = resolve_path(args.output)
    else:
        output = video.parent / f"{video.stem}_4k.mp4"

    info = ffprobe_info(video)
    print(
        f"Input : {video.name}  {info['width']}×{info['height']}  "
        f"{info['fps']:.3f} fps  {info['duration']:.1f}s  "
        f"audio={'yes' if info['has_audio'] else 'no'}"
    )
    print(f"Output: {output}")
    print(
        f"Model : {args.model}  half={args.half}  "
        f"batch={args.batch_size}  tile={args.tile}"
    )

    if args.dry_run:
        print("Dry run — no processing performed.")
        return 0

    if args.model not in MODEL_INFO:
        print(f"Unknown model '{args.model}'. Available: {list(MODEL_INFO)}")
        return 1

    use_batch_mode = args.batch_size > 1 and args.tile == 0
    if args.batch_size > 1 and args.tile > 0:
        print(
            "Warning: batch_size > 1 requires tile=0 for direct net_g batching; "
            "falling back to single-frame + tiling."
        )
        use_batch_mode = False

    upsampler = load_upsampler(args.model, args.device, args.half, args.tile)

    tmp_root = Path(tempfile.mkdtemp(prefix="video_enhance_"))
    frames_dir = tmp_root / "frames"
    sr_dir = tmp_root / "sr_frames"

    try:
        print("Extracting frames…")
        n_frames = extract_frames(video, frames_dir)
        print(f"  {n_frames} frames extracted.")

        print(f"Running Real-ESRGAN ({'batch' if use_batch_mode else 'single'} mode)…")
        process_frames(upsampler, frames_dir, sr_dir, args.batch_size, args.half, use_batch_mode)

        assemble_video(sr_dir, video, output, info["fps"], args.crf, info["has_audio"])
        print(f"Done → {output}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale video to 4K with Real-ESRGAN + ffmpeg (H.265 output)."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input video path (relative to project root or absolute).")
    parser.add_argument("-o", "--output", default="",
                        help="Output path. Default: {input_dir}/{stem}_4k.mp4")
    parser.add_argument("--model", default="RealESRGAN_x4plus",
                        choices=list(MODEL_INFO),
                        help="SR model. Use anime_6B for animated content. Default: x4plus.")

    # ── device & precision ────────────────────────────────────────────────────
    parser.add_argument("--device", default="cuda",
                        help="Torch device. Default: cuda.")
    parser.add_argument("--half", action="store_true",
                        help="fp16 inference. Halves VRAM, ~1.5× faster.")

    # ── batching ──────────────────────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Frames per GPU call. Requires --tile 0. "
                             "H20: use --h20 or set 4-8 for 1080p input.")
    parser.add_argument("--tile", type=int, default=128,
                        help="Tile size for tiled inference (reduces peak VRAM). "
                             "0 = no tiling (required for batch mode). "
                             "H20 can use 0. Default: 128.")

    # ── H20 preset ────────────────────────────────────────────────────────────
    parser.add_argument("--h20", action="store_true",
                        help="H20 preset: half=True, batch=8, tile=0.")

    # ── output encoding ───────────────────────────────────────────────────────
    parser.add_argument("--crf", type=int, default=18,
                        help="H.265 CRF quality (lower = better). Default: 18.")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show video info and exit without processing.")
    return parser.parse_args()


def main() -> int:
    try:
        return run(parse_args())
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        print(f"ffmpeg/ffprobe failed:\n{stderr}")
        return 1
    except Exception as exc:
        print(f"video_enhance failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
