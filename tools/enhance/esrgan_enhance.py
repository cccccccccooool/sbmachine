"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：使用 Real-ESRGAN 对图片进行超分辨率增强。

启动方式：python tools/enhance/esrgan_enhance.py --input ... --output ...
输入数据流：低分辨率图片文件。
输出数据流：高分辨率图片文件。
用法用途：对模糊的 CS2 录像帧进行 4K 超分辨率，提升后续 YOLO/VLM 识别精度。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.enhance.video_enhance import MODEL_INFO, load_upsampler, resolve_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def enhance_directory(
    directory: Path,
    *,
    model: str,
    device: str,
    half: bool,
    tile: int,
    overwrite: bool,
    dry_run: bool,
) -> int:
    paths = [path for path in sorted(directory.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES]
    if dry_run:
        for path in paths:
            print(path)
        return len(paths)

    import cv2

    upsampler = load_upsampler(model, device, half, tile)
    count = 0
    for path in paths:
        output_path = path if overwrite else path.with_name(f"{path.stem}_esrgan{path.suffix}")
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"skip unreadable image: {path}")
            continue
        sr, _ = upsampler.enhance(frame, outscale=4)
        if not cv2.imwrite(str(output_path), sr):
            raise RuntimeError(f"failed to write image: {output_path}")
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upscale image directories with Real-ESRGAN.")
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--model", default="RealESRGAN_x4plus_anime_6B", choices=list(MODEL_INFO))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Write enhanced pixels back to the original filenames.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for value in args.dirs:
        directory = resolve_path(value)
        if not directory.exists():
            print(f"skip missing directory: {directory}")
            continue
        count = enhance_directory(
            directory,
            model=args.model,
            device=args.device,
            half=args.half,
            tile=args.tile,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(f"{directory}: {count} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
