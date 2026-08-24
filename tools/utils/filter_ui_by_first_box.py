#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：根据第一个 box 和第二个 box 坐标比较，将相同 UI 布局签名的图片归为同一赛事分组。

启动方式：python tools/utils/filter_ui_by_first_box.py
输入数据流：dataset_index.json 和各数据集的 annotations.jsonl。
输出数据流：按赛事分组的子目录。
用法用途：读取各数据集的 annotations.jsonl，同时比较第一个 box 和第二个 box 坐标，
    以此作为赛事 UI 布局的特征签名，将相同签名的图片归为同一赛事分组。

支持三种导出模式：
  --mode copy    (默认) 将图片和标注文件物理拷贝到子数据集目录，路径随之更新。
  --mode inplace 不移动/复制任何图片和标注文件；生成的 annotations.jsonl 中 image/label
                 路径保持指向原始位置，data.yaml 也指向原始数据集的 images 目录。
                 适用场景：筛选出的数据仍在原始目录下标注，本脚本只起"视图索引"作用。
  --mode group   将图片按双 box 布局分类并拷贝到 dataset_dir/1、dataset_dir/2 等数字目录下，
                 只复制图片、跳过 empty 分组，方便标注工具直接加载并原地修改。

使用方法 (Usage):
    # 默认拷贝模式（生成独立子数据集）
    python tools/utils/filter_ui_by_first_box.py

    # 原地索引模式（不移动文件，仅生成筛选视图）
    python tools/utils/filter_ui_by_first_box.py --mode inplace

    # 仅处理指定数据集 ID（支持多个）
    python tools/utils/filter_ui_by_first_box.py --dataset 22 blast

    # 指定输出根目录（仅对 copy 模式生效）
    python tools/utils/filter_ui_by_first_box.py --mode copy --out-dir data/vision/yolo_filtered

参数说明:
    --mode {copy,inplace,group}   导出模式 (默认: copy)
    --dataset ID [ID ...]         只处理指定 dataset_id 的数据集，不填则处理全部
    --out-dir PATH                copy 模式下的输出根目录（默认在各 dataset_dir/filtered_ui 下）
    --dry-run                     只打印分组摘要，不实际写入任何文件
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data" / "vision" / "yolo_background" / "dataset_index.json"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def box_key(annotations: list[dict]) -> str:
    """以第一个 box + 第二个 box 的坐标联合构造赛事 UI 特征签名 key。

    缺失对应 box 时用 "x" 代替；完全没有标注则归入 "empty" 分组。
    """
    def fmt(box) -> str:
        if isinstance(box, list) and len(box) == 4:
            return "_".join(map(str, box))
        return "x"

    if not annotations:
        return "empty"

    first_box = annotations[0].get("box") if len(annotations) >= 1 else None
    second_box = annotations[1].get("box") if len(annotations) >= 2 else None

    if first_box is None:
        return "empty"
    if second_box is None:
        return f"b1_{fmt(first_box)}"
    return f"b1_{fmt(first_box)}__b2_{fmt(second_box)}"


def to_posix_rel(path: Path) -> str:
    """把绝对路径转成相对项目根目录的 POSIX 字符串。"""
    return path.relative_to(PROJECT_ROOT).as_posix()


def write_filtered_annotations(dst_path: Path, records: list[dict]) -> None:
    """将筛选后的标注记录逐行写入 JSONL 文件。"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_data_yaml(src_yaml: Path, dst_yaml: Path, path_line: str) -> None:
    """
    从 src_yaml 复制 data.yaml 内容，仅替换 path: 一行。
    """
    if not src_yaml.exists():
        return
    lines = src_yaml.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("path:"):
            lines[index] = f"path: {path_line}"
            break
    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    dst_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_classes_json(src: Path, dst: Path) -> None:
    """若源 classes.json 存在则原样拷贝到目标位置。"""
    if src.exists():
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# 分组逻辑
# ---------------------------------------------------------------------------

def group_records(ann_path: Path) -> dict[str, list[dict]]:
    """读取 annotations.jsonl，按 (box1 + box2) 联合 key 分组。"""
    groups: dict[str, list[dict]] = {}
    image_to_record: dict[str, dict] = {}

    with open(ann_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                print(f"    [!] 解析 JSON 行失败: {exc}")
                continue

            img_rel = record.get("image")
            if img_rel:
                image_to_record[img_rel] = record

    for record in image_to_record.values():
        key = box_key(record.get("annotations", []))
        groups.setdefault(key, []).append(record)

    return groups


# ---------------------------------------------------------------------------
# 导出模式：copy
# ---------------------------------------------------------------------------

def export_copy(
    groups: dict[str, list[dict]],
    dataset_dir: Path,
    filtered_root: Path,
    dry_run: bool,
) -> None:
    """物理拷贝图片和标注文件，生成完全独立的子数据集。"""
    classes_src = dataset_dir / "classes.json"
    data_yaml_src = dataset_dir / "data.yaml"

    for key, records in groups.items():
        sub_dir = filtered_root / key
        print(f"      -> 分组 '{key}': {len(records)} 条记录")

        if dry_run:
            continue

        # 清理并重建目录结构
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        for split in ["train", "val"]:
            (sub_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (sub_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        copied_images = 0
        copied_labels = 0
        sub_records = []

        for record in records:
            img_rel = record.get("image")
            lbl_rel = record.get("label")
            split = record.get("split", "train")
            if not img_rel or not lbl_rel:
                continue

            img_src = PROJECT_ROOT / img_rel
            lbl_src = PROJECT_ROOT / lbl_rel
            img_dst = sub_dir / "images" / split / img_src.name
            lbl_dst = sub_dir / "labels" / split / lbl_src.name

            if img_src.exists():
                shutil.copy2(img_src, img_dst)
                copied_images += 1

            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dst)
                copied_labels += 1

            # 更新 record 中的路径为子数据集路径
            sub_record = record.copy()
            sub_record["image"] = to_posix_rel(img_dst)
            sub_record["label"] = to_posix_rel(lbl_dst)
            sub_records.append(sub_record)

        write_filtered_annotations(sub_dir / "annotations.jsonl", sub_records)
        write_classes_json(classes_src, sub_dir / "classes.json")
        write_data_yaml(data_yaml_src, sub_dir / "data.yaml", to_posix_rel(sub_dir))

        print(f"         [OK] 已拷贝 {copied_images} 张图片, {copied_labels} 个标注")


# ---------------------------------------------------------------------------
# 导出模式：inplace（原地索引）
# ---------------------------------------------------------------------------

def export_inplace(
    groups: dict[str, list[dict]],
    dataset_dir: Path,
    filtered_root: Path,
    dry_run: bool,
) -> None:
    """
    不移动任何文件。
    在 filtered_ui/<key>/ 下生成:
      - annotations.jsonl   : image/label 路径完全保持原始路径不变
      - data.yaml           : path 指向原始 dataset_dir（images 仍在原始位置）
      - classes.json        : 直接拷贝自父级
    """
    classes_src = dataset_dir / "classes.json"
    data_yaml_src = dataset_dir / "data.yaml"
    # inplace 模式：data.yaml 的 path 仍指向原始 dataset_dir
    original_dataset_rel = to_posix_rel(dataset_dir)

    for key, records in groups.items():
        sub_dir = filtered_root / key
        print(f"      -> 分组 '{key}': {len(records)} 条记录")

        if dry_run:
            continue

        sub_dir.mkdir(parents=True, exist_ok=True)

        # annotations.jsonl：image/label 路径原样写入，不更改
        write_filtered_annotations(sub_dir / "annotations.jsonl", records)

        # data.yaml：path 指向原始 dataset_dir（使用原始 images/ 目录）
        write_data_yaml(data_yaml_src, sub_dir / "data.yaml", original_dataset_rel)

        # classes.json
        write_classes_json(classes_src, sub_dir / "classes.json")

        print(f"         [OK] 已生成原地索引 (annotations={len(records)} 条, 文件路径未变)")


# ---------------------------------------------------------------------------
# 导出模式：group（纯图片数字分组）
# ---------------------------------------------------------------------------

def export_group(
    groups: dict[str, list[dict]],
    dataset_dir: Path,
    dry_run: bool,
) -> None:
    """
    将分组后的图片复制到数据集根目录下以数字命名的临时文件夹中 (如 22/1/, 22/2/)。
    只复制图片文件，不复制任何标注或其他文件。
    跳过 'empty' 分组。
    """
    group_idx = 1
    # 按照图片数量降序排列，保证最常用的分组在前面
    sorted_groups = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)

    for key, records in sorted_groups:
        if key == "empty":
            print(f"      -> 跳过 'empty' 分组 (共 {len(records)} 张图片)")
            continue

        group_dir = dataset_dir / str(group_idx)
        print(f"      -> 分组 {group_idx} (Key: {key}): {len(records)} 条记录 -> {group_dir.relative_to(PROJECT_ROOT)}")

        if dry_run:
            group_idx += 1
            continue

        if group_dir.exists():
            shutil.rmtree(group_dir)
        group_dir.mkdir(parents=True, exist_ok=True)

        copied_images = 0
        for record in records:
            img_rel = record.get("image")
            if not img_rel:
                continue
            img_src = PROJECT_ROOT / img_rel
            if img_src.exists():
                shutil.copy2(img_src, group_dir / img_src.name)
                copied_images += 1

        print(f"         [OK] 已复制 {copied_images} 张图片到 {group_dir.name}/")
        group_idx += 1


# ---------------------------------------------------------------------------
# 主处理逻辑
# ---------------------------------------------------------------------------

def process_dataset(
    dataset: dict,
    mode: str,
    out_dir: Path | None,
    dry_run: bool,
) -> None:
    """处理单个数据集：按 UI 签名分组后按指定模式导出。"""
    dataset_id = dataset.get("dataset_id")
    dataset_dir_rel = dataset.get("dataset_dir")
    if not dataset_dir_rel:
        print(f"[-] 数据集 {dataset_id} 缺少 dataset_dir 字段，跳过。")
        return

    dataset_dir = PROJECT_ROOT / dataset_dir_rel
    ann_path = dataset_dir / "annotations.jsonl"
    if not ann_path.exists():
        print(f"[-] 未找到标注文件: {ann_path}，跳过。")
        return

    print(f"\n[+] 数据集: {dataset_id}  模式: {mode}")
    print(f"    标注文件: {ann_path}")

    groups = group_records(ann_path)
    print(f"    发现 UI/赛事分组数量: {len(groups)}")

    if mode == "group":
        export_group(groups, dataset_dir, dry_run)
        return

    # 输出根目录
    if out_dir is not None and mode == "copy":
        filtered_root = out_dir / dataset_id
    else:
        suffix = "filtered_ui_inplace" if mode == "inplace" else "filtered_ui"
        filtered_root = dataset_dir / suffix

    print(f"    输出目录: {filtered_root}")

    if mode == "copy":
        export_copy(groups, dataset_dir, filtered_root, dry_run)
    else:
        export_inplace(groups, dataset_dir, filtered_root, dry_run)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> int:
    """解析命令行参数，加载索引并逐个数据集执行筛选导出。"""
    parser = argparse.ArgumentParser(
        description="按 (box1+box2) 联合 key 筛选赛事 UI，支持 copy / inplace / group 三种导出模式。"
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "inplace", "group"],
        default="copy",
        help=(
            "导出模式。copy: 物理拷贝文件生成独立子数据集；"
            "inplace: 不移动文件，只生成索引视图；"
            "group: 将图片按双 box 布局分类并拷贝到 dataset_dir/1, dataset_dir/2... 数字目录下，"
            "方便 PyQt5 工具直接加载并原地修改。"
        ),
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        metavar="ID",
        help="只处理指定 dataset_id（可传多个），不填则处理全部。",
    )
    parser.add_argument(
        "--out-dir",
        metavar="PATH",
        default=None,
        help="copy 模式下的自定义输出根目录（相对或绝对路径）。其他模式下忽略此参数。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印分组摘要，不实际写入任何文件。",
    )
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        print(f"[-] 未找到索引文件: {INDEX_PATH}")
        return 1

    try:
        index_data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[-] 加载索引文件失败: {exc}")
        return 1

    datasets = index_data.get("datasets", [])
    if not datasets:
        print("[-] 索引文件中没有定义任何数据集。")
        return 0

    # 白名单过滤
    if args.dataset:
        wanted = set(args.dataset)
        datasets = [item for item in datasets if item.get("dataset_id") in wanted]
        if not datasets:
            print(f"[-] 未找到指定的数据集 ID: {args.dataset}")
            return 1

    out_dir: Path | None = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir

    if args.dry_run:
        print("[DRY-RUN] 以下为分组预览，不会写入任何文件。\n")

    for dataset in datasets:
        try:
            process_dataset(dataset, mode=args.mode, out_dir=out_dir, dry_run=args.dry_run)
        except Exception as exc:
            print(f"[-] 处理数据集 {dataset.get('dataset_id')} 时出错: {exc}")

    print("\n[+] 全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
