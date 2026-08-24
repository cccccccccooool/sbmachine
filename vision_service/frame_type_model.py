"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：帧类型分类器工具。

启动方式：被 tools/slicing/run_frame_type_slicer.py 和 tools/slicing/train_frame_type_classifier.py 导入。
输入数据流：图像文件（磁盘路径）或视频帧（numpy 数组）。
输出数据流：返回 torch tensor（供模型推理）或预测结果 dict（含 label/confidence/probs）。
用法用途：提供帧类型分类器的模型构建、图像预处理、checkpoint 加载和单帧预测功能。
默认使用 MobileNetV3-Small：训练时迁移学习效果更强，推理仍足够轻量，适合本地低成本运行。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FRAME_LABELS = ["game", "break"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def resolve_device(value: str):
    """解析设备字符串为 torch.device。被 train_frame_type_classifier.py 和 run_frame_type_slicer.py 调用。

    Parameters
    ----------
    value : str
        设备字符串，"auto" 或 None 时自动选择 cuda/cpu。

    Returns
    -------
    torch.device
    """
    import torch

    if value and value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_frame_type_model(num_classes: int, pretrained: bool = False):
    """构建 MobileNetV3-Small 分类模型。被 train_frame_type_classifier.py 和 load_checkpoint() 调用。

    Parameters
    ----------
    num_classes : int
        分类数（如 2: game/break）。
    pretrained : bool
        是否加载 ImageNet 预训练权重。

    Returns
    -------
    nn.Module
        修改了分类头的 MobileNetV3-Small 模型。
    """
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def read_image_rgb(path: str | Path) -> np.ndarray:
    """读取图片为 RGB numpy 数组。被 image_to_tensor() 调用。

    Parameters
    ----------
    path : str | Path
        图片路径。

    Returns
    -------
    np.ndarray
        RGB 格式的图像数组。
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_image_array(image: np.ndarray) -> np.ndarray:
    """对图像数组做 ImageNet 归一化并转置为 CHW。被 frame_to_tensor() 和 image_to_tensor() 调用。

    Parameters
    ----------
    image : np.ndarray
        RGB 图像数组（HWC）。

    Returns
    -------
    np.ndarray
        归一化后的 CHW 数组。
    """
    array = image.astype("float32") / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))


def frame_to_tensor(frame_bgr: np.ndarray, img_size: int):
    """将 BGR 视频帧预处理为模型输入 tensor。被 predict_frame() 和 run_frame_type_slicer.py 调用。

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR 格式的视频帧。
    img_size : int
        目标尺寸（正方形）。

    Returns
    -------
    torch.Tensor
        形状为 (1, 3, img_size, img_size) 的 tensor。
    """
    import cv2
    import torch

    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    array = normalize_image_array(image)
    return torch.from_numpy(array).unsqueeze(0)


def image_to_tensor(path: str | Path, img_size: int):
    """将图片文件预处理为模型输入 tensor。被 train_frame_type_classifier.py 调用。

    Parameters
    ----------
    path : str | Path
        图片路径。
    img_size : int
        目标尺寸。

    Returns
    -------
    torch.Tensor
        形状为 (3, img_size, img_size) 的 tensor。
    """
    import cv2
    import torch

    image = read_image_rgb(path)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    array = normalize_image_array(image)
    return torch.from_numpy(array)


def load_checkpoint(path: str | Path, device: Any):
    """加载训练好的 checkpoint。被 run_frame_type_slicer.py 调用。

    Parameters
    ----------
    path : str | Path
        checkpoint 文件路径（.pt）。
    device : Any
        torch.device。

    Returns
    -------
    tuple
        (model, labels, img_size, checkpoint) 四元组。
    """
    import torch

    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    except Exception:
        # checkpoint 含非 tensor 数据（labels/img_size 等）时回退，允许受信任的本地模型
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    labels = list(checkpoint.get("labels") or DEFAULT_FRAME_LABELS)
    img_size = int(checkpoint.get("img_size", 224))
    model = build_frame_type_model(len(labels), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, labels, img_size, checkpoint


def predict_frame(model, labels: list[str], frame_bgr: np.ndarray, img_size: int, device: Any) -> dict:
    """对单帧进行分类预测。被 run_frame_type_slicer.py 调用。

    Parameters
    ----------
    model : nn.Module
        已加载权重的分类模型。
    labels : list[str]
        类别标签列表。
    frame_bgr : np.ndarray
        BGR 格式的视频帧。
    img_size : int
        模型输入尺寸。
    device : Any
        torch.device。

    Returns
    -------
    dict
        含 label（预测标签）、confidence（置信度）、probs（各类概率）。
    """
    import torch

    tensor = frame_to_tensor(frame_bgr, img_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    index = int(np.argmax(probs))
    return {
        "label": labels[index],
        "confidence": round(float(probs[index]), 6),
        "probs": {label: round(float(probs[i]), 6) for i, label in enumerate(labels)},
    }


def predict_frames_batch(
    model,
    labels: list[str],
    frames_bgr: list[np.ndarray],
    img_size: int,
    device: Any,
    batch_size: int = 32,
) -> list[dict]:
    """对多帧做分批推理，返回与 predict_frame 结构一致的结果列表。

    GPU 场景下单进程批推理吞吐远高于逐帧调用；CPU 场景批量亦可减少 Python
    层开销。输出顺序与输入 frames_bgr 一一对应。
    """
    import torch

    chunk_size = max(1, int(batch_size))
    outputs: list[dict] = []
    for start in range(0, len(frames_bgr), chunk_size):
        chunk = frames_bgr[start : start + chunk_size]
        tensors = torch.cat(
            [frame_to_tensor(frame, img_size) for frame in chunk],
            dim=0,
        ).to(device)
        with torch.no_grad():
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        for row in probs:
            index = int(np.argmax(row))
            outputs.append(
                {
                    "label": labels[index],
                    "confidence": round(float(row[index]), 6),
                    "probs": {label: round(float(row[i]), 6) for i, label in enumerate(labels)},
                }
            )
    return outputs
