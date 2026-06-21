"""
本文件功能：轻量级 VLM 推理服务（Qwen2.5-VL-3B-Instruct），提供 OpenAI 兼容的 /v1/chat/completions 和 batch 端点。

启动方式：python tools/simple_vlm_server.py（监听 0.0.0.0:23333）
输入数据流：HTTP POST 请求（含 base64 图片和 prompt 文本）。
输出数据流：返回 OpenAI 兼容格式的 JSON 响应（含画面描述文本）。
用法用途：作为 Phase 2 VLM 推理后端，自动加载基础模型和可选的 LoRA 微调权重，支持单帧和批量推理。
"""
import os
os.environ["HF_HOME"] = "/workspace/.hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

import base64
from io import BytesIO
import torch
from fastapi import FastAPI, Request
from PIL import Image
import uvicorn
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

app = FastAPI()

print("正在加载 Qwen2.5-VL-3B-Instruct 基础模型到 GPU...")
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 检测并加载微调的 LoRA 权重
lora_dir = os.path.join("models", "vlm", "qwen2_5_vl_lora")
if os.path.exists(lora_dir) and os.path.isdir(lora_dir):
    try:
        from peft import PeftModel
        print(f"检测到微调 LoRA 权重，正在加载: {lora_dir}")
        model = PeftModel.from_pretrained(model, lora_dir)
        print("LoRA 权重加载成功！")
    except ImportError:
        print("[警告] 未安装 peft 库，跳过加载 LoRA。请运行 pip install peft 安装。")
    except Exception as e:
        print(f"[错误] 加载 LoRA 权重失败: {e}")

processor = AutoProcessor.from_pretrained(base_model_id)
print("模型加载成功，服务准备就绪！")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages", [])
    
    # 1. 提取 Prompt 和 Base64 图片
    if not messages or not isinstance(messages[0].get("content"), list):
        return {"choices": [{"message": {"content": "错误：请求缺少有效 messages"}}]}
    prompt = ""
    pil_image = None
    for content in messages[0]["content"]:
        if content.get("type") == "text":
            prompt = content.get("text", "")
        elif content.get("type") == "image_url":
            try:
                url_str = content["image_url"]["url"]
                base64_data = url_str.split(",")[1]
                image_bytes = base64.b64decode(base64_data)
                pil_image = Image.open(BytesIO(image_bytes))
            except Exception:
                return {"choices": [{"message": {"content": "错误：图片 base64 解码失败"}}]}
            
    if not pil_image:
        return {"choices": [{"message": {"content": "错误：未在请求中检测到有效图片"}}]}

    # 2. 转换为 Qwen2.5-VL 的输入格式
    qwen_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(qwen_messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 3. 原生 PyTorch 推理生成（避开 Triton 编译）
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=int(payload.get("max_tokens", 128)))
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # 4. 返回符合 OpenAI 规范的响应格式
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": output_text
                }
            }
        ]
    }

@app.post("/v1/chat/completions/batch")
async def chat_completions_batch(request: Request):
    payload = await request.json()
    batch = payload.get("batch", [])
    max_new_tokens = int(payload.get("max_new_tokens", 128))

    if not batch:
        return {"results": []}

    qwen_messages_list = []
    decode_errors = []
    for idx, item in enumerate(batch):
        prompt = item.get("prompt", "")
        image_b64 = item.get("image_b64", "")
        try:
            image_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            decode_errors.append(idx)
            qwen_messages_list.append(None)
            continue
        qwen_messages_list.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        })
    valid_indices = [i for i, m in enumerate(qwen_messages_list) if m is not None]
    valid_messages = [qwen_messages_list[i] for i in valid_indices]
    if not valid_messages:
        return {"results": ["错误：图片解码失败"] * len(batch)}

    # 每条消息单独 apply_chat_template，再合并为 batch texts（只处理有效帧）
    texts = [
        processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
        for msg in valid_messages
    ]

    # 逐条提取图像输入，合并成一个列表供 processor 使用
    all_image_inputs = []
    for msg in valid_messages:
        img_inputs, _ = process_vision_info([msg])
        if img_inputs:
            all_image_inputs.extend(img_inputs)

    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # 把有效帧结果与错误占位符按原始顺序合并
    valid_results = [t.strip() for t in output_texts]
    final_results = [""] * len(batch)
    for out_idx, orig_idx in enumerate(valid_indices):
        final_results[orig_idx] = valid_results[out_idx]
    for orig_idx in decode_errors:
        final_results[orig_idx] = "错误：图片解码失败"
    return {"results": final_results}


if __name__ == "__main__":
    # 启动本地 23333 端口服务
    uvicorn.run(app, host="0.0.0.0", port=23333)
