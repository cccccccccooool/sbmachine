"""多模态 VLM 请求客户端。负责将帧图像编码为 base64，组装提示词并向后端的 VLM 服务发送推理请求，返回画面描述。"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import requests

from core.prompt_loader import load_prompt

GLOBAL_SYSTEM_HINT = load_prompt("vlm_system")


class VlmClient:
    def __init__(
        self,
        config: dict,
        *,
        default_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        default_hint: str = GLOBAL_SYSTEM_HINT,
    ) -> None:
        self.config = config or {}
        self.endpoint = (os.getenv("AI6657_VLM_ENDPOINT") or str(self.config.get("endpoint", ""))).strip()
        if not self.endpoint:
            raise ValueError("Phase 2 VLM requires an endpoint")
        self.model = str(self.config.get("model", default_model))
        self.timeout_sec = int(self.config.get("timeout_sec", 120))
        self.temperature = float(self.config.get("temperature", 0.2))
        self.max_tokens = int(self.config.get("max_tokens", 160))
        self.system_hint = str(self.config.get("system_hint", default_hint))

    @classmethod
    def local(cls, config: dict) -> "VlmClient":
        return cls(config)

    @classmethod
    def global_scene(cls, config: dict) -> "VlmClient":
        return cls(config)

    def describe(self, frame: Any, hint: str = "") -> str:
        image_url = "data:image/jpeg;base64," + self._frame_to_jpeg_b64(frame)
        prompt = self.system_hint
        if hint:
            prompt += "\n" + hint
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            import sys
            print(f"[vlm_client] warn: empty choices from {self.endpoint}", file=sys.stderr)
            return ""
        content = (choices[0].get("message") or {}).get("content") or ""
        return content.strip()

    def describe_batch(self, frames: list[Any], hints: list[str]) -> list[str]:
        """批量推理:一次发送 N 张图,返回 N 个响应字符串。

        自动推导 batch 端点(把 /v1/chat/completions 替换为 /v1/chat/completions/batch)。
        如果 server 端不支持 batch(404 / 任何网络异常),自动降级为逐张串行调用。
        hints 与 frames 一一对应;长度不一致时截断到较短一方。
        """
        if not frames:
            return []

        # 推导 batch 端点
        base = self.endpoint
        if base.endswith("/v1/chat/completions"):
            batch_endpoint = base + "/batch"
        else:
            batch_endpoint = base.rstrip("/") + "/batch"

        batch_items = []
        paired_hints = list(hints) + [""] * max(0, len(frames) - len(hints))
        for frame, hint in zip(frames, paired_hints):
            prompt = self.system_hint
            if hint:
                prompt += "\n" + hint
            batch_items.append({
                "prompt": prompt,
                "image_b64": self._frame_to_jpeg_b64(frame),
            })

        try:
            payload = {
                "batch": batch_items,
                "max_new_tokens": self.max_tokens,
            }
            resp = requests.post(
                batch_endpoint,
                json=payload,
                timeout=self.timeout_sec * max(1, len(frames)),
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            # 保证返回长度与输入一致
            while len(results) < len(frames):
                results.append("")
            return [str(r).strip() for r in results[: len(frames)]]
        except Exception as _batch_exc:
            import sys
            from urllib.parse import urlparse
            p = urlparse(self.endpoint)
            health_url = f"{p.scheme}://{p.netloc}/health"
            try:
                probe = requests.get(health_url, timeout=5)
                reachable = probe.status_code < 500
            except Exception:
                reachable = False
            if not reachable:
                est_secs = len(frames) * self.timeout_sec
                raise RuntimeError(
                    f"[vlm_client] VLM server unreachable ({_batch_exc}); "
                    f"serial fallback would block ~{est_secs}s for {len(frames)} frames. Aborting."
                )
            print(f"[vlm_client] batch failed ({_batch_exc}), falling back to serial", file=sys.stderr)
            return [self.describe(frame, hint) for frame, hint in zip(frames, paired_hints)]

    def _frame_to_jpeg_b64(self, frame: Any) -> str:
        import cv2

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("failed to encode video frame as JPEG")
        return base64.b64encode(buffer.tobytes()).decode("ascii")
