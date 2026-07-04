"""Phase 2 OpenAI-compatible multimodal API client."""
from __future__ import annotations

import base64
from typing import Any

from core.prompt_loader import load_prompt
from sbmachine.llm_shim import _dump_api_log, _load_secrets, _post_openai_with_retry

GLOBAL_SYSTEM_HINT = load_prompt("vlm_system")


class VlmApiClient:
    def __init__(
        self,
        config: dict,
        *,
        default_hint: str = GLOBAL_SYSTEM_HINT,
    ) -> None:
        self.config = config or {}
        api_cfg = self.config.get("api", {}) if isinstance(self.config.get("api", {}), dict) else {}
        secrets = _load_secrets()
        vlm_secrets = secrets.get("vlm", {}) if isinstance(secrets.get("vlm", {}), dict) else {}
        self.base_url = (
            vlm_secrets.get("base_url")
            or secrets.get("base_url")
            or api_cfg.get("base_url")
            or ""
        ).strip()
        if not self.base_url:
            raise ValueError("Phase 2 VLM API requires base_url")
        self.api_key = (
            vlm_secrets.get("api_key")
            or secrets.get("api_key")
            or api_cfg.get("api_key")
            or ""
        )
        self.model = str(
            vlm_secrets.get("model")
            or secrets.get("model")
            or api_cfg.get("model")
            or self.config.get("model")
            or ""
        )
        if not self.model:
            raise ValueError("Phase 2 VLM API requires model")
        self.timeout_sec = int(self.config.get("timeout_sec", 120))
        self.temperature = float(self.config.get("temperature", 0.2))
        self.max_tokens = int(self.config.get("max_tokens", 160))
        self.system_hint = str(self.config.get("system_hint", default_hint))

    @classmethod
    def global_scene(cls, config: dict) -> "VlmApiClient":
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        data = _post_openai_with_retry(url, payload, headers, self.timeout_sec)
        _dump_api_log(url, payload, data)
        choices = data.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content") or ""
        return content.strip()

    def describe_batch(self, frames: list[Any], hints: list[str]) -> list[str]:
        paired_hints = list(hints) + [""] * max(0, len(frames) - len(hints))
        return [self.describe(frame, hint) for frame, hint in zip(frames, paired_hints)]

    def _frame_to_jpeg_b64(self, frame: Any) -> str:
        import cv2

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("failed to encode video frame as JPEG")
        return base64.b64encode(buffer.tobytes()).decode("ascii")
