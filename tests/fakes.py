"""Zero-network fake model backends for pipeline tests."""
from __future__ import annotations

import itertools
import wave
from pathlib import Path
from typing import Any


class FakeLLM:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or ["{}"])
        self.prompts: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self._cursor = 0

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        self.calls.append({"prompt": prompt, "args": args, "kwargs": kwargs})
        if not self.responses:
            return "{}"
        index = min(self._cursor, len(self.responses) - 1)
        self._cursor += 1
        return self.responses[index]

    def chat_completion(self, messages: list[dict[str, str]], *args: Any, **kwargs: Any) -> str:
        prompt = "\n".join(str(item.get("content", "")) for item in messages)
        return self.generate(prompt, *args, **kwargs)


class FakeVLM:
    def __init__(self, description: str = "fake visual description") -> None:
        self.description = description
        self.frames: list[Any] = []

    def describe_frame(self, frame: Any, *args: Any, **kwargs: Any) -> str:
        self.frames.append({"frame": frame, "args": args, "kwargs": kwargs})
        return self.description

    def describe_frames(self, frames: list[Any], *args: Any, **kwargs: Any) -> list[str]:
        return [self.describe_frame(frame, *args, **kwargs) for frame in frames]


class FakeTTS:
    def __init__(self, duration_sec: float = 0.5, sample_rate: int = 16000) -> None:
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.requests: list[dict[str, Any]] = []

    def synthesize(self, text: str, output_path: str | Path, *args: Any, **kwargs: Any) -> Path:
        self.requests.append({"text": text, "output_path": str(output_path), "args": args, "kwargs": kwargs})
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nframes = int(self.duration_sec * self.sample_rate)
        silence = b"\x00\x00"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(b"".join(itertools.repeat(silence, nframes)))
        return path
