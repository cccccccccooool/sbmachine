"""Phase3a LLM 后端：mock（预定义响应）与 replay（fixture 回放）。

每个后端都返回 _ApiChatResult 实例，保持与 llma_api.generate() 的契约兼容。
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from sbmachine.llm_shim import _ApiChatResult

# ── Mock 后端 ──────────────────────────────────────────────────────────────

_MOCK_CASES: list[dict[str, Any]] = [
    # (after_strip, raw_content, reasoning_content, description)
    {
        "after_strip": '{"neutral":"A区爆弹进攻"}',
        "raw_content": '{"neutral":"A区爆弹进攻"}',
        "reasoning": None,
        "desc": "正常 neutral",
    },
    {
        "after_strip": '{"neutral":""}',
        "raw_content": '{"neutral":""}',
        "reasoning": None,
        "desc": "合法空 neutral",
    },
    {
        "after_strip": '{"think":"分析过程","neutral":"战术配合"}',
        "raw_content": '{"think":"分析过程","neutral":"战术配合"}',
        "reasoning": None,
        "desc": "JSON 级 think 字段",
    },
    {
        "after_strip": '{"neutral":"闪光弹致盲"}',
        "raw_content": '{"neutral":"闪光弹致盲"}',
        "reasoning": "这是推理过程，解释为什么选择这个闪光弹描述...",
        "desc": "response 层 reasoning_content",
    },
    {
        "after_strip": '{"neutral":"有效输出"}',
        "raw_content": '<think>需要先分析战术背景，确认是默认A区爆弹后再写解说。</think>\n{"neutral":"有效输出"}',
        "reasoning": None,
        "desc": "<think> 标签包裹",
    },
    {
        "after_strip": '```json\n{"neutral":"Markdown围栏"}\n```',
        "raw_content": '```json\n{"neutral":"Markdown围栏"}\n```',
        "reasoning": None,
        "desc": "Markdown JSON 围栏",
    },
    {
        "after_strip": '{"neutral":"文本","extra_field":true}',
        "raw_content": '{"neutral":"文本","extra_field":true}',
        "reasoning": None,
        "desc": "未知额外字段",
    },
    {
        "after_strip": "这不是JSON",
        "raw_content": "这不是JSON",
        "reasoning": None,
        "desc": "非法 JSON",
    },
    {
        "after_strip": '{"neutral": 123}',
        "raw_content": '{"neutral": 123}',
        "reasoning": None,
        "desc": "neutral 非字符串",
    },
]

_mock_index = 0


def _build_mock_raw_response(
    raw_content: str, reasoning: str | None, model_name: str
) -> dict:
    """构建与 vLLM OpenAI-compatible 格式一致的模拟响应。"""
    message: dict[str, Any] = {"role": "assistant", "content": raw_content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": 1720000000,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": len(raw_content) // 2,
            "total_tokens": 200 + len(raw_content) // 2,
        },
    }


def mock_generate(
    prompt: str,
    llm_cfg: dict,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
) -> str:
    """轮转返回预定义响应，覆盖所有回归场景。

    环境变量 MOCK_RESPONSE_INDEX 可锁定特定索引（0-based）。
    """
    global _mock_index
    forced = os.getenv("MOCK_RESPONSE_INDEX")
    if forced is not None:
        try:
            idx = int(forced) % len(_MOCK_CASES)
        except (ValueError, TypeError):
            idx = _mock_index
    else:
        idx = _mock_index
        _mock_index = (_mock_index + 1) % len(_MOCK_CASES)

    case = _MOCK_CASES[idx]
    model = str(llm_cfg.get("model", "mock-model"))
    raw_response = _build_mock_raw_response(
        case["raw_content"], case.get("reasoning"), model
    )

    messages = [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": prompt},
    ]
    request_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_cfg.get("temperature", 0.3)),
    }
    if max_tokens and int(max_tokens) > 0:
        request_payload["max_tokens"] = int(max_tokens)

    result = _ApiChatResult(
        case["after_strip"],
        scope="llma",
        source_run_id=uuid.uuid4().hex,
        request_payload=request_payload,
        log_ctx=log_ctx,
        raw_response=raw_response,
        reasoning_content=case.get("reasoning"),
    )
    # 打印调试信息，方便本地集成测试观察
    print(f"  [mock] #{idx} {case['desc']}", flush=True)
    return result


# ── Replay 后端 ─────────────────────────────────────────────────────────────

def create_replay_generator(fixture_dir: Path):
    """返回 match llma_api.generate 签名的 replay 生成函数。

    从 fixture_dir（output/debug/<run_id>/phase3a/）读取已保存的 debug JSON，
    用 log_ctx 中的 round/scene 匹配对应 fixture，重建 _ApiChatResult 回放。
    """
    fixtures: dict[tuple[int, int], dict] = {}
    for path in sorted(Path(fixture_dir).glob("r*_w*.json")):
        m = re.match(r"r(\d+)_w(\d+)\.json", path.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        try:
            fixtures[key] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

    if not fixtures:
        print(
            f"[replay] WARNING: no fixtures found in {fixture_dir}",
            flush=True,
        )

    def _replay_generate(
        prompt: str,
        llm_cfg: dict,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        log_ctx: dict | None = None,
    ) -> str:
        # 从 log_ctx 解析 round/window
        round_no = 0
        win_idx = 0
        if log_ctx:
            round_str = str(log_ctx.get("round", ""))
            scene_str = str(log_ctx.get("scene", ""))
            m_round = re.search(r"(\d+)", round_str)
            m_win = re.search(r"(\d+)", scene_str)
            if m_round:
                round_no = int(m_round.group(1))
            if m_win:
                win_idx = int(m_win.group(1))

        key = (round_no, win_idx)
        fixture = fixtures.get(key)
        if fixture is None:
            # 宽松匹配：只按 round 匹配的第一个 fixture
            for (r, w), f in fixtures.items():
                if r == round_no:
                    fixture = f
                    print(
                        f"  [replay] round {round_no}: using w{w} fixture for w{win_idx} (fallback match)",
                        flush=True,
                    )
                    break
            if fixture is None:
                raise RuntimeError(
                    f"[replay] no fixture for round={round_no} window={win_idx} in {fixture_dir}"
                )

        raw_vllm = fixture.get("vllm_raw_response")
        http_body = fixture.get("http_request_body")
        if raw_vllm is None:
            raise RuntimeError(
                f"[replay] fixture r{round_no:03d}_w{win_idx:02d}.json has no vllm_raw_response"
            )

        message = (
            (raw_vllm.get("choices", [{}])[0].get("message", {}) or {})
            if isinstance(raw_vllm, dict)
            else {}
        )
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content")

        # 复现 _execute_openai_chat 的 think 剥离逻辑
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[-1]
        content = content.strip()

        print(
            f"  [replay] r{round_no:03d}_w{win_idx:02d} → content={content[:60]}...",
            flush=True,
        )

        return _ApiChatResult(
            content,
            scope="llma",
            source_run_id=uuid.uuid4().hex,
            request_payload=http_body if isinstance(http_body, dict) else {},
            log_ctx=log_ctx,
            raw_response=raw_vllm if isinstance(raw_vllm, dict) else None,
            reasoning_content=reasoning if isinstance(reasoning, str) and reasoning else None,
        )

    return _replay_generate
