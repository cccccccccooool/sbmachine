#!/usr/bin/env python3
"""Phase3a 单窗口调试工具 —— 脱离完整流水线，快速复现一个 window 的处理链。

用法：
  # 从 rounds_with_neutral.json 选一个窗口，展示已有结果
  python -m tools.debug_phase3a --input output/sbmachine/rounds_with_neutral.json --round 1 --window 3

  # Mock 后端：用预定义响应测试解析/契约/fallback
  python -m tools.debug_phase3a --input ... --round 1 --window 3 --backend mock

  # Replay 后端：回放已保存的 vLLM 原始响应
  python -m tools.debug_phase3a --input ... --round 1 --window 3 --backend replay --fixture output/debug/<run_id>/phase3a/

  # 本地小模型
  python -m tools.debug_phase3a --input ... --round 1 --window 3 --backend openai_compatible --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sbmachine.commentary_planner import fallback_neutral
from sbmachine.debug_phase3a import DebugRecorder, DebugWindowRecord
from sbmachine.llm_backends import create_replay_generator, mock_generate
from sbmachine.llm_projection import build_llm_window_projection
from sbmachine.phase3a_prompt import (
    _build_analyst_system,
    _build_window_prompt,
    _first_json_obj,
    _parse_window_neutral_response,
)


def _find_window(rounds: list[dict], target_round: int, target_window: int) -> tuple[dict, dict] | None:
    """在 rounds_with_neutral.json 中定位指定 round/window。"""
    for rnd in rounds:
        if rnd.get("round_no") != target_round:
            continue
        scenes = rnd.get("scenes", [])
        if not isinstance(scenes, list):
            continue
        if 1 <= target_window <= len(scenes):
            return rnd, scenes[target_window - 1]
    return None


def _build_debug_record(
    *,
    round_no: int,
    window_idx: int,
    scene: dict,
    plan: dict,
    window_payload: dict,
    system_prompt: str,
    user_prompt: str,
    raw_result: str | None,
    neutral: str,
    neutral_source: str,
) -> DebugWindowRecord:
    """从窗口处理结果构建完整的 DebugWindowRecord。"""
    fb_text = fallback_neutral(plan)

    raw_http_body = None
    raw_vllm = None
    msg_content = None
    reasoning = None
    think = None
    cleaned = None

    if raw_result is not None:
        raw_http_body = getattr(raw_result, "request_payload", None)
        raw_vllm = getattr(raw_result, "raw_response", None)
        msg_content = str(raw_result)
        reasoning = getattr(raw_result, "reasoning_content", None)
        if raw_vllm:
            orig_content = (
                (raw_vllm.get("choices", [{}])[0].get("message", {}) or {})
                .get("content") or ""
            )
            if "</think>" in orig_content:
                think = orig_content
                cleaned = orig_content.rsplit("</think>", 1)[-1].strip()

    # 解析诊断
    json_parsed = None
    parse_err = None
    contract_ok = False
    contract_err = None
    parsed_neu = None

    if raw_result is not None:
        json_parsed = _first_json_obj(str(raw_result))
        if json_parsed is None:
            parse_err = "not a valid JSON object"
        else:
            for _key in ("think", "reasoning", "reasoning_content"):
                json_parsed.pop(_key, None)
            extra = set(json_parsed) - {"neutral"}
            if extra:
                contract_err = f"unexpected fields: {sorted(extra)}"
            elif "neutral" not in json_parsed:
                contract_err = "missing 'neutral' field"
            elif not isinstance(json_parsed.get("neutral"), str):
                contract_err = "neutral is not a string"
            else:
                contract_ok = True
                parsed_neu = json_parsed["neutral"].strip()

    return DebugWindowRecord(
        round_no=round_no,
        window_idx=window_idx,
        t_start=float(scene.get("t_start", 0)),
        t_end=float(scene.get("t_end", 0)),
        scene=str(scene.get("scene", "")),
        raw_plan=plan,
        llm_projection=window_payload,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        http_request_body=raw_http_body if isinstance(raw_http_body, dict) else None,
        llm_config={},
        vllm_raw_response=raw_vllm if isinstance(raw_vllm, dict) else None,
        message_content=msg_content,
        reasoning_content=reasoning,
        think_text=think,
        cleaned_content=cleaned,
        json_parse_result=json_parsed,
        parse_error=parse_err,
        contract_valid=contract_ok,
        contract_error=contract_err,
        parsed_neutral=parsed_neu,
        final_neutral=neutral,
        neutral_source=neutral_source,
        equals_fallback=(neutral == fb_text),
        fallback_text=fb_text,
    )


def _print_separator(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _print_json(label: str, data: object) -> None:
    print(f"\n── {label} ──")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(str(data))


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase3a 单窗口调试")
    ap.add_argument("--input", required=True, help="rounds_with_neutral.json 路径")
    ap.add_argument("--round", type=int, required=True, help="回合编号")
    ap.add_argument("--window", type=int, required=True, help="窗口编号 (1-based)")
    ap.add_argument(
        "--backend",
        choices=["mock", "replay", "openai_compatible"],
        default=None,
        help="LLM 后端；省略则只展示已有 neutral（不调模型）",
    )
    ap.add_argument("--fixture", help="replay 模式的 fixture 目录")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="openai_compatible 的 base_url")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B", help="模型名称")
    ap.add_argument("--output", help="调试 JSON 输出目录（默认 output/debug/tool/<uuid>/）")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    data = json.loads(input_path.read_text(encoding="utf-8"))
    rounds = data.get("rounds", []) if isinstance(data, dict) else []
    if not rounds:
        print("ERROR: no rounds in input", file=sys.stderr)
        return 2

    found = _find_window(rounds, args.round, args.window)
    if found is None:
        print(
            f"ERROR: round={args.round} window={args.window} not found in input",
            file=sys.stderr,
        )
        return 2

    rnd, scene = found
    plan = scene.get("commentary_plan", {})
    if not isinstance(plan, dict) or not plan:
        print("ERROR: scene has no commentary_plan", file=sys.stderr)
        return 2

    round_no = args.round
    window_idx = args.window

    # 1. 原始 plan
    _print_separator(f"Round {round_no} / Window {window_idx}")
    print(f"  scene: {scene.get('scene', '?')}")
    print(f"  t_start: {scene.get('t_start')}  t_end: {scene.get('t_end')}")
    print(f"  existing neutral: {scene.get('neutral', '')[:80]}")
    print(f"  existing neutral_source: {scene.get('neutral_source', '?')}")

    _print_json("1. Raw Plan (commentary_plan)", plan)

    # 2. LLM 投影
    window_payload = build_llm_window_projection(plan)
    _print_json("2. LLM Projection (whitelist)", window_payload)

    # 3. System / User Prompt
    system_prompt = _build_analyst_system()
    user_prompt = _build_window_prompt({"commentary_plan": plan})
    _print_separator("3. System Prompt")
    print(system_prompt)
    _print_separator("4. User Prompt")
    print(user_prompt)

    # 4. LLM 调用（可选）
    neutral = str(scene.get("neutral", ""))
    neutral_source = str(scene.get("neutral_source", "fallback"))
    raw_result = None

    if args.backend:
        _print_separator(f"5. LLM Call (backend={args.backend})")

        if args.backend == "mock":
            gen_fn = mock_generate
            llm_cfg = {"model": "mock", "temperature": 0.3}
        elif args.backend == "replay":
            fixture_dir = Path(args.fixture) if args.fixture else Path("output/debug")
            if not fixture_dir.is_dir():
                print(f"ERROR: fixture dir not found: {fixture_dir}", file=sys.stderr)
                return 2
            gen_fn = create_replay_generator(fixture_dir)
            llm_cfg = {"model": "replay", "temperature": 0.3}
        elif args.backend == "openai_compatible":
            from sbmachine import llma_api as _llma_backend

            gen_fn = _llma_backend.generate
            llm_cfg = {
                "model": args.model,
                "temperature": 0.3,
                "max_tokens": 256,
                "timeout_sec": 120,
            }
            if args.base_url:
                llm_cfg["base_url"] = args.base_url
            print(f"  base_url: {args.base_url or '(from env AI6657_BASE_URL)'}")
            print(f"  model: {args.model}")
        else:
            print(f"ERROR: unknown backend: {args.backend}", file=sys.stderr)
            return 2

        log_ctx = {"round": f"round{round_no}", "scene": f"win{window_idx}"}
        try:
            raw_result = gen_fn(
                user_prompt,
                llm_cfg,
                system_prompt=system_prompt,
                max_tokens=256,
                log_ctx=log_ctx,
            )
        except Exception as exc:
            print(f"ERROR: LLM call failed: {exc}", file=sys.stderr)
            raw_result = None

        if raw_result is not None:
            raw_text = str(raw_result)
            _print_json("5a. Raw Response Content", raw_text[:500])

            # 检查 think/reasoning 属性
            reasoning = getattr(raw_result, "reasoning_content", None)
            if reasoning:
                _print_json("5b. Reasoning Content", str(reasoning)[:300])

            raw_vllm = getattr(raw_result, "raw_response", None)
            if raw_vllm:
                _print_json("5c. vLLM Raw Response (sanitized)", {
                    "model": raw_vllm.get("model", ""),
                    "choices[0].message.role": (
                        raw_vllm.get("choices", [{}])[0].get("message", {}).get("role", "")
                    ),
                    "content_preview": str(
                        raw_vllm.get("choices", [{}])[0].get("message", {}).get("content", "")
                    )[:200],
                    "has_reasoning": bool(
                        raw_vllm.get("choices", [{}])[0].get("message", {}).get("reasoning_content")
                    ),
                    "usage": raw_vllm.get("usage", {}),
                })

            # 6. 解析
            _print_separator("6. Parse & Contract Check")
            parsed = _parse_window_neutral_response(raw_text, debug=True)
            if parsed is None:
                print("  PARSE FAILED → will use fallback")

                # 诊断失败原因
                json_obj = _first_json_obj(raw_text)
                if json_obj is None:
                    print("  reason: not a valid JSON object")
                else:
                    for _key in ("think", "reasoning", "reasoning_content"):
                        json_obj.pop(_key, None)
                    extra = set(json_obj) - {"neutral"}
                    if extra:
                        print(f"  reason: unexpected fields: {sorted(extra)}")
                    elif "neutral" not in json_obj:
                        print("  reason: missing 'neutral' field")
                    elif not isinstance(json_obj.get("neutral"), str):
                        print("  reason: neutral is not a string")
                    else:
                        print("  reason: unknown (contract OK but parse returned None)")
            else:
                print(f"  PARSE OK: neutral = {parsed[:100]}")

            # 7. Fallback
            _print_separator("7. Fallback Decision")
            fb_text = fallback_neutral(plan)
            print(f"  fallback_neutral(plan): {fb_text[:100]}")

            if neutral_source == "fallback":
                neutral = fb_text
                print("  → USING FALLBACK")
            else:
                print(f"  → USING LLM OUTPUT (neutral_source={neutral_source})")
        else:
            # LLM 调用失败
            _print_separator("6-7. LLM Failed")
            neutral = fallback_neutral(plan)
            neutral_source = "fallback"
            print(f"  → fallback: {neutral[:100]}")
    else:
        # 不调用 LLM，只展示已有结果
        _print_separator("5-7. Existing Result (no LLM call)")
        print(f"  neutral: {neutral[:100]}")
        print(f"  neutral_source: {neutral_source}")
        fb_text = fallback_neutral(plan)
        print(f"  fallback_neutral(plan): {fb_text[:100]}")
        print(f"  equals_fallback: {neutral == fb_text}")

    # 8. 构建 DebugWindowRecord 并落盘
    _print_separator("8. Debug Record")
    record = _build_debug_record(
        round_no=round_no,
        window_idx=window_idx,
        scene=scene,
        plan=plan,
        window_payload=window_payload,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_result=raw_result,
        neutral=neutral,
        neutral_source=neutral_source,
    )

    output_dir = (
        Path(args.output)
        if args.output
        else PACKAGE_ROOT / "output" / "debug" / "tool" / uuid.uuid4().hex / "phase3a"
    )
    recorder = DebugRecorder(enabled=True, output_dir=output_dir)
    recorder.record_window(record)
    print(f"  debug JSON written to: {output_dir / f'r{round_no:03d}_w{window_idx:02d}.json'}")

    # 9. 最终结果
    _print_separator("9. Final Result")
    print(f"  neutral: {neutral}")
    print(f"  neutral_source: {neutral_source}")
    print(f"  equals_fallback: {record.equals_fallback}")
    print(f"  contract_valid: {record.contract_valid}")
    if record.contract_error:
        print(f"  contract_error: {record.contract_error}")
    if record.parse_error:
        print(f"  parse_error: {record.parse_error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
