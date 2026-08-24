"""Phase3a 调试记录器：收集每个窗口的完整处理链数据，仅 debug 模式启用。

非 debug 模式时 DebugRecorder 内部为 no-op，不产生任何 I/O 开销。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DebugWindowRecord:
    """单窗口完整调试记录 —— 从原始 plan 到最终 neutral 的全链路快照。"""

    # ── 定位 ──
    round_no: int
    window_idx: int
    t_start: float
    t_end: float
    scene: str
    run_id: str = ""

    # ── 输入侧 ──
    raw_plan: dict = field(default_factory=dict)
    llm_projection: dict = field(default_factory=dict)
    system_prompt: str = ""
    user_prompt: str = ""
    http_request_body: dict | None = None
    llm_config: dict = field(default_factory=dict)

    # ── 输出侧 ──
    vllm_raw_response: dict | None = None
    message_content: str | None = None
    reasoning_content: str | None = None
    think_text: str | None = None
    cleaned_content: str | None = None

    # ── 解析侧 ──
    json_parse_result: dict | None = None
    parse_error: str | None = None
    contract_valid: bool = False
    contract_error: str | None = None
    parsed_neutral: str | None = None

    # ── 最终侧 ──
    final_neutral: str = ""
    neutral_source: str = "fallback"
    generation_status: str = ""  # success | transport_error | http_error | response_error | truncated | parse_error | contract_error
    error_stage: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    content_present: bool = False
    content_chars: int = 0
    reasoning_present: bool = False
    reasoning_chars: int = 0
    raw_response_saved: bool = False
    equals_fallback: bool = False
    fallback_text: str = ""


class DebugRecorder:
    """调试记录器：disabled 时所有方法为 no-op，零 I/O 开销。"""

    def __init__(self, enabled: bool, output_dir: Path) -> None:
        self._enabled = enabled
        self._output_dir = output_dir
        if self._enabled:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def record_window(self, record: DebugWindowRecord) -> None:
        """写入单窗口调试 JSON。disabled 时直接返回。"""
        if not self._enabled:
            return
        fname = f"r{record.round_no:03d}_w{record.window_idx:02d}.json"
        path = self._output_dir / fname
        path.write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
