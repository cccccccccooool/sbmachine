"""Phase3b → Phase4 之间的空回合决策钩子（临时交互接口）。

运行语义：当 commentary 产物里存在「空回合」（round.status == "empty"，
p3b 阶段因 required 窗口 style_failed 占比超阈值而整回合留空）时，run_all
在进入 Phase4 前调用本模块，向用户提供三个选项：

- continue：接受空回合，继续进入 Phase4（p4 侧跳过留空回合）；
- retry   ：重新运行 Phase3b（LLM-B）重出该场解说（接口预留，当前由调用方后续实现）；
- cancel  ：保留当前已产出结果并退出流水线（接口预留，同上）。

当前实现只落地 continue 的默认交互（纯 input()），retry/cancel 仅预留
返回值契约，具体重跑/退出逻辑由后续接入，不在本模块实现。
"""
from __future__ import annotations

from typing import Any, Callable

DEFAULT_PROMPT_TEXT = (
    "\n检测到空回合（round.status == empty，p3b 已整回合留空）。"
    "请选择下一步：\n"
    "  [continue] 保留空回合，继续进入 Phase4（跳过留空回合）\n"
    "  [retry]    重新运行 Phase3b(LLM-B) 重出解说（接口已预留）\n"
    "  [cancel]   保留当前产物并退出流水线（接口已预留）\n"
    "输入选择（continue/retry/cancel）："
)

_VALID_ACTIONS = frozenset({"continue", "retry", "cancel"})


def has_empty_rounds(commentary_manifest: dict[str, Any]) -> bool:
    """判断 commentary 产物是否包含空回合。"""
    rounds = commentary_manifest.get("rounds") if isinstance(commentary_manifest, dict) else None
    if not isinstance(rounds, list):
        return False
    return any(
        isinstance(round_data, dict) and round_data.get("status") == "empty"
        for round_data in rounds
    )


def _default_prompt(manifest: dict[str, Any]) -> str:
    empty_rounds = [
        r.get("round_no") for r in manifest.get("rounds", [])
        if isinstance(r, dict) and r.get("status") == "empty"
    ]
    suffix = f"（空回合：round_no={empty_rounds}）" if empty_rounds else ""
    return DEFAULT_PROMPT_TEXT + suffix


def decide_empty_rounds(
    commentary_manifest: dict[str, Any],
    *,
    prompt: Callable[[str], str] | None = None,
) -> str:
    """空回合三选一交互；返回 continue/retry/cancel。

    prompt：注入输入读取函数（默认内置 input()）；便于测试与后续接入
    retry/cancel 的调度逻辑。任何非识别输入默认回落 continue（不阻断流水线）。
    """
    if not has_empty_rounds(commentary_manifest):
        return "continue"
    readline = prompt or input
    try:
        action_input = readline(_default_prompt(manifest=commentary_manifest)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "continue"
    if action_input in _VALID_ACTIONS:
        return action_input
    # 兼容单字母/数字快捷输入
    shortcuts = {"c": "continue", "r": "retry", "x": "cancel", "1": "continue", "2": "retry", "3": "cancel"}
    return shortcuts.get(action_input, "continue")
