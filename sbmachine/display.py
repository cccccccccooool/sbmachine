"""终端 GUI 渲染层。

`run.py` 默认模式的唯一渲染出口：把 `run_all()` 的阶段回调实时画成 rich 进度条 +
阶段产物，结束后渲染最终汇报面板。本模块只做展示，不含任何业务逻辑，也不 import
任何 phase* 模块；所有渲染都是防御式的，绝不因缺字段或路径异常打断流水线。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.config_loader import ConfigError
from sbmachine.common import load_config, require_path
from sbmachine.preflight import enabled_phases
from sbmachine.run_all import run_all

# run_all 只会回调这些阶段名；publish / preflight / lock 等内部状态名静默忽略。
_KNOWN_STAGES = {
    "demo_parse", "video_marking", "phase1",
    "phase2", "phase3a", "phase3b", "phase3c", "phase4",
}

# 进度条固定展示顺序（与流水线实际执行顺序一致）。
_STAGE_ORDER = ("demo_parse", "video_marking", "phase1", "phase2", "phase3a", "phase3b", "phase3c", "phase4")

# 中文显示名。同时收录回调阶段名与 config `phases:` key 两套词汇：
# 回调传的是 phase1/phase2/...，而 run_all 返回的 enabled_phases/checkpointed_stages
# 也是这套短名；config key 版本一并列出，保证任何一侧的写法都能查到显示名。
_STAGE_LABELS = {
    "demo_parse": "Demo 解析",
    "video_marking": "视频标记",
    "phase1": "预处理切片",
    "phase2": "Phase2 YOLO/OCR",
    "phase3a": "Phase3a 中性稿",
    "phase3b": "Phase3b 风格化",
    "phase3c": "Phase3c LLM-C 润色",
    "phase4": "Phase4 TTS 合成",
    # config `phases:` key 别名
    "preprocess_slice": "预处理切片",
    "phase1_slice": "预处理切片",
    "phase2_yolo": "Phase2 YOLO/OCR",
    "phase3a_semantic": "Phase3a 中性稿",
    "phase3b_semantic": "Phase3b 风格化",
    "phase3c_render": "Phase3c LLM-C 润色",
    "phase4_assemble": "Phase4 TTS 合成",
}

_BAR_WIDTH = 20


def _console() -> Console:
    """构造 Console。

    Windows 中文控制台默认 GBK，emoji / 框线字符会让 rich 抛 UnicodeEncodeError，
    直接打断流水线收尾。这里就地把 stdout 切到 UTF-8（errors="replace"）。
    刻意不另包一层 TextIOWrapper：包装对象被 GC 时会连带关掉 sys.stdout.buffer，
    害得本进程后续所有 stdout 写入报 "I/O operation on closed file"。
    legacy_windows 交给 rich 自行探测，避免在老 conhost 上吐裸 ANSI 转义。
    """
    try:
        stdout = sys.stdout
        encoding = (getattr(stdout, "encoding", "") or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stdout, "reconfigure"):
            stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    return Console()


def _safe_print(console: Console, renderable: Any, *, fallback: str | None = None) -> None:
    """打印且绝不抛：编码/渲染失败时退化为 ASCII 化纯文本。"""
    try:
        console.print(renderable)
        return
    except Exception:
        pass
    try:
        text = fallback if fallback is not None else str(renderable)
        sys.stdout.write(text.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _fmt_duration(seconds: float | None) -> str:
    """秒数格式化为 MM:SS，异常输入返回 --:--。"""
    try:
        total = int(max(0.0, float(seconds)))
    except (TypeError, ValueError):
        return "--:--"
    return f"{total // 60:02d}:{total % 60:02d}"


def _fmt_size(num_bytes: int | None) -> str:
    """字节数格式化为人类可读大小。"""
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _describe_path(path: Any) -> str | None:
    """产物路径描述：文件给大小，目录给文件数，不存在返回 None。"""
    try:
        target = Path(path)
        if target.is_file():
            return f"{target.name}  {_fmt_size(target.stat().st_size)}"
        if target.is_dir():
            files = [item for item in target.iterdir() if item.is_file()]
            return f"{target.name}/  {len(files)} 个文件"
    except (OSError, ValueError, TypeError):
        return None
    return None


def _enabled_stages(config: Any) -> list[str]:
    """从 effective config 的 `phases:` 解析已启用阶段，返回回调阶段名列表。

    直接复用 `sbmachine.preflight.enabled_phases()`：config key → 阶段名的映射与
    默认值是它的既有职责，重复实现一份只会在新增阶段时产生漂移。
    """
    try:
        stages = enabled_phases(config if isinstance(config, dict) else {})
    except Exception:
        return []
    result: list[str] = [name for name in _STAGE_ORDER if name in set(stages)]
    return result


def _stage_label(name: Any) -> str:
    """阶段名 → 显示名；未知名称原样返回。"""
    return _STAGE_LABELS.get(str(name), str(name))


class _Display:
    """区域 A（进度条）+ 区域 B（阶段产物）的状态机与渲染器。

    所有回调都吞掉自身异常：GUI 层永远不能因渲染问题打断 run_all 的流水线。
    """

    def __init__(self, stages: list[str]) -> None:
        self._lock = threading.RLock()
        self.stages = list(stages)
        self.status: dict[str, str] = {stage: "pending" for stage in self.stages}
        self.started: dict[str, float] = {}
        self.elapsed: dict[str, float] = {}
        self.attempt: dict[str, int] = {stage: 0 for stage in self.stages}
        self.completed: dict[str, int | None] = {stage: None for stage in self.stages}
        self.total: dict[str, int | None] = {stage: None for stage in self.stages}
        self.unit: dict[str, str | None] = {stage: None for stage in self.stages}
        self.detail: dict[str, str | None] = {stage: None for stage in self.stages}
        self.artifacts: dict[str, list[str]] = {}
        self.error: tuple[str, str] | None = None
        self.run_started = time.monotonic()

    @property
    def total_elapsed(self) -> float:
        return time.monotonic() - self.run_started

    def _track(self, stage: str) -> None:
        """确保 stage 在进度表内（config 未列出但仍回调时补一行）。"""
        if stage not in self.status:
            self.status[stage] = "pending"
            self.attempt[stage] = 0
            self.completed[stage] = None
            self.total[stage] = None
            self.unit[stage] = None
            self.detail[stage] = None
            self.stages.append(stage)
            self.stages.sort(key=lambda name: _STAGE_ORDER.index(name) if name in _STAGE_ORDER else 99)

    # ── run_all 回调入口 ──
    def on_stage_start(self, stage: Any) -> None:
        try:
            name = str(stage)
            if name not in _KNOWN_STAGES:
                return
            with self._lock:
                self._track(name)
                self.attempt[name] += 1
                self.started[name] = time.monotonic()
                self.elapsed.pop(name, None)
                self.completed[name] = None
                self.total[name] = None
                self.unit[name] = None
                self.detail[name] = None
                self.status[name] = "running"
        except Exception:
            pass

    def on_stage_progress(
        self,
        stage: Any,
        completed: Any = None,
        total: Any = None,
        unit: Any = None,
        detail: Any = None,
    ) -> None:
        """接收真实工作单元计数；非法、乱序和终态后的事件全部忽略。"""
        try:
            name = str(stage)
            if name not in _KNOWN_STAGES:
                return
            if completed is not None and (isinstance(completed, bool) or not isinstance(completed, int) or completed < 0):
                return
            if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
                return
            if total == 0 and completed not in (None, 0):
                return
            if completed is not None and total is not None and completed > total:
                return
            if unit is not None and (not isinstance(unit, str) or not unit):
                return
            if detail is not None and not isinstance(detail, str):
                return
            with self._lock:
                if self.status.get(name) != "running":
                    return
                prior_total = self.total.get(name)
                if total is not None and prior_total is not None and total != prior_total:
                    return
                prior_completed = self.completed.get(name)
                if completed is not None and prior_completed is not None and completed < prior_completed:
                    return
                if total is not None:
                    self.total[name] = total
                if completed is not None:
                    self.completed[name] = completed
                if unit is not None:
                    self.unit[name] = unit
                if detail is not None:
                    self.detail[name] = detail
        except Exception:
            pass

    def on_stage_done(self, stage: Any, artifacts: Any = None) -> None:
        try:
            name = str(stage)
            if name not in _KNOWN_STAGES:
                return
            with self._lock:
                self._track(name)
                started = self.started.get(name)
                self.elapsed[name] = (time.monotonic() - started) if started is not None else 0.0
                self.status[name] = "done"
                described = []
                for item in list(artifacts or []):
                    item_text = _describe_path(item)
                    if item_text:
                        described.append(item_text)
                self.artifacts[name] = described
        except Exception:
            pass

    def on_error(self, stage: Any, message: Any = "") -> None:
        try:
            name = str(stage)
            if name not in _KNOWN_STAGES:
                return
            with self._lock:
                self.error = (name, str(message))
                self._track(name)
                started = self.started.get(name)
                self.elapsed[name] = (time.monotonic() - started) if started is not None else 0.0
                self.status[name] = "failed"
        except Exception:
            pass

    def on_stage_canceled(self, stage: Any, message: Any = "") -> None:
        try:
            name = str(stage)
            if name not in _KNOWN_STAGES:
                return
            with self._lock:
                self._track(name)
                started = self.started.get(name)
                self.elapsed[name] = (time.monotonic() - started) if started is not None else 0.0
                self.detail[name] = str(message) or self.detail.get(name)
                self.status[name] = "canceled"
        except Exception:
            pass

    def callbacks(self) -> dict:
        return {
            "on_stage_start": self.on_stage_start,
            "on_stage_progress": self.on_stage_progress,
            "on_stage_done": self.on_stage_done,
            "on_stage_canceled": self.on_stage_canceled,
            "on_error": self.on_error,
        }

    # ── 渲染 ──
    def _bar(self, stage: str) -> Text:
        with self._lock:
            state = self.status.get(stage, "pending")
            completed, total = self.completed.get(stage), self.total.get(stage)
            if state == "done":
                return Text("█" * _BAR_WIDTH, style="green")
            filled = 0
            if isinstance(completed, int) and isinstance(total, int) and total > 0:
                filled = _BAR_WIDTH * completed // total
            if state in {"failed", "canceled"}:
                color = "red" if state == "failed" else "yellow"
                return Text("█" * filled, style=color) + Text("▒" * (_BAR_WIDTH - filled), style="dim")
            if state == "running" and total is not None:
                color = "cyan" if total == 0 or completed != total else "bright_cyan"
                return Text("█" * filled, style=color) + Text("▒" * (_BAR_WIDTH - filled), style="dim")
            if state == "running":
                started = self.started.get(stage)
                spent = (time.monotonic() - started) if started is not None else 0.0
                position = int(max(0.0, spent)) % ((_BAR_WIDTH - 1) * 2)
                if position >= _BAR_WIDTH:
                    position = (_BAR_WIDTH - 1) * 2 - position
                return Text("░" * position, style="dim") + Text("▰", style="cyan") + Text("░" * (_BAR_WIDTH - position - 1), style="dim")
            return Text("░" * _BAR_WIDTH, style="dim")

    def _status_text(self, stage: str) -> Text:
        with self._lock:
            state = self.status.get(stage, "pending")
            if state == "done":
                return Text(f"✓ 完成   {_fmt_duration(self.elapsed.get(stage))}", style="green")
            if state == "failed":
                return Text("✗ 失败", style="bold red")
            if state == "canceled":
                return Text("⊘ 未执行/已取消", style="yellow")
            if state == "running":
                started = self.started.get(stage)
                spent = (time.monotonic() - started) if started is not None else 0.0
                completed, total, unit = self.completed.get(stage), self.total.get(stage), self.unit.get(stage)
                if total == 0:
                    return Text(f"⟳ 无工作单元，校验中 {_fmt_duration(spent)}", style="cyan")
                if isinstance(total, int) and isinstance(completed, int):
                    label = f"{completed}/{total} {unit or ''}".rstrip()
                    if total > 0 and completed == total:
                        return Text(f"⟳ {label} 处理完成，校验/提交中", style="bright_cyan")
                    return Text(f"⟳ {label} {_fmt_duration(spent)}", style="cyan")
                suffix = f"，已处理 {completed} {unit or ''}" if isinstance(completed, int) else ""
                return Text(f"⟳ 运行中{suffix} {_fmt_duration(spent)}", style="cyan")
            return Text("○ 等待", style="dim")

    def _progress_table(self) -> Table:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True)
        for stage in self.stages:
            table.add_row(_stage_label(stage), self._bar(stage), self._status_text(stage))
        if not self.stages:
            table.add_row(Text("（无已启用阶段）", style="dim"), Text(""), Text(""))
        return table

    def _artifact_table(self) -> Table:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", no_wrap=True)
        table.add_column(overflow="fold")
        rows = 0
        for stage in self.stages:
            state = self.status.get(stage, "pending")
            if state == "running":
                table.add_row(Text(f"⟳ {_stage_label(stage)}", style="cyan"), Text("进行中...", style="dim"))
                rows += 1
            elif state in {"done", "failed"}:
                items = self.artifacts.get(stage) or []
                mark = "✓" if state == "done" else "✗"
                style = "green" if state == "done" else "red"
                detail = "  ·  ".join(items) if items else "（无产物）"
                table.add_row(Text(f"{mark} {_stage_label(stage)}", style=style), Text(detail))
                rows += 1
        if not rows:
            table.add_row(Text("（尚无产物）", style="dim"), Text(""))
        return table

    def renderable(self) -> Group:
        header = Text("🎙  6657 解说流水线", style="bold") + Text(
            f"    总耗时 {_fmt_duration(self.total_elapsed)}", style="dim"
        )
        return Group(
            Panel(header, border_style="cyan"),
            Panel(self._progress_table(), title="进度", title_align="left", border_style="cyan"),
            Panel(self._artifact_table(), title="阶段产物", title_align="left", border_style="cyan"),
        )


class _LiveView:
    """让 rich Live 的自动刷新每帧重新求值，从而让运行中阶段的计时真正走字。

    Live 只会重画同一个 renderable 对象；实现 `__rich__` 才能在每次刷新时拿到新快照。
    """

    def __init__(self, display: "_Display") -> None:
        self._display = display

    def __rich__(self) -> Any:
        try:
            return self._display.renderable()
        except Exception:
            return Text("（渲染中断）", style="dim")


def _join_stages(names: Any) -> str:
    """阶段名列表 → `Demo 解析 · 视频标记` 形式；空列表给占位符。"""
    try:
        labels = [_stage_label(name) for name in list(names or [])]
    except TypeError:
        labels = []
    return " · ".join(labels) if labels else "（无）"


def _render_startup_error(message: str) -> None:
    """区域 C 的启动期形态：配置路径/加载失败，尚未进入流水线。"""
    body = Group(Text("配置错误", style="bold"), Text(f"  {message}"))
    _safe_print(_console(), Panel(body, title="启动失败", title_align="left", border_style="red"),
                fallback=f"启动失败: 配置错误: {message}")


def _dry_run_panel(report: dict) -> Panel:
    """区域 C 的 dry-run 形态。"""
    valid = bool(report.get("config_valid", False))
    lines: list[Any] = []
    lines.append(Text("已启用阶段", style="bold"))
    lines.append(Text(f"  {_join_stages(report.get('enabled_phases'))}"))
    lines.append(Text(""))
    lines.append(Text("所需输入", style="bold"))
    inputs = report.get("required_inputs") or []
    if not isinstance(inputs, list) or not inputs:
        lines.append(Text("  （无）", style="dim"))
    for item in inputs if isinstance(inputs, list) else []:
        item = item if isinstance(item, dict) else {}
        exists = bool(item.get("exists", False))
        mark = Text("✓ 存在", style="green") if exists else Text("✗ 缺失", style="red")
        lines.append(Text(f"  {item.get('name', '?')}  {item.get('path', '?')}  ") + mark)
    lines.append(Text(""))
    lines.append(Text("将启动的服务", style="bold"))
    services = report.get("services_started") or []
    joined = "  ".join(str(name) for name in services) if isinstance(services, list) and services else "（无）"
    lines.append(Text(f"  {joined}"))
    errors = report.get("errors") or []
    if isinstance(errors, list) and errors:
        lines.append(Text(""))
        lines.append(Text("错误", style="bold red"))
        for err in errors:
            lines.append(Text(f"  {err}", style="red"))
    title = "🔍 Dry-run 预检通过" if valid else "🔍 Dry-run 预检失败"
    return Panel(
        Group(*lines), title=title, title_align="left",
        border_style="green" if valid else "red",
    )


def _success_panel(result: dict, display: _Display) -> Panel:
    """区域 C 的成功形态。"""
    lines: list[Any] = [
        Text(f"run_id: {result.get('run_id', '?')}    总耗时 {_fmt_duration(display.total_elapsed)}"),
        Text(""),
        Text("已完成阶段", style="bold"),
        Text(f"  {_join_stages(result.get('enabled_phases'))}"),
        Text(""),
        Text("输出产物", style="bold"),
    ]
    rows = 0
    for stage in display.stages:
        for item in display.artifacts.get(stage) or []:
            lines.append(Text(f"  {item}"))
            rows += 1
    if not rows:
        lines.append(Text("  （无）", style="dim"))
    checkpointed = result.get("checkpointed_stages") or []
    if isinstance(checkpointed, list) and checkpointed:
        lines.append(Text(""))
        lines.append(Text("已 checkpoint 阶段", style="bold"))
        lines.append(Text(f"  {_join_stages(checkpointed)}"))
    if not result.get("publishable", True):
        lines.append(Text(""))
        lines.append(Text("注意：结果标记为不可发布（publishable=false）", style="yellow"))
    return Panel(Group(*lines), title="✅ 成功", title_align="left", border_style="green")


def _failure_panel(result: dict, display: _Display) -> Panel:
    """区域 C 的失败形态。"""
    stage = result.get("failed_stage") or (display.error[0] if display.error else "?")
    message = result.get("error") or (display.error[1] if display.error else "未提供错误信息")
    lines: list[Any] = [
        Text(f"阶段 {_stage_label(stage)}    run_id: {result.get('run_id', '?')}"
             f"    总耗时 {_fmt_duration(display.total_elapsed)}"),
        Text(""),
        Text("错误", style="bold"),
        Text(f"  {message}", style="red"),
    ]
    checkpointed = result.get("checkpointed_stages") or []
    lines.append(Text(""))
    lines.append(Text("已完成的上游 checkpoint（可复用）", style="bold"))
    lines.append(Text(f"  {_join_stages(checkpointed) if isinstance(checkpointed, list) else '（无）'}"))
    error_dir = result.get("error_dir")
    if error_dir:
        lines.append(Text(""))
        lines.append(Text(f"诊断目录  {error_dir}", style="dim"))
    return Panel(Group(*lines), title="❌ 失败", title_align="left", border_style="red")


def _exit_code(result: Any) -> int:
    """从 run_all 返回值取 exit_code；缺失或非法一律按失败(1)处理。"""
    if not isinstance(result, dict):
        return 1
    try:
        value = result.get("exit_code", 1)
        return 1 if value is None else int(value)
    except (TypeError, ValueError):
        return 1


def _render_result(result: dict, display: _Display, *, dry_run: bool) -> None:
    """渲染区域 C；渲染自身出错时退化为纯文本，绝不抛出。"""
    console = _console()
    try:
        result = result if isinstance(result, dict) else {}
        if dry_run:
            panel = _dry_run_panel(result)
        elif str(result.get("status", "")) == "complete" and _exit_code(result) == 0:
            panel = _success_panel(result, display)
        else:
            panel = _failure_panel(result, display)
    except Exception as exc:  # 组装失败不能影响退出码
        _safe_print(console, f"最终汇报渲染失败: {exc}", fallback=str(result))
        return
    _safe_print(console, panel, fallback=str(result))


def _prompt_empty_rounds(console: Console, live: Live, prompt_text: str) -> str:
    """暂停 Live 刷新并显示空回合决策，避免裸 input() 被 TUI 覆盖。"""
    live.stop()
    try:
        console.print(Panel(
            Text(str(prompt_text).strip()),
            title="需要确认",
            title_align="left",
            border_style="yellow",
        ))
        return console.input("[bold cyan]请输入 continue / retry / cancel：[/]")
    finally:
        live.start(refresh=True)


def run_with_display(config_str: str, *, dry_run: bool = False, turbo: bool = False) -> int:
    """渲染 rich GUI，内部处理所有异常（含 require_path 失败），返回 exit_code。"""
    try:
        config_path = require_path(config_str, "--config")
    except (ConfigError, OSError, ValueError) as exc:
        _render_startup_error(str(exc))
        return 2

    stages: list[str] = []
    try:
        stages = _enabled_stages(load_config(config_path))
    except (ConfigError, OSError, ValueError, TypeError):
        stages = []  # 配置读不出来时先画空表，真正的错误由 run_all 返回后汇报

    display = _Display(stages)
    console = _console()
    if dry_run:
        result = _guarded_run(config_path, display, dry_run=True, turbo=turbo)
        _render_result(result, display, dry_run=True)
        return 0 if isinstance(result, dict) and result.get("config_valid", False) else 2

    # result 先置哨兵：Live 收尾若自己抛异常，绝不能触发流水线二次执行。
    result: dict | None = None
    try:
        with Live(_LiveView(display), console=console, refresh_per_second=4, transient=True) as live:
            callbacks = display.callbacks()
            callbacks["prompt_empty_rounds"] = lambda prompt: _prompt_empty_rounds(console, live, prompt)
            result = _guarded_run(
                config_path, display, dry_run=False, turbo=turbo, callbacks=callbacks,
            )
    except Exception:
        # Live 起不来（无 tty / 编码异常）时退化为无实时刷新的直跑。
        if result is None:
            result = _guarded_run(config_path, display, dry_run=False, turbo=turbo)

    _safe_print(console, display.renderable())
    _render_result(result, display, dry_run=False)
    return _exit_code(result)


def _guarded_run(
    config_path: Path,
    display: _Display,
    *,
    dry_run: bool,
    turbo: bool = False,
    callbacks: dict | None = None,
) -> dict:
    """调用 run_all；即使它意外抛出，也折算成一份失败 dict，GUI 层不向上冒泡。

    Ctrl-C 不吞：转成 exit_code 130 的失败 dict，让面板正常收尾而非裸 traceback。
    """
    try:
        result = run_all(
            config_path, dry_run=dry_run,
            callbacks=callbacks or display.callbacks(), debug_mode=False, turbo=turbo,
        )
    except KeyboardInterrupt:
        return _fallback_result("用户中断（Ctrl-C）", display, dry_run=dry_run, exit_code=130)
    except Exception as exc:  # noqa: BLE001 - GUI 层兜底，避免裸 traceback
        return _fallback_result(f"{type(exc).__name__}: {exc}", display, dry_run=dry_run, exit_code=1)
    return result if isinstance(result, dict) else {}


def _fallback_result(message: str, display: _Display, *, dry_run: bool, exit_code: int) -> dict:
    """run_all 未能返回结果时，按 run.py 既有字段形状伪造一份，保证汇报可渲染。"""
    if dry_run:
        return {
            "config_valid": False, "enabled_phases": [], "required_inputs": [],
            "services_started": [], "writes_performed": False, "errors": [message],
        }
    return {
        "status": "failed", "publishable": False,
        "failed_stage": display.error[0] if display.error else "unknown",
        "error": message, "exit_code": exit_code,
    }
