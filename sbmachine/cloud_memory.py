"""云端会话缓存与成本护栏（仅 backend == "api" 路径使用）。

- `MatchConversation`：按 run_id 隔离的会话；历史上限轮数内保留成功轮（同窗口重试原位替换），
  失败轮不入历史，历史永远是"已确认成功"链。
- 成本护栏：`_ApiChatResult.usage` 累计；`cloud_token_budget_per_match` 固定预算或
  动态预算（窗口数 × 单窗均值 × factor，第二回合首次调用时锁定）；超限后后续窗口走
  silence（不发起请求、不入历史）。
- `generate(...)`：签名与 `llma_api.generate` / `llmb_api.generate` 兼容（含 phase3b 的
  system 位置传参），内部构造 messages = [system] + 历史轮 + 当前轮，复用
  `llm_shim._execute_openai_chat`；异常原样上抛（4xx 不可重试判定依赖异常契约）。

  通过 `make_generate(secret_scope, semantic_cfg)` 工厂创建，scope 决定 secrets 读取
  与训练样本归属（llma/llmb），避免在共用层暴露云端配置。
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Callable

from sbmachine import cloud_cache
from sbmachine.common import _output_cap
from sbmachine.llm_shim import _ApiChatResult, _execute_openai_chat

_DEFAULT_BUDGET_ROUNDS = 12  # 对齐一个半场；max_rounds=0（禁用会话）时预算仍按此兜底
_sessions: dict[str, "MatchConversation"] = {}
_sessions_lock = threading.Lock()  # 保护 _sessions 的并发创建（回合级并发下多线程首调）

# 阶段 4：source_rounds 跟踪（仅 cloud_cache_enabled 时维护）。
# 按 (scope, run_id) 记录已验收轮（round/scene/user/assistant），其 sha256 参与缓存键，
# 使 llmb 会话历史变化自动使键失效；llma 无状态路径不记录（键恒用 ""）。
_cache_tracking_enabled = False
_source_round_entries: dict[tuple[str, str], list[dict]] = {}
_source_round_entries_lock = threading.Lock()


def _set_cache_tracking(enabled: bool) -> None:
    global _cache_tracking_enabled
    _cache_tracking_enabled = bool(enabled)


def _record_source_round(scope: str, run_id: str, entry: dict) -> None:
    with _source_round_entries_lock:
        entries = _source_round_entries.setdefault((scope, run_id), [])
        entries.append(dict(entry))
        encoded = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cloud_cache.set_source_rounds(scope, run_id, hashlib.sha256(encoded.encode("utf-8")).hexdigest())


class MatchConversation:
    """单场比赛（run_id）的会话历史与预算状态。"""

    def __init__(
        self,
        *,
        run_id: str,
        scope: str,
        max_rounds: int = 12,
        max_tokens: int = 0,
        budget_per_match: int = 0,
        budget_factor: float = 2.0,
        budget_enabled: bool = False,
    ) -> None:
        self.run_id = run_id
        self.scope = scope
        self.max_rounds = max(0, int(max_rounds))
        self.max_tokens = max(0, int(max_tokens))
        # 预算护栏默认关闭（预算已放开）：仅 cloud_token_budget_enabled=true 时启用。
        self.budget_enabled = bool(budget_enabled)
        self.budget_per_match = max(0, int(budget_per_match))
        self.budget_factor = float(budget_factor) if float(budget_factor) > 0 else 2.0
        self.rounds: list[dict] = []  # {"round","scene","user","assistant"}，最早在前
        self.usage_total = 0
        self._lock = threading.Lock()  # 回合级并发下保护 rounds/usage 的读写一致性
        self._seen_windows = 0
        self._mean_tokens = 0.0
        self._round_windows: dict[str, int] = {}
        self._round_order: list[str] = []
        self._budget_locked = False
        self._budget_total = 0

    # ── 会话历史 ──

    def messages(self, system_prompt: str | None, current_user: str) -> list[dict]:
        with self._lock:
            history = list(self.rounds)
        msgs = [{"role": "system", "content": system_prompt or ""}]
        for entry in history:
            msgs.append({"role": "user", "content": entry["user"]})
            msgs.append({"role": "assistant", "content": entry["assistant"]})
        msgs.append({"role": "user", "content": current_user})
        return msgs

    def add_round(self, user: str, assistant: str, log_ctx: dict | None) -> None:
        if self.max_rounds <= 0:
            return
        round_label = str((log_ctx or {}).get("round") or "round0")
        scene_label = str((log_ctx or {}).get("scene") or "win0")
        entry = {
            "round": round_label,
            "scene": scene_label,
            "user": user,
            "assistant": assistant,
        }
        with self._lock:
            for index, existing in enumerate(self.rounds):
                if existing["round"] == round_label and existing["scene"] == scene_label:
                    self.rounds[index] = entry  # 同窗口重试：原位替换，不重复累积
                    break
            else:
                self.rounds.append(entry)
            while len(self.rounds) > self.max_rounds:
                self.rounds.pop(0)  # 超限裁剪最早轮
            if self.max_tokens > 0:
                while self.rounds and self._input_tokens() > self.max_tokens:
                    self.rounds.pop(0)  # 会话输入 token 上限兜底

    def _input_tokens(self) -> int:
        return sum(
            self._estimate_tokens(entry["user"]) + self._estimate_tokens(entry["assistant"])
            for entry in self.rounds
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # 中英混合粗估：2 字符 ≈ 1 token（保守），保证不低估
        return max(1, len(text) // 2)

    # ── 成本护栏 ──

    def observe_round_start(self, log_ctx: dict | None) -> None:
        """调用观察：按回合计数窗口，供动态预算锁定使用。"""
        round_label = str((log_ctx or {}).get("round") or "round0")
        with self._lock:
            if round_label not in self._round_windows:
                self._round_order.append(round_label)
                self._round_windows[round_label] = 0
            self._round_windows[round_label] += 1

    def record_usage(self, usage: object, log_ctx: dict | None) -> None:
        """仅累计调用 token（预算护栏）；不写会话历史。

        会话历史由调用方在业务校验通过后显式 commit_round 提交，
        避免失败轮（如 HTTP 200 但空稿/无效 JSON）污染历史诱导后续窗口。"""
        del log_ctx
        total = 0
        if isinstance(usage, dict):
            total = int(usage.get("total_tokens") or 0)
            if total <= 0:
                total = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        with self._lock:
            self.usage_total += max(0, total)
            self._seen_windows += 1
            if self._seen_windows > 0:
                self._mean_tokens = self.usage_total / self._seen_windows
        self._try_lock_budget()

    def _try_lock_budget(self) -> None:
        if not self.budget_enabled or self.budget_per_match > 0 or self._budget_locked:
            return
        with self._lock:
            if not self.budget_enabled or self.budget_per_match > 0 or self._budget_locked:
                return
            # 回合级并发下"第二回合首次出现"时，回合窗口可能尚未全部完成：
            # 过早锁定会让首回合窗口数低估预算 → 提前 silence 掉后续窗口。
            # 延迟到已见窗口覆盖每回合平均 ≥2 窗（seen >= 2×回合数）再锁定，
            # 并用已见回合的最大窗口数估算（并发下首个回合计数未必完整）。
            if (len(self._round_order) < 2
                    or self._seen_windows < 2 * len(self._round_order)
                    or self._mean_tokens <= 0):
                return
            windows_per_round = max(self._round_windows.values()) if self._round_windows else 0
            if windows_per_round <= 0:
                return
            rounds_cap = self.max_rounds if self.max_rounds > 0 else _DEFAULT_BUDGET_ROUNDS
            self._budget_total = (
                windows_per_round * rounds_cap * self._mean_tokens * self.budget_factor
            )
            self._budget_locked = True

    def budget_exceeded(self) -> bool:
        if not self.budget_enabled:
            return False  # 预算已放开：默认不做 token 限额，永不 silence
        with self._lock:
            if self.budget_per_match > 0:
                return self.usage_total > self.budget_per_match
            return self._budget_locked and self.usage_total > self._budget_total

    def silence_result(self, log_ctx: dict | None) -> _ApiChatResult:
        """预算超限的静默响应：phase3a 解析为空 neutral（静默跳过），phase3b 解析为
        contract-leak（本地零成本重试后窗口静默）。均不发起真实请求。"""
        if self.scope == "llmb":
            content = '{"commentary": "", "felt_intensity": 0}'
        else:
            content = '{"neutral": ""}'
        return _ApiChatResult(
            content,
            scope=self.scope,
            source_run_id=self.run_id,
            request_payload={},
            log_ctx=dict(log_ctx or {}),
            raw_response={
                "choices": [{"message": {"content": content, "role": "assistant"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 0},
            },
            finish_reason="stop",
            http_status=200,
            usage={"total_tokens": 0},
            budget_silence=True,
        )


def commit_round(response: object, assistant_text: str | None = None) -> None:
    """业务校验通过后，把一轮 user/assistant 显式写入会话历史。

    必须在业务验收（如 phase3a 的严格 JSON 校验 / phase3b 的 validation）成功
    后才调用；失败轮不入历史，避免空稿/无效稿污染后续窗口的上下文。
    同窗口重试：按 round+scene 原位替换（与 add_round 语义一致）。"""
    if not isinstance(response, _ApiChatResult):
        return
    # 阶段 4：验收成功路径 —— 先把缓存 pending 转正（与是否建会话无关，
    # llma 无状态路径同样在此确认）。
    cloud_cache.confirm_cached(response)
    log_ctx = getattr(response, "log_ctx", None)
    run_id = str((log_ctx or {}).get("run_id") or getattr(response, "source_run_id", None) or "default")
    conversation = _sessions.get(run_id)
    payload = getattr(response, "request_payload", None)
    if not isinstance(payload, dict):
        return
    messages = payload.get("messages")
    user = ""
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
                user = message["content"]
                break
    if not user:
        return
    assistant = str(response) if assistant_text is None else assistant_text
    if conversation is not None:
        conversation.add_round(user, assistant, log_ctx)
    # 阶段 4：把本次验收轮计入 source_rounds（llmb 无论是否建会话都记，
    # 使缓存重跑时历史增长与首跑一致；llma 仅带会话的 A/B 模式记）。
    if _cache_tracking_enabled and (conversation is not None or str(getattr(response, "scope", "") or "") == "llmb"):
        scope_label = str(getattr(conversation, "scope", "") or getattr(response, "scope", "") or "llma")
        _record_source_round(scope_label, run_id, {
            "round": str((log_ctx or {}).get("round") or "round0"),
            "scene": str((log_ctx or {}).get("scene") or "win0"),
            "user": user,
            "assistant": assistant,
        })


def make_generate(secret_scope: str, semantic_cfg: dict | None = None) -> Callable:
    """创建签名与 llma_api.generate / llmb_api.generate 兼容的云端 generate。

    secret_scope 决定 secrets 读取与训练样本归属（"llma"/"llmb"）。
    semantic_cfg 读取 cloud_* 云端特化键；缺省用安全默认值。

    会话按 scope 分流：Phase3a（llma）无上下文需求，走**无状态路径**
    （不建会话、不入历史、零锁竞争，可自由并发）；Phase3b（llmb）保留
    会话历史（`cloud_conversation_max_rounds`）。Phase3a 专用键
    `cloud_analyst_conversation_max_rounds`（默认 0=无会话）可显式覆盖。
    """
    scope = secret_scope if secret_scope in {"llma", "llmb", "llmc"} else "llma"
    semantic = semantic_cfg if isinstance(semantic_cfg, dict) else {}
    cloud_model = str(semantic.get("cloud_model") or "").strip()
    if scope in {"llma", "llmc"}:
        # 3a 无状态：默认 0（无会话）；显式配置 >0 才启用（仅用于特殊 A/B）。
        conv_max_rounds = int(semantic.get("cloud_analyst_conversation_max_rounds", 0) or 0)
    else:
        conv_max_rounds = int(semantic.get("cloud_conversation_max_rounds", 12) or 0)
    # 注意：不能命名为 max_tokens —— generate 的参数同名会遮蔽本闭包变量。
    conv_max_tokens = int(semantic.get("cloud_conversation_max_tokens", 0) or 0)
    budget_per_match = int(semantic.get("cloud_token_budget_per_match", 0) or 0)
    budget_factor = float(semantic.get("cloud_token_budget_factor", 2.0) or 2.0)
    # 预算护栏默认关闭：仅显式 cloud_token_budget_enabled=true 启用。
    budget_enabled = bool(semantic.get("cloud_token_budget_enabled", False))
    # 阶段 4：成功响应缓存默认关闭；仅显式 cloud_cache_enabled=true 时包装请求路径。
    cache_enabled = bool(semantic.get("cloud_cache_enabled", False))

    def generate(
        prompt: str,
        llm_cfg: dict,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        log_ctx: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        # 3a 无会话（或会话/预算全禁用）时直接无状态请求：不建会话、不碰锁。
        stateless = conv_max_rounds <= 0 and not budget_enabled
        conversation = None
        if not stateless:
            run_id = str((log_ctx or {}).get("run_id") or "default")
            conversation = _sessions.get(run_id)
            if conversation is None:
                with _sessions_lock:
                    conversation = _sessions.get(run_id)
                    if conversation is None:
                        conversation = MatchConversation(
                            run_id=run_id,
                            scope=scope,
                            max_rounds=conv_max_rounds,
                            max_tokens=conv_max_tokens,
                            budget_per_match=budget_per_match,
                            budget_factor=budget_factor,
                            budget_enabled=budget_enabled,
                        )
                        _sessions[run_id] = conversation
            conversation.observe_round_start(log_ctx)
            if conversation.budget_exceeded():
                return conversation.silence_result(log_ctx)
        effective_cfg = dict(llm_cfg or {})
        if cloud_model:
            effective_cfg["model"] = cloud_model
        # 阶段 1 云端护栏键透传：总时限、scope 并发、队列等待。
        for cfg_key, semantic_key in (
            ("total_timeout_sec", "cloud_total_timeout_sec"),
            ("cloud_request_concurrency", "cloud_request_concurrency"),
            ("cloud_queue_timeout_sec", "cloud_queue_timeout_sec"),
        ):
            if semantic.get(semantic_key) is not None:
                effective_cfg[cfg_key] = semantic[semantic_key]
        if conversation is None:
            messages = [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = conversation.messages(system_prompt, prompt)
        cap = _output_cap(effective_cfg, max_tokens)
        raw = _execute_openai_chat(
            messages,
            effective_cfg,
            max_tokens=cap,
            log_ctx=log_ctx,
            secret_scope=scope,
            response_format=response_format,
        )
        if conversation is not None:
            conversation.record_usage(getattr(raw, "usage", None), log_ctx)
        return raw

    if cache_enabled:
        _set_cache_tracking(True)
        return cloud_cache.make_cached_generate(scope, semantic, generate)
    return generate


def confirm_cache(response: object) -> None:
    """业务验收通过后的缓存确认入口（透传 cloud_cache.confirm_cached）。

    正常链路由 commit_round 在验收成功路径自动触发；独立调用方
    （如 run 编排层在 accept_api_response 成功路径）可直接调用。"""
    cloud_cache.confirm_cached(response)


def cleanup_pending_cache(run_id: str | None = None, semantic_cfg: dict | None = None) -> int:
    """运行结束清理未确认的 pending 缓存条目（透传 cloud_cache.cleanup_pending）。"""
    return cloud_cache.cleanup_pending(run_id, semantic_cfg)


def clear_sessions() -> None:
    """清空会话注册表（测试隔离用）。"""
    _sessions.clear()
    global _cache_tracking_enabled
    _cache_tracking_enabled = False
    with _source_round_entries_lock:
        _source_round_entries.clear()
    cloud_cache.clear_registry()
