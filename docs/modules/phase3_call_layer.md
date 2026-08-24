# Phase3 调用层：协议核心 / cloud-local adapter / 请求护栏 / 并发 / 缓存

> 基于代码核对(2026-08-16),仅记录代码/配置/测试已确认的事实,运行行为以源码为准。
> 本文对应当前源码中的调用层实现；历史整洁迁移/速度优化计划不在当前 Git 跟踪内容中，不作为仓库内链接。

## 1. 分层总览（迁移计划书 §2）

```text
Phase3 Core（云端/本地共用）
   schemas / neutral_contract / commentary_planner / scene_context / llm_projection /
   tactic_* / hype_score / emotion_policy / phase3b_prompt
                    │
                    ▼
LLM Protocol Core（llm_protocol.py）
   请求/响应 DTO（_ApiChatResult）、_load_secrets 密钥合同、_dump_api_log 脱敏日志、
   错误分类（_request_error_category）、SSE 归一（_consume_sse_response）、
   传输重试（_post_openai_with_retry）、_build_chat_payload / _finalize_chat_result
           │                          │
           ▼                          ▼
Cloud Adapter（cloud_adapter.py）   Local Adapter（local_adapter.py）
   cloud_generate：远端 SSE/总时限/    local_generate：回环/非流式/
   信号量/in_flight/probe              request_interval 节流
                    │
                    ▼
llm_shim.py（薄兼容入口）
   re-export 全部符号 + _execute_openai_chat 按 _is_loopback_url 分流
```

## 2. 模块边界（迁移计划书 §3.1）

| 模块 | 职责 | 关键符号 |
|---|---|---|
| `llm_protocol.py` | 共用协议层：DTO、密钥、日志、校验、错误分类、SSE 归一、传输重试、payload 构建 | `_ApiChatResult`、`_load_secrets`、`_dump_api_log`、`record_validation_reason`、`_request_error_category`、`_consume_sse_response`、`_post_openai_with_retry`、`_build_chat_payload`、`_finalize_chat_result` |
| `cloud_adapter.py` | 云端私有：SSE 流式、`total_timeout_sec` 硬总时限、`cloud_request_concurrency` 信号量、`cloud_queue_timeout_sec` 排队上限、`probe_api_connectivity` | `cloud_generate`、`_scope_semaphore`、`_in_flight_start/_end` |
| `local_adapter.py` | 本地私有：非流式、`request_interval_sec` 节流 | `local_generate`、`_request_throttle` |
| `llm_shim.py` | 薄兼容入口：re-export 全部符号 + `_execute_openai_chat` 按 loopback 分流 | `_execute_openai_chat` |

既有调用方（`phase3a_analyst`、`phase3b_style`、`cloud_memory`、`cloud_cache`、`llma_api`、`llmb_api`、`llm_backends`）继续 `from sbmachine.llm_shim import ...` 零改动。

## 3. 密钥合同（_load_secrets，迁移计划书 §4.3）

解析优先级 `profile → scope → fallback`：

```text
AI6657_CLOUD_<SCOPE>_* > AI6657_CLOUD_* > 旧 AI6657_<SCOPE>_* > 旧 AI6657_*（仅 cloud 回退）
scope ∈ {LLMA, LLMB, VLM}
AI6657_LOCAL_BASE_URL（缺省 http://127.0.0.1:8000/v1，允许占位 key）
```

- 返回结构含 `api_key/base_url/model`（cloud 通用）、`llma/llmb/vlm`（scope）、`local`、`warnings`（旧 scoped 键迁移提示）。
- 进程环境变量优先于 `.env`；远程端点必须提供 key，回环允许占位。
- 模板 `.env.example`；`docs/operations.md` §3 已同步。

## 4. 请求护栏（速度优化计划 §阶段1）

| 配置键（semantic 段） | 语义 | 缺省 |
|---|---|---|
| `connect_timeout_sec` | 连接超时 | 回退 `timeout_sec` |
| `read_idle_timeout_sec` | SSE 读空闲超时 | 回退 `timeout_sec` |
| `total_timeout_sec` | SSE 硬总时限，超时抛 `requests.Timeout`（可重试基础设施错误） | 不启用 |
| `cloud_request_concurrency` | scope 级有界信号量（llma/llmb/llmc 逻辑域独立） | 当前配置 6；代码缺省 0=不限 |
| `cloud_queue_timeout_sec` | 信号量排队等待上限（超时抛 Timeout） | 0=不限 |

- 不做任何 thinking 调配：模型原生 reasoning 保留，响应统一剥离 `<think>`/`reasoning_content` 只取 content。
- 诊断字段（§阶段0）：`request_id`、`model`、`streaming`、`connect_ms`、`ttfb_ms`、`in_flight`、`queue_ms`、`finish_reason`、`retry_category`、`usage_tokens`、`validation_reason`（`record_validation_reason` 业务验收失败后补记）。

## 5. 并发（速度优化计划 §阶段2/3）

| 配置键 | 阶段 | 语义 | 缺省 |
|---|---|---|---|
| `analyst_window_concurrency` | 3 | LLM-A 两阶段：规则预计算（`_WindowRequest`）→ 窗口请求并发 → 按 window_id 顺序验收 | 当前配置 7；代码缺省 API=4、本地 vLLM=1 |
| `style_concurrent_scenes` | 2 | LLM-B 三段式：确定性预计算（`_StyleWindowPlan`）→ 滑动窗口并发 → 主线程顺序验收/更新短语 | 当前配置 6；代码缺省 API=4、本地 vLLM=1 |
| `cloud_conversation_max_rounds` | 2 | LLM-B 云端会话历史上限；当前配置 6，0=无会话 | 代码缺省 0 |

- 结果验收/`recent_style_phrases`/惊叹配额/`commit_round` 全部在主线程按时间顺序执行；worker 只返回候选，不写共享结构。
- 回合末兜底重试仅当窗口从未有效重试或失败为可恢复基础设施错误时执行。

## 6. 成功响应缓存（速度优化计划 §阶段4，`cloud_cache.py`）

- 配置：当前仓库 `cloud_cache_enabled=true`；`cloud_cache_dir` 支持 `{run_id}` 占位，另有 `cloud_cache_pending_ttl_sec`、`cloud_source_rounds_sha256`。
- 键：`cache_version + scope + model + endpoint + prompt_hash + system_prompt_hash + generation_config + source_rounds_sha256`；Prompt 原文不落盘。
- 生命周期：HTTP 200 → 写 `pending` → 业务验收成功（`cloud_memory.commit_round` 路径）`confirm_cached` 转正 → 未确认由 TTL/`cleanup_pending_cache` 清理。
- 命中返回 `_ApiChatResult` 同构对象（`cache_hit=True`），验收器照常运行；缓存版本/损坏/键不符自动回退网络请求。

## 7. 后端选择（唯一开关）

- 无 profile 覆盖层：`config/llm.yaml` 单份直白配置，`semantic.analyst_backend`/`style_backend` 即唯一后端开关（cloud=api / local=vllm）。
- cloud 后端不启动本地 talk 服务（`run_all._phase3_local_service_name` 对 api backend 返回 None）；local 后端不读远端 key/SSE/cloud session。

## 8. 验证方法

```bash
python -m pytest tests/unit/test_llm_shim.py -q
python -m pytest tests/unit/test_env_secrets.py -q
python -m pytest tests/unit/test_cloud_cache.py -q
python -m pytest tests/unit/test_config_profile.py -q
python -m pytest tests/unit/test_phase3a_concurrency.py -q
python -m pytest tests/unit/test_phase3b_concurrency.py -q
python -m compileall -q sbmachine core tools tests
```

## 9. 已知偏差

- `tests/unit/test_production_gates.py` 的历史 flaky 原因尚未由当前代码证实；该文件明确不使用随机采样，运行结果应以实际复测为准。
- 计划书步骤 3/5（`phase3a_core.py`/`phase3b_core.py` 物理拆分、Prompt 目录移动）未执行；相关历史计划与报告不在当前 Git 跟踪内容中，不能作为仓库内链接或当前验收证据。
