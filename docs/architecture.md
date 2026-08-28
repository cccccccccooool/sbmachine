> 基于代码核对(2026-08-24),仅记录代码/配置/测试已确认的事实,运行行为以源码为准。

# 系统架构总览

## 1. 定位与数据流

以 CS2 demo 为唯一硬事实源,由确定性规则完成场景/动作/选题,再由当前配置的 `legacy_llma` 或可选 `rule_template` 生成中性稿、LLM-B 写风格稿,最后经 GPT-SoVITS 生成逐局解说的离线流水线。唯一 CLI 入口为 `run.py`(`run.py:29`),它调用 `sbmachine/run_all.py:run_all()`(`run_all.py:158`)。

DFD 0 层数据流(代码支持的完整链路;阶段是否执行取决于开关):

```text
操作者 / config/ ── 控制阶段、路径、后端与服务生命周期
CS2 .dem ─> demo 解析 ─> output/demo/
比赛视频 ─> 帧分类/切片/回合对齐 ─> rounds.json
                 ▼
        Phase2:DEM 时间轴 + YOLO/OCR/POV ─> rounds_with_yolo.json
                 ▼
         Phase3a:规则切窗/规划 + 中性稿生成(当前 legacy_llma) ─> rounds_with_neutral.json
                  ▼
         Phase3b:LLM-B 风格化(稀疏候选任务单) ─> rounds_with_commentary.json + commentary.json
                  ├─> llmb_draft_package.json
                  ▼
         Phase3c:LLM-C 独立渲染交接 ─> commentary_render_package.json(v2)
                  ▼
         Phase4:GPT-SoVITS / PCM 画布 / FFmpeg ─> rounds_final.json + assemble_manifest.json
                                       + round_NNN.wav(可选 round_NNN.mp4)
```

阶段开关以 `config/pipeline.yaml` 为准：当前启用 `phase3c_render` 与 `phase4_assemble`，其余阶段关闭并复用既有 checkpoint。代码缺省与配置约束见 `sbmachine/preflight.py`。

## 2. 事实优先级

固定为:`DEM > 人工复核地图 > 确定性规则派生事实 > Tiny-LLM/LLM-B文本`。DEM 是唯一硬事实源,比分/击杀/C4/血量/胜负不得从画面、音频或模型文本推断；Tiny-LLM 与 LLM-B 都不能新增事实（`fact_catalog`/`required_fact_ids`/模板neutral/规则保底句均由规则层产生）。

## 3. 阶段总表

| 阶段 | 入口文件 | 输入 | 输出 | 模块文档 |
|---|---|---|---|---|
| demo 解析 | `tools/demo/parse_demo.py`(`upstream_jobs.py:26`) | `.dem` | `output/demo/`(rounds/kills/roster 等 + demo_manifest.json) | 见 [modules/phase2.md](modules/phase2.md) demo_query 查询层 |
| 视频标记 | `tools/slicing/run_frame_type_slicer.py`(`run_all.py:89`) | 视频、帧分类权重、可选 replay 权重、demo rounds | `detector_rows.jsonl`、`segments.json` | 见 [modules/phase2.md](modules/phase2.md) §4 回合对齐 |
| Phase1 预处理切片 | `sbmachine/phase1_preprocess_slice.py`(`run_all.py:17`) | 视频、segments 或 detections | `rounds.json`、`round_list.json`、`segments.json`、可选逐局 clip | 见 [modules/phase2.md](modules/phase2.md) §4 切片期对齐 |
| Phase2 时间轴/YOLO/POV | `sbmachine/phase2_yolo.py`(`run_all.py:18`) | `rounds.json`、视频、demo 工件、YOLO/OCR | `rounds_with_yolo.json`、`rounds_with_yolo_semantic.json` | [modules/phase2.md](modules/phase2.md) |
| Phase3a 中性稿 | `sbmachine/phase3a_analyst.py`(`run_all.py:19`) | `rounds_with_yolo.json` | `rounds_with_neutral.json`(默认 schema v4) | [modules/phase3a.md](modules/phase3a.md) |
| Phase3b 风格化 | `sbmachine/phase3b_style.py`(`run_all.py:20`) | neutral 与同源 yolo 输入 | `rounds_with_commentary.json`、`commentary.json`(v2/v3 双契约) | [modules/phase3b_phase4.md](modules/phase3b_phase4.md) |
| Phase3c LLM-C 渲染交接 | `sbmachine/phase3c_llmc.py`、`phase3c_cli.py` | `llmb_draft_package.json`（v1/v2 分流） | `commentary_render_package.json`（v1/v2） | [modules/phase3c.md](modules/phase3c.md) |
| Phase4 合成 | `sbmachine/phase4_assemble.py`、`phase4_av.py`、`phase4_media.py` | legacy commentary，或严格路径的 render package v2、TTS 配置与源视频 | `rounds_final.json`、`assemble_manifest.json`、逐局 WAV/MP4、严格切片 sidecar | [modules/phase3b_phase4.md](modules/phase3b_phase4.md) |

编排顺序见 `run_all.py:231-295`;各阶段发布契约见 `preflight.py:393-461`。

## 4. 事务模型

单次运行的输出走"暂存→原子提升"事务(`sbmachine/run_context.py`)。目录布局:

```text
output/.staging/<run_id>/   # 本轮 publish/ 与 diagnostics/、effective_config.yaml
output/sbmachine/           # 只保存最近一次成功结果
output/demo/                # 只保存最近一次成功的 demo 投影
output/error/<run_id>/      # 失败运行的诊断与 failure.json
```

`run_id = <UTC时间戳>-<uuid8>`(`run_context.py:55-56`)。

- `prepare`(`run_context.py:66`):建 `publish/`、`diagnostics/`;将现有 `output/demo`、`output/sbmachine` 复制进 `publish/` 作基线;按启用阶段失效最靠前启用阶段及其全部下游工件(`_invalidate_stale_outputs` `:83-100`);生成有效配置并把所有输出路径重写进 `publish/`,并**强制 `debug.phase3=False`**(`:135-136`)。
- `checkpoint`(`run_context.py:174`):每个已启用阶段都必须在 `execute → require_outputs（若有）→ validator → checkpoint → on_stage_done` 后才进入 UI 的绿色完成态；校验或 checkpoint 失败时不会先显示完成。checkpoint 原子提升已校验工件，同时保留暂存树供下游继续消费。
- `_promote_from`(`run_context.py:223-252`):先把已存在正式目录备份到 `diagnostics/previous_success/`,再 `os.replace` 提升新工件;任一步异常则逆序回滚新旧,成功后删备份(原子提升 + 回滚)。
- `complete`(`:208`)写 `run_manifest.json`(`status:complete`、`publishable:true`)后提升并清 staging;`fail`(`:254`)写 `failure.json` 并 `os.replace(staging→error/<run_id>)`,`checkpointed_stages` 列出已提交阶段。
- 运行锁:`run_all` 用 `FileLock(output/.run.lock)` 包裹整个执行(`run_all.py:187`);取锁失败返回 `failed_stage:"lock"`、`exit_code:3`(`:219-228`),即**第二个并发运行立即失败**。多容器 Phase4 另加 `output/.sovits.lock`(`:470`)。

## 5. 服务拓扑两模式

由 `runtime.manage_services` 二选一(`run_all.py:235`,变量 `single_container`)。**命名与直觉相反**:

- `manage_services: true` → `ServiceManager`(单环境进程托管,`service_manager.py`)。phase2/3/4 以**独立子进程**运行:`python -m sbmachine.phase_yolo / phase_semantic / phase_tts`(`run_all.py:331/351/378`)。phase3a+phase3b 合并在同一个 `phase_semantic` 子进程内。
- `manage_services: false` → `ComposeManager`(多容器错峰,`compose_manager.py`)。phase2/3a/3b/4 为**进程内直接函数调用** `run_phase2/run_phase3a/run_phase3b/run_phase4`(`run_all.py:419/441/448/471`)。
- **两模式都自动拉起/关闭服务**:ServiceManager 自 `subprocess.Popen` 起本地进程(且会复用用户手动已起的服务,`service_manager.py:132-136`);ComposeManager 恒 `docker compose up -d`(无外部复用逻辑)。`one_model_at_a_time`(默认 True,`run_all.py:317`)控制单容器模式下是否逐阶段独占显存;gpu_guard 仅 `use_gpu_guard=True` 时生效(默认 False)。

## 6. 运行期进度与跨进程事件

默认 Rich 界面只展示阶段已真实完成的工作单元：Phase2、Phase3a、Phase3b、Phase4 按 `round`，视频标记按计划采样 `frame`；Demo 与 Phase1 没有可靠分母时保持不定进度。`completed == total` 仅表示工作处理完毕，UI 显示“校验/提交中”，仅在编排器完成 validator 与 checkpoint 后才变为绿色完成。

回调 dict 保持旧 `on_stage_start/on_stage_done/on_error` 兼容，并新增可选 `on_stage_progress(stage, completed, total, unit, detail)` 与 `on_stage_canceled(stage, message)`；缺失新键或 `callbacks=None` 不改变业务路径。受管子进程写入各自的 `diagnostics/progress/<stage>.jsonl`，父进程校验 schema、run ID 和单 writer sequence 后转发；通道损坏、畸形行或错误事件仅记入诊断并退化为不定进度，绝不改变产物、checkpoint 或退出码。JSONL 与摘要不属于 Phase1～Phase4 正式 JSON，也不随 publish 目录发布。

受管 Phase3 的父进程只在真实开始时启动 Phase3a；子进程完成 3a 工作后才发送 Phase3b 的 start。两种服务拓扑因而共享 `phase3a work_complete < phase3b start` 语义；权威 done 仍由父编排器在最终门禁后产生。

## 6. 配置合并规则

`core/config_loader.py:load_config()`(`:104`):

- 传目录时按**文件名字母序**递归合并所有 `*.yaml`(`:127`);双方均为 dict 则递归,标量相等跳过。
- **冲突 = 不同文件对同一标量键给出不同值**(相等值不冲突),报 `ConfigError` 并给出点分键路径与冲突双方来源文件(`_merge_config` `:32-59`)。
- `train.yaml`/`audio.yaml` 已移入 `tools_config/`，不再参与 `config/` 目录合并（调用端不读取其段）。
- demo 路径归一(`_normalize_demo_path` `:82-102`):`paths.demo_output_dir` 与 `demo.parsed_dir` 解析为绝对路径后若都存在且不等则报错,否则取非空者并写回两键。
- 配置路径不存在、YAML 根非 mapping 均报 `ConfigError`(`:121/70-72`)。

### 6.1 后端选择（唯一开关）

- 无 profile 覆盖层：`config/llm.yaml` 是单份直白配置，`semantic.analyst_backend`/`style_backend`（或 `llm.backend`、环境变量 `AI6657_<STAGE>_BACKEND`）即唯一后端开关。当前仓库配置使用 `api`，Phase3a/3b 窗口并发为 `7/6`，云端请求并发为 `6`，成功响应缓存开启；本地 `vllm` 由代码按后端降为串行。

## 7. Phase3 调用层分层（整洁迁移，计划书 §2）

```text
Phase3 Core（规则/契约/验收/情绪/发布：schemas、neutral_contract、commentary_planner、
           scene_context、llm_projection、tactic_*、hype_score、emotion_policy、phase3b_prompt）
                    │
                    ▼
LLM Protocol Core（llm_protocol.py：DTO、_load_secrets、日志脱敏、错误分类、SSE 归一、传输重试）
           │                          │
           ▼                          ▼
  Cloud Adapter（cloud_adapter.py）   Local Adapter（local_adapter.py）
  远端 SSE/总时限/信号量/in_flight/   回环节流/非流式/本地 request_interval_sec
  probe
                    │
                    ▼
  llm_shim.py（薄兼容入口：re-export + _execute_openai_chat 按 _is_loopback_url 分流）
```

- 云端/本地各自的端点策略、Prompt、并发、会话、缓存与密钥归属已移入对应 adapter；`llm_shim.py` 仅保留薄入口供既有调用方零改动使用。
- 会话/缓存/预算：`cloud_memory.py`（会话+预算护栏）、`cloud_cache.py`（成功响应缓存）、`cloud_prompts.py`（云端 Prompt 组合），均只在 `backend=api` 路径被引用。
- 请求护栏（速度优化计划 §阶段1）：超时拆分 `connect_timeout_sec/read_idle_timeout_sec/total_timeout_sec`（SSE 硬总时限）、scope 级信号量 `cloud_request_concurrency`、队列上限 `cloud_queue_timeout_sec`。模型原生 reasoning 保留，响应统一剥离 `<think>`/`reasoning_content` 只取 content。

## 已知偏差

README/记忆与代码现状的冲突不在本文展开,统一见 [docs/known_discrepancies.md](known_discrepancies.md)。
