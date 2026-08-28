> 基于代码核对(2026-08-20),仅记录代码/配置/测试已确认的事实,运行行为以源码为准。

# 安装 / 配置 / 运行 / 排查

## 1. 唯一入口 run.py

```bash
python run.py                  # 读 config/(默认),终端 GUI 模式
python run.py --config config/ # 显式指定配置目录或文件
python run.py --dry-run        # 纯读预检,不调 AI
python run.py --debug          # 调试模式:透传所有 print,末尾输出原始 JSON(机器可解析)
```

参数解析见 `run.py:51-59`;`run_all(config_path, dry_run=...)` 为唯一编排入口(`--debug` 直调 `run.py:64`,默认模式经 `display.run_with_display` 调 `run.py:74`)。正式运行退出码取 `exit_code`(0 成功 / 1 阶段失败 / 2 配置或 preflight 失败 / 3 运行锁被占),dry-run 按 `config_valid` 返回 0 或 2(`run.py:68-70`、`display.py:452/466`)。

默认模式渲染 rich 终端 GUI(`sbmachine/display.py`),**stdout 不再是可解析的 JSON**;只有 `--debug` 会把结果字典以 JSON 打印到 stdout(`run.py:67`)。解析 stdout 的脚本须改用 `--debug`。`--debug` 同时经环境变量 `AI6657_DEBUG_PHASE3` 打开 Phase3 调试落盘(`run_all.py:198-200`),替代了原先的 `debug.phase3` 配置项。

`--dry-run` 的真实语义(`run_all.py:160-182`):仅 `load_config` + `preflight_config` 后立即返回报告。**不创建 RunContext、不取运行锁、不启动任何服务/容器、不调 gpu_guard、零写盘**。preflight 只做:正数字段校验(`preflight.py:58-67`)、`semantic.window_min_sec ≤ window_max_sec`(`:143`)、按启用阶段列出并校验必需输入是否存在(`_required_inputs` `:69-131`)、phase2 启用而 demo_parse 未启用时校验已存在 parsed demo(`:151-158`)。

## 2. 配置文件

`config/` 下三个 YAML,按文件名字母序合并(见 architecture.md 第 6 节),训练/工具专用配置在 `tools_config/`(`train.yaml`、`audio.yaml`),不参与调用端合并:

| 文件 | 用途 |
|---|---|
| `config/pipeline.yaml` | 阶段开关、服务拓扑(runtime)、主路径、Phase4 参数、帧分类与粗切参数(slicer) |
| `config/llm.yaml` | Phase3 模型、后端、窗口与生成参数(semantic/llm);方案 R 恢复配置 `semantic.recovery.enabled`(bool,仓库默认生产值 true；字段缺失或显式 false 时为零容忍)与 `semantic.recovery.max_retries`(int,默认 3);Phase3b 预算弹性 `semantic.style_budget_hard_tolerance`(float,默认 0.5);K 配额 `semantic.style_k_enabled`(bool,默认 false);强事实依据模式 `semantic.strong_fact_mode`(bool,默认 false=全面相信 LLM,统一控制 B0 unexpected_fact 与 C3 事实作用域,空稿/情绪/预算硬线不受影响);语速配置 `semantic.speech_rate`(base_char_per_sec + char_budget_factor + tts_speed_factor,可由 `hype_rules.json` 降级);Phase3a 生成器 `semantic.phase3a_generator`（当前仓库配置为 `legacy_llma`，`rule_template` 为另一模式;tiny 候选块默认 enabled=false、shadow_only=true);配音任务单 `semantic.voice_task`(enabled 默认 false、candidate_policy=sparse_v1、speech_profile_id=speech-profile-v1、require_validated_profile=true);LLM-C 阶段 `semantic.phase3c`（当前仓库配置为 `optional`，可选 `off/shadow/optional/required`；**须引号包裹如 `mode: "off"` 防止 YAML 解析为布尔 False**;temperature=0.6;max_retries=2;render_timebase_fps=30) |
| `config/yolo.yaml` | demo、YOLO、OCR、时间轴采样参数 |
| `tools_config/audio.yaml` | ASR、说话人与音频工具参数(`tools/audio/` 专用,**不属推理主链**;文件头有隔离注释) |
| `tools_config/train.yaml` | 训练 profile、数据集、输出目录(不参与 `config/` 目录模式合并) |



原 `config/slicer.yaml` 已并入 `pipeline.yaml` 的 `slicer:` 节(`pipeline.yaml:59-66`,命名空间 `slicer.*` 不变,读取方无感知);原 `config/tts.yaml` 已删除,唯一有效字段迁为 `phase4.tts_config`(`pipeline.yaml:56`)。

### 2.1 Phase4 维护键

| 键 | 维护含义 |
|---|---|
| `phase4.publish_profile` | `legacy`、`strict_av`、`strict_c`、`broadcast` 的发布门禁 |
| `phase4.clip_mode` | 严格 profile 必须为 `strict_decode` |
| `phase4.media_probe` | FFmpeg/FFprobe 路径与边界探测开关 |
| `phase4.media_tolerances` | 视频帧边界、音频边界与字幕越界容差 |
| `phase4.tts_config` | GPT-SoVITS 唯一运行配置入口 |

严格 Phase4 依赖 `commentary_render_package_v2`、源视频、FFmpeg/FFprobe 和 TTS 配置。`media_sync_status=not_checked` 不能发布为严格结果。`max_frame_boundary_error_sec` 当前仍是固定秒值，输入帧率变化时按 [`known_discrepancies.md`](known_discrepancies.md) E13 复核。

### 2.2 语音时长 profile 标定（speech_units_v1）

commentary v3 的风险分级与候选安全上界只认 `validated` profile（`data/speech_profiles/<profile_id>/profile.json`），**不得用 `base_char_per_sec=5.0` 冒充分级依据**。离线标定入口：

```bash
python tools/calibrate_speech_profile.py \
  --manifest data/speech_profiles/calibration_manifest.jsonl \
  --profile-id speech-profile-v1 \
  --out data/speech_profiles/speech-profile-v1 \
  --engine-fingerprint <engine> --voice-fingerprint <voice> --preprocess-fingerprint <pre>
```

- 独立文本最低 160 条、目标 200 条（60/20/20 拟合/校准/留出；近重复不跨集）；另抽 20-30 条 1.25/1.5 速度验证子集。
- 不足 160 条只能产出 `status=exploration`；留出覆盖率 ≥95%、关键标签 ≥90%、最大低估 ≤0.20s、中位过估 ≤25%、速度缩放误差 P95 ≤5% 全部满足才写 `validated`。
- profile 不可变：修改 voice/参考音频/引擎/采样率/预处理或情绪策略必须生成新 `profile_id`；Phase4 累计 50 条真实结果后离线比较，漂移则旧 profile 转 `stale`（只影响后续 run，不回写当前 run）。

## 3. .env 与后端选择

后端解析优先级(`common.py:resolve_backend` `:95-100`):
`环境变量 AI6657_<STAGE>_BACKEND` > `semantic.<stage>_backend` > `llm.backend` > 默认 `"vllm"`。`<stage>` 为 `analyst`(Phase3a)或 `style`(Phase3b)。**backend 环境变量只读真实进程环境,不从 .env 切换**。

API 连接信息从根目录 `.env` 读取(模板 `.env.example`),规则:

- 进程环境变量优先于 `.env`。
- 新密钥合同按 `profile → scope → fallback` 解析(`llm_protocol._load_secrets`):
  `AI6657_CLOUD_<SCOPE>_*` > `AI6657_CLOUD_*`(通用) > 旧 `AI6657_<SCOPE>_*` > 旧 `AI6657_*`(仅 cloud 回退);
  scope 为 `LLMA`/`LLMB`/`VLM`。本地回环端点用 `AI6657_LOCAL_BASE_URL`(缺省 `http://127.0.0.1:8000/v1`),允许占位 key。
- 旧 `AI6657_LLMA_*`/`AI6657_LLMB_*`/`AI6657_VLM_*` 在迁移窗口仍生效,但 `_load_secrets` 会给出迁移提示(进 `warnings` 字段)。
- 远程端点必须提供 API key;本机回环端点允许占位 key。
- 不得提交真实 `.env` 或 API 调试日志。

模型原生 reasoning 保留（不做任何 thinking 调配）；所有后端统一剥离响应中的 `<think>…</think>` 与 `reasoning_content`，只取 content 供业务验收。

**云端 API 与本地 vLLM 的请求差异**（`llm_shim._execute_openai_chat` 按 `_is_loopback_url` 分流到 `cloud_adapter.cloud_generate` / `local_adapter.local_generate`）:
- **节流**:请求间隔锁 `request_interval_sec` 仅对本机回环端点生效;云端官方 API 不节流(自带服务端配额管理)。
- **流式**:云端自动 `stream:true` + `stream_options.include_usage`,SSE 聚合 content/reasoning/finish_reason/usage 后按原 `_ApiChatResult` 契约返回;本地 vLLM 保持非流式。
- **max_tokens**:云端预算由 `semantic.cloud_*_output_max_tokens` 放开,不被本地字数公式截断;`style_runtime_config` 透传 `cloud_style_output_max_tokens` 供 phase3b 读取。
- **护栏**(速度优化计划 §阶段1):`total_timeout_sec` 为 SSE 硬总时限(超时按可重试基础设施错误中断);`cloud_request_concurrency` 为 scope 级信号量上限、`cloud_queue_timeout_sec` 为排队上限。
- **并发**(§阶段2/3):当前仓库配置为 `analyst_window_concurrency=7`、`style_concurrent_scenes=6`；代码缺省按后端降级，API 为 4、本地 vLLM 为 1。两者均按顺序验收以保持产物确定性。
- **缓存**(§阶段4):当前仓库配置 `cloud_cache_enabled=true`，仅缓存业务验收成功的响应(`cloud_cache.py`),键含 scope/model/endpoint/prompt 哈希；关闭时直连。

## 4. 正式输出目录约定

正式运行先取全局运行锁 `output/.run.lock`(`run_all.py:187`),再使用事务目录:

```text
output/.staging/<run_id>/   # 本轮过程产物与有效配置
output/sbmachine/           # 只保存最近一次完整成功结果
output/demo/                # 只保存最近一次完整成功的 demo 投影
output/error/<run_id>/      # 失败运行、诊断与 failure.json
```

- 全局锁:第二个并发运行立即失败(`failed_stage:"lock"`、`exit_code:3`,`run_all.py:219-228`)。
- Checkpoint 复用:demo_parse / video_marking / phase1 / phase2 各自通过契约后原子提升到正式目录(`run_context.py:174-192`);后续阶段失败时,已 checkpoint 的上游产物保留,失败结果 `checkpointed_stages` 列出已提交阶段,可在关闭已完成阶段后复用。
- Checkpoint 失效:`prepare` 时按启用阶段找到最靠前启用阶段,删除该阶段及其全部下游工件与旧 `run_manifest.json`(`run_context.py:83-100`),避免重跑上游后新旧事实混合发布。
- 发布判据:`output/sbmachine/run_manifest.json` 的 `status:complete` 且 `publishable:true`(`run_context.py:208-213`)。单阶段 CLI 由 `require_debug_output` 阻止覆盖正式目录(`common.py:33-44`)。
- 进度诊断:默认 Rich 只读取回调快照；受管 Phase2/3/4 与视频标记可在 `output/.staging/<run_id>/diagnostics/progress/` 写入独占 JSONL。它们是瞬态 UI/诊断数据，不进入 `output/sbmachine/*.json` 或 checkpoint。通道读取失败、错误 run ID、乱序或畸形 JSON 均不会使运行失败，UI 仅退化为不定进度；失败运行迁移到 `output/error/<run_id>/diagnostics/` 后可供排查。

## 5. 排查顺序

遇到运行问题,依 [docs/order.md](order.md) 顺序检查:

```text
配置合并 → 输入文件与 manifest → demo_parse
→ video_marking / Phase2 → VLM/LLM 服务健康状态
→ Phase3 schema 契约 → Phase4 TTS 与音视频输出
```

对应代码定位:配置合并 `core/config_loader.py`;输入/契约校验 `sbmachine/preflight.py`;服务健康 `service_manager.py:_poll_health/_health_url`(`:52/72`)、`compose_manager.py:up_one`(`:41`);失败诊断落 `output/error/<run_id>/failure.json`。失败运行也可用 MCP 只读工具查询(`mcp/server.py`)。

## 已知偏差

README/记忆与代码现状的冲突不在本文展开,统一见 [docs/known_discrepancies.md](known_discrepancies.md)。
