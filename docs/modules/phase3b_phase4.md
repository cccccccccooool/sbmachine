> 基于代码核对(2026-08-24),仅记录代码/配置/测试已确认的事实,运行行为以源码为准。

# Phase3b 风格化 与 Phase4 合成

本文档记录 `phase3b_style`(LLM-B 风格化)与 `phase4_assemble`(TTS + 逐局混流)两阶段的职责、接口、约束与验证方法,不逐行复述源码。

## Phase3b:LLM-B 风格化

> 从属声明:Phase3 的唯一权威设计是 [../phase3_commentary_final_plan.md](../phase3_commentary_final_plan.md)。本节只记录 Phase3b 的代码现状,不复活已被否决的分支(MatchMemory、历史 few-shot、跨局情绪状态、口癖 JSON 库、API/vLLM 两套 style prompt)。风格重复门禁已于代码移除(不再校验 `high_repetition`),残句门禁(不再校验 `incomplete_sentence`)亦已移除。

### 职责与入口

- 入口:`sbmachine/phase_semantic.py:22`,子进程 `python -m sbmachine.phase_semantic`,phase3a→phase3b 同进程顺序执行。
- 主逻辑:`sbmachine/phase3b_style.py:104` `run_phase3b()`,逐回合逐 scene 调 LLM-B、情绪定档、拼稿,产 `rounds_with_commentary.json` 与 `commentary.json`。
- 后端分流:`resolve_backend` 只允许 `api`/`vllm`(`phase3b_style.py:118-120`)。本地 `vllm` 走 `llmb_api.generate`(`llmb_api.py:64`)→ `llm_shim._execute_openai_chat`(分流到 `local_adapter.local_generate`);云端 `api` 走 `cloud_memory.make_generate`(`phase3b_style.py:239-241`)→ `cloud_adapter.cloud_generate`,system 拼装用 `cloud_prompts.build_cloud_style_system` 且 user prompt 注入窗口类型。HTTP 层对云端（非回环）自动启用流式（SSE）并把 `max_tokens` 放开到 `cloud_style_output_max_tokens`。云端会话历史仅在业务验收(validation)通过后 `cloud_memory.commit_round` 写入,失败轮/空稿不入历史；当前仓库配置 `cloud_conversation_max_rounds=6`，代码缺省为 0。
- **窗口级有界并发(速度优化计划 §阶段2)**:`style_concurrent_scenes` 控制三段式——按时间顺序预计算 `_StyleWindowPlan`(不可变计划)→ `ThreadPoolExecutor(max_workers)` 滑动窗口并发请求 → 主线程按原顺序验收/更新 `recent_style_phrases`/惊叹配额。当前仓库配置为 6；代码缺省按后端为 API=4、本地 vLLM=1。`dispatch_order/completion_order` 只进诊断。回合末兜底重试仅当窗口从未有效重试(`retry_count==0`)或最后失败为可恢复基础设施错误时执行。
- **请求护栏(§阶段1)**:`total_timeout_sec`/`cloud_request_concurrency`/`cloud_queue_timeout_sec` 经 `cloud_memory.make_generate` 透传。
- **成功响应缓存(§阶段4)**:`cloud_cache_enabled=true` 时仅缓存业务验收成功响应,`commit_round` 路径转正;命中结果照常过验收器。

### 1. LLM-B 输入集合与明确不读取项

user_prompt 是严格 JSON，包含：

- `neutral`：已经通过 Phase3A 契约的中性事实基线；
- `fact_anchors`：人物、阵营、数字、事件、结果、地点、武器七类允许事实；
- `delivery`：节奏、情绪上限、`target_chars` 与 `hard_char_limit`；
- `aliases`：仅限 neutral 中已出现人物的别名；
- `recent_style_phrases`：最近窗口的非事实风格残余签名；
- `allowed_event_terms`：本窗口事件可用的中文表达列表(由 `_EVENT_TERMS` 导出,按 anchors 中的 events/results 过滤),模型应从对应列表中选用至少一个词；
- 重试时追加结构化 `retry_feedback`。

system 三段拼接(`phase3b_style.py:145-151`):`style_system`(注入 persona)+ `Prompt/cs_rules.txt` + 可选 `style_skill`。

明确不读取：`commentary_plan`、DEM/VLM、原始事件账本、地图关系、历史解说库和绝对时间。最近风格残余只用于复读约束，不作为事实来源。

### 2. 目标响应与验收防线

目标格式 `{"commentary": "...", "felt_intensity": 0.xx}`,严格解析(`phase3b_style.py:58-80`):

- `_extract_json_obj` 只接受恰好一个 JSON 对象,代码围栏与前后散文一律拒绝(`phase3b_prompt.py:65-71`)。
- 键集必须恰好 `{"commentary","felt_intensity"}`,多/缺字段判 unparseable。
- 泄漏防线:commentary 须为非空 str、不含 `_LEAK_MARKERS`(```json/"scenes"/"t_start"/【中性稿】等)、不以 `{` 开头(`phase3b_prompt.py:57`;`phase3b_style.py:64-69`)。
- `felt_intensity` 须为有限 float∈[0,1] 且非 bool(`:72-77`)。
- 污染防线：命中任务说明、JSON 泄漏或提示词残片即拒绝。
- 事实防线：人物、阵营、数字、地点、武器、事件和结果必须受锚点约束，不得新增未授权事实。
- 预算防线：双层预算——软上限=``hard_char_limit`` (超了标记 ``budget_overage``,不阻断)；硬上限=``hard_char_limit×(1+tolerance)``(默认 1.5 倍)阻断,容忍刻支持超出的文本由 Phase4 动态加速补足。
- 所有失败按结构化原因重试最多 2 次(重试时回传上次输出的原始文本,类似 Plan R 方案)；重试耗尽记为 `style_failed`,不静默吞掉。
- retry 失败后,回合末尾兜底再补一次调用,利用本回合完整上下文曲线就。`style_failed` 窗口的 ``retry_count`` 保持不变(标记初始重试次数,不加额外计数)。
- **llmb 诊断落盘**(`_init_style_diagnostics`/`_write_style_diagnostic`):非 debug 也写脱敏摘要(`output_rounds_path.parent/diagnostics/phase3b/<run_id>_diagnostics.jsonl`,每窗每次尝试一条),字段含 `max_tokens/validation_ok/validation_reason/finish_reason/http_status/reasoning_chars/usage`(含 reasoning_tokens),用于重试根因观测;不落 prompt/正文。

### 3. 情绪定档体系(EmotionPolicy)

README 未记录,以代码为准(`sbmachine/emotion_policy.py`):

- 融合:硬事实权重 0.7 + LLM-B 权重 0.3,夹到 [0,1](`blend_intensity:18-24`);从 `hype_rules.json` 装载(`from_rules:84-92`)。
- 三档:平述/激动/惊叹,阈值默认激动 0.35、惊叹 0.72(`:71-92`);历史样本足够时按目标分布自适应微调,偏移受 `max_threshold_shift` 限幅,样本不足回退基础阈值(`_thresholds:102-114`)。
- 每回合最多 1 次「惊叹」:`round_scream_used` 逐回合重置(`phase3b_style.py:163,238-247`)。
- 「惊叹」额外要求 `scream_eligible`(硬事实),防 LLM-B 单独拉满(`emotion_policy.py:120-122`)。
- `normalize_commentary_emotion` 把句内标签钳到最终档以内,并保证以 `[标签]` 开头(`:36-57`)。

### 4. 空 neutral、逐窗账本与回合状态

- `intentional_empty`：不调 LLM-B，记录 `skipped_intentional_empty`，不得生成 scene。
- `unrecoverable`：记录 `skipped_unrecoverable`，使回合不可发布，不得伪装为正常静默。
- 普通非空 neutral：必须得到 `ok`/`retry_success` scene，否则记录 `style_failed`。
- 每个 Phase3A 窗口都写入一个 `window_results`；成功 scene 反向记录唯一 `published_scene_index`。
- **回合末兜底重试**(`phase3b_style.py:301-375`)：主循环结束后,对本轮所有 `style_failed` 窗口各补一次调用,利用本回合已成功窗口的完整上下文(``recent_style_phrases`` 含所有成功签名)。重试时带相同的分类专属 retry_feedback；补救成功则更新状态为 `retry_success`,补写入 `scenes_manifest` 和 `scene_commentaries`。
- 成功 scene 的 manifest 记录 `budget_overage`(output_chars / char_budget),供 Phase4 动态调语速。
- 回合状态为 `ok`、`silent`、`partial`、`style_failed` 或 `analyst_failed`。发布校验只放行 `ok` 与全部窗口均为合法主动静默的 `silent`。

### 5. 两产物同源、发布身份与剥离 _phase2_yolo

- `commentary.json` 顶层使用 `commentary_schema_version=2`，记录 `source_neutral_run_id`、`source_neutral_sha256` 与 `source_window_count`。发布门禁核对来源 neutral 身份、逐窗一一对应、scene 身份、预算以及文本/情绪重渲染。
- `commentary.json` 与 `rounds_with_commentary.json` 同源：Phase4 继续强校验视频、地图、回合、起止时间、文本和情绪分段逐项相等。
- 写出前剥离 `_phase2_yolo`:每回合 `phase2_yolo=None`(`phase3b_style.py:343-347`),视觉时间线权威副本只在 `rounds_with_yolo.json`,防产物膨胀;发布契约也不读 `_phase2_yolo`。
- `scenes[].text` 只剥首个最终档前缀、保留句内标签(`:283`);`commentary_text` 保留首标签(`:296`)。

### 6. 训练样本副作用

仅 `status=="ok"` 的回合把最终通过全部验收的输入输出对落训练样本；`partial`、重试耗尽、污染或最终写盘异常均不采纳。

### 6.1 commentary v3 稀疏候选任务单（`voice_task.enabled=true` 时生效）

- 分流：neutral 输入按 `schema_version` 分流——v4 走 `_run_phase3b_v4`，v3 走既有单稿路径（commentary v2 输出不变）。v4 路径还需 `voice_task.enabled=true` 且 profile 就绪（`speech_measure.load_profile` 返回 `validated` 且引擎/声线/预处理三指纹匹配），否则回退 v2 单稿并写 `risk_class=unknown` 影子诊断。
- **风险分级**（`_classify_risk`）：预判用 `speech_measure.measure_text(neutral).safe_duration_upper_bound_at_base_speed_sec`（U）对比固定 slot（S）与 `max_speed_factor=1.5`（M）：`U<=S`→green；`S<U<=S*M`→amber；`U>S*M`→red。red 预判直接生成 compact（primary 不进入生产候选）；非 red 生成 primary 后按实际文本**终判**，若终判变 red 则 primary 只进诊断并补生成 compact。候选 `preserved_fact_ids` 一律由原子校验器计算，required 未全覆盖的候选不进生产集合。
- **候选规则**（sparse_v1）：green 只交付 primary+capsule（不调 compact）；amber 交付 primary+compact+capsule（LLM-B 最多两次调用）；capsule 来自 Phase3a 的 `rule_capsule`（source=rule_capsule，不消耗 LLM-B token），情绪用 `emotion_policy.capsule_emotion` 钳到硬事实档位（≤0.45）。`selection_order=["primary","compact","capsule"]`。
- **commentary v3 产物**：`commentary_schema_version=3`、`voice_task_contract_version=1`、`candidate_policy="sparse_v1"`、`speech_metric_version="speech_units_v1"`、`source_neutral_run_id/sha256`；每 voice_task 含 `render_slot/required_fact_ids/speech_profile_id/risk_class/selection_order/max_speed_factor/candidates[primary/compact/capsule]`。`rounds_with_commentary.json` 只保留 primary 文本 + `voice_task_id` + `primary_variant_id`，不复制备选稿。落盘前先过 `voice_task_contract.validate_commentary_v3`（fail-closed）。
- 模型调用输入（`phase3b_prompt`）带 delivery 块：`variant_kind/target_units/hard_units/slot_duration_sec/max_speed_factor`；compact 额外携带"不得删除 required facts"压缩约束。响应仍是单对象 `{"commentary","felt_intensity"}`。

## Phase4:TTS + 逐局混流

### 职责与入口

- 入口:`sbmachine/phase_tts.py:23`,子进程 `python -m sbmachine.phase_tts`;清理旧产物后取 `FileLock output/.sovits.lock`(`:53`)再调 `run_phase4`。
- 主逻辑:`sbmachine/phase4_assemble.py:242` `run_phase4()`;音视频辅助在 `phase4_av.py`;TTS 客户端在 `audio_service/gpt_sovits_client.py`。

### 7. 逐 scene 合成与铺画布

- 每 scene 独立 TTS:`_synthesize_with_cache`(`phase4_assemble.py:31`)→ `synthesize_emotional`(按情绪分组请求、内存拼接 PCM,`gpt_sovits_client.py:210-244`)。
- 铺放:`relative_start = scene.t_start - round.start_sec`(`phase4_assemble.py:298`),`_assemble_scene_wav` 建 `round_duration` 长静音画布按帧写入(`phase4_av.py:116-172`),输出 `round_NNN.wav`(`:287,333`)。
- 可选逐局 MP4:需 `phase4.make_video=true` + 源视频存在 + 该回合有 scenes + 非 dry_run(`:337`),复用/切 clip 后 `_mux_round_video`→`round_NNN.mp4`;不用 `-shortest`,两路音频 apad+atrim 到画面时长,无游戏音轨时只输出解说轨(`phase4_av.py:46-73`)。
- **执行分流**：legacy 路径继续按 `commentary.json.commentary_schema_version` 消费 v2/v3。严格路径要求 `commentary_render_package_v2`，校验 `artifact_identity`、`package_status`、内容策略、回合和单元顺序后，只合成 `render_units[].final_text`；不得从 B 包或 commentary 回退取稿。
- **惰性选稿**（`_select_voice_variant`）：按 `selection_order` 逐候选合成，primary 适配即停；超长只能在 `max_speed_factor` 内再试一次，仍失败进入下一候选；全部失败写 `render_unfit` 并阻止关键回合发布（不写最终 WAV/MP4）。`attempted_variants` 与每次 TTS 调用一一对应。
- **固定 slot**：legacy 任务单保留 tick 适配校验；严格路径按 PCM sample 计算资产、slot 与回合画布边界，超界即 `render_unfit`，不移动原始视频时间轴。
- **输出增量**：`rounds_final.json`/`assemble_manifest.json` 每 scene 记录 `voice_task_id/selected_variant_id/selected_text/actual_duration_sec/applied_speed_factor/audio_start_tick/audio_end_tick/fit_state/attempted_variants/render_unfit_reason`（v2 输入时这些字段可缺省）。
- **缓存指纹**：`gpt_sovits_client.tts_cache_fingerprint` 缓存键含 `variant_id/profile_id/speed_factor`，任一变化即失效。

### 7.1 严格媒体链路

- `media_clock.py` 使用有理数和 half-even 规则，将秒独立映射为视频 PTS 与 PCM sample；语义 `render_timebase_fps` 不参与源视频帧率判断。
- `media_probe.py` 读取源视频身份、time base、stream start PTS、帧率与 VFR 状态，并执行有界解码边界探测。探测缺失时状态为 `not_checked`，不得记为通过。
- `phase4_media.py` 只接受 `strict_decode` 切片及已验证 sidecar；混流不使用 `-shortest`，输出后再次探测时长和边界。
- `assemble_manifest.json` v2 分别记录 `media_sync_status`、`content_gate_status`、`delivery_status`。严格发布必须通过对应 profile 的发布门禁。

### 8. silent、超窗与动态语速

- silent:`skipped = not scenes`(`phase4_assemble.py:291`),空输入直接写纯静音(`phase4_av.py:124-126`)。
- 超窗失败(RuntimeError,不截断/不挤占):音频时长 > 窗口(`phase4_av.py:135-138`)、铺画布后 end_frame 越界(`:160-164`);WAV 格式不一致也报错(`:141-144`)。
- 动态 TTS 语速：若 Phase3b 产物的 `budget_overage > 1.0`(即 output_chars 超出 char_budget),TTS 合成时自动加速以容下更长文本。加速公式：`speed = emotion_speed × max(1.0, min(1.5, budget_overage))`；emotion_speed 来自 `config/llm.yaml` 的 `semantic.speech_rate.tts_speed_factor`。上限 1.5 倍防声音失真。该 `speed_factor` 已纳入缓存指纹(`gpt_sovits_client.py:156-183`),速度变化自动触发缓存重。TTS 缓存指纹:文件名 `sha256(fingerprint\0text)`(`phase4_assemble.py:26-28`)。
- clip cache 指纹:`sha256(source_sha256\0start\0end)`(`phase4_assemble.py:193-196`),优先复用 phase1 `segment_video`,否则指纹缓存 `clip_{fp}.mp4`(`:199-239`),不跨比赛复用。

### 9. Phase4 配置全部来自 pipeline.yaml 的 phase4 节

- Phase4 输出目录/缓存/音量/采样率全部读 `config/pipeline.yaml` 的 `phase4` 节(`phase4_assemble.py:266-273`)。
- GPT-SoVITS 运行时配置路径同在该节:`phase4.tts_config`(默认 `audio_service/gpt_sovits_runtime.yaml`),经 `require_path` 解析(`phase4_assemble.py:260-263`);preflight 按同一键登记必需输入(`preflight.py:130`)。
- 发布与媒体键为 `publish_profile`、`clip_mode`、`media_probe.*`、`media_tolerances.*`。严格 profile 要求 `clip_mode=strict_decode`；固定 `max_frame_boundary_error_sec=0.05` 的帧率适配问题见 `known_discrepancies.md` E13。
- 原 `config/tts.yaml` 已删除(其 `output_dir`、`final_audio`、`video.*` phase4 从未读取,属旧流程死配置);现 phase4 无第二处配置来源。

### 10. Phase3b 发布校验

发布门禁 `validate_commentary_publishable`(`preflight.py:520`)执行逐窗校验:

- `commentary_schema_version=2`、SHA-256、run_id、window_count 一一核对。
- 预算校验: `output_chars > int(char_budget × (1.0 + tolerance))` 才报 `exceeds char_budget`,与 Phase3b 的双层预算一致。
- `retry_count` 必须在 0-2 之前(回合末兜底不改变原值)。
- 缩放 K 配额 `style_k_enabled`(`config/llm.yaml`,默认 false=零容忍):仅当启用时,`style_failed` 窗口数 ≤ `max(2, floor(0.03 × total_windows))` 才赦免 round 的非发布状态。

### 11. 配置速查

Phase3b 专属配置集中在 `config/llm.yaml` 的 `semantic` 节:

| 键 | 默认值 | 说明 |
|---|---|---|
| `style_max_retries` | 当前配置 3；代码最多钳制为 2 | 单窗重试次数 |
| `style_budget_hard_tolerance` | 0.5 | 预算弹性系数,软上限×(1+tolerance)=硬上限 |
| `style_k_enabled` | false | K 配额开关 |
| `style_temperature` | 0.55 | 采样温度 |
| `style_top_p` | 0.85 | nucleus sampling |
| `style_frequency_penalty` | 0.2 | token 重复惩罚 |
| `style_output_max_tokens` | 1024 | 本地输出 max_tokens |
| `style_concurrent_scenes` | 当前配置 6；代码缺省 API=4、本地 vLLM=1 | LLM-B 窗口级有界并发 |
| `cloud_style_output_max_tokens` | 当前配置 16384 | 云端输出预算(`style_runtime_config` 透传,phase3b api 分支直接按此值发送,不被字数公式截断) |
| `cloud_conversation_max_rounds` | 当前配置 6；代码缺省 0 | 云端会话历史上限(轮);0=禁用会话 |
| `cloud_token_budget_enabled` | false | 成本护栏;false=永不 silence |
| `cloud_total_timeout_sec` | 600 | SSE 硬总时限 |
| `cloud_request_concurrency` | 6 | scope 级信号量上限 |
| `cloud_queue_timeout_sec` | 120 | 信号量排队上限 |
| `cloud_cache_enabled` | 当前配置 true | 成功响应缓存总开关 |

### 10. 验证方法

- Phase3b 响应契约：`tests/unit/test_phase3b_response.py`（严格 JSON、事实/预算/完整句/复读、重试、训练样本提交与回滚）。
- Phase3b 静默与逐窗账本：`tests/unit/test_phase3b_silence.py`（intentional empty、unrecoverable、partial、`window_results`、剥离 `_phase2_yolo`）。
- 情绪定档:`tests/unit/test_emotion_policy.py`(0.7 硬事实权重、无硬事实不能高档、惊叹需 scream_eligible、短场次不需满历史、句内钳档)。
- Phase4 缓存与合成:`tests/test_phase4_cache.py`(相对时间铺放、逐 scene 独立合成、模型/参考/语速指纹变化、silent 纯静音、超窗失败、dry_run 不写、clip 指纹、拒绝异源/文本不符/旧无 scenes 契约、无游戏音轨混流)。
- Phase4 v2 与媒体门禁：`tests/unit/test_phase4_execution_v2.py`、`test_phase4_media_gate.py`、`test_media_clock.py`、`tests/contract/test_phase4_sync_contract.py`。
- 发布体检:`sbmachine/preflight.py` 的 `validate_commentary_publishable`(`:450`)、`validate_final_manifest`(`:350`);dry-run 走 `preflight_config`。

## 已知偏差

代码与 README/其它文档的措辞冲突统一记录在 [../known_discrepancies.md](../known_discrepancies.md)。
