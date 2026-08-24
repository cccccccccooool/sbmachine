# 已知偏差清单（文档 / 代码 / 记忆冲突）

> 基于截至 2026-08-20 的当前树只读代码核对整理。本文件满足 `docs/AI_DOCUMENTATION.md` 第 7 条：
> 无法确定代码与文档哪方正确时，如实报告冲突，不自行编造解释。
>
> 处置原则：**一律以源码为准**。本次任务为只读，仅登记冲突、不修改任何代码、数据、README 或记忆。
> 每条给出：现象、代码事实（`path:line`）、建议处置方。

分三类：A. README 与代码不一致；B. 代码级瑕疵（只读未改）；C. 记忆库与代码不一致。
另附 D. 数据缺失现状（属设计内 fail-closed，非缺陷）。

> **对抗复核结论（2026-07-27，6 组独立子个体逐条证伪）**：26 项**无一被推翻**。
> 20 项完全确认；5 项实质成立但原措辞/行号已订正（A5 扁平键范围、A7 归属子仓库 README、A14 行号 :469、B6「不进 prompt」纠正、C2 收窄为「运行时代码链」）；
> 1 项（A12）经复核判定 README 并无偏差，已撤销。订正处均在对应条目内以「对抗复核」标注。

> **对抗复核（2026-07-27）**：26 项经 6 组对抗子个体独立证伪，**0 项被推翻**。20 项完全确认；5 项实质成立但措辞/行号需订正（A5、A7、A14、B6、C2，已在各条内标注"对抗复核修正"）；1 项撤销（A12，README 实无偏差）。

---

## A. README.md / 文档 与 代码不一致

### A1. 默认阶段开关与 README「当前状态」表相反（重要）

- README `README.md:20-29` 称默认仅开 Phase3a/3b/4，且未列 Phase3c。
- 实际 `config/pipeline.yaml` 为八个阶段开关：当前只启用 Phase3c/Phase4，其余阶段关闭并复用既有 checkpoint。
- 处置：README 当前状态表已过时；以配置为准。见 `docs/architecture.md` 与 `docs/operations.md`。

### A2. 默认服务模式与 README 相反（重要）

- README `README.md:305-308` 称默认 `manage_services: false`（Docker Compose 多容器）。
- 实际 `config/pipeline.yaml:9` 为 `manage_services: true`（`ServiceManager` 单容器进程托管）。
- 附带反直觉点：`run_all.py:235` `single_container = manage_services`，`True` 走进程托管、`False` 走 Compose，变量名与直觉相反。

### A3. `run.py` 顶部 docstring 与代码矛盾

- `run.py:9-11` docstring 称 `false`=用户手动启动服务、`true`=run.py 自动拉起。
- 实际两种模式都会自动管理服务：`false` 走 `ComposeManager` 也是自动 `up -d`/`stop`。docstring 应视为过时。

### A4.〔已消除 — `debug.phase3` 配置项已删除，改由 `--debug` 驱动〕

- ~~`pipeline.yaml` 设 `debug.phase3: true`，但事务运行无条件改写为 `False`，README 未提~~。
- **消除方式**：`config/pipeline.yaml` 的 `debug:` 块已删除（现该文件无 `debug` 键）。Phase3 调试落盘改由 `run.py --debug`（`run.py:54-58`）经环境变量 `AI6657_DEBUG_PHASE3` 驱动（`run_all.py:198-200`），读取方 `phase3a_analyst.py:170`、`phase3b_style.py:116` 仍保留 `config.debug.phase3` 或 env var 的双路判断。`run_context.py:133-136` 仍无条件把 `debug.phase3` 改写为 `False`，注释自承其目的已变为「防止旧配置文件残留字段静默激活 debug 落盘」，对 env var 路径无影响。

### A5. `background_info` 字段名与 README 严重不符（重要）

- README `README.md:153` 称 key frame 背景新增扁平字段 `hud_detected/timer_value/timer_raw/timer_confidence/score`。
- 实际 `phase2_background.py:196` 产出嵌套结构 `when/who/where/events`：timer 存于 `when.timer`、比分存于 `events.score_ocr`，`build_background_info` 产物中**不存在**上述扁平字段。README 该段描述的是旧 schema。
- 对抗复核修正：范围应限定为"`build_background_info` 产物中不存在"，而非全代码库不存在。诊断脚本 `tools/diagnose_hud_ocr.py:67/70/79` 另有同名报表键 `hud_detected/timer_raw/timer_value`（自建本地统计字典，与 key frame 背景结构无关）；仅 `timer_confidence` 全代码库（代码侧）确无出现。

### A6. L0 score OCR 未接线，README 只点 L2

- README `README.md:148` 只说 L2 onset 未接线。
- 实际 `round_aligner` 的唯一调用点（`run_frame_type_slicer.py:265-267`）既未传 `score_ocr_per_segment`（L0）也未传 `onset_per_segment`（L2），切片期仅 L1 duration DP 生效。
- 另注：`round_aligner` 只在切片期跑一次给 `align_offset` 初值；Phase2 运行期实际用 `time_align.RoundTimeAlign`。README 把三层写在 `round_aligner` 名下，易误解为运行期三层对齐。

### A7. 子仓库 README 营销表述有 VLM/画面理解，实际 Phase2 没有

- **归属订正（对抗复核）**：营销表述**仅存在于 `sbmachine/README.md`**（`:23/:34/:35`「看画面 / 视觉模型描述战场」「大模型结合画面 + 全场记忆，写带情绪标签的口播稿」）。**顶层 `README.md` 不含此表述**，反而与代码一致地明确声明 VLM 冷藏（`README.md:37`「VLM 不属于当前 Phase2 运行依赖」、`:549`「VLM 已退出运行主链」、`:559` 禁 MatchMemory/跨局情绪）。原条目误将其挂到"README 顶层 `:33-35`"，应改归子仓库 README。
- 实际 `phase2_yolo.py` 全流程只有 HUD YOLO + OCR + DEM 注入，**无任何 VLM 调用**；LLM-A（`analyst_system.txt:1-18`、投影白名单 `llm_projection.py`）只写一句 ≤100 字中性稿，不看画面帧、无全场记忆、不加情绪标签。情绪标签与主播风格在 Phase3b。README 正文（`:54/:168/:197-229`）对此描述准确，仅顶层营销段与子仓库 README 过时。与记忆「已砍复杂证据层，证据层=朴素逐帧视觉 dump」一致。

### A8. README 未提情绪定档体系（EmotionPolicy）

- `emotion_policy.py:71-128` 是 Phase3b 核心：硬事实 0.7 + LLM-B 0.3 融合定三档（平述/激动/惊叹），阈值默认激动 0.35 / 惊叹 0.72 并按历史分布自适应；**每回合最多 1 次「惊叹」**（`phase3b_style.py:163,238-247`），最高档另需 `scream_eligible` 硬事实。README `README.md:231-262` 只字未提。

### A9. README 未提 Phase3b 两道二次防线

- `phase3b_style.py:231-263`：相邻完全复读抑制、污染标记（任务/注:/字数预算约/根据以上信息等）过滤。README 未提。

### A10.〔已消除 — `config/tts.yaml` 已删除，死配置随文件移除〕

- ~~`config/tts.yaml` 的 output/video 段 phase4 不读取~~。
- **消除方式**：`config/tts.yaml` 已从仓库删除。其中唯一有效项迁为 `config/pipeline.yaml` 的 `phase4.tts_config`（`pipeline.yaml:56`，值 `audio_service/gpt_sovits_runtime.yaml`），读取方 `phase4_assemble.py:260-263` 与 `preflight.py:130` 已同步改为读 `phase4.tts_config`。死字段 `tts.output_dir`、`tts.final_audio` 及整个 `video:` 段（`make_filtered_video`/`clip_dir`/`final_video`）随文件一并删除，不再存在可误配的入口。phase4 输出目录仍来自 `phase4.output_dir`（`phase4_assemble.py:268`）。

### A11. TTS 指纹措辞与代码分层不完全对应

- README `README.md:296` 称 TTS 指纹「包含文本」。实际文本不在 `tts_cache_fingerprint`（`gpt_sovits_client.py:156-183`）内，而在外层缓存键 `sha256(fingerprint\0text)`（`phase4_assemble.py:26-28`）。功能等价，分层措辞不同。

### A12.〔已撤销 — 经对抗复核判定 README 无偏差〕

- ~~README「唯一 Prompt/style_system.txt」表述略简~~。
- **撤销理由**：`README.md:239` 原文为「唯一 `Prompt/style_system.txt`、persona、CS 边界和可选 style skill」，**已完整列出三部分**（persona + CS 边界=cs_rules.txt + 可选 style skill），与代码三段拼接（`phase3b_style.py:145-151`）一一对应。此处「唯一」是针对 `README.md:249`「API/vLLM 两套 style prompt（列于'当前不存在'下）」而言，强调只有一份 style_system、不分后端，并非表述简化。原判「略简/未道出三段」不成立，本条撤销。

### A13. README 未写出 LLM 基座具体型号

- 基座 = `Qwen/Qwen3-14B-AWQ`（`tools_config/train.yaml:14-15`、`verify_talk.py:14`、`train_config_*.yaml`），served-model 名 `qwen3`。README 全文只在思考开关处提 Qwen（`:356`），未点明型号。

### A14. 宿主 pip 脚本的训练环境边界未点破

- `cloud_prepare.sh` / `install_training_deps.sh` 用宿主 pip 装依赖，但 README `README.md:469`（对抗复核订正，原记 :465 为空行）明说「宿主 Python 和 requirements.txt 不是训练环境契约」。这两个脚本是云端便利脚本，不满足 `require_training_container` 门禁（`scripts/_train_common.sh:3-12`），不能用于正式可复现训练。

### A15. MCP `check_services` 探测口径只覆盖两个服务

- README `README.md:488` 笼统说「探测口径与 service_manager.py 一致」。实际 `service_manager` 有三个健康 URL（vllm/vlm/sovits），MCP `check_services`（`mcp/server.py:191-208`）只探 talk_service(vllm) 与 audio_service(sovits)，未探 vlm(`:23333`)。已核实覆盖的两个口径确与 `service_manager.py:78/84` 一致。

### A16.〔已修复〕`docs/order.md` 曾引用不存在的 `docs/readme/README.md`

- 历史 `docs/order.md:5` 曾指向 `readme/README.md`，但 `docs/readme/` 目录不存在；现已改为仓库根 `../README.md`。项目 README 在仓库根 `README.md`。

---

## B. 代码级瑕疵（只读核对发现，本次未修改）

> 以下为代码内的名实不符、死代码或潜在 bug。均按只读约束**不改动**，仅登记待授权后处理。

### B1. ~~`tests/conftest.py` 的 `fake_backends` 引用未定义变量~~（2026-07-29 已修复）

- `tests/conftest.py:94-99` 已导入并构造 `FakeVLM`;fixture 可返回完整的 `llma`、`llmb`、`vlm`、`tts` fake 集合。

### B2. `ffmpeg`/`slow` marker 重复声明且零使用

- 在 `pytest.ini` 与 `conftest.py:71-73` 各声明一次（冗余无害）；且当前无任何测试实际打这两个 marker，`-m "not ffmpeg and not slow"` 目前是空过滤，属为未来预留的约定。

### B3. `round_emotion` 传参名实不符

- `hype_score.py:105` `dominant_round_emotion` 形参名 `avg_hype`、docstring 称「回合平均」，但本地（`analyst.py:211`）与云端（`cloud_runner.py:137`）都传入 `peak_hype`。命名/注释误导，行为是按峰值定情绪。

### B4. `run_phase1` 旧接口未被主链调用

- `phase1_slice.py:50` `run_phase1` 无生产调用点；主链 `run_preprocess_slice` 自带一份 `load_or_build_segments`，两处切片逻辑并存（潜在重复代码）。

### B5. `manual_notes` 已实现但未接入主链

- `manual_notes.py` 的 `load_manual_notes`/`lookup_manual_note` 无任何生产调用点；`test_phase3a_manual_notes.py:41-43` 反而断言人工笔记**不进入** prompt。若文档宣称支持人工笔记注入，与现状不符。

### B6. `fallback_neutral` 绕过 LLM 投影白名单

- LLM 路径的 main_topic summary 被 `_project_main_topic`（`llm_projection.py:74`）对多数类型清空；但 `fallback_neutral`（`commentary_planner.py:226`）直接取未投影的 `plan.main_topic.summary`，可能含 callout/选手名，两条路径信息面不一致。
- **对抗复核订正**：原记「仅进最终 neutral、不进 prompt，不构成信息泄露」不准确。fallback 产物只是不进 Phase3a 的 **LLM-A window prompt**（该 prompt 由投影后的 `window_payload` 构造，`phase3a_analyst.py:244-253`）；但它写入 scene 的 `neutral` 字段（`phase3a_analyst.py:287`），会随中性稿流入 **Phase3b 的 LLM-B prompt**（`phase3b_style.py:184` 读 `neutral`、`:207` 拼入 `【中性稿】` 块）。故未投影的 callout/选手名确会到达下游 LLM-B，泄漏不止于 neutral。

### B7. `style_runtime_config` 曾过滤云端特化键（已修复，留痕）

- 2026-08-15 修复前：`llmb_api.style_runtime_config` 只用 `STYLE_DEFAULTS` 的 key 集从 `semantic` 提取配置，`cloud_style_output_max_tokens`（4096）被过滤 → `phase3b_style.py` 云端分支实际只发 1024（思考模型 reasoning 吃满预算 → content 截断 → `unparseable`）。诊断 `diagnostics/phase3b/*.jsonl` 显示 `completion_tokens == max_tokens == reasoning_tokens`。
- 修复：`llmb_api.py` 单独透传该键；`phase3b_style.py` 主调用与回合末兜底重试云端分支直接按 `cloud_style_output_max_tokens` 发送，不再被 `min(字数公式)` 截断。
- 验证：`tests/unit/test_llmb_api_config.py`、`tests/unit/test_cloud_prompts.py::test_cloud_style_branch_sends_full_cloud_max_tokens_budget`。

### B8. 云端节流/流式与本地差异（设计内，登记防误读）

- `llm_shim._execute_openai_chat` 按 `_is_loopback_url` 分流：本地 vLLM 保留请求间隔节流 + 非流式；云端官方 API 不节流、自动 SSE 流式（`stream:true` + `include_usage`）。该差异是刻意设计（云端自带配额管理），非缺陷。

---

## C. 记忆库（MEMORY）与代码不一致

> 以下为本次核对发现的、与用户长期记忆文件冲突之处。**本次不改记忆**，仅如实上报，是否更新由用户裁定。

### C1. 声纹筛选脚本位置

- 记忆 `[[audio-cleaning-pipeline]]` 记「06 说话人声纹筛选脚本现在 `tools/data_clean/`」。
- 实际 `tools/data_clean/` 只有两个 **SFT 配对**脚本（`label_commentary_pairs.py`、`build_commentary_pairs.py`）；真正的音频清洗链在 `tools/audio/`（`06_speaker_filter.py` 等），由 `scripts/clean_commentary.sh` 编排。
- 记忆记录的「04→02→06→prepare 顺序」正确，仅目录归属有误。

### C2. LLM 基座与推理框架

- 记忆 `[[model-baseline]]` 记「LLM=qwen3-8b（Ollama 部署）」、`[[inference-framework-decision]]` 记「2026-06 定档 llama.cpp 单后端」。
- 实际**运行时代码链**一致指向 **`Qwen/Qwen3-14B-AWQ` + vLLM**（`tools_config/train.yaml:14-15`、`config/llm.yaml:2` `backend: vllm`、`scripts/verify_talk.py:15` `SOURCE_MODEL`），运行链内未见 qwen3-8b / Ollama / llama.cpp。
- **对抗复核订正**：原记「未见 qwen3-8b / Ollama / llama.cpp **任何引用**」是过度断言。仅限"运行时代码链"成立；全库范围内 docs/plan、docs/report、`.remember/*`、`tests/test_vllm_runtime_tools.py`（断言其**不**在运行时）及 `.git` 历史仍有大量历史引用，连本清单文件自身（本条）也在引用。行号亦订正：`config/llm.yaml` 的 `backend: vllm` 在第 2 行（第 1 行为 `llm:`）。
- 按「以代码为准」，当前基座/推理框架应为 Qwen3-14B-AWQ/vLLM。记忆与代码的冲突需用户确认：是代码已演进、还是记忆描述的是另一环节（如 Ollama 仅用于某侧部署）。

---

## D. 数据缺失现状（设计内 fail-closed，非缺陷）

> 以下不是 bug，是「无数据即静默/兜底」的既定设计，登记以免误判为缺陷。

### D1. `database/tactics/` 目录不存在

- `tactic_book.load_tactic_book`（`tactic_book.py:186-188`）只读 `database/tactics/<map>.json`，目录当前不存在 → 战术书恒为空、战术匹配恒不触发。`database/` 现有 `match_notes/` 与 `player_aliases.json`。

### D2. 仅部分地图具备人工复核模板

- `database/maps/ancient.json` 已存在且为 v2，`source.manual_reviewed=true`；`spatial_context.load_map_template`（`spatial_context.py:23`）可将 Ancient 解析为 `reviewed_graph`。未提供 reviewed 模板的其他地图仍回退 `coordinate_fallback`，对外 `nearby` 为空。

---

## E. 本计划（配音任务单改造）实施登记

> 2026-08-16 配音任务单改造的历史实施登记。原计划文件已不在当前 Git 跟踪内容中。原则：以源码为准；契约版本原则——新语义必须升版，不允许同一 schema 版本改变旧字段含义。

### E1. 阶段-1 基线归并（`config/llm.yaml` 采用 .env 解耦版）

- 当前 `config/llm.yaml` 使用 `.env` 解耦的 API 配置，显式设置 `backend=api`、`analyst_concurrent_rounds=5` 以及 cloud 配置块。历史“两树哈希一致”和 `baseline_manifest.md` 路径当前无法复核，不作为现状结论。
- 阶段-1 登记的代码修复以当前源码和测试为准，不再引用已移除的计划树。

### E2. `5.0 字/秒` 不再承担 v3 安全分级

- `semantic.speech_rate.base_char_per_sec=5.0`（`config/llm.yaml`）仅属 legacy 预算解释；commentary v3 的 green/amber/red 分级与 `safe_duration_upper_bound_at_base_speed_sec` 只认 `data/speech_profiles/<id>/profile.json`（`validated` 状态，`sbmachine/speech_measure.py`）。无 validated profile 时 Phase3b 只写 `risk_class=unknown` 影子诊断并回退 v2。
- Phase3a v4 的 `rule_contract_unfit` 前置检查在 profile 未标定期使用 5 字/秒 × 1.5 倍的**保守护栏**（不冒充 validated 分级），profile 到位后由 profile 接管。

### E3. preflight 统一计量口径

- `preflight.py` 曾用 `len(re.sub(...))` 重算 `output_chars`，与 `phase3b_prompt/phase3b_style` 的 `count_spoken_chars` 不一致（阶段-1 记录在案缺口）。已统一为 `count_spoken_chars`；v3 候选的 spoken 复核用 `speech_measure.measure_text`。历史 v2 产物的 `output_chars` 语义不变、不重算。

### E4. 预存测试同步（基线漂移修复）

- `tests/unit/test_phase3b_response.py`：`_call_style` 已由主工作区改为返回 3 元组 `(text, felt, meta)`，旧测试按 2 元组解包导致 4 项确定性失败；已同步测试解包（不改代码）。
- `tests/unit/test_phase3a_audit.py::test_training_samples_not_accepted_when_manifest_not_publishable`：主工作区加入规则层 fallback 后，truncated 窗口会被 `fallback_neutral` 兜底为 success，原"整场不可发布"预期已不成立；已更新为验证"fallback（rule 源）窗口不产生训练样本"。

### E5. v4 静默 scene 表示

- v4 静默窗口沿用 v3 惯例：`neutral=""` + `neutral_source="intentional_empty"`；计划书未定义 v4 静默显式标记，`validate_neutral_v4` 对静默 scene 豁免 rule_capsule/fact_catalog 等字段要求。golden fixture 的静默 scene 缺 `neutral_source/fact_anchors/char_budget`，preflight v4 门禁在内存补齐后再校验（只改内存，不写回文件）。

### E6. fact_id tick 补零位数

- 计划书 §7.3 规定"5 位补零"且契约正则 `\d{5}`；§7.4/§9.4 示例写 `000360`（6 位）。按 §7.3 正则执行（fixture 用 `00360`/`00450`/`00090`）。建议计划侧统一口径。

### E8. `test_production_gates_detect_real_budget_violations` 全量顺序下偶发

- 该用例单独/按文件运行稳定通过，全量 `pytest tests` 顺序下偶发失败（`tests/unit/test_production_gates.py`，依赖全局随机/配置状态），属预存 flaky（阶段2/3 的 agent 亦报告同一现象），与本计划改动无关。复跑该文件即绿。

### E9. LLM-C 已独立为 Phase3c（2026-08-17 实施登记）

- 旧内嵌 LLM-C（`phase3b_style.py` 的 `_run_round_integration`/`_rebuild_scenes_from_integration`）已移除；`config/llm.yaml` 的 `semantic.llmc` 段已删除，残留配置被 `preflight_config` fail-closed（`preflight.py`）。
- 新增 `sbmachine/phase3c_llmc.py`（C0~C7 门禁 + 四态 `semantic.phase3c.mode`）、`phase3c_cli.py`；Phase3b 出口新增 `llmb_draft_package_v1` 封存导出（`_export_llmb_draft_package`）。
- 流水线接线：`run_all.py` 两模式、`phase_semantic.py` 传 `draft_package_path`、`run_context.py` `_PATH_OUTPUTS/_STAGE_ARTIFACTS`、`pipeline.yaml` `phase3c_render` + 2 路径键。
- 云端凭证：`llm_protocol._load_secrets` 新增 llmc（回退 cloud 通用键）；`cloud_memory.make_generate` 接受 llmc scope（无状态、缓存/统计独立）。
- 门禁数值：B 恒硬线 1.5x；C 目标 1.0x/硬线 1.25x、逐窗 `r_C<=min(1.25,max(1.0,r_B))`，无 tolerance 叠加。
- 权威维护文档：`docs/modules/phase3c.md`；原计划文件已从当前 Git 跟踪内容移除。

### E10.〔已完成〕Phase4 消费 render package v2

- `run_all.py` 将 `commentary_render_package.json` 传入 Phase4；严格路径校验 `commentary_render_package_v2` 的身份、状态和策略，只使用 `render_units[].final_text`。
- legacy commentary v2/v3 路径仍保留；严格发布配置不得回退到该路径。

### E13. Phase4 帧边界容差仍是固定秒值（重要，待修复）

- `phase4.media_tolerances.max_frame_boundary_error_sec` 当前固定为 `0.05` 秒，并同时用于严格切片边界、切片时长和混流时长验收。
- 媒体探测已取得 `avg_frame_rate`、`r_frame_rate` 与 VFR 标志，但门禁尚未据实际帧率换算容差。低于 20 fps 或 VFR 素材可能误拒；高帧率素材则会获得过宽的帧数容忍。
- 维护要求：将帧边界容差按探测到的实际时间基或帧时长计算；在完成前，变更输入素材帧率时必须重新核对边界验收口径。

### E11. 端到端真实调用验证记录

- 历史 scratch 链路曾验证合法 manifest → 导出 B 包 → `run_phase3c(mode=optional)` → `validate_render_package`；scratch 文件已清理，当前没有可复核的运行产物或测试结果，不能把历史通过数当作当前基线。

### E12. 强事实依据模式（2026-08-17 实施登记）

- 新增 `semantic.strong_fact_mode`（默认 `false`）总开关，统一控制 B0 `unexpected_fact`（`phase3b_prompt.validate_style_commentary`）与 C3 事实作用域（`phase3c_llmc.check_fact_scope`）两个门禁；开启才执行，关闭=全面相信 LLM。
- 关闭范围仅限 unexpected_fact/C3：空稿、情绪标签、预算硬线（B 1.5x / C 1.25x 非退化）等运行基础门禁不受开关影响，始终生效。
- `phase3c.fact_scope_enabled` 已并入总开关（同语义，键替换）。
- C3 中的阵营（teams）子检查已移除；玩家实体检查升级为混合 token（字母+数字）+ leet 规范化（`dev1ce`↔`device`），武器前缀兜底（`AK47`）放行；地点/武器/数字词表检查保留。
- 测试覆盖 `test_style_strong_fact_mode_off_trusts_llm_for_facts`（B 侧）及 C 侧开关用例；当前通过数量以实际 pytest 运行结果为准。

### E7. 待人工执行的验收项（本计划环境无法完成）

- 真实 TTS 标定采集（最低 160 条独立文本 + 20-30 条多速度验证）需正式 voice/参考音频与 GPT-SoVITS 服务；工具链（`tools/calibrate_speech_profile.py`）与测试已就绪，采集与验收待人工执行。
- 87 窗历史场景重放与旧超预算样本重分类需要 `data/temp/parse-test2` 等输入；本登记不把历史工作树样例或单回合结果当作当前全量验收。
- Tiny-LLM 训练（2000/300/300 数据门禁）与成对下游盲评（阶段5B）不在本计划执行范围。
- 主工作区另一 agent 的云端速度优化/云端-本地层清理与本计划重叠文件（`llm_shim.py`、`phase3a_analyst.py`、`phase3b_style.py`、`config/llm.yaml`）两树已一致；后续归并以主工作区为准。

---

## 处置建议汇总

| 类别 | 条目 | 建议 |
|---|---|---|
| A（README 过时） | A1–A16（A4、A10 已消除；A12 已撤销） | 以代码为准；待授权后同步修订 README（尤其 A1/A2/A5 属会误导运行的重要项）。本次已在 `docs/` 各文档按代码事实记录。A4/A10 已随 config 合并落地消除（`debug:` 块与 `config/tts.yaml` 均已删除），条目保留并划除以留痕。A12 经对抗复核判定 README 无偏差，已划除。 |
| B（代码瑕疵） | B1–B8 | B1（NameError）与 B3（peak/avg 命名）建议优先修，其余为死代码/冗余清理，须绕开 `vlm/` 冷藏区。B7 已修复（2026-08-15，含回归测试）；B8 为设计内差异，无需处理。 |
| C（记忆冲突） | C1–C2 | 交用户裁定是否更新记忆文件，本次未改。 |
| D（数据缺失） | D1–D2 | D1 仍为无战术书的 fail-closed；D2 已有 Ancient reviewed 模板，其他地图缺失时自动回退。 |
| E（实施与遗留登记） | E1–E13 | Phase3c/Phase4 v2 接线已登记；E7 为待人工验收项，E8 为预存 flaky，E13 为待修复的帧率容差问题。 |


