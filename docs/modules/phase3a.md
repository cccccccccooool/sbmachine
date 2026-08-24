# Phase3a：规则规划与中性稿生成

> 基于代码核对(2026-08-20),仅记录代码/配置/测试已确认的事实,运行行为以源码为准。

本文是 [`docs/phase3_commentary_final_plan.md`](../phase3_commentary_final_plan.md) 的**实现细节伴随文档**。该权威设计定义 Phase3 的职责边界与不可恢复的旧能力(MatchMemory、历史 few-shot、跨局情绪、让 LLM 自行从原始 players/events 推断规则、用画面覆盖 DEM),本文只描述"以代码为准的当前实现",不与权威设计矛盾,也不描述或复活被其否决的分支。README 及记忆中与代码不符之处,统一见文末"已知偏差"。

## 1. 职责与流水

Phase3a 把 DEM 硬事实经确定性规则层压成每窗口唯一的 `CommentaryPlan`,再由 LLM-A 只写一句中性稿。规则层完成事实判断,LLM 不重做。本地主流程 `run_phase3a`(`sbmachine/phase3a_analyst.py:160`),逐回合(`_process_round:205`)执行:

1. 规范化时间线:物理帧取自 `rounds_with_yolo_semantic.json`(`load_semantic_frames`),经 `_semantic_payload`→`_normalize_planning_frames`(`sbmachine/phase3a_payload.py:32`:标记鞭尸、裁字段、保留坐标供规则用);`rounds_with_yolo.json` 只作契约/身份锚点(`phase3a_analyst.py:187-196`)。
2. 切场景窗:`build_scene_contexts`(`sbmachine/scene_context.py:74`)。先按 `classify_frame_scene:35`(准备/未下包/炸弹/收尾)切段,再按 `window_max_sec/window_min_sec` 二次切分;跨阶段击杀补 `context_start`。切分后自动合并时长小于 `_effective_min_sec`(取 `window_min_sec` 与语速极限其中较大者)的短窗口,按场景优先级(收尾>炸弹>未下包>准备)向前合并(`_merge_short_windows:97-128`)。
3. 逐窗口提取:`build_window_rule_projection` 先战术匹配、再 `plan_window`。`scene_context.extract_actions` 先调用 `rule_compare` 比较击杀直达字段与跨 tick 的 snapshots/fire/equip 证据，再交给 `kill_semantics`、空间关系、hype/情绪、配额和字数预算。
4. 生成唯一 CommentaryPlan v2：`plan_window` 返回事件账本、窗口归属、炸弹状态/转变、唯一主话题、被抑制事件、`required_facts`、空间审计和约束。
5. LLM-A 只看到 projection v2 白名单并写一句 neutral；响应解析后还要通过必需事实、阵营、数字、实体、事件、结果、地点、武器和字符预算校验。
6. 方案 R 恢复层（`_recover_analyst_window`）:S2 分类后对可恢复错误（contract_error/parse_error/truncated）走错误反馈 prompt 重试 ≤3 次；基建错误（transport/http/response）走原样重调用+退避 ≤3 次。耗尽则标 unrecoverable 留白。仓库默认生产配置为 `semantic.recovery.enabled=true,max_retries=3`；旧自定义配置缺失 recovery 或显式关闭时退化为单次调用，与现行零容忍一致。Phase3a manifest 记录本次有效 recovery 值。

## 2. 模块与入口

| 模块 | 入口 | 职责 |
|---|---|---|
| `phase3a_analyst.py` | `run_phase3a:160` | 本地主流程:切窗、逐窗规划、调 LLM-A、写 manifest |
| `scene_context.py` | `build_scene_contexts:74`、`classify_frame_scene:35`、`extract_actions:152` | 确定性场景分类、切窗、动作抽取 |
| `commentary_planner.py` | `plan_window:104`、`PlannerState:13`、`fallback_neutral:226` | 选主话题、跨窗去重与配额、空间过滤,产出 CommentaryPlan |
| `spatial_context.py` | `resolve_spatial_context:177` | 局部空间归属(锚点、附近队友/敌人),fail-closed |
| `kill_semantics.py` | `build_kill_topics:115` | 击杀语义识别(串杀/扫转/武器压制/多杀/特殊),需几何或弹药证据 |
| `rule_compare.py` | `compare_kill()`、`enrich_kill_actions()` | fail-closed 击杀规则比较：空中/盲狙/背身/单向发现/甩枪/移动/单发/切装备等；记录 POV 角色 |
| `hype_score.py` | `compute_hype:18`、`dominant_round_emotion:105`、`_scene_scream_eligible:124`、`_compute_char_budget:291` | 硬事实强度、情绪档位、惊叹资格、字数预算 |
| `llm_projection.py` | `build_llm_window_projection:164`、`build_rule_state_delta:118` | 把规则 plan 白名单化为 LLM 可见事实;增量队伍存活/HP |
| `tactic_book.py` | `load_tactic_book:182`、`compile_tactic_book:159` | 战术规则书严格加载与编译 |
| `tactic_matcher.py` | `match_window:323` | 无未来泄露的窗口内战术匹配 |
| `tactic_projection.py` | `build_window_rule_projection:21` | 串联战术匹配与 planner,分离公开 plan 与 debug 证据 |
| `neutral_contract.py` | `new_manifest_metadata:21`、`validate_neutral_manifest:30` | 产物契约常量与 fail-closed 校验 |
| `phase3a_prompt.py` | `_build_window_prompt:30`、`_parse_window_neutral_response:66` | 本地 prompt 组装与严格响应解析 |
| `phase3a_audit.py` (S5 新增) | `build_audit_artifact:57`、`build_window_statistics:122`、`read_audit_artifact:172` | 审计产物 `llma_input.json` 升级（contract version 2）、窗口级统计、源哈希；向后兼容 v1 读取 |
| `phase3a_cloud_runner.py` | `run_cloud_phase3a:62` | 云端主流程(每回合一次请求) |
| `rule_neutral_renderer.py` | `render_neutral:56`、`render_capsule:95`、`validate_preserved_facts:161` | 纯模板中性句渲染器(v4 `rule_template` 路径)：白名单连接词、capsule 最短完整句、原子级 preserved 校验 |

规则参数来自 `Prompt/json/cs_game_rules.json`(`load_cs_game_rules`)与 `Prompt/json/hype_rules.json`(`load_hype_rules`),进程内缓存(`sbmachine/common.py:70-85`)。LLM-A 提示词模板取自 `Prompt/analyst_system.txt`、`Prompt/analyst_round.txt`(`core/prompt_loader.py`)。

`rule_compare` 遵循四层次序：单事件布尔/坐标字段 → 跨 tick 行为证据 → 回合比分/消费上下文 → 去重与播报配额。任一增量字段或 capability 缺失就不触发对应标签。`ct_score/t_score` 先还原为无未来泄露的 `score_before`；`money_spent_this_round` 只按阵营汇总为实际消费额，明确不冒充当前装备价值，因此不会直接套用参考项目的 ECO/全起结论。POV 角色只来自 `who.pov_player`：killer/victim 获得主角加权，observer 降权，unavailable 不加权并交由原空间锚点降级；不会把降级锚点伪装成 POV。

> `manual_notes.py`(人工逐局笔记)已实现但**未接入 Phase3a 主链**:无任何生产调用点,`tests/test_phase3a_manual_notes.py:41-43` 反而断言笔记不进入 prompt。

## 3. LLM-A 可见事实白名单

传给 LLM-A 的窗口 payload 由 `build_llm_window_projection` 白名单化，包含 `projection_version=2`、`window_id`、`main_topic`、`selected_actions`、`required_facts`、完整 T/CT `rule_state`、字符约束和可选 `tactic_hint`。原始坐标、未选事件、suppressed topics、POV/空间审计、原始帧和 evidence 不进入 prompt。

关键约束：主话题摘要以 `required_facts[].canonical_text` 为权威；LLM 可以自然组织语序，但必须覆盖必需文本与锚点，不能依靠自由发挥补齐规则未投影的事实。完整 snapshot 同时保留 T/CT，`changed_teams` 只表达变化范围，不删除未变化方。

## 4. 战术规则书(tactic_*)

- 数据来源:`load_tactic_book` 只读 `database/tactics/<map>.json`(`sbmachine/tactic_book.py:186-188`),任何异常静默回退空集。
- 严格编译:`compile_tactic_book:159` 要求顶层恰为 `{version:1, map, tactics}` 且 `map` 与传入一致;任一条规则格式非法即整本书失效(`:173-175`);rule_id 去重(`:177`)。条件 kind 限 `{zone_count, alive_count, event_count, bomb_planted}`。
- 无未来泄露:`match_window`(`tactic_matcher.py:323`)只取候选时刻及之前的证据(`_event_rows:149`,`time <= candidate_time`);唯一最高优先级,平级则放弃(`:366`);`active_rule_ids` 抑制同一规则跨窗重复触发(`:369-372`)。
- tactic_hint 三层剥离:`to_prompt_payload`(`tactic_matcher.py:22`)给出 `rule_id/label/hint/matched_at`(不带 evidence)→ `plan_window`(`commentary_planner.py:186-197`)校验后写入 plan → `_project_tactic_hint`(`llm_projection.py:63`)最终只留 `rule_id(≤120)/label/hint(≤240)`,**丢弃 matched_at**。evidence 只进 debug(`tactic_projection.py:44-45`,落 `output/debug_phase3/r*_w*_tactic_match.json`)。

> 现状:`database/tactics/` 目录当前不存在,`tactic_book.tactics` 恒为空,**战术匹配恒不触发**,tactic 相关 kind 无数据可走。

## 5. 空间关系 fail-closed

- 地图模板:`load_map_template`(`sbmachine/spatial_context.py:23`)读 `database/maps/<map>.json`,且仅当 `_is_reviewed_template` 通过才作权威,否则降级为无模板(`:36`)。
- reviewed 硬门槛:`_is_reviewed_template:87` 要求 `version≥2`、`source.manual_review_required==True` 且 `manual_reviewed==True`、callouts 非空,且每节点含非空 `zh`、int `level`、非空 `layer`。
- `map_precision`(`:153`):通过=`reviewed_graph`,否则=`coordinate_fallback`。coordinate_fallback 下 `nearby` 恒空(`:194`),`_callout_relation` 直接返回 `coordinate_unverified`,不产生邻接/路径/跨层关系。
- 只信人工边:`_trusted_edges:109` 只认 `source ∈ {manual, manual+observed}` 的 `directed_transitions`,observed-only 边不算空间事实。
- 锚点:优先 POV(`source=pov`,confidence 1.0,`:185`);无 POV 时按场景取对方孤立选手,需连续两帧同一人(`_stable_isolated`,confidence 0.75)。`plan_window:121` 只要锚点存在即用 `local_actions` 过滤 owned_actions(`_local_actions:156`:保留锚点/附近人参与动作 + bomb_planted/defuse_started 全局事实)。

> 现状：`database/maps/ancient.json` 已是 v2 且人工复核通过，可进入 `reviewed_graph`；其他没有 reviewed 模板的地图仍回退 `coordinate_fallback`。

## 6. 产物契约 `rounds_with_neutral.json`

顶层 metadata 按生成器分流：`legacy_llma` 写 `schema_version=3`、`phase3a_mode`（本地 `llma_slicer_then_llma_analyze`；云端 `cloud_round_timeline`）；`rule_template` 写 `schema_version=4`、`phase3a_mode="rule_neutral_renderer"`。两者均含 `run_id`、`source_rounds_sha256`，本地另加 `video_path/map_name/model/rounds`。

- round 字段(`:295-305`):`round_no/start_sec/end_sec/demo_round_hint/round_emotion/peak_hype/avg_hype/analyst_failed/scenes`。
- scene 字段：`window_id/t_start/t_end/context_start/context_end/scene/actions/commentary_plan/neutral/fact_anchors/hype/scream_eligible/char_budget`。方案 R 字段：`retry_count`、`first_attempt_status`、`first_attempt_detail`；`neutral_source` 可为 `llm`、`llm_retry`、`intentional_empty` 或 `unrecoverable`。
- `commentary_plan` 是审计记录,保留 `spatial`(锚点、附近人名、callout、距离)等原始信息,与"给 LLM 的投影"是两份不同数据。

校验：v3 由 `validate_neutral_manifest` fail-closed 校验；v4 由 `validate_neutral_v4`/`validate_neutral_v4_publishable` 校验。两条路径都要求身份、时间槽、事实字段和窗口边界满足对应 schema，不能用 v3 校验器验证 v4 产物。

LLM-A 响应仍要求严格 `{"neutral": string}`，随后执行确定性语义和预算验收；失败按方案 R 分类重试，不能只因 JSON 契约通过就采纳。

### 6.1 v4 模式：原子事实与规则中性句（`phase3a_generator.mode=rule_template`）

由 `semantic.phase3a_generator.mode` 二选一：当前仓库配置为 `legacy_llma`，走旧 LLM-A 路径；`rule_template` 不调用 LLM-A，纯规则产出。两模式对外契约分离：v4 写 `schema_version=4` + `phase3a_mode="rule_neutral_renderer"`，legacy 维持 v3。

- **原子事实**：`commentary_planner.build_atomic_fact_units`（`commentary_planner.py:1047`）只消费结构化 `event_ledger`/`selected_actions`/`main_topic`，**禁止从 summary 反推事实**。稳定 ID 格式 `fact:v1:<window_id>:<kind>:<tick 5位>:<fingerprint8>`；fingerprint 只哈希规范化的事实类型/tick/主体/客体/结果（不含 canonical_clause、不含数组序号）；派生事实（round_result 等）显式 `origin="derived"` + `source_tick_range`；语义重复事件去重；同 ID 映射到不同 payload 时 fail-closed 抛错；旧 `topic:*` 只存在于 legacy 适配器。
- **中性渲染**：`rule_neutral_renderer.render_neutral` 按 `priority desc, anchor_tick asc, fact_id asc` 稳定排序，单事实直接用完整 canonical_clause，多事实用白名单连接词（"，随后"/"，同时"），禁止因果/转折/强度判断；`render_capsule` 用各事实族最短完整句模板覆盖全部 required 事实，**禁止对子串截断**。输出后必须过原子级 `validate_preserved_facts`（`preserved_fact_ids` 由校验器计算，模型不得自报）。
- **v4 scene 增量字段**：`neutral_renderer{selected,policy}`、`rule_capsule`、`fact_catalog`、`required_fact_ids`、`render_slot{start_sec,end_sec,start_tick,end_tick,continuity_group_id,gap_policy}`（tick=秒×30 取整；连续组默认 `null`/`independent_window`）、`speech_budget{target_units,hard_units,profile_id:"speech-profile-v1"}`；`char_budget` 保留为 legacy 诊断。静默窗口保持 `neutral_source="intentional_empty"`。
- **前置失败**：最短 capsule 在最高语速（1.5×）安全上界仍超固定 slot 时，写 JSON 前以 `RendererUnfitError`（rule_contract_unfit）失败，不交付不可执行的任务单。无 validated profile 期间用保守护栏（5 字/秒 × 1.5 倍）估算，不冒充分级。
- **契约**：`neutral_contract.validate_neutral_v4` fail-closed 校验（v4 身份、mode、slot tick 与 t_start/t_end 一致性、fact ID 正则、required⊆catalog、capsule 非空、speech_budget 区间）；preflight `validate_neutral_v4_publishable` 按 schema_version 分流。
- **配置**：`phase3a_generator.tiny` 块仅声明 Tiny 候选（`enabled=false, shadow_only=true, fallback_to_template=true`），本计划不实现 tiny_assembler；Tiny 须通过独立数据门禁（训练 2000/验证 300/测试 300）与下游盲评后才可写入生产配置。
- 同一 v4 scene 只交付一个 `neutral`，绝不并行保存模板/Tiny 双份正文；A/B 双臂只存在于独立离线评估集。

## 7. 本地 vs 云端

本地 `run_phase3a` 与云端 `run_cloud_phase3a` 是两条独立主流程（云端仅 `run_llma_api.py` 调用）；`phase_semantic` 子进程内的 3a 走本地流程，但 `semantic.analyst_backend: api` 时其 LLM 调用改用云端会话封装（见下）。

| 维度 | 本地 `run_phase3a` | 云端 `run_cloud_phase3a` |
|---|---|---|
| 调度入口 | `phase_semantic`、`run_all`(流水线内,`run_all.py:441`) | 仅 `run_llma_api.py:32`(独立,**未接入 run_all**) |
| LLM 粒度 | 每窗口一次 | 每回合一次 |
| 产 scene | 每窗一条(失败用 fallback) | 每回合至多一条(LLM 选一个 window_id,可静默 0 条) |
| 响应契约 | `{neutral}` | `{window_id, neutral}` + `response_format=json_object` |
| neutral 长度 | `effective_char_limit = min(100, max(1, char_budget))`，每窗预算 `char_budget = max(8, int(duration × base_char_per_sec × emotion_factor))`；语速 `base_char_per_sec` 默认 5.0，三档 `char_budget_factor`(平述 0.95/激动 1.1/惊叹 0.88)取自 `config/llm.yaml` 的 `semantic.speech_rate`，可由 `hype_rules.json` 降级 | 硬校验 ≤100 字(`phase3a_cloud_runner.py:34`) |
| phase3a_mode | `llma_slicer_then_llma_analyze` | `cloud_round_timeline` |
| 契约校验 | 由调用方外部 `validate_neutral_publishable` | 内联 `validate_neutral_manifest`(`:148`) |
| 采样默认 | temp 0.3 / top_p 0.9 / max_tokens 256 / repeat_penalty 1.3 | temp 0.2 / top_p 0.9 / max_tokens 2048 |
| 训练样本 | `accept_api_response` 采纳(`:347`) | 无 |
| 中间产物 | 额外写 `llma_input.json`(`:326`) | 无;原子替换写盘 |
| 输入 sha 防变 | 无 | 跑前后比对,变则报错(`:141`) |
| 失败熔断 | >50% 回合失败即 `sys.exit(1)`(`:332-337`) | 无(`analyst_failed` 恒 False) |

**本地流程的云端后端**（`analyst_backend: api` 时，`phase3a_analyst.py:888-892` 改用 `cloud_memory.make_generate`）：
- 每窗口一次请求，经 `cloud_memory.MatchConversation` 会话（按 run_id 缓存，`cloud_conversation_max_rounds` 上限，成功轮业务验收后 `commit_round` 才入历史，失败轮不入防污染）。**Phase3a 云端默认无会话**（`cloud_analyst_conversation_max_rounds=0`，无状态路径，可自由并发）。
- 回合级并发：api 后端 `analyst_concurrent_rounds` 默认 5（vllm 保持 1），会话/预算状态线程安全。
- **窗口级并发（速度优化计划 §阶段3）**：`analyst_window_concurrency` 控制两阶段方案——规则预计算（`_WindowRequest`，`PlannerState` 仅此阶段修改）→ 窗口请求并发（全局共享 pool）→ 按 `window_id` 顺序验收/兜底/诊断/落盘。当前仓库配置为 7；缺少配置时 API 缺省 4、本地 vLLM 缺省 1。
- 预算护栏：`cloud_token_budget_enabled` 默认 false=永不 silence；开启后超限窗口返回 `budget_silence`（`_is_budget_silence` 识别为预期留白，不判错、不重试、`neutral_source=intentional_empty`）。
- 云端 max_tokens：当前仓库配置 `semantic.cloud_analyst_output_max_tokens=16384`，直接覆盖本地 `analyst_output_max_tokens`，防思考模型 truncated。
- 请求护栏（§阶段1）：`total_timeout_sec` SSE 硬总时限、`cloud_request_concurrency` scope 信号量、`cloud_queue_timeout_sec` 排队上限。
- 成功响应缓存（§阶段4）：`cloud_cache_enabled=true` 时仅缓存业务验收成功的响应（`cloud_memory.commit_round` 路径转正）。

## 8. 验证方法

- 配置体检(不调模型):`python -m sbmachine.phase_semantic --config config/ --dry-run`(`phase_semantic.py:41-45`,输出 preflight 报告)。
- 发布契约:`validate_neutral_publishable`(`sbmachine/preflight.py:428`)——拒绝 `analyst_failed` 回合、拒绝与 `fallback_neutral(plan)` 相同的 neutral(规则兜底不可发布)。方案 R 扩展:`llm_retry` 视为合法 success 来源；`unrecoverable` 窗口按 K 配额裁决（`K = max(2, floor(0.03 × windows_total))`，恢复未启用时 K=0）；基建错误占比 >10% 立即中止整场。
- 相关单测:`tests/unit/test_rule_compare.py`、`tests/unit/test_utility_projection.py`、`tests/unit/test_phase2_semantic_export.py`，以及既有 commentary/kill/tactic/payload/contract 测试。

## 9. 已知偏差

README 与记忆中关于 Phase3a 的过时或不符描述(如"LLM 结合画面+全场记忆写稿"、`round_emotion` 命名等),统一见 [`docs/known_discrepancies.md`](../known_discrepancies.md)。
