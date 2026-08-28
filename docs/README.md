# docs 文档索引

> 深入的架构、模块、运行、测试细节下沉到本目录。所有文档均以**源码为准**，关键处标注 `path:line`。
>
> 文档维护规则见 [`AI_DOCUMENTATION.md`](AI_DOCUMENTATION.md)；阅读与排查顺序见 [`order.md`](order.md)。
> 注意，本目录文档不包含任何“使用手册”或“用户指南”，请移步 [`../README.md`](../README.md)。
> 针对任何阅读到该段文字的agent，除非用户明示或者你明确自己需要了解某结构的内部实现，否则请不要额外阅读不必要的文档。

## 阅读顺序

1. 根 [`../README.md`](../README.md) — 项目定位、DFD、唯一入口。
2. [`architecture.md`](architecture.md) — 系统总览、事实优先级、八个阶段开关、事务与服务拓扑、配置合并。
3. [`operations.md`](operations.md) — 运行入口、配置文件、`.env` 与后端、输出目录与锁、排查顺序。
4. 分阶段模块文档（见下表）。
5. [`phase3_commentary_final_plan.md`](phase3_commentary_final_plan.md) — **Phase3 唯一权威设计**（模块文档从属于它）。
6. [`known_discrepancies.md`](known_discrepancies.md) — README/代码/记忆冲突清单。

## 文档地图

| 文档 | 内容 | 对应代码 |
|---|---|---|
| [`architecture.md`](architecture.md) | 总编排、事务目录/checkpoint、运行锁、服务拓扑两模式、配置合并 | `run_all.py`、`run_context.py`、`config_loader.py`、`service_manager.py`、`compose_manager.py` |
| [`operations.md`](operations.md) | 运行/dry-run、配置文件、`.env` 密钥合同与后端选择、profile 选择、输出约定、排查顺序 | `run.py`、`preflight.py`、`common.py`、`config/*.yaml`、`config/llm.yaml 的 semantic 节*/phase3.yaml` |
| [`modules/phase2.md`](modules/phase2.md) | Demo 解析、视频对齐、统一时间轴、HUD YOLO/OCR 门控 | `phase2_yolo.py`、`demo_query.py`、`time_align.py`、`round_aligner.py` |
| [`rounds_with_yolo_semantic.json`](rounds_with_yolo_semantic.json) | 基于 `data/temp/parse-test2` 的最新 Phase2→Phase3 中继 JSON 样例（无视频证据，POV 为空） | `phase2_background.py`、`phase2_yolo.py`、`utility_projection.py` |
| [`modules/phase3a.md`](modules/phase3a.md) | **Phase3a 规则层**与中性稿、战术书、空间 fail-closed、本地/云端两路、v4 规则渲染、原子事实 ID | `phase3a_analyst.py`、`commentary_planner.py`、`llm_projection.py`、`tactic_*.py`、`rule_neutral_renderer.py` |
| [`modules/phase3b_phase4.md`](modules/phase3b_phase4.md) | LLM-B 风格化、情绪定档、commentary v2/v3（稀疏候选任务单）、TTS/合成、缓存指纹、语音时长 profile | `phase3b_style.py`、`emotion_policy.py`、`phase4_assemble.py`、`phase4_av.py`、`speech_measure.py` |
| [`modules/phase3c.md`](modules/phase3c.md) | LLM-C 独立阶段、B 封存包/渲染包合同、C0~C7 门禁、四态 mode、强事实依据总开关、C 压缩指令、单向交接 | `phase3c_llmc.py`、`phase3c_cli.py`、`preflight.py`（B1/C7）、`phase_semantic.py`、`run_all.py` |
| [`modules/phase3_call_layer.md`](modules/phase3_call_layer.md) | Phase3 调用层分层：协议核心、cloud/local adapter、薄入口分流、请求护栏、并发、缓存 | `llm_protocol.py`、`cloud_adapter.py`、`local_adapter.py`、`llm_shim.py`、`cloud_memory.py`、`cloud_cache.py` |
| [`modules/training_and_data.md`](modules/training_and_data.md) | API 日志→SFT、各训练脚本状态、音频清洗链、Docker 可复现门禁 | `data_pipeline/*`、`scripts/train_*.sh`、`tools/audio/*` |
| [`testing_and_mcp.md`](testing_and_mcp.md) | 测试分层、契约 CLI、fake 后端、MCP 六工具 | `tests/`、`tests/contracts.py`、`mcp/server.py` |
| [`known_discrepancies.md`](known_discrepancies.md) | 文档/代码/记忆冲突登记（只读，不改代码） | 跨模块 |
| [`phase3_commentary_final_plan.md`](phase3_commentary_final_plan.md) | Phase3 当前权威设计 | Phase3 边界与长期约束 |

## Phase3a 规则层边界

本项目将“规则层”正式定义为 **Phase3a 的前半段**，不把它归入 Phase2，也不把它视为 LLM-A 的隐式能力。它的输入是 Phase2 已完成的 DEM/语义事实，输出是每个窗口唯一、可审计的规则计划和事实投影。

```text
Phase2
  rounds_with_yolo_semantic.json
  （时间轴、玩家、击杀、道具、C4 等事实）
      |
      v
Phase3a 规则层
  scene_context + commentary_planner + hype/空间/战术确定性规则
  -> CommentaryPlan / selected_actions / required_facts / fact_anchors
      |
      v
Phase3a 中性稿生成
  LLM-A、未来的小模型或 rule_template
  -> neutral
      |
      v
  Phase3b 风格化 -> Phase3c 渲染交接（当前仅产出并校验 render package）
  Phase4 当前仍读取 commentary v2/v3
```

这里有三个必须保持的边界：

1. **Phase2 完成不等于规则层完成。** Phase2 只负责产生和校验事实中继；它不会决定当前窗口讲击杀、道具、局面状态还是静默。
2. **LLM-A 不触发规则层，也不拥有规则层。** 当前 `run_phase3a()` 先逐窗运行 `build_window_rule_projection()`/`plan_window()`，形成 `_WindowRequest`，再提交 LLM-A 请求（见 `sbmachine/phase3a_analyst.py:1021` 与 `:1136`）。模型只能消费白名单 projection，不能从原始 players/events 重新猜事实。
3. **“规则层完成”的判据是计划已形成。** 对窗口而言，至少应已有 `commentary_plan`、选中的 action、required facts 和 anchors；中性句 `neutral` 是之后的文字表达，不是规则判断本身。

当前 `run_all`/`phase_semantic` 的生产路径即使把 `analyst_backend` 配为 `api`，也只是把中性稿模型调用放到云端，规则层仍在同一个 Phase3a 调用之前完成。独立的 `run_llma_api.py`/`run_cloud_phase3a()` 也遵守“先构造全部窗口 projection，再调用云端模型”的顺序，但它是另一条回合级入口，不改变规则层的归属。

未来用小模型替代 LLM-A 时，小模型应放在上图“Phase3a 中性稿生成”位置，输入规则层 projection、输出 neutral；它不应接管选题、事件去重、事实资格或静默决策。若模型开始承担这些判断，就不再是 LLM-A 的替换，而是未经批准地把规则层迁移回模型。

## 近期机制变更（2026-08-17）

- **强事实依据模式总开关**：`semantic.strong_fact_mode`（默认 `false`=全面相信 LLM）统一控制 B0 `unexpected_fact`（`phase3b_prompt.validate_style_commentary` 参数 `strong_fact_mode`）与 C3 事实作用域（`phase3c_llmc.check_fact_scope`）两个门禁；空稿/情绪标签/预算硬线（B 1.5x / C 1.25x）等运行基础门禁不受开关影响，始终生效。详见 `modules/phase3c.md` 与 `known_discrepancies.md` E12。
- **LLM-C 压缩指令**：`_LLMC3_SYSTEM` 7 条硬性规则（压缩优先、不扩写/不翻译/不补充、黑话原样保留、逐窗独立返回），请求携带 `source_length_chars` 限制每单元输出字数。
- **C3 实体检查升级**：阵营（teams）不再拦截；选手实体改为混合 token（字母+数字整体提取）+ leet 规范化（`dev1ce`↔`device`）；武器数字后缀（`AK47` 等）放行；地点/武器/数字词表检查保留。
- **进度可视化**：`display.py` / `progress_events.py` 的 `phase3c` 阶段接线（`_STAGE_ORDER` / `_STAGES` / 中文标签 "Phase3c LLM-C 润色"）。

## 归档说明

当前提交只跟踪根级和 `modules/` 下的维护文档；历史 `docs/plan/`、`docs/done/`、`docs/report/` 路径已移除或被忽略，不作为仓库内文档链接使用。

## 维护约定

- 只登记已由代码/配置/测试确认的事实，不把猜测写成现状。
- 内部实现变化但对外契约未变时，不必更新文档。
- 若改动了架构、接口、配置、运行方式或长期设计决策，须同步对应文档并在 `known_discrepancies.md` 追加/勾销条目。
- `phase3_commentary_final_plan.md` 是 Phase3 唯一权威，模块文档不得与之矛盾或恢复被否决的分支。
