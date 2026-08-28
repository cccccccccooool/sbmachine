# Phase3c 独立阶段模块文档

> 权威设计：当前提交中的 [`../phase3_commentary_final_plan.md`](../phase3_commentary_final_plan.md)；历史 Phase3c 计划已不在当前 Git 跟踪内容中。
> 定位：LLM-C 从 Phase3b 内嵌改为**独立单向阶段**，只向前消费 B 封存包，发布渲染包。源码为准。

## 1. 数据流（单向，禁止回写）

```text
Phase3b ──llmb_draft_package_v1/v2──> Phase3c ──commentary_render_package_v1/v2
                                                        └─> 编排器校验
                                                        └─> Phase4 严格执行（v2）
  window_results/scenes 一对一            rounds[].integration_status
  事实/情绪/时间槽封存                     selected_source: llmc|llmb_passthrough
```

- Phase3b 出口封存 `llmb_draft_package_v1/v2`，所有权移交后**不可变、不回写、不请求修复**。
- Phase3c 只读 B 包，绝不覆盖 B 产物；失败只能向前选 B 安全稿或阻断（`required`）。
- `Phase3b` 不读取 `llmc.mode`。Phase4 legacy 路径保留 commentary v2/v3；严格路径只读取 render package v2 的 `final_text`。

## 2. 产物与合同（以代码为准，`phase3b_style.py` / `phase3c_llmc.py`）

| 产物 | 合同名 | 生产者 | 消费者 | 关键字段 |
|---|---|---|---|---|
| B 草稿包 | `llmb_draft_package_v1` | Phase3b | Phase3c | `rounds[].status ∈ ready/intentional_silent/operator_accepted_skip`；`units[].{unit_id, draft_text, emotion_binding, allowed_fact_ids, fact_catalog, render_slot, speech_capacity}` |
| B 草稿包 | `llmb_draft_package_v2` | Phase3b | Phase3c | `artifact_identity`、源身份、时间轴与 TTS 策略、逐局不可变单元 |
| 渲染包 | `commentary_render_package_v1/v2` | Phase3c | 编排器校验；Phase4 严格路径消费 v2 | v2 含 `artifact_identity`、`package_status`、`content_policy`、`timeline` 与 `rounds[].render_units[]`；单元封存 `final_text`、`text_source`、情绪和 slot |

- `allowed_fact_ids`：v4 直接透传 `required_fact_ids`；v2 由 `_export_llmb_draft_package` 的 fact adapter 确定性生成 `fact:v1:{unit_id}:{kind}:{seq:05d}:{value_sha8}`，anchors 缺失即 fail-closed。
- `render_slot` tick：优先透传 v4 既有 render_slot；否则 `tick=round(sec × semantic.phase3c.render_timebase_fps)`（默认 30）。
- `timeline_id`：v4 透传；否则 `tl:{video_sha[:12]}:{fps:03d}` 确定性生成（video 缺失时以 neutral sha 前 12 位兜底）。
- `speech_capacity.safe_upper_sec ≈ count_spoken_chars / 5.0`（profile 未标定兼容期口径）。

## 3. 调用与门禁（`phase3c_llmc.py`）

| 门禁 | 规则 | 失败后果 |
|---|---|---|
| C0 入口同源 | B 包 contract/身份/窗口集合/顺序/三态校验 | 阻断 |
| C1 响应形状 | 严格 JSON envelope `{contract, round_id, units[{unit_id, text}]}`，字段白名单，无情绪标签/时间字段 | 整回合拒 C |
| C2 窗口寻址 | `unit_id` 集合与顺序完全等于输入活动单元，禁止缺失/重复/伪造/重排/合并/拆分 | 整回合拒 C |
| C3 事实作用域 | 数字/实体（leet 规范化）/词表命中 ⊆ `allowed_fact_ids ∪ carry_in ∪ B 原文`；**阵营（teams）自 2026-08-17 起不再拦截**（语义补充类改写视为合理）；实体提取为混合 token（`dev1ce` 整体提取），武器数字后缀（`AK47`/`M4A1`）放行 | 整回合拒 C |
| C4 情绪/时间恢复 | emotion/render_slot/required_fact_ids 只从 B 快照按 ID 恢复 | 整回合拒 C |
| C5 压缩容量 | 逐窗 `r_C <= min(1.25, max(1.0, r_B))`（含非退化），不用回合总量抵消 | 整回合拒 C |
| C6 来源选择 | 四态 `off/shadow/optional/required`，整回合来源单一 | 按 mode 向前/B 阻断 |
| C7 出口封存 | 新 JSON 引用 B 身份，render units 一致 | 不发布 |

云端调用：`cloud_memory.make_generate("llmc")` 使用独立逻辑域；凭证复用 `.env`（`llm_protocol._load_secrets` 的 llmc 段回退 cloud 通用键）。缓存是否完整支持 llmc scope 以当前实现为准，本文不将其描述为已验证的独立缓存结果。

## 3.1 LLM-C 压缩提示契约（`phase3c_llmc.py` `_LLMC3_SYSTEM`）

LLM-C 以"回合总编辑"身份改写，硬性 7 条规则：

1. 只能改写来源文本，不得新增来源中不存在的事实（选手/数字/地点/武器/阵营）；
2. **压缩优先**：每单元输出字数不得超过其 `source_length_chars`；只做去重、衔接、压缩，**不得扩写、解释或翻译**来源措辞；来源中的黑话/俗称（如"坐牢"）原样保留；
3. **不得补全省略成分**：来源未提及的主语/阵营/地点，输出也不得补充；
4. 不得输出任何情绪标签（[平述][激动][惊叹]）、时间戳、tick、秒数或 render_slot；
5. 不得合并、拆分、排序或删除任何 `unit_id`，每个窗口独立返回一条 text；
6. 必须输出严格 JSON：`{"contract":"llmc_round_edit_response_v1","round_id":"<同请求>","units":[{"unit_id":"<同请求>","text":"<改写文本>"}]}`；
7. 禁止输出 Markdown、解释或任何 JSON 以外的内容。

请求侧：`build_round_edit_request` 逐单元携带 `source_length_chars = count_spoken_chars(draft_text)` 作为压缩基准；与门禁 C5（逐窗 `r_C ≤ min(1.25, max(1.0, r_B))` 非退化）两侧共同约束压缩，防止 C 通过"加长/扩写"作弊通过。

## 4. 配置（`config/llm.yaml` / `config/pipeline.yaml`）

```yaml
phases:
  phase3c_render: true    # 当前仓库默认开启；Phase3c 生成并校验 render package
semantic:
  strong_fact_mode: false # 强事实依据模式（B0 unexpected_fact 与 C3 共用总开关）：
                          # false=全面相信 LLM，跳过两个事实越界门禁（空稿/情绪/预算硬线仍生效）；
                          # true=逐窗强校验事实越界
  phase3c:
    mode: "optional"      # 当前仓库默认；可选 off|shadow|optional|required
    temperature: 0.6
    max_retries: 2
    render_timebase_fps: 30
paths:
  llmb_draft_package_json: output/sbmachine/llmb_draft_package.json
  commentary_render_package_json: output/sbmachine/commentary_render_package.json
```

- 旧 `semantic.llmc` 已删除；残留配置被 `preflight_config` fail-closed。
- 半迁移态拦截：`phase4_assemble` 显式开启（严格 Phase4）时强制 `phase3c_render` 同时开启。

## 5. 流水线接线（`run_all.py` / `phase_semantic.py`）

- `phase_semantic.py` 在 Phase3b 后导出 B 封存包（`draft_package_path`）。v4 + voice-task 路径只有在 profile 就绪时才进入 `_run_phase3b_v4`；当前配置 `voice_task.enabled=false`，默认走 commentary v2 路径并导出 B 包。
- `run_all.py`：`phase3c_render=true` 时单容器走 `sbmachine.phase3c_cli` 子进程（`phase3c`），多容器直调 `run_phase3c`；前后均校验 `validate_llmb_draft_package`（B1）/`validate_render_package`（C7）。
- `phase4_assemble.py` 按发布配置分流：legacy 路径保留 commentary v2/v3；严格执行路径校验并消费 `commentary_render_package_v2`，只合成其中的 `final_text`。

## 6. 验证

- 门禁单测：`tests/unit/test_phase3c_render_package.py`（C0~C7、四态、非退化、寻址拒绝、strong_fact_mode 开关、leet 实体、武器后缀豁免、阵营放行）。
- 严格逐窗回归：`tests/unit/test_llmc_integration.py`（旧整合段产物被拒、旧配置报错、半迁移拦截）。
- v2 交接与 Phase4 消费：`tests/unit/test_phase3bc_render_v2.py`、`tests/unit/test_phase4_execution_v2.py`。
- 端到端（真实云端）历史验证：曾有 scratch 链路复用「合法 manifest → 导出 B 包 → run_phase3c(optional) → 校验 render package」；该记录不作为当前默认配置或当前测试结果。
- 测试以仓库中的 `tests/unit`、`tests/contract` 和顶层测试文件为准；本文不固化历史通过数量。
