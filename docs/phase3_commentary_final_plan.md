# Phase 3 解说链路最终设计

本文是 Phase 3 的唯一权威设计说明。Phase 3 将 DEM 事实、确定性规则和两个 LLM 的职责严格分离。

## 1. 总体原则

```text
DEM 硬事实
  → 规则层提取场景、动作、选题和空间关系
  → LLM-A 生成中性解说
  → LLM-B 只做风格化
```

- DEM 是击杀、C4、玩家状态、道具和回合结果的唯一事实来源。
- 画面/VLM 只能提供定位、POV、OCR 和风味信息，不能覆盖 DEM 事实。
- 复杂 CS 规则必须在确定性规则层完成，LLM 不负责猜测事实。
- 证据不足时保持静默，不为了覆盖率编造内容。

## 2. Phase 3a：规则规划与中性解说

Phase3a 读取 `rounds_with_yolo_semantic.json`（`rounds_with_yolo.json` 只作完整视觉审计与身份锚点），按回合窗口生成唯一的 `CommentaryPlan v2`。规则层负责：

1. 场景窗口切分；
2. 击杀、道具、炸弹和回合结束等动作提取；
3. POV、队友/敌人、地图锚点和附近人物关系筛选；
4. 先比较单事件字段，再比较跨 tick 的移动/朝向/fire/equip/snapshot 证据，并使用无未来泄露的回合开始比分与实际消费汇总，最后计算击杀语义、选题和播报优先级；
5. 为每个硬事件生成稳定身份、归属与抑制原因；
6. 生成带 `required_facts` 和字符预算的有限上下文。

POV 是选题主角而不是普通空间锚点：`who.pov_player` 是击杀者或受害者时分别记录 `pov_role=killer/victim` 并加权；受害者 POV 的中性摘要仍以 POV 玩家为句子主语。只有 POV 缺失时，空间层才允许按稳定孤立选手降级，且降级锚点不得被标成 POV。所有新增规则均 fail-closed，缺字段即不触发。

LLM-A 只负责把计划写成一句中性事实稿，输出必须是严格 JSON：

```json
{"neutral":"一句中性事实解说"}
```

不得输出 Markdown、解释文字或额外字段。空计划保持静默，不调用模型。

## 3. Phase 3b：风格化

LLM-B 只接收当前 neutral、事实锚点、窗口时长、字数预算、选手别名、上一句短尾部和最近风格残余。它不读取 DEM、VLM、原始事件、地图关系或历史解说库，也不重新判断击杀和战术。

输出格式：

```json
{
  "commentary": "风格化解说",
  "felt_intensity": 0.55
}
```

neutral 为空时仅在 `intentional_empty` 下作为正常静默；`unrecoverable`、风格失败或逐窗对账缺失均不可伪装为静默发布。LLM-B 输出还必须通过事实锚点、字符预算、完整句和风格残余复读校验。

## 4. 产物契约

`rounds_with_neutral.json` 必须包含：

```json
{
  "schema_version": 3,
  "phase3a_mode": "llma_slicer_then_llma_analyze",
  "run_id": "...",
  "source_rounds_sha256": "...",
  "rounds": []
}
```

每个 scene 必须携带完整形状的 `fact_anchors`。Phase3b 必须拒绝旧 schema、错误 mode、缺失 run ID 或 source SHA-256 不匹配的输入。

`commentary.json` 使用 `commentary_schema_version=2`，记录来源 neutral 身份、来源窗口总数和覆盖全部窗口的 `window_results`。仅 `ok` 与合法 `silent` 回合可发布；`partial`、`style_failed`、`upstream_failed` 均被发布门禁拒绝。`rounds_with_commentary.json`、`commentary.json` 和最终产物必须来自同一回合、同一窗口、同一文本和同一情绪分段。

## 5. 空间与地图

地图模板位于 `database/maps/<map>.json`。没有 `manual_reviewed: true` 的模板不得产生附近人物、路径或跨层关系，只能使用坐标兜底。人工地图边和名称必须标记 `source: manual` 或 `manual+observed`，并遵守 `one_way` 方向约束。

## 6. 不允许恢复的旧能力

- MatchMemory、历史 few-shot 和跨回合情绪状态；
- 旧 commentary schema 和并行事实文件；
- 让 LLM 自行从原始 players/events 推断规则；
- 用画面或模型文本覆盖 DEM 事实。

任何 schema、规则或 Prompt 变更都必须同步更新契约测试和本文件。
