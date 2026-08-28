# 文档阅读顺序

建议按以下顺序阅读项目文档：

1. [`../README.md`](../README.md)：项目目标、数据流和运行入口。
2. [`../config/pipeline.yaml`](../config/pipeline.yaml)：阶段开关、路径和服务编排。
3. [`../config/yolo.yaml`](../config/yolo.yaml)：Demo、YOLO、OCR 和时间轴配置。
4. [`../config/llm.yaml`](../config/llm.yaml)：LLM-A、LLM-B 和后端配置。
5. [`phase3_commentary_final_plan.md`](phase3_commentary_final_plan.md)：Phase3 当前唯一权威设计。
6. [`modules/`](modules/)：按阶段阅读实现细节。
7. [`known_discrepancies.md`](known_discrepancies.md)：记录文档、代码与记忆的已确认冲突。

## 排查顺序

遇到运行问题时，依次检查：

```text
配置合并
→ 输入文件与 manifest
→ demo_parse
→ video_marking / Phase2
→ VLM/LLM 服务健康状态
→ Phase3 schema 契约
→ Phase4 TTS 与音视频输出
```

当前提交不跟踪 `docs/plan/`、`docs/done/`、`docs/report/` 下的历史归档，不应把这些路径作为仓库内文档链接。

## 地图模板工具（tools/maps）

### 交互流程

`initialize_map_template.py` 分三轮收集人工信息，生成 v2 地图 JSON：

1. **第一轮** — 为每个点位填写中文名（回车保留默认值）。
2. **第二轮** — 为每个点位填写层级整数（任意整数，默认 `1`；回车接受当前值）。
3. **第三轮** — 用编号选择起终点和关系类型，添加真实可走连接。

关系类型编号：

```
1=walk  2=stairs  3=ladder  4=ramp  5=vent  6=drop  7=lift
```

### 连接填写规范

- **"相接"** 表示存在真实可移动路径（walk/stairs/ladder/ramp/vent/drop/lift），不等同于"能看到"。
- **"能看到"** 不计入相接，不要填写视野关系。
- 非单向连接自动生成反向边；`drop` 默认提示单向，仍可人工改为双向。
- Nuke 三楼到 A 包只有存在跳下、楼梯或梯子等真实路线时才填写。

### 示例

数字输入：

```
起点编号：1
终点编号：3
关系类型编号 [1=walk]：6
单向？[Y/n]：
层级变化（起→终）[-2]：
```

生成的 JSON 片段：

```json
{
  "from": "ThirdFloor", "to": "ASite", "kind": "drop",
  "level_delta": -2, "one_way": true, "samples": 0, "source": "manual"
}
```

### 重要限制

- `build_map_template.py` **不负责推断关系**，只统计 tick 坐标与人工边的观测计数。
- 不自动从坐标、z 值或玩家移动创建连接。
- 不新增 `visibility_relations`；视野关系永远不产生邻接判定。

### 命令

```bash
# 初始化新地图模板
python -m tools.maps.initialize_map_template --map de_nuke --ticks ticks.jsonl

# 仅打印点位目录，不写文件
python -m tools.maps.initialize_map_template --map de_nuke --ticks ticks.jsonl --show-catalog

# 用 tick 数据标定已有人工模板
python -m tools.maps.build_map_template ticks.jsonl --map de_nuke \
  --base-template database/maps/de_nuke.json --output database/maps/de_nuke_calibrated.json
```

## 战术规则填写器（tools/tactics）

### 命令

```bash
# 交互式新增战术规则
python -m tools.tactics.initialize_tactic_book

# 指定地图名跳过第一步输入
python -m tools.tactics.initialize_tactic_book --map de_ancient
```

### 交互流程

1. 输入地图名（或 `--map` 参数指定）。
2. 输入**战术名**，例如"假爆A真打B"。
3. 输入**解说提示词**；回车则与战术名相同。
4. 选择**阵营**：`1=T`、`2=CT`。
5. 程序展示区域编号表，例如：
   ```
   1. A小（A_Short）
   2. B区（B_Main）
   3. 中路（Mid）
   ```
6. 选择**分区编号**，输入该分区**人数范围**（见下方格式），再选择**动作条件**（0–3）。
7. 询问"继续添加分区"：是则回到第 6 步，否则进入预览。
8. 预览后选 `s=保存 / a=放弃 / q=退出`。

### 人数范围格式

| 输入 | 含义 |
|------|------|
| `1`  | 恰好 1 人 |
| `3-5` | 3 到 5 人 |
| `4+` | 4 人及以上 |

### 动作编号

| 编号 | 动作语义 |
|------|---------|
| `0`  | 不需要动作条件 |
| `1`  | 投掷道具（`utility_throw`） |
| `2`  | 击杀（`kill`） |
| `3`  | 闪光（`flash`） |

动作条件固定生成"该阵营在该区至少一人做过一次"，`count=[1,null]`。

### 输出位置

`database/tactics/<map_name>.json`；新规则追加在已有规则末尾，旧规则顺序和内容不变。
保存前经 `compile_tactic_book()` 校验；校验失败时原文件不被覆盖。

### 运行测试

```bash
python -m pytest tests/unit/test_tactic_authoring.py -q
```

---

1. 仅维护可复用的 `database/tactics/<map>.json`；不填写、也不读取 `database/match_notes/`。callout、side、道具 type 必须先从真实 debug/语义帧枚举中确认，未知值不猜。

       python -m pytest tests/unit/test_tactic_matcher.py -q
       python -m sbmachine.phase_semantic --config config/

2. 审计 `output/sbmachine/llma_input.json` 的每个窗口：只有命中窗口可有 `tactic_hint`，且它只含 `rule_id`、`label`、`hint`；不得含 matcher evidence、`where.players`、`events`、原始 callout 或坐标。
3. 抽查两条首批规则：A 小 T 方 1 人在候选时刻前投至少两颗规则指定道具、B 区 T 方 3–5 人时才出现“假爆A真打B”；T 方中路 4–5 人时才出现“中路摆谱中期反清”。同 priority 规则并列、坏规则书、未知字段均必须静默。
4. 云端输入也只可包含窗口规则投影（`main_topic`、`selected_actions`、`rule_state`、短 `tactic_hint`）；`rule_state` 只能是 T/CT 存活人数与总 HP 的首窗快照或增量，不得出现 roster/raw timeline。抽查链路为 `rule_id → matched_at → 原始 frame/event`；事实违背率超过 5% 即停用对应规则。
5. 确认 LLM-A 正常生成中性稿；无命中时走普通规则规划/静默，不能凭空生成战术。
