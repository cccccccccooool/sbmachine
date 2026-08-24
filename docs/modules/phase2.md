> 基于代码核对(2026-08-20),仅记录代码/配置/测试已确认的事实,运行行为以源码为准。

# Phase2 视觉链

## 1. 职责边界

Phase2 只做四件事,不做视觉语义描述:

1. 为每个回合生成固定采样率的 DEM 时间轴。
2. 用 HUD YOLO 定位 timer / score / POV 姓名栏坐标(YOLO 只回坐标,不读数值)。
3. 在坐标框上跑 timer / score / POV OCR。
4. 按 tick 从 demo 注入 `when/who/where/events` 事实。

全流程无 VLM、无"描述战场"环节:`sbmachine/phase2_yolo.py` 的主循环只调用 YOLO 门控、OCR 与 `DemoQuery`,不引用任何视觉描述模型。YOLO 明确声明"仅返回坐标、不读取玩家数值/计时器文本/C4 状态/击杀栏内容"(`sbmachine/phase2_yolo_gate.py:20-25`)。硬事实(比分、击杀、炸弹、玩家状态)一律来自 demo 解析,视觉只提供坐标与 OCR 文本。

## 2. 模块与入口

| 模块 | 职责 | 关键函数 (path:line) |
|---|---|---|
| `sbmachine/phase2_yolo.py` | Phase2 主循环:建时间轴、解码、YOLO、OCR 调度、背景注入、写盘 | `run_phase2()` :54;`should_sample_alignment_ocr()` :30;`write_semantic_frames()` :328 |
| `sbmachine/phase2_yolo_gate.py` | HUD 区域路由器:检测框 → 结构化坐标,判白屏 | `YoloGate.decide()` :55;`structure_background()` :91;别名 `YoloUiDetector` :145 |
| `sbmachine/phase2_ocr.py` | OCR、候选增强、短窗共识、区域筛选 | `read_ocr_text()` :59;`OcrConsensus` :100;`_detect_pov_ocr()` :176;`_detect_score_ocr()` :218;`pov_white_text_ratio()` :201 |
| `sbmachine/phase2_background.py` | 单帧背景行构建、回合号补齐、C4/点位判定 | `build_background_info()` :125;`resolve_demo_round_hints()` :21;`_demo_round_hint()` :7;`_c4_planted_at()` :92 |
| `sbmachine/phase2_timeline.py` | 生成固定间隔 `(time_sec, requires_frame)` 网格 | `build_timeline()` :5 |
| `sbmachine/phase2_quality.py` | 把连续无检测帧合并成告警区间,只提示不阻断 | `coalesce_yolo_gaps()` :5 |
| `sbmachine/demo_query.py` | demo 产物只读查询层(状态/事件/道具活跃) | `DemoQuery.load()` :168;`state_at()` :241;`kills_between()` :276;`match_player()` :200;`round_by_no()` :215;`tick_rate` :174 |
| `sbmachine/utility_projection.py` | 把 `grenades.json` 投影为每次投掷唯一的稳定事件，并校正烟/火生效 tick | `project_grenade()`；`project_grenades()` |

## 3. 统一时间轴与采样

- **固定 1FPS 网格**:`build_timeline` 按 `phase2_interval_sec`(默认 1.0s,`config/yolo.yaml:43`)生成 `(ts, requires_frame=True)` 序列,所有帧 `requires_frame` 恒为 True(`phase2_timeline.py:5`)。这是采样网格,不是视频真实 fps。
- **每局单次 seek + 顺序 grab**:每回合起点 `cap.set(CAP_PROP_POS_MSEC)` 只 seek 一次(`phase2_yolo.py:145`),之后靠 `cap.grab()` 顺序推进到目标 ts(`phase2_yolo.py:166-173`);视频真实 fps 仅用于步进估算,不改变采样密度。
- **timer/score OCR 窗口调度**:由 `should_sample_alignment_ocr`(`phase2_yolo.py:30`)控制。开局 `[0,10)` 秒连续采(`alignment_initial_sec`),随后等 `alignment_period_sec`(20s),在 `alignment_window_sec`(5s)完整窗口内采,即 `[30,35)`、`[50,55)`……片段尾部不足完整 5 秒则跳过该窗口。非调度帧 timer/score 置 `skipped:alignment_schedule`(`phase2_yolo.py:221-222`)。
- **POV OCR 白像素门控**:POV OCR 对每个解码帧都跑(不受上面窗口调度约束,`phase2_yolo.py:223`)。仅使用 YOLO 定位框、无固定 ROI 生产回退;先算框内白色像素比例(HSV 饱和度 ≤ `white_saturation_max`、明度 ≥ `white_value_min`,`phase2_ocr.py:201`),低于 `white_ratio_threshold`(0.01)则 `skipped:pov_white_gate` 不调 OCR(`phase2_ocr.py:180-185`)。结果经 `OcrConsensus` 短窗共识,空/低置信会让旧 POV 过期(`phase2_ocr.py:110-118`)。
- **DEM 背景无条件注入**:`build_background_info` 对每个 ts 都调用(`phase2_yolo.py:251`),即使解码失败(`gate_reason=decode_failed`、`has_frame=False`)仍注入完整 demo 事实,时间轴不留洞(`phase2_yolo.py:233-234`、`schemas.py:40`)。

## 4. 回合对齐:两套对齐器

Phase 相关代码存在**两套独立对齐器**,职责不同,不要混淆:

- **切片期 `sbmachine/round_aligner.py`(L0/L1/L2)**:唯一调用点是切片器 `tools/slicing/run_frame_type_slicer.py:265-267` 的 `validate_segments_with_demo`,只传 `segments/rounds/tick_rate`。因此:
  - **L1 duration DP** 是唯一生效层(Needleman-Wunsch 时长子序列对齐,`round_aligner.py:200`),产出写入 `rounds.json` 的 `align_offset`。
  - **L0 score OCR** 代码完整(`round_aligner.py:27/:209`)但调用处未传 `score_ocr_per_segment`(为 None),分支永不进入 → 未接线。
  - **L2 onset 互相关** 代码完整(`round_aligner.py:133/:244`)但调用处未传 `onset_per_segment` → 未接线。
  - demo 缺失时全标 `unmatched`;对齐异常时全标 `error:<类型>`,绝不静默退回按位置映射(`run_frame_type_slicer.py:250-272`)。
- **运行期 `sbmachine/time_align.py` 的 `RoundTimeAlign`**:Phase2 逐帧实际使用的对齐器。
  - timer OCR 命中即 `add_anchor` 写锚(`phase2_background.py:143-145`);
  - 连续 `plant_empty_timer_frames`(3)空 timer 且已有锚点、demo 有 `bomb_planted_tick`、预测植弹时刻与当前 ts 在容差内 → `align.freeze` 冻结偏移(`phase2_yolo.py:236-249`);
  - **偏移优先级**:frozen > 锚点中位数 > provisional(即 L1 的 `align_offset`)> `freeze_end_tick`(`time_align.py:125-131`)。

## 5. `rounds_with_yolo.json` 契约

`run_phase2` 输出仍是 `MatchPackage` schema,每回合 `_phase2_yolo` 填充为 `YoloData`(dict 键带下划线前缀,`schemas.py:134-135`)。

- **YoloData 字段**(`schemas.py:43`):`background[]`、`key_frames[]`、`yolo_required`、`yolo_model`、`detector_mode="demo_timeline_yolo_ocr"`、`sample_interval_sec`、`total_yolo_frames`、`detection_warnings[]`。
- **KeyFrame 字段**(`schemas.py:32`):`time_sec`、`gate_reason`、`yolo_tags`、`yolo_confidence`、`ui_regions`、`background_info`、`has_frame`。
- **background_info 真实结构为嵌套四段**(`phase2_background.py:196-236`),**不是**扁平的 `hud_detected/timer_value/timer_raw/timer_confidence` 等字段:
  - `when`:`video_time / timer / timer_source / tick / tick_rate / round_no / relative_sec / phase(pre_round|in_round|post_round) / align_frozen / align_warnings`。timer 存于 `when.timer`。
  - `who`:`pov_player / ocr_raw / match_score / ocr_engine / pov_source / view`。
  - `where`:`pov_callout / players[]`(players 已剔除 steamid)；玩家新增 `yaw/pitch/is_walking/is_airborne/is_ducking/is_scoped/zoom_level/in_bomb_zone/money_spent_this_round`。
  - `events`:`kills / utilities / damages / flashes / smokes_active / infernos_active / weapon_fires / item_equips / event_snapshots / c4{...} / score_ocr{ct,t,raw,source,confidence}`。`utilities` 每条投掷仅出现一次，优先用 `stable_event_id` 去重；比分 OCR 仍存于 `events.score_ocr`。
- **向后兼容**:读旧盘时 `has_frame` 缺失回退自 `has_vlm`,并丢弃旧 `background_info.what` 字段(`schemas.py:194-198`)。
- 另可选输出精简语义产物(`write_semantic_frames`)。每回合保留旧字段 `{round_no, frames[]}`，并新增 `demo_round_no/map_name/capabilities/round_result`；旧 Phase3 文件仍可读取。真实字段样例见 [`../rounds_with_yolo_semantic.json`](../rounds_with_yolo_semantic.json)。

## 6. demo_manifest 校验(fail-closed)

`DemoQuery._load` 每次加载先调 `validate_demo_manifest`,失败即抛 `DemoManifestError` 中止(`demo_query.py:343`)。校验规则(`tools/demo/demo_manifest.py:103`):

- `demo_manifest.json` 存在、`schema_version == 1`、`status == "complete"`、`source_demo_sha256` 为合法 64 位 hex;
- 9 个必需 JSON 产物齐全 + 至少一个 tick 产物(`ticks.parquet` 或 `ticks.jsonl`); 当 capabilities 声明 `weapon_fire/item_equip/event_snapshots=true` 时，相应 `fired.json/equips.json/event_snapshots.json` 也必须进入 manifest；
- 逐文件校验 `sha256`,并对 json/jsonl 校验行数;
- `manifest.tick_rate == demo_meta.json.tick_rate` 且 > 0。

清单由 `parse_demo.py` 在所有产物落盘后最后写(`tools/demo/parse_demo.py:168-173`),配合临时目录原子替换,保证清单只指向完整数据。

## 7. 验证方法

- **HUD/OCR 覆盖率复测**(按生产 1FPS 调度顺序执行,`tools/diagnose_hud_ocr.py:190`):

  ```bash
  python tools/diagnose_hud_ocr.py \
    --video <clip.mp4> \
    --model <vision.yolo.model_path> \
    --sequential
  ```

  可选参数:`--samples`(默认 30)、`--seeds`(默认 `6657,20260713,42`)。

- **Phase2 契约体检**:`sbmachine/preflight.py:416` 的 `validate_phase2_publishable(path)` 对产物跑 `validate_vision_timeline`,由 `sbmachine/run_all.py` 在 Phase2 结束后自动调用(`run_all.py:332/:426`),契约不符即抛 `PublishContractError`。

- **单元测试**:`tests/unit/test_phase2_alignment.py`、`test_phase2_ocr.py`、`test_phase2_quality.py`、`test_round_aligner.py`、`test_frame_type_slicer.py`。

## 已知偏差

README / 记忆与代码现状的冲突不在本文展开,统一见 `docs/known_discrepancies.md`。
