# 配置文件说明

`config/` 目录下共 7 个 YAML 文件，按职责分组说明如下。

---

## pipeline.yaml — 总控

项目的主调度文件，控制运行模式、阶段开关和全局路径。

### `profile`
当前运行档位（目前仅为标识，未实际分档）。

### `runtime` — 服务生命周期

| 参数 | 类型 | 说明 |
|------|------|------|
| `manage_services` | bool | `false` = 多容器模式（docker compose），`true` = 单容器内多进程错峰 |
| `compose_file` | str | 多容器模式使用的 compose 文件路径 |
| `one_model_at_a_time` | bool | 单卡错峰：每阶段前启动对应服务，结束后立即停止，避免多模型同时占显存 |
| `use_gpu_guard` | bool | 调试用 GPU 隔离钩子，**正式运行保持 false** |

#### `runtime.services` — 各服务启动配置

每个服务（`vlm` / `sovits`）支持以下子字段：

| 子字段 | 说明 |
|--------|------|
| `enabled` | 是否由调度器自动管理（false = 需手动提前启动） |
| `start` | 启动命令（shell 字符串） |
| `startup_timeout_sec` | 健康轮询超时（秒），超时则报错退出 |

### `phases` — 阶段开关

按流水线顺序依次执行，`true` = 执行，`false` = 跳过。

| 字段 | 对应阶段 |
|------|---------|
| `demo_parse` | 调用 Go 二进制解析 `.dem` → `output/demo/`（硬事实源） |
| `video_marking` | MobileNetV3 预测视频帧类型（game/break） |
| `preprocess_slice` | 回合分段 + 可选视频切片 → `rounds.json` |
| `phase2_vision` | YOLO 门控 + VLM 逐帧描述 → 视觉记录 |
| `phase3_semantic` | LLM 分析（3a 中性稿）+ 风格化（3b 口播稿） |
| `phase4_assemble` | TTS 合成 + 音视频拼装 |

### `debug`

| 字段 | 说明 |
|------|------|
| `phase3` | `true` 时将 phase3 每局 LLM 输入/输出完整落盘到 `output/debug_phase3/` |

### `paths` — 全局路径

| 字段 | 说明 |
|------|------|
| `demo` | 输入 `.dem` 文件路径 |
| `video` | 输入视频文件路径（`.mp4`） |
| `map_name` | 地图名，用于检索口径数据库 |
| `hud_detections_jsonl` | 离线 HUD 检测 JSONL（无 segments 时备用） |
| `segments_json` | 已有分段 JSON，提供则跳过 round_segmenter |
| `clip_dir` | 可选：将每局切出独立小视频的目录 |
| `rounds_json` ~ `assemble_manifest_json` | 各阶段中间/最终产物路径（一般无需修改） |

---

## llm.yaml — 语言模型

控制 Phase 3（语义分析 + 风格化）使用的 LLM 后端和行为。

### `llm` — 后端与调用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `backend` | `api` | 固定为 `api`（云端 OpenAI 兼容接口）。密钥在 `config.yaml` 的 `secrets` 段配置 |
| `temperature` | `0.75` | 生成温度；越高越随机，解说风格类任务建议 0.7~0.85 |
| `timeout_sec` | `300` | 单次 API 请求超时（秒） |
| `num_ctx` | `16384` | 上下文长度（保留字段，云端 API 按模型限制为准） |
| `enable_thinking` | `false` | 关闭 Qwen3 思考链（`/no_think` 前缀注入 system），避免慢速和额外 token 消耗 |
| `frequency_penalty` | `0.3` | 防复读惩罚（API 参数） |

### `semantic` — 双模型与分析参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `model` | `qwen3:8b` | analyst/style 均未指定时的统一回退模型名 |
| `analyst_model` | `qwen3:8b` | Phase 3a 分析模型（中性事实白描） |
| `style_model` | `qwen3:8b` | Phase 3b 风格模型（口播解说，加情绪标签） |
| `analyst_concurrent_rounds` | `5` | 并发分析局数（API 模式下受账号并发限额约束） |
| `analyst_max_tokens` | `3072` | 分析模型单次输出上限；过低会导致 JSON 截断（analyst_failed） |
| `segment_long_rounds` | `true` | 超长回合策略：`true` = 分段分析后合并（无损，K次调用）；`false` = 降采样（快但有损） |
| `segment_overlap_frames` | `2` | 分段时的重叠帧数，仅供上下文连续，输出按归属去重 |

### `paths` — 提示词与数据库路径

| 字段 | 说明 |
|------|------|
| `catchphrase_library` | 口癖词库 JSON（phase3b few-shot 注入） |
| `commentary_demos` | 真实解说稿示例 JSON（API 路径 few-shot） |
| `callouts_zh` | 地图区位中文对照表 |
| `terms_zh` | CS2 术语中文词典 |
| `lineups_dir` | 手雷 lineup 数据库目录 |

---

## vision.yaml — 视觉感知

控制 Phase 2（视频理解）的所有视觉处理行为。

### `demo` — demo 解析

| 参数 | 说明 |
|------|------|
| `parsed_dir` | demo 解析输出目录 |
| `anchor_tolerance_sec` | demo 与视频对齐时的锚点容差（秒） |
| `plant_empty_timer_frames` | 炸弹安置事件起始空帧数 |

### `vision.yolo` — YOLO 门控

| 参数 | 说明 |
|------|------|
| `enabled` | 是否启用 YOLO 预过滤（false = 全帧交 VLM，成本高） |
| `model_path` | YOLO 权重路径 |
| `classes_file` | 类别定义 JSON |
| `conf_threshold` | 置信度阈值（低于此值的检测忽略） |
| `white_frame_mean_threshold` | 白屏过滤均值阈值（>245 = 闪光帧，跳过） |
| `skip_labels` | 不触发 VLM 的标签列表（如 `flash` 闪光帧） |
| `prompt_labels` | 标签 → VLM 提示词映射（告知 VLM 哪些区域已遮蔽） |

### `vision.pov_ocr` — 视角识别

| 参数 | 说明 |
|------|------|
| `box` | 归一化坐标 `x1,y1,x2,y2`，框定"Currently watching: XXX"姓名栏 |
| `min_match_score` | OCR 结果与选手名单最低匹配分 |
| `enabled` | 是否启用视角 OCR |
| `spectator_min_frames` | 旁观模式最少帧数（调优 OCR box 后可降至 3-5） |

### `vision.timer_ocr` / `vision.score_ocr`

YOLO 未能检测到计时器/比分时的固定 ROI 备用 OCR，一般保持 `enabled: false`。

### `vision.align` — 时间轴对齐

| 参数 | 说明 |
|------|------|
| `duration_gap_penalty` | DP 对齐丢段罚分（秒） |
| `onset_tolerance_sec` | onset 容差（秒） |
| `veto_threshold` | 低于此匹配分时降低置信度 |

### `vision.sampling` — 采样策略

| 参数 | 说明 |
|------|------|
| `demo_interval_sec` | 背景行采样间隔（含 demo 事实，不解码画面） |
| `vlm_interval_sec` | 常规 VLM 喂图间隔（秒） |
| `dense_pre_sec` | 事件前加密窗口（击杀/炸弹前 N 秒密集喂图） |
| `dense_post_sec` | 事件后加密窗口 |
| `dense_fps` | 事件加密窗口帧率 |

### `vision.vlm` — VLM 服务

| 参数 | 说明 |
|------|------|
| `endpoint` | VLM 推理服务地址（OpenAI 兼容 `/v1/chat/completions`） |
| `model` | 模型名（如 `Qwen/Qwen2.5-VL-3B-Instruct`） |
| `temperature` | 生成温度（视觉描述任务建议低值，0.1~0.3） |
| `max_tokens` | 单帧描述最大 token 数 |
| `timeout_sec` | 单次请求超时 |
| `batch_size` | 单次打包发送帧数（`1` = 串行；高显存卡可调高） |

---

## tts.yaml — 语音合成与视频输出

控制 Phase 4（TTS + 拼装）的输出路径。

### `tts`

| 参数 | 说明 |
|------|------|
| `config` | GPT-SoVITS 运行时配置文件路径 |
| `output_dir` | 各句音频片段输出目录 |
| `final_audio` | 最终合并语音文件路径（`.wav`） |

### `video`

| 参数 | 说明 |
|------|------|
| `make_filtered_video` | 是否生成带解说的最终视频 |
| `clip_dir` | 视频片段输出目录 |
| `final_video` | 最终合并视频路径（`.mp4`） |

---

## slicer.yaml — 视频帧分类

用于 `video_marking` 阶段，通过 MobileNetV3 分类每秒帧类型（游戏画面 / 间歇画面）。

| 参数 | 说明 |
|------|------|
| `model` | 分类模型权重路径 |
| `interval_sec` | 采样间隔（秒） |
| `min_live_sec` | 判定为游戏片段的最短持续时长（秒） |
| `bridge_gap_sec` | 两段游戏画面之间的最大间隙，小于此值则合并为一段 |
| `device` | 推理设备（`auto` / `cuda` / `cpu`） |
| `timer_roi` | 计时器 ROI，归一化坐标（用于辅助分类校准） |
| `score_roi` | 比分 ROI，归一化坐标 |

---

## audio.yaml — 音频工具（训练数据预处理）

**注意：此文件为训练数据准备阶段的工具配置，不参与主推理流水线。**

用于从直播录音中提取、过滤解说音频片段，为 LLM 微调构建数据集。

### `audio_tools.loud_onset` — 强瞬态检测

| 参数 | 说明 |
|------|------|
| `sample_rate` | 采样率（Hz） |
| `flux_threshold` | onset 动态阈值系数（相对均值的倍数） |
| `peak_mult` | 峰值与均值之比，超过此倍数才算强瞬态 |
| `min_gap_sec` | 相邻 onset 最小间隔（秒），防止重复触发 |

### `audio_tools.asr` — 语音识别

| 参数 | 说明 |
|------|------|
| `model` | ASR 模型（FunASR `paraformer-zh`） |
| `vad_model` | VAD 模型（语音活动检测） |
| `punc_model` | 标点恢复模型 |
| `max_single_segment_time` | 单段最大时长（毫秒） |

### `audio_tools.speaker` — 说话人分离

| 参数 | 说明 |
|------|------|
| `diarization_model` | 说话人分离模型 |
| `sv_model` | 声纹验证模型 |
| `asr_model` | 分离后 ASR 模型 |
| `sim_threshold` | 声纹相似度阈值（高于此值判定同一说话人） |
| `sample_per_spk` | 每说话人采样段数 |
| `min_dur` / `max_dur` | 有效片段时长范围（秒） |

### `audio_tools.paths`

| 字段 | 说明 |
|------|------|
| `input` | 原始直播音频（待处理） |
| `segments` | AI 分段结果 JSONL |
| `filtered_audio` | 过滤后音频 |
| `segments_final` | 最终分段 JSONL（用于 SFT 数据构建） |

---



### `output_dirs` / `system_prompts` / `input_files`

各 profile 的 LoRA adapter 输出目录、system prompt 文件和输入 JSONL 路径，
统一在此集中管理，`scripts/` 和 `data_pipeline/` 均读此文件。
