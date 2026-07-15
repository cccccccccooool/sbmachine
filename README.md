# sbmachine 推理发布件

> 本目录是 `ai-6657` 仓库的**推理链发布拷贝**，按 `docs/sbmachine_inventory.md` 白名单同步，只含端到端跑通 `python run.py` 所需资产。训练、数据清洗、诊断工具、测试、VLM 冷藏区**不在此目录**，开发请回主仓库。
> 一句话：CS2 录像 + .dem → demo 硬事实 + YOLO/OCR 时间轴 → 规则集规划 → 双 LLM 中性稿/口播稿 → GPT-SoVITS 出声。
> **铁律：硬事实唯一来源是 demo 文件**，画面只做定位锚点，不参与比分/击杀/炸弹判断。

## 目录构成（与清单一一对应）

| 路径 | 角色 |
|---|---|
| `run.py` | 唯一启动入口（`sbmachine.run_all:run_all`） |
| `docker-compose.yml`、`docker/` | 多容器服务拓扑与 vLLM/SoVITS entrypoint |
| `.env.example`、`requirements.txt` | API 密钥模板与 Python 依赖 |
| `sbmachine/` | 流水线全部阶段实现（phase1~4、preflight、schemas、llm_shim 等） |
| `core/` | config_loader、prompt_loader、各 schema |
| `vision_service/` | region_crops（phase2）、frame_type_model 等 |
| `audio_service/` | emotion（phase3b）、gpt_sovits_client（phase4）、runtime yaml |
| `config/` | pipeline/llm/vision/slicer/tts/audio/train.yaml 全套 |
| `Prompt/` | analyst/style 提示词、`json/{cs_game_rules,hype_rules}.json`、skill |
| `database/player_aliases.json` | 选手绰号注入 |
| `database/maps/` | 人工地图空间关系模板（**当前为空**，spatial 层 fail-closed 降级，补模板即自动启用附近人归类） |
| `models/` | YOLO / 帧分类权重 |
| `tools/` | 仅 run 链子集：demo 解析（Go）、demo_manifest、debug/phase2、slicing 三件、simple_vlm_server |

明确排除：`tools/start/gpu_guard.py`（个人工具，代码已做缺失守卫，静默跳过）、`tests/`、`training/`、`data_pipeline/`、`vlm/`、其余 tools 子目录。

## 运行

```bash
python run.py --dry-run   # JSON 链路自检，不调任何 AI 模型；config_valid: true 即依赖链齐全
python run.py             # 正式运行，读 config/
```

所有阶段开关、路径、服务拓扑都在 `config/pipeline.yaml`，不靠命令行参数。云端 API 版 phase3a 单独入口：`run_llma_api.py`（如未同步可从主仓库取）。

## 数据流（当前架构）

```
.dem ─ tools/demo/parse_demo.py（Go 解析器，含 ammo/坐标/callout）→ output/demo/{rounds,kills,grenades,roster}.json + ticks.parquet
mp4  ─ tools/slicing/run_frame_type_slicer.py（帧分类粗切）
phase1   回合切分            → rounds.json
phase2   phase2_yolo.py      → rounds_with_yolo.json + rounds_with_yolo_semantic.json（精简 DEM 事实帧）
phase3a  phase3a_analyst.py  → 确定性规则层（scene 切窗 → 击杀语义 串/扫转/特 → 空间锚点/附近人 → commentary_plan）
                               + 增量状态报告（首窗全量、之后仅报变化）
                               → llma_input.json（LLM-A 输入中间产物，落盘留档）
                               → LLM-A 逐窗中性稿 → rounds_with_neutral.json
phase3b  phase3b_style.py    → 6657 风格口播 + 情绪标签 → rounds_with_commentary.json / commentary.json
phase4   phase4_assemble.py  → TTS + 时间戳对齐混音 → rounds_final.json + rounds/round_NNN.wav/.mp4
```

关键设计：**LLM-A 拿不到原始 players/events/ammo/坐标**——它们由规则层内部消化成 plan 与状态增量；模型只见 `commentary_plan` + 状态报告 + when/who 氛围帧，防止模型重做规则决策。

## 服务与模型

| 角色 | 模型 | 后端 |
|---|---|---|
| LLM（analyst + style） | Qwen3 系（见 `config/llm.yaml`） | vLLM 本地容器 或 OpenAI 兼容 API（`.env` 配置，LLMA/LLMB 前缀可分别覆盖） |
| TTS | GPT-SoVITS | `audio_service/` + api_v2 |

单卡错峰由 `config/pipeline.yaml` 的 `runtime.manage_services` / `one_model_at_a_time` 控制，任意时刻卡上只有一个模型。

## 关键产物

| 文件 | 说明 |
|---|---|
| `output/sbmachine/rounds_with_yolo_semantic.json` | 精简 DEM 事实帧时间轴（players/events/ammo/坐标的磁盘落点） |
| `output/sbmachine/llma_input.json` | LLM-A 输入定版（plan + state_block + 氛围帧），审计/复跑用 |
| `output/sbmachine/rounds_with_neutral.json` | 中性稿 + hype |
| `output/sbmachine/commentary.json` | 口播稿 + 情绪段 |
| `output/sbmachine/rounds/round_NNN.wav/.mp4` | 逐局成品 |

## 同步方式

本目录内容由主仓库脚本重铺（保留 `.git` 与本 README）：

```bash
cd /d D:\code\ai-6657
python _publish_sbmachine.py
```

同步后在本目录跑 `python run.py --dry-run` 验证。**不要在本目录直接改代码**——改动会在下次同步时被覆盖，一律回主仓库改完再发布。

## 权益提醒

请只在授权、自用或合规二创范围内使用真人声音和人设素材，不用于冒充本人、商业牟利、诈骗或误导观众。
