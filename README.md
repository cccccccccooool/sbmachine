# sbmachine 推理发布件

> 一句话：CS2 录像 + .dem → demo数据 + YOLO/OCR 时间轴 → 规则集规划 → 双 LLM 中性稿/口播稿 → GPT-SoVITS 出声。

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

## 使用方式
调配好config/ 下的相关配置后直接在根目录下运行
```bash
python run.py
```
即可按照预设方案全链路拉起
只不过当前第三阶段和第四阶段环境需求不同，docker我还没整理好公开出来，所以还是运行不了

## 配置需求
| 模式 | 显存 | 调用模型 |
|---|---|---|
| 完全本地调用 | 12G | qwen3 14b + GPT-SoVITS|
| 第三部分使用api模式 | 6g | `GPT-SoVITS |

## 当前进度
原本打算用还是用vlm的，但是产出太过于糟糕，干脆直接砍了用povplayer来充当导播风味了

基本链路在cnb上能正常运行，GitHub这边我整理下在推送docker镜像

## 未来计划
- 后面有精力将map的地图模型完善下，以启用附近人归类功能
- 自己通过api模式收集到充足的stf数据后，尝试训练一个更适合的llm模型
- 十分的想在8GB显存上部署，但是qwen3 8b性能实在过于弱，不得以才上qwen3 14b，后续数据上来了我看能不能蒸回8b
- 严格来说第四部分还没完全完善，代表音频还没切分出来，并且调用端用的还是web页面，但我单轮测试通过了，后续在慢慢更改
- 现在最大问题还是在第一二阶段速度过慢，未来搜寻别的方案
- 真的无法复刻玩宝宝那种多姿多样的解说风格🥺🥺，我拼尽全力现在也只能说的是cs demo评判器，只不过是玩宝宝声音说出来的，等一轮数据收集完了再看看
- GPT-SoVITS模型权重就不打算公布出来了

## 致谢

  - [demoinfocs-golang](https://github.com/markus-wa/demoinfocs-golang) —— 本项目的 CS2 demo 解析能力
    完全构建在这个出色的 Go 库之上（MIT License）。感谢 @markus-wa 及所有贡献者