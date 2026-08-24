# 训练与数据链

> 基于代码核对(2026-08-20),仅记录代码/配置/测试已确认的事实,运行行为以源码为准。

## 1. 链路分离与模型基座

训练与推理是两条独立链路,各用独立镜像:

- 训练:只在 `docker/train.Dockerfile` 镜像内(见第 6 节门禁)。
- 推理:talk 镜像跑 vLLM 服务,不视为训练镜像。`config/llm.yaml:1-2` `backend: vllm`、`base_url: http://127.0.0.1:8000/v1`。
- 音频清洗 / GPT-SoVITS:audio 镜像。

唯一模型基座(所有 LLM profile 共用):

- LLM 基座 = `Qwen/Qwen3-14B-AWQ`,revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`(`tools_config/train.yaml:14-15`、`training/train_config_style.yaml`、`scripts/verify_talk.py:14`)。注释注明与 vLLM 同源,adapter 可直挂。
- served-model 名统一为 `qwen3`(`config/llm.yaml:12,15-16`、`tools_config/audio.yaml:29`)。
- HUD YOLO 权重唯一来源 = `models/yolo/yolo_cs2.pt`(`config/yolo.yaml:10`);YOLO 训练 base 用 `yolo11n.pt`(`scripts/train_yolo_ui_locator.sh:23`)。

## 2. API 日志 → SFT 流程

三步,产物逐级下游:

1. `data_pipeline/export_api_sft.py` — 扫 `logs/api_training_*.jsonl`(:56)。硬编码筛选:`scope==llmb`(可传参)**且** `accepted is True`(:30),四字段 `sample_id/source_run_id/input/output` 均须非空字符串(:36);目标经 `normalise_style_target` 严格校验为 `{commentary, felt_intensity∈[0,1]}` JSON(:38,`builder.py:217-245`),不合法整条丢弃。按 `(state,commentary)` 去重(:84),原子写出(:93)。输出字段 `{state, commentary, sample_id, source_run_id, source_id, [ts/round/scene]}`(:41-52)。
2. `data_pipeline/build_sft_dataset.py` — 转 ShareGPT,每条 `conversations=[system, human=state, gpt=commentary]`(:76-82);`profile==style` 时对 commentary 再校验一次(:69)。默认 input/output/system_prompt 由 `tools_config/train.yaml` 提供(:118-127)。委托 `DatasetBuilder` 按 source 分组切分,原子写出 train/val/test + split-manifest(:93-101)。
3. `scripts/train_llm_lora.sh` — Docker 内训练(见第 6 节)。

真实命令:

```bash
python -m data_pipeline.export_api_sft --logs-dir logs --scope llmb --output data/sft/style_aligned.jsonl
python data_pipeline/build_sft_dataset.py --config tools_config/train.yaml --profile style
bash scripts/train_llm_lora.sh tools_config/train.yaml   # 须在训练 Docker 内
```

分组切分约束(`data_pipeline/builder.py`):按 `source_id → match → video` 优先级定泄漏边界(:65-91);同一 source 不跨 train/val/test(:112-168);切分确定性(seed=6657),写出走原子事务(:281-376)。

## 3. train_*.sh 真实状态

| 脚本 | 真实状态 |
|---|---|
| `scripts/train_llm_lora.sh` | 真实训练入口。profile(full/lite/analyst/style)→ `training/train_config_*.yaml`(:42-48),LLaMA-Factory LoRA。analyst 默认禁用,需 `AI6657_ENABLE_ANALYST_TRAINING=1`(:37-40) |
| `scripts/train_yolo_ui_locator.sh` | 真实训练入口。`yolo detect train`(:49)训 HUD UI 定位器,含独立 test 集 mAP50 评测(:51-75)+ finalize 门禁(:76) |
| `scripts/train_voice_gpt_sovits.sh` | 仅预检/引导,不真正训练。只检查 GPT-SoVITS 目录与 list(:20-30),打印进 WebUI 手动训练提示(:32-35) |
| `vlm/scripts/train_vlm_{global,local,minimap}_lora.sh` | 冷藏。dataset 已注册,数据生产器需显式提供;`vlm/` 禁改禁删 |
| `scripts/_train_common.sh` | 公共库,非入口。提供 `require_training_container`(:3-12)、`train_yaml_value`/`train_dataset_path`/`train_split_manifest_path`、`training_preflight`(:54-65) |

## 4. 音频清洗链(tools/audio/)

`scripts/clean_commentary.sh` 编排,执行顺序 **04 → 02 → 06 → prepare**(数字非递增,此为真实顺序):

- `tools/audio/04_vocal_isolation.py` — Demucs 人声分离去噪(`clean_commentary.sh:95`)。
- `tools/audio/02_ai_audio_filter.py` — FunASR ASR 转写 + 可选本地 vLLM 内容粗筛(:122);vLLM 8000 端口不可达时自动 `--skip-llm`(:113-119)。
- `tools/audio/06_speaker_filter.py` — 声纹筛选核心。ModelScope CAM++ 说话人分离 + 声纹对比,需 `REF_AUDIO` 参考音频,筛出解说员本人干净句标 `keep_ai=True`(`clean_commentary.sh:145`)。
- `audio_service/prepare_gpt_sovits_dataset.py` — 切片 + 生成 GPT-SoVITS list(`clean_commentary.sh:155`)。

相关但不在此链:`01_audio_merge.py`(前置,产 `6657_merged.wav`)、`07/07slim_speaker_emotion.py`(情绪分析)、`ras_asr_process.py`(独立 ras 数据处理)。

`tools/data_clean/` 是 **SFT 配对**,与音频清洗无关:

- `label_commentary_pairs.py` — 终端审核 UI,给 ASR 片段打 `game/chitchat` 标签(:134-189)。
- `build_commentary_pairs.py` — `game` 句 × VLM/demo 事件按视频时间窗 join 成 state+commentary 配对(:231-339);含 faithfulness 校验,只留 `pass`(:390-395);有无 `--analyst-neutral` 切换情感/事实模型模式(:307-318)。

## 5. tools/cloud/one_click_train.py 真实身份

名不副实:实为一键云端 **faster-whisper ASR 转写器,不是训练启动器**(文件 docstring :1-6)。从 `--material-dir`(优先 `gameplay_only.*`)或 `--audio` 抽 16k 单声道音频(:135-158)→ 转写 → 写 ASR JSONL(带 `keep_ai:True` :200-207);可选包裹 `tools/start/gpu_guard.py release/resume`(:66-72,217-243);刻意不 import 主流水线(:5)。

## 6. Docker 唯一训练环境与可复现门禁

**容器门禁**(`scripts/_train_common.sh:3-12`,三条件):`AI6657_TRAINING_CONTAINER==1` + `/.dockerenv` 存在 + `AI6657_TRAINING_IMAGE_DIGEST` 匹配 `^sha256:[0-9a-f]{64}$`。

**镜像可复现**(`docker/train.Dockerfile`):build-arg `TRAIN_BASE_IMAGE` 必须 `@sha256:<64 小写 hex>` 完整 pin,构建期强校验(:4-11);`COPY training/requirements.lock` 安装后 `pip freeze --all | sort > /opt/ai6657/requirements.resolved.txt` 并断言非空(:32-35)。

**preflight**(`tools/training/build_info.py preflight`,经 `training_preflight`):

- `validate_dataset_counts` — train/eval/test 行数各达 min 门槛(:106-125)。
- `validate_split_manifest` — manifest `schema_version==1`;逐 split 校验本地文件名、行数、`source_ids`、**SHA-256** 与实际字节一致;禁止 source_id 跨 train/val/test 泄漏;`all == train+val+test`(:128-179)。
- `validate_training_environment` — digest 格式 + requirements.lock/resolved 非空,返回两者 SHA-256(:182-198)。

**finalize**(`build_info.py finalize` → `_create_build_info` :328):复跑上述三校验(:358-371);
- adapter:`model_revision` 须 40 位 commit SHA(:376);`adapter_config.json` 的 `base_model_name_or_path` 须等于配置 base_model、`peft_type==LORA`、`target_modules` 非空、权重可加载(:235-284);`eval_loss ≤ max_eval_loss`(:384);`test_loss` 从 `test_results.json` 读,其 `metric/dataset_sha256/samples` 三重校验后 `≤ max_test_loss`(:390-398,201-222)。
- yolo:artifact 可 `YOLO()` 加载、`results.csv` mAP50 ≥ 门槛、独立 test mAP50 ≥ 门槛(:399-421)。
- 写 `build_info.json`(schema_version=2),含 artifact/train/eval/test/split-manifest/test_results/requirements 全套 SHA-256、镜像 digest、seed、base_model+revision(:425-467);失败删残留 proof(:471-479)。

**加载侧**(`verify_build_info_artifact` :482):推理侧复核 `build_info.json` 的 `artifact_sha256` 与当前 artifact 字节一致。

## 7. 边界:非可复现的便利脚本

`scripts/cloud_prepare.sh` 与 `tools/cloud/install_training_deps.sh` 用宿主 `pip` 装依赖(cloud_prepare.sh:15-16、install_training_deps.sh:17-45),**不满足** `require_training_container` 门禁,只是云端裸机/依赖便利脚本,不能用于正式可复现训练。

---

已知偏差:README / 记忆与代码不一致处(音频清洗链位置、LLM 基座型号/推理框架等)统一见 `docs/known_discrepancies.md`。
