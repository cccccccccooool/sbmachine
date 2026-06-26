# sbmachine

根据 CS2 比赛视频和对应的 `.dem` demo 文件，自动生成具有 **6657 直播间风格**的中文解说语音。

---

## 项目简介

输入一场 CS2 录像（视频 + demo），输出一段带情绪标签、口播腔调的解说语音。整条流水线全自动：

- **视觉理解**：YOLO 门控 + 由事件驱动的VLM描述画面
- **语义分析**：LLM 从帧序列提炼中性事实稿（analyst）
- **风格化**：LLM 将事实稿转换为玩机器风格口播解说的解说稿（style）
- **语音合成**：GPT-SoVITS 输出音频，与小局视频拼装为最终成品

---

## 流水线结构

```
.dem + .mp4
    │
    ▼
[demo parse]      解析 demo → demo的JSON格式数据
    │
    ▼
[video marking]   MobileNetV3 分类帧类型，然后输出一份标记的每秒的类别的json格式数据
    │
    ▼
[preprocess]      根据video marking给出的数据进行回合分段并将原视频切割成若干小对局 + 对齐视频和demo的时间轴
    │
    ▼
[phase 2: vision] YOLO 过滤非游戏画面内容 → 由事件驱动VLM进行描述画面内容 → 输出一份带有povplayer视角内容的demo数据
    │
    ▼
[phase 3: semantic]
    ├── 3a analyst  输入phase 2产出的数据，根据提示词和窗口机制输出中性稿（未来将支持单独让3a使用api调用）
    └── 3b style    输入3a产出的中性稿，根据内置database进行rag检索出相对应解说专业术语搭配经过训练过的模型（待做）输出一份具有玩机器风格的口播解说稿
    │
    ▼
[phase 4: assemble] GPT-SoVITS TTS + 音视频拼装
    │
    ▼
视频/音频
```

---

## 硬件要求

| 场景 | 显存需求 |
|------|---------|
| 完整运行（VLM + TTS） | 约 8~12 GB |

---

## 安装

### 1. Python 依赖

```bash
pip install -r requirements.txt
```

### 2. VLM 服务（Phase 2）

项目使用 [lmdeploy](https://github.com/InternLM/lmdeploy) 部署 `Qwen2.5-VL-3B-Instruct`：

```bash
pip install lmdeploy

# 首次运行会自动下载权重（约 7GB）
python tools/simple_vlm_server.py
# 服务默认监听 http://127.0.0.1:23333
```


### 3. GPT-SoVITS（Phase 4）

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS /opt/GPT-SoVITS
cd /opt/GPT-SoVITS
pip install -r requirements.txt

# 下载预训练权重（见项目 README）
# 启动 API 服务（由 sbmachine 自动调用，也可手动启动）
python api_v2.py -a 0.0.0.0 -p 9880
```

### 4. 配置 LLM API 密钥

在项目根目录创建 `config.yaml`：

```yaml
secrets:
  base_url: https://your-api-endpoint/v1   
  api_key: sk-xxx
  model:                           
```

---

## 配置

编辑 `config/` 目录下的各 YAML 文件：

| 文件 | 职责 |
|------|------|
| `pipeline.yaml` | 阶段开关、路径、运行模式 |
| `llm.yaml` | LLM 后端参数、语义分析配置 |
| `vision.yaml` | VLM、YOLO、OCR、采样策略 |
| `tts.yaml` | TTS 输出路径 |
| `slicer.yaml` | 视频帧分类器配置 |
| `audio.yaml` | 训练数据音频工具（不参与推理） |
| `train.yaml` | LLM 微调配置（不参与推理） |

详细参数说明见 [docs/config.md](docs/config.md)。

最重要的几项：

```yaml
# pipeline.yaml — 指定输入文件
paths:
  demo:  data/raw/match.dem
  video: data/raw/demo.mp4
  map_name: de_dust2

# pipeline.yaml — 按需开关阶段
phases:
  phase2_vision:   true   # 已有 rounds_with_vision.json 可设 false 跳过
  phase3_semantic: true
  phase4_assemble: true
```

---

## 运行

### 空跑自检

不调用任何 AI，只验证回合/时间轴链路是否通畅：

```bash
python run.py --dry-run
```

### 完整运行

```bash
python run.py
```

调度器会按 `pipeline.yaml` 的 `phases` 开关依次执行，并自动启停 VLM / TTS 服务。

### 单阶段运行

跳过调度器，手动启动所需服务后单独执行某阶段：

```bash
python -m sbmachine.phase_vision   --config config/   # 仅视觉分析
python -m sbmachine.phase_semantic --config config/   # 仅语义分析
python -m sbmachine.phase_tts      --config config/   # 仅 TTS 拼装
```

---

## 开发现状

- VLM 和 LLM 尚未针对 CS2 场景做专项微调，输出质量有待提升
- 提示词仍在调优中，phase 3 在复杂回合下可能出现幻觉或截断
- 还未对第四部分放置参考音频片段，故若直接运行的话可能会报错运行不了
- 此次上传了phase 3相关模型本地调用方案，但未做真正适配（如模型下载文档和database为空未做兼容），可能无法照常使用

---

## 路线图

- [ ] 微调 VLM，提升画面理解准确率
- [ ] 微调 LLM analyst/style adapter
- [ ] 补充并校准 `database/` 中的地图数据和术语表
- [ ] Web UI 可视化运行与进度监控
- [√] 调优第四步