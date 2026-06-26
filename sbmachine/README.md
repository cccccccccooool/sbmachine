<div align="center">

<img src="https://YOUR-AVATAR-LINK-HERE.png" width="160" alt="项目头像占位 —— 替换成你的图床/仓库内图片链接" />

# 🎙️ 6657 · CS2 录像 AI 解说

**把一整段沉默的 CS2 录像，变成一场有血有肉、会上头的中文解说。**

[![预览视频](https://img.shields.io/badge/▶_预览视频-点击观看-ff0000?style=for-the-badge)](https://YOUR-PREVIEW-VIDEO-LINK-HERE)

</div>

---

## 这是什么

你有一段 CS2 的比赛录像，几十分钟，安安静静，只有游戏原声。

把它丢进来。等一会儿。

出来的是一段**配好解说的视频**——一个语气、口癖、上头方式都神似主播「6657」的 AI，从头到尾给你把这局比赛讲完。比分焦灼时它会紧张，残局翻盘时它会破音，连跪几局它甚至会开始阴阳怪气。

它不是念稿机器。它**看得懂画面**（谁在架枪、烟往哪扔），**记得住全场**（上半场谁是大腿、哪一局是转折点、现在是不是赛点），所以越到后面讲得越带劲——就像一个真的从头看到尾的解说，而不是每局现场失忆。

---

## 它是怎么做到的

整条流水线分四步，每一步把"沉默的像素"往"会说话的解说"推进一程：

| 步骤 | 干的事 | 通俗讲 |
|---|---|---|
| **① 切片对齐** | 录像粗切成一局一局，对齐到 demo 的绝对回合号 | 先把比赛**分回合**，知道现在讲的是第几局 |
| **② 看画面** | 逐帧抽样，让视觉模型描述战场（走位、烟闪、架枪） | 给 AI 一双**眼睛**，把画面翻译成文字 |
| **③ 写解说** | 大模型结合画面 + 全场记忆，写出带情绪标签的口播稿 | 给 AI 一张**嘴和一段记性**，写出像人说的话 |
| **④ 配音合成** | 按情绪标签换音色，合成语音、拼回视频 | 给 AI 一副**嗓子**，把稿子念成有起伏的声音 |

> 一条铁律贯穿始终：**比分、击杀、炸弹这些硬事实，只认 demo 解析结果，AI 绝不允许瞎编**。画面模型只负责"描述长什么样"，绝不负责"判定发生了什么"。这是它不胡说八道的根本。

每一步的产物都是一个 JSON 检查点，跑断了能从中间接着跑，不用从头再来。

---

## 快速上手

> 完整的环境安装（系统依赖、PyTorch、Ollama、TTS 权重）见 **[`docs/deploy.md`](../docs/deploy.md)**。这里只讲跑起来。

### 1. 配一下要跑什么

编辑 `config/pipeline.yaml`，告诉它录像在哪、跑哪几步：

```yaml
paths:
  video: data/raw/demo.mp4      # 你的录像
  map_name: de_ancient          # 地图

phases:                         # 想跑哪步开哪步
  preprocess_slice: true        # 切片对齐
  phase2_vision: true           # 看画面
  phase3_semantic: true         # 写解说
  phase4_assemble: true         # 配音合成
```

### 2. 让它自己管好模型服务

单机单卡（8–12G 显存就够）推荐让流水线自己错峰起停模型——用到哪个起哪个，跑完立刻让出显存：

```yaml
# config/pipeline.yaml
runtime:
  manage_services: true         # 流水线全程托管 VLM / LLM / TTS 的生命周期
  one_model_at_a_time: true     # 单卡错峰：任意时刻卡上只有一个模型
```

### 3. 一行跑起来

```bash
# 先空跑自检：不调任何 AI，只验证回合/时间轴链路通不通
python run.py --dry-run

# 正式出片
python run.py
```

看到日志依次冒出 `[services] ... healthy`、每步结束 `stop`、最后 `[run.py] 全部阶段完成`，就成了。产物在 `output/` 下。

### 只想重跑某一步？

每一步都能单独拎出来跑（前提是它依赖的服务已经起着）：

```bash
python -m sbmachine.phase_vision   --config config/   # 只重跑「看画面」
python -m sbmachine.phase_semantic --config config/   # 只重跑「写解说」
python -m sbmachine.phase_tts      --config config/   # 只重跑「配音」
```

---

## 显存为什么够

四步里最吃显存的也就单个模型的量级。靠 `one_model_at_a_time` 错峰——看画面时只有视觉模型在卡上，写解说时只有语言模型在卡上，配音时只有 TTS 在卡上：

```
看画面 → 起视觉模型 → 跑完 → 让出整张卡 ┐
写解说 → 起语言模型 → 跑完 → 让出整张卡 ├ 峰值显存 ≈ 最大的那一个，不是三个相加
配音  → 起 TTS     → 跑完 → 让出整张卡 ┘
```

所以一张 8–12G 的卡，就能把整条链路从头跑到尾。

---

<div align="center">

**把录像丢进去，等一段会上头的解说出来。**

▶ [预览视频](https://YOUR-PREVIEW-VIDEO-LINK-HERE) · 📖 [部署文档](../docs/deploy.md)

</div>
