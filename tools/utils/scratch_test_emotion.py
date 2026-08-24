"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：测试语音情感标签提取逻辑和情感音频参考源解析逻辑的临时测试脚本。

它验证文本中 `[激动]`、`[惋惜]` 等情绪修饰语的解析，以及在合并、匹配时的回退逻辑。

启动方式：python tools/utils/scratch_test_emotion.py
输入数据流：无（使用脚本内嵌的测试文本）。
输出数据流：打印解析结果到 stdout。
用法用途：验证文本中情绪修饰语的解析及合并、匹配时的回退逻辑是否正确。
"""
from voice.emotion import parse_emotional_text, resolve_emotion_ref

sample_text = "[激动]秒了！又秒一个！这枪太硬了！[平述]现在 CT 这边还剩三个人，经济也起来了。[惋惜]哎，老汤这大狙又空了。[激动]但是！这球又抢回来了！"
print("--- 基本切分 ---")
for segment in parse_emotional_text(sample_text):
    print(repr(segment.emotion), "|", segment.text)

print("--- 无标签开头 ---")
for segment in parse_emotional_text("开局没标签的一句话。[紧张]残局了。"):
    print(repr(segment.emotion), "|", segment.text)

print("--- 相邻同情绪合并 ---")
for segment in parse_emotional_text("[激动]秒了！[激动]又一个！"):
    print(repr(segment.emotion), "|", segment.text)

print("--- ref 回退 ---")
default = {"audio_path": "data/voice/reference/6657_ref.wav", "prompt_text": "默认", "prompt_lang": "zh"}
refs = {"激动": {"audio_path": "x/excited.wav", "prompt_text": "秒了"}}
print("激动 ->", resolve_emotion_ref("激动", refs, default))
print("未知情绪 ->", resolve_emotion_ref("紧张", refs, default))
