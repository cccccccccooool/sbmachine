# AI 玩机器 Skill

本目录仅使用以下本地来源提炼：

`C:\Users\sihasiha\.claude\skills\dot-skill\skills\celebrity\wanjiji`

没有混用此前项目中的旧 Skill 或其他人物资料。

## 运行入口

`style_skill.md` 是 Phase3B 唯一运行入口，继续由 `config/llm.yaml` 的 `paths.style_skill` 加载。运行时不会递归读取本目录，也不会把研究资料或固定语录送给模型。

## 目录

- `style_skill.md`：当前流水线实际加载的自包含入口。
- `wanjiji/core/`：可复用的人格、认知、语言与边界内核。
- `wanjiji/scenarios/`：特定工作场景的适配规则。
- `wanjiji/extraction_manifest.md`：来源、保留内容与排除原则。
- `reference_original/wanjiji/`：用户指定真实 Skill 的原样离线备份，仅供追溯，运行时禁止加载。

当前只接入 CS2 实时解说场景。后续聊天、复盘或直播互动应复用同一人格内核，并各自增加场景适配器，而不是改写人物定义。

`style_skill.md` 是面向模型的编译产物，不是原始文件的简单拼接。原版中的研究过程、历史版本和固定语录只保存在 `reference_original/`，不得加入 `paths.style_skill`。
