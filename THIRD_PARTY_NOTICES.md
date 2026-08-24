# THIRD-PARTY NOTICES

本仓库通过外部调用或依赖使用以下第三方组件。**本仓库不包含这些组件的源码、
模型权重或音频素材**，相关版权与许可以上游为准；如需再分发上游内容，请自行
遵守其许可证义务。

| 组件 | 用途 | 使用方式 | 许可证 | 上游 |
|---|---|---|---|---|
| demoinfocs-golang | CS2 demo 解析 | Go 工具外部调用 | MIT | https://github.com/markus-wa/demoinfocs-golang |
| GPT-SoVITS | TTS 语音合成 | HTTP API 外部调用 | MIT | https://github.com/RVC-Boss/GPT-SoVITS |
| Real-ESRGAN 权重 | 画面增强（tools/enhance） | 运行时下载权重 | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN |
| Python 依赖 | requirements.txt 所列包 | pip 安装 | 各包许可 | https://pypi.org |

## 说明

- 通过包管理器（pip）安装的依赖按各自 PyPI 许可证使用，不随本仓库再分发。
- 语音参考音频、GPT-SoVITS 模型权重与图像增强权重，需由使用者自行确认来源与授权，本仓库不打包发布。