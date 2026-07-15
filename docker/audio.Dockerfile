# ═══════════════════════════════════════════════════════════════════════════════
# ai-6657 统一镜像：CNB 云端开发 / 训练 / 音频清洗，一个镜像全包。
#
# ★ 关键决策：不用 Conda，直接用 Ubuntu 22.04 原生 Python 3.10。
#   Conda base 环境自带的 python 是 3.13，会导致 torch / GPT-SoVITS 不兼容。
#   Ubuntu 22.04 的 python3 就是 3.10，完美匹配。
#
# ★ 用法（只用构建一次，之后永远 image: 拉取，不再 rebuild）：
#   1. 本地/任意有 docker 的地方构建：
#        docker build --platform linux/amd64 \
#          -f .ide/Dockerfile \
#          -t <你的镜像仓库地址>/ai6657:latest .
#   2. 推送：
#        docker login <你的镜像仓库地址>
#        docker push <你的镜像仓库地址>/ai6657:latest
#   3. 把 .cnb.yml 里的 image: 占位符替换为你的仓库地址。
#   4. 之后 CNB 每次创建工作区直接拉镜像，零构建、零下载依赖。
#
#   如果你没有镜像仓库账号，可先推送到：
#     - 腾讯云 TCR：ccr.ccs.tencentyun.com/<命名空间>/ai6657:latest
#     - CNB 自带制品库（在 CNB 项目页「镜像仓库」查看地址）
# ═══════════════════════════════════════════════════════════════════════════════
FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    GPT_SOVITS_HOME=/opt/GPT-SoVITS \
    HF_HUB_DOWNLOAD_TIMEOUT=60

# ── 系统依赖 ──────────────────────────────────────────────────────────────────
# python-is-python3 让 `python` 指向 python3（而非 conda），避免混淆。
# swig 是 GPT-SoVITS 部分组件的编译依赖。
# git, git-lfs, wget, curl, ca-certificates, unzip, p7zip-full: 常用包与解压工具。
# openssh-client, openssh-server, procps, locales: CNB云端 VS Code 运行与同步依赖。
RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
    && sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python-is-python3 python3-venv build-essential \
    curl ca-certificates git git-lfs wget unzip p7zip-full zstd \
    openssh-client openssh-server procps locales \
    cmake swig libgl1 libglib2.0-0 ffmpeg libsndfile1 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── pip 源：主源腾讯云（CNB 同机房最快）──────────────────────────────
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple

# ── PyTorch cu126（南京大学镜像，避开 download.pytorch.org 被墙）─────────────────
# torchcodec 解决 demucs save_with_torchcodec 报错。
RUN pip install torch torchaudio torchcodec \
    --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu126 \
    --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple

# ── GPT-SoVITS 本体（装在 /opt，不会被 /workspace 挂载覆盖）────────────────────
ARG GPTSOVITS_REPO=https://github.com/RVC-Boss/GPT-SoVITS.git
RUN git clone --depth=1 "${GPTSOVITS_REPO}" "${GPT_SOVITS_HOME}"

WORKDIR ${GPT_SOVITS_HOME}

# 修 install.sh 的 PyTorch 死源
RUN sed -i 's#download.pytorch.org/whl#mirrors.nju.edu.cn/pytorch/whl#g' install.sh

# GPT-SoVITS 依赖（torch 已装，pip 自动跳过）
RUN pip install -r requirements.txt && \
    if [ -f extra-req.txt ]; then pip install -r extra-req.txt; fi

# GPT-SoVITS webui 兼容版本锁定
RUN pip install "gradio==4.44.1" "gradio-client==1.3.0" "httpx==0.27.2" \
    "fastapi==0.115.6" "starlette==0.41.3"

# 修 gradio 云端自检 bug
RUN sed -i '1i import gradio.networking; gradio.networking.url_ok = lambda *a, **k: True' webui.py

# ── 项目基础依赖（推理服务用，音频清洗依赖后续按需加）─────────────────────────
RUN pip install \
    imageio-ffmpeg \
    requests \
    pyyaml \
    numpy \
    fastapi \
    uvicorn \
    pydantic

# ── 音频清洗全家桶（含人声分离与声纹所需的各种零碎依赖）───────────────────────
# 包含之前排雷发现的 modelscope 幽灵依赖（addict, datasets, simplejson 等）
RUN pip install \
    funasr "modelscope[audio]" demucs librosa soundfile scikit-learn scipy \
    addict datasets simplejson sortedcontainers && \
    pip install --upgrade pypinyin

# ── 预装 NLTK 英文发音语法补丁包 ──────────────────────────────────────────────
# 增加 until 循环防 ghproxy 偶发抽风导致报错退出
RUN mkdir -p /root/nltk_data/taggers && cd /root/nltk_data/taggers && \
    until wget https://ghproxy.net/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip; do echo "重试下载 NLTK eng..."; sleep 3; done && \
    unzip -o averaged_perceptron_tagger_eng.zip && \
    until wget https://ghproxy.net/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip; do echo "重试下载 NLTK base..."; sleep 3; done && \
    unzip -o averaged_perceptron_tagger.zip

# ── 提前修复 fast_langdetect 的底层文件夹缺失 Bug ──────────────────────────
# 顺便尝试离线下载语种识别小模型（防止首次运行合成时卡顿）
RUN mkdir -p ${GPT_SOVITS_HOME}/GPT_SoVITS/pretrained_models/fast_langdetect && \
    wget -O ${GPT_SOVITS_HOME}/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin || true

# ── 预装 GPT-SoVITS 的 7G 预训练模型（开箱即用）───────────────────────────────
# 利用国内满速镜像源，并且排除掉没用的 TF/Flax 庞大权重，大幅缩减镜像体积
# 增加 until 循环实现断点续传：防止 CNB 云端网络波动导致长达20分钟的下载突然中断报错
# HF_HUB_DOWNLOAD_TIMEOUT=60 延长单次请求超时（默认10s太短，网络波动频繁触发重试）
# max-workers=2 限制并发下载数，避免高频并发连接触发镜像源限流/超时
RUN export HF_ENDPOINT="https://hf-mirror.com" && \
    until huggingface-cli download hfl/chinese-roberta-wwm-ext-large --local-dir ${GPT_SOVITS_HOME}/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large --exclude "*.h5" "*.msgpack" --max-workers 2; do echo "网络波动，正在重试下载 Roberta..."; sleep 5; done && \
    until huggingface-cli download TencentGameMate/chinese-hubert-base --local-dir ${GPT_SOVITS_HOME}/GPT_SoVITS/pretrained_models/chinese-hubert-base --exclude "*.h5" "*.msgpack" --max-workers 2; do echo "网络波动，正在重试下载 Hubert..."; sleep 5; done && \
    until huggingface-cli download LJ1995/GPT-SoVITS --local-dir ${GPT_SOVITS_HOME}/GPT_SoVITS/pretrained_models --max-workers 2; do echo "网络波动，正在重试下载底座..."; sleep 5; done

# ── 端口 ──────────────────────────────────────────────────────────────────────
# GPT-SoVITS: 9874(webui) 9872(推理) | 本项目 API: 7860
EXPOSE 9874 9872 7860

WORKDIR /workspace
