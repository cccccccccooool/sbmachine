FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    AI6657_RECOMMENDED_SKILL_MODEL=Qwen/Qwen3-32B

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
    && sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev python-is-python3 python3-venv \
      build-essential cmake git git-lfs curl wget ca-certificates \
      unzip p7zip-full zstd openssh-client openssh-server procps locales \
      ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
      --index-url https://mirrors.cloud.tencent.com/pypi/simple

RUN pip install \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
      --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu126 \
      --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple

RUN pip install \
      "transformers==4.51.3" \
      "accelerate>=0.26.0" \
      "llamafactory" \
      "peft<0.18.0" \
      "bitsandbytes" \
      "datasets" \
      "deepspeed" \
      "sentencepiece" \
      "protobuf" \
      "tiktoken" \
      "einops" \
      "hf_transfer" \
      "pandas" \
      "pyarrow" \
      "scipy" \
      "numpy" \
      "pyyaml" \
      "requests" \
      "tqdm" \
      "tenacity" \
      "soundfile" \
      "librosa" \
      "faster-whisper" \
      "imageio-ffmpeg" \
      --index-url https://mirrors.cloud.tencent.com/pypi/simple \
      --extra-index-url https://mirrors.aliyun.com/pypi/simple || \
    pip install \
      "transformers==4.51.3" \
      "accelerate>=0.26.0" \
      "llamafactory" \
      "peft<0.18.0" \
      "bitsandbytes" \
      "datasets" \
      "deepspeed" \
      "sentencepiece" \
      "protobuf" \
      "tiktoken" \
      "einops" \
      "hf_transfer" \
      "pandas" \
      "pyarrow" \
      "scipy" \
      "numpy" \
      "pyyaml" \
      "requests" \
      "tqdm" \
      "tenacity" \
      "soundfile" \
      "librosa" \
      "faster-whisper" \
      "imageio-ffmpeg" \
      --index-url https://pypi.org/simple

WORKDIR /workspace
COPY . /workspace

RUN python - <<'PY'
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

CMD ["/bin/bash"]
