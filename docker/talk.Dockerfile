# syntax=docker/dockerfile:1.7
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
    AI6657_HF_ENDPOINTS=https://hf-mirror.com,https://huggingface.co \
    AI6657_VLLM_SERVED_MODEL_NAME=qwen3 \
    AI6657_VLLM_MAX_MODEL_LEN=16384 \
    AI6657_VLLM_GPU_MEMORY_UTILIZATION=0.90

USER root
WORKDIR /workspace

# ── SSH server ──────────────────────────────────────────────────────────────
# 允许 VS Code Remote-SSH 直连 talk 容器。
# 密码通过 AI6657_SSH_PASSWORD 环境变量传入；不设则启动时随机生成并打印到日志。
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && mkdir -p /var/run/sshd \
    # 允许 root 密码登录（匹配注释和未注释行）
    && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    # Docker 容器 PAM 经典修复：防止登录后立即被踢
    && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd \
    # 保活：防止空闲断连
    && echo 'ClientAliveInterval 30' >> /etc/ssh/sshd_config \
    && echo 'ClientAliveCountMax 3' >> /etc/ssh/sshd_config \
    # 生成 host keys（构建时做一次，启动时 entrypoint 再兜底检查）
    && ssh-keygen -A \
    && rm -rf /var/lib/apt/lists/*

COPY docker/talk_entrypoint.sh /opt/ai6657/talk_entrypoint.sh
RUN chmod +x /opt/ai6657/talk_entrypoint.sh \
    && mkdir -p /root/.cache/huggingface

# Optional pre-bake: download model to HF cache at build time so it's in the image.
# Pass model ID via build-arg; empty = skip, runtime entrypoint will download.
ARG AI6657_PRELOAD_TALK_MODELS=
RUN <<HEREDOC
export AI6657_PRELOAD_TALK_MODELS="${AI6657_PRELOAD_TALK_MODELS}"
python3 <<'PYEOF'
import os, time, sys
from huggingface_hub import snapshot_download, try_to_load_from_cache

model = os.environ.get('AI6657_PRELOAD_TALK_MODELS', '')
if not model:
    print('[build] skip model pre-bake; runtime entrypoint will download to HF cache')
    sys.exit(0)

for raw in model.split(','):
    m = raw.strip()
    if not m:
        continue
    cached = try_to_load_from_cache(m, 'config.json')
    if cached:
        print(f'[build] cache hit: {m}')
        continue
    endpoints = ['https://hf-mirror.com', 'https://huggingface.co']
    for ep in endpoints:
        os.environ['HF_ENDPOINT'] = ep
        for attempt in range(5):
            print(f'[build] download {m} from {ep} (attempt {attempt+1}/5)...')
            try:
                path = snapshot_download(
                    m,
                    resume_download=True,
                    max_workers=4,
                    ignore_patterns=['.gitattributes', 'README.md'],
                )
                print(f'[build] done: {path}')
                break
            except Exception as e:
                print(f'[build] attempt {attempt+1} failed: {e}')
                if attempt < 4:
                    wait = min(60 * (2 ** attempt), 300)
                    print(f'[build] retry in {wait}s...')
                    time.sleep(wait)
        else:
            print(f'[build] FATAL: endpoint {ep} exhausted for {m}')
            continue
        break
    else:
        print(f'[build] FATAL: all endpoints failed for {m}')
        sys.exit(1)
PYEOF
HEREDOC

EXPOSE 8000 22
ENTRYPOINT ["/opt/ai6657/talk_entrypoint.sh"]
