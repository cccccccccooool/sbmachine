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

COPY docker/talk_entrypoint.sh /opt/ai6657/talk_entrypoint.sh
RUN chmod +x /opt/ai6657/talk_entrypoint.sh \
    && mkdir -p /root/.cache/huggingface

# This is the optional vLLM runtime only.  It deliberately never bakes model
# weights into the image: `sbmachine setup --install` performs the explicit
# model preparation into the attached cache volume.
EXPOSE 8000
ENTRYPOINT ["/opt/ai6657/talk_entrypoint.sh"]