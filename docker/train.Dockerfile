ARG TRAIN_BASE_IMAGE
FROM ${TRAIN_BASE_IMAGE}

ARG TRAIN_BASE_IMAGE
RUN digest="${TRAIN_BASE_IMAGE##*@sha256:}" \
    && reference="${TRAIN_BASE_IMAGE%@sha256:*}" \
    && [ -n "${reference}" ] \
    && [ "${reference}" != "${TRAIN_BASE_IMAGE}" ] \
    && [ "${#digest}" -eq 64 ] \
    && case "${digest}" in *[!0-9a-f]*) exit 1 ;; esac \
    || { echo "TRAIN_BASE_IMAGE must be a complete image reference pinned by @sha256:<64 lowercase hex>" >&2; exit 1; }

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    AI6657_TRAINING_CONTAINER=1 \
    AI6657_TRAINING_IMAGE_REVISION=w09-v1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python-is-python3 git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple

RUN mkdir -p /opt/ai6657
COPY training/requirements.lock /opt/ai6657/requirements.lock
RUN pip install --no-cache-dir -r /opt/ai6657/requirements.lock \
    && python3 -m pip freeze --all | LC_ALL=C sort > /opt/ai6657/requirements.resolved.txt \
    && test -s /opt/ai6657/requirements.resolved.txt

WORKDIR /workspace
CMD ["bash"]
