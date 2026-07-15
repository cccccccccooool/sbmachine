FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip --index-url https://mirrors.cloud.tencent.com/pypi/simple

# 安装基础网关依赖
RUN pip install fastapi python-multipart uvicorn requests pydantic pyyaml \
    --index-url https://mirrors.cloud.tencent.com/pypi/simple

WORKDIR /workspace

EXPOSE 7860
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "7860"]
