#!/usr/bin/env bash
set -Eeuo pipefail

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

SERVED_NAME="${AI6657_VLLM_SERVED_MODEL_NAME:-qwen3}"
HOST="${AI6657_VLLM_HOST:-0.0.0.0}"
PORT="${AI6657_VLLM_PORT:-8000}"
MODEL_ID="${AI6657_TALK_MODEL:-${AI6657_TALK_MODELS:-Qwen/Qwen3-14B-AWQ}}"
HF_ENDPOINTS="${AI6657_HF_ENDPOINTS:-https://hf-mirror.com,https://huggingface.co}"

# ── Model download ──────────────────────────────────────────────────────────

download_model() {
  if python3 -c "
from huggingface_hub import try_to_load_from_cache
path = try_to_load_from_cache('${MODEL_ID}', 'config.json')
print(path or '')
exit(0 if path else 1)
" 2>/dev/null; then
    echo "[talk] model ${MODEL_ID} already in HF cache, skip download"
    return 0
  fi

  IFS=',' read -r -a ENDPOINT_ARRAY <<< "$HF_ENDPOINTS"
  for endpoint in "${ENDPOINT_ARRAY[@]}"; do
    echo "[talk] downloading ${MODEL_ID} from ${endpoint} ..."
    if HF_ENDPOINT="$endpoint" hf download "$MODEL_ID" 2>&1; then
      echo "[talk] download complete: ${MODEL_ID}"
      return 0
    fi
    echo "[talk] download failed from ${endpoint}, trying next..." >&2
  done
  return 1
}

# ── vLLM launch ─────────────────────────────────────────────────────────────

append_extra_args() {
  if [[ -n "${AI6657_VLLM_ARGS:-}" ]]; then
    read -r -a EXTRA <<< "${AI6657_VLLM_ARGS}"
    VLLM_ARGS+=("${EXTRA[@]}")
  fi
  if [[ -n "${AI6657_VLLM_LORA_MODULES:-}" ]]; then
    VLLM_ARGS+=(--enable-lora --lora-modules)
    read -r -a LORAS <<< "${AI6657_VLLM_LORA_MODULES}"
    VLLM_ARGS+=("${LORAS[@]}")
  fi
}

launch_model() {
  VLLM_ARGS=(
    --host "$HOST"
    --port "$PORT"
    --model "$MODEL_ID"
    --served-model-name "$SERVED_NAME"
    --trust-remote-code
    --max-model-len "${AI6657_VLLM_MAX_MODEL_LEN:-16384}"
    --gpu-memory-utilization "${AI6657_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
    # Qwen3 思考默认开启，从 chat template 层关闭，content 才是干净 JSON
    --reasoning-parser qwen3
    --default-chat-template-kwargs '{"enable_thinking": false}'
  )
  if [[ -n "${AI6657_VLLM_MAX_NUM_SEQS:-}" ]]; then
    VLLM_ARGS+=(--max-num-seqs "${AI6657_VLLM_MAX_NUM_SEQS}")
  fi
  append_extra_args

  echo "[talk] launching vLLM: served=${SERVED_NAME} model=${MODEL_ID}"
  exec python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
}

# ── main ────────────────────────────────────────────────────────────────────

mkdir -p "$HF_HOME"

MAX_ATTEMPTS="${AI6657_VLLM_MAX_ATTEMPTS:-2}"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[talk] attempt ${attempt}/${MAX_ATTEMPTS}: ${MODEL_ID}"
  if download_model; then
    launch_model
  fi
  if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
    echo "[talk] retrying in 10s..." >&2
    sleep 10
  fi
done

echo "[talk] ${MODEL_ID} could not be served after ${MAX_ATTEMPTS} attempts." >&2
exit 1
