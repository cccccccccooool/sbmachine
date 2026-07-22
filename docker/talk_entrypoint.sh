#!/usr/bin/env bash
set -Eeuo pipefail

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

SERVED_NAME="${AI6657_VLLM_SERVED_MODEL_NAME:-qwen3}"
HOST="${AI6657_VLLM_HOST:-0.0.0.0}"
PORT="${AI6657_VLLM_PORT:-8000}"
MODEL_ID="${AI6657_TALK_MODEL:-${AI6657_TALK_MODELS:-Qwen/Qwen3-14B-AWQ}}"
MODEL_REVISION="${AI6657_TALK_REVISION:-}"
HF_ENDPOINTS="${AI6657_HF_ENDPOINTS:-https://hf-mirror.com,https://huggingface.co}"
READY_DIR="${HF_HOME}/ai6657-talk-ready"

model_key() {
  python3 - "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import hashlib
import sys
print(hashlib.sha256((sys.argv[1] + "@" + sys.argv[2]).encode("utf-8")).hexdigest())
PY
}

READY_MARKER="${READY_DIR}/$(model_key).json"

snapshot_is_available() {
  python3 - "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, revision = sys.argv[1:]
kwargs = {"repo_id": model_id, "local_files_only": True}
if revision:
    kwargs["revision"] = revision
try:
    print(snapshot_download(**kwargs))
except Exception:
    raise SystemExit(1)
PY
}

marker_matches() {
  python3 - "$READY_MARKER" "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import json
import sys
from pathlib import Path

path, model_id, revision = map(str, sys.argv[1:])
try:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload == {"model_id": model_id, "revision": revision} else 1)
PY
}

write_marker() {
  mkdir -p "$READY_DIR"
  python3 - "$READY_MARKER" "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import json
import sys
from pathlib import Path

path, model_id, revision = map(str, sys.argv[1:])
Path(path).write_text(json.dumps({"model_id": model_id, "revision": revision}) + "\n", encoding="utf-8")
PY
}

verify_model() {
  if ! marker_matches; then
    echo "[talk] model ${MODEL_ID} is not prepared; run python run.py setup --backend container --install" >&2
    return 1
  fi
  if ! snapshot_is_available >/dev/null; then
    echo "[talk] model cache for ${MODEL_ID} is incomplete; rerun python run.py setup --backend container --install" >&2
    return 1
  fi
  echo "[talk] verified prepared model ${MODEL_ID}"
}

prepare_model() {
  mkdir -p "$HF_HOME"
  if marker_matches && snapshot_is_available >/dev/null; then
    echo "[talk] model ${MODEL_ID} is already prepared"
    return 0
  fi

  IFS=',' read -r -a endpoints <<< "$HF_ENDPOINTS"
  for endpoint in "${endpoints[@]}"; do
    echo "[talk] preparing ${MODEL_ID} from ${endpoint} ..."
    args=(download "$MODEL_ID")
    if [[ -n "$MODEL_REVISION" ]]; then
      args+=(--revision "$MODEL_REVISION")
    fi
    if HF_ENDPOINT="$endpoint" hf "${args[@]}"; then
      if snapshot_is_available >/dev/null; then
        write_marker
        echo "[talk] model preparation complete: ${MODEL_ID}"
        return 0
      fi
      echo "[talk] download completed but the local cache did not verify" >&2
    fi
    echo "[talk] preparation failed from ${endpoint}, trying next..." >&2
  done
  return 1
}

append_extra_args() {
  if [[ -n "${AI6657_VLLM_ARGS:-}" ]]; then
    read -r -a extra <<< "${AI6657_VLLM_ARGS}"
    VLLM_ARGS+=("${extra[@]}")
  fi
  if [[ -n "${AI6657_VLLM_LORA_MODULES:-}" ]]; then
    VLLM_ARGS+=(--enable-lora --lora-modules)
    read -r -a loras <<< "${AI6657_VLLM_LORA_MODULES}"
    VLLM_ARGS+=("${loras[@]}")
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

case "${1:-serve}" in
  prepare)
    prepare_model
    ;;
  verify)
    verify_model
    ;;
  serve|"")
    verify_model
    launch_model
    ;;
  *)
    exec "$@"
    ;;
esac