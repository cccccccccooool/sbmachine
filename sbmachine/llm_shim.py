import os
import json
import datetime
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_DIR = _PROJECT_ROOT / "logs"


def _load_secrets() -> dict:
    """从根目录 config.yaml 加载密钥，文件不存在则返回空 dict。"""
    p = _PROJECT_ROOT / "config.yaml"
    if not p.exists():
        return {}
    import yaml
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("secrets", {})


def _dump_api_log(url: str, request_payload: dict, response_raw: dict, log_ctx: dict | None = None) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y%m%d")
    log_path = _LOG_DIR / f"api_debug_{today}.jsonl"
    entry: dict = {}
    if log_ctx:
        entry.update(log_ctx)   # round/scene 排最前，一眼看清请求批次
    entry.update({
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "url": url,
        "request": request_payload,
        "response": response_raw,
    })
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True
)
def _post_openai_with_retry(url: str, payload: dict, headers: dict, timeout: int):
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _no_think_system(system_prompt: str, model: str, llm_config: dict) -> str:
    """qwen3 关思考：system 首行插 /no_think（前置确保截断时也生效）。"""
    if not bool(llm_config.get("enable_thinking", False)) and "qwen" in model.lower():
        return "/no_think\n\n" + (system_prompt or "")
    return system_prompt


def _build_openai_request(llm_config: dict) -> tuple[str, dict, str]:
    """从 config / secrets / env 解析 endpoint、auth headers、model name。"""
    _s = _load_secrets()
    base_url = _s.get("base_url") or os.getenv("AI6657_base_url") or llm_config.get("base_url", "https://api.openai.com/v1")
    url = f"{base_url.rstrip('/')}/chat/completions"
    api_key = _s.get("api_key") or os.getenv("AI6657_api_key") or llm_config.get("api_key", "sk-xxx")
    model = _s.get("model") or os.getenv("AI6657_LLM_MODEL") or llm_config.get("model", "gpt-4o-mini")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return url, headers, model


def _build_payload(model: str, messages: list[dict], llm_config: dict, max_tokens: int | None) -> dict:
    """组装 chat completions payload（model/messages/temperature/max_tokens/frequency_penalty）。"""
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_config.get("temperature", 0.75)),
    }
    if max_tokens and int(max_tokens) > 0:
        payload["max_tokens"] = int(max_tokens)
    fp = float(llm_config.get("frequency_penalty", 0.0) or 0.0)
    if fp:
        payload["frequency_penalty"] = fp
    return payload


def generate_commentary_openai(
    prompt: str,
    llm_config: dict,
    system_prompt: str,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
) -> str:
    """调用大模型 API（无历史，分析模型专用）。"""
    url, headers, model = _build_openai_request(llm_config)
    messages = [
        {"role": "system", "content": _no_think_system(system_prompt, model, llm_config)},
        {"role": "user", "content": prompt},
    ]
    payload = _build_payload(model, messages, llm_config, max_tokens)
    timeout = int(llm_config.get("timeout_sec", 120))
    data = _post_openai_with_retry(url, payload, headers, timeout)
    _dump_api_log(url, payload, data, log_ctx=log_ctx)
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def generate_commentary_chat_openai(
    messages: list[dict],
    llm_config: dict,
    max_tokens: int | None = None,
) -> str:
    """调用大模型 API（带对话历史，风格模型专用）。"""
    url, headers, model = _build_openai_request(llm_config)
    # qwen3 关思考：首条 system 消息尾追加 /no_think
    if messages and not bool(llm_config.get("enable_thinking", False)) and "qwen" in model.lower():
        msgs = [dict(m) for m in messages]
        if msgs[0].get("role") == "system":
            msgs[0]["content"] = (msgs[0].get("content") or "") + "\n/no_think"
        else:
            msgs.insert(0, {"role": "system", "content": "/no_think"})
        messages = msgs
    payload = _build_payload(model, messages, llm_config, max_tokens)
    timeout = int(llm_config.get("timeout_sec", 120))
    data = _post_openai_with_retry(url, payload, headers, timeout)
    _dump_api_log(url, payload, data)
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
