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


def _execute_openai_chat(
    messages: list[dict],
    llm_config: dict,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
) -> str:
    """内部通用的 OpenAI 格式 API 调用器。"""
    _s = _load_secrets()
    base_url = _s.get("base_url") or os.getenv("AI6657_base_url") or llm_config.get("base_url", "https://api.openai.com/v1")
    url = f"{base_url.rstrip('/')}/chat/completions"
    api_key = _s.get("api_key") or os.getenv("AI6657_api_key") or llm_config.get("api_key", "sk-xxx")
    model = _s.get("model") or os.getenv("AI6657_LLM_MODEL") or llm_config.get("model", "gpt-4o-mini")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # qwen3 关思考：前置确保截断时也生效
    if messages and not bool(llm_config.get("enable_thinking", False)) and "qwen" in model.lower():
        msgs = [dict(m) for m in messages]
        if msgs[0].get("role") == "system":
            if not msgs[0].get("content", "").startswith("/no_think"):
                msgs[0]["content"] = "/no_think\n\n" + (msgs[0].get("content") or "")
        else:
            msgs.insert(0, {"role": "system", "content": "/no_think"})
        messages = msgs

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_config.get("temperature", 0.75)),
    }
    if max_tokens and int(max_tokens) > 0:
        payload["max_tokens"] = int(max_tokens)
    fp = float(llm_config.get("frequency_penalty", 0.0) or 0.0)
    if fp:
        payload["frequency_penalty"] = fp

    timeout = int(llm_config.get("timeout_sec", 120))
    data = _post_openai_with_retry(url, payload, headers, timeout)
    _dump_api_log(url, payload, data, log_ctx=log_ctx)

    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def generate_commentary_openai(
    prompt: str,
    llm_config: dict,
    system_prompt: str,
    max_tokens: int | None = None,
    log_ctx: dict | None = None,
) -> str:
    """调用商业大模型 API（无历史，分析模型专用）。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return _execute_openai_chat(messages, llm_config, max_tokens, log_ctx)


def generate_commentary_chat_openai(
    messages: list[dict],
    llm_config: dict,
    max_tokens: int | None = None,
) -> str:
    """调用商业大模型 API（带对话历史，风格模型专用）。"""
    return _execute_openai_chat(messages, llm_config, max_tokens)
