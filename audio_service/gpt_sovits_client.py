"""6657 风格离线录像解说 AI 项目
项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
本文件功能：GPT-SoVITS 语音合成 API 客户端。

启动方式：被 sbmachine/phase4_assemble.py 的 run_phase4() 导入调用；也可独立运行（python audio_service/gpt_sovits_client.py --text ... --output ...）。
输入数据流：解说文本字符串和 GPT-SoVITS 运行时配置（YAML）。
输出数据流：写入 WAV 音频文件到指定路径。
调用方式：通过 synthesize() 合成普通语音，通过 synthesize_emotional() 按情绪分段合成并拼接，
通过 set_weights() 切换模型权重。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import wave
from pathlib import Path

import requests

try:
    import yaml
except ImportError:
    print("Missing pyyaml. Please install pyyaml first.")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sbmachine.file_lock import FileLock


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _resolve_api_url(config: dict) -> str:
    api_settings = config.get("api", {})
    host = os.getenv("GPT_SOVITS_API_HOST") or api_settings.get("host", "127.0.0.1")
    port = int(os.getenv("GPT_SOVITS_API_PORT") or api_settings.get("port", 9880))
    return f"http://{host}:{port}"


def set_weights(api_url: str, gpt_weights: str, sovits_weights: str) -> None:
    gpt_weights = os.getenv("GPT_SOVITS_GPT_WEIGHTS") or gpt_weights
    sovits_weights = os.getenv("GPT_SOVITS_SOVITS_WEIGHTS") or sovits_weights
    if gpt_weights:
        response = requests.get(
            f"{api_url}/set_gpt_weights",
            params={"weights_path": gpt_weights},
            timeout=60,
        )
        response.raise_for_status()
    if sovits_weights:
        response = requests.get(
            f"{api_url}/set_sovits_weights",
            params={"weights_path": sovits_weights},
            timeout=60,
        )
        response.raise_for_status()


def _emotion_speed_factors() -> dict[str, float]:
    """加载实际发给 GPT-SoVITS 的各情绪语速系数（speech_rate.tts_speed_factor，只管 TTS 语速）。"""
    try:
        rules_path = PROJECT_ROOT / "Prompt" / "json" / "hype_rules.json"
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        values = rules.get("speech_rate", {}).get("tts_speed_factor", {})
        return {str(emotion): float(value) for emotion, value in values.items()}
    except Exception:
        return {}


def _build_tts_payload(
    text: str,
    reference: dict,
    media_type: str = "wav",
    speed_factor: float = 1.0,
) -> dict:
    return {
        "text": text,
        "text_lang": reference.get("text_lang", reference.get("prompt_lang", "zh")),
        "ref_audio_path": str(
            resolve_path(reference.get("audio_path", "data/voice/reference/6657_ref.wav"))
        ),
        "prompt_text": reference.get("prompt_text", ""),
        "prompt_lang": reference.get("prompt_lang", "zh"),
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": media_type,
        "streaming_mode": False,
        "speed_factor": speed_factor,
    }


def _validate_wav_bytes(wav_bytes: bytes) -> tuple[int, int, int, int]:
    """拒绝非完整、空的、非 PCM 的 WAV 响应体，防止把坏音频写入磁盘。"""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError(f"unsupported WAV compression: {wav_file.getcomptype()}")
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if channels <= 0 or sample_width <= 0 or sample_rate <= 0 or frame_count <= 0:
                raise ValueError("WAV has invalid or empty audio parameters")
            audio_frames = wav_file.readframes(frame_count)
            expected_size = frame_count * channels * sample_width
            if len(audio_frames) != expected_size:
                raise ValueError(
                    f"truncated WAV data: expected {expected_size} bytes, got {len(audio_frames)}"
                )
    except (EOFError, wave.Error) as exc:
        raise ValueError("GPT-SoVITS response is not a decodable WAV") from exc
    return channels, sample_width, sample_rate, frame_count


def _compute_reference_fingerprint(reference: dict, speed_factor: float) -> dict:
    tts_payload = _build_tts_payload("", reference, speed_factor=speed_factor)
    audio_path = Path(tts_payload["ref_audio_path"])
    audio_sha256 = None
    if audio_path.is_file():
        audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    return {
        "audio_path": str(audio_path),
        "audio_sha256": audio_sha256,
        "prompt_text": tts_payload["prompt_text"],
        "prompt_lang": tts_payload["prompt_lang"],
        "text_lang": tts_payload["text_lang"],
        "speed_factor": tts_payload["speed_factor"],
    }


def _compute_file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_weight_fingerprint(weight_path: str | Path) -> dict:
    resolved_path = resolve_path(weight_path) if weight_path else None
    content_sha256 = None
    if resolved_path is not None and resolved_path.is_file():
        # 文件被快速替换为同大小模型时，元数据可能保持不变。
        # 对文件内容取哈希，避免复用过期的 TTS 资源。
        content_sha256 = _compute_file_sha256(str(resolved_path))
    return {
        "path": str(resolved_path) if resolved_path is not None else "",
        "sha256": content_sha256,
    }


def tts_cache_fingerprint(
    config: dict,
    text: str,
    *,
    speed_factor: float = 1.0,
    variant_id: str | None = None,
    profile_id: str | None = None,
    budget_overage: float = 1.0,
) -> str:
    """对所有可能改变情绪 TTS 输出的运行时值取指纹，用于缓存命中判定。

    缓存键除 model/引用音频/情绪语速外，还包含调用方显式指定的
    speed_factor、variant_id、profile_id 与 budget_overage：任一变化都会
    生成新的缓存文件。budget_overage 参与 v2 合成时的实际倍速
    （overage_mult），因此必须进入缓存身份（审计 §5.2：相同文本在不同
    char_budget 下不得复用另一速度合成的 WAV）。
    """
    from audio_service.emotion import parse_emotional_text, resolve_emotion_ref

    model_config = dict(config.get("model", {}))
    model_config["gpt_weights"] = os.getenv("GPT_SOVITS_GPT_WEIGHTS") or model_config.get("gpt_weights", "")
    model_config["sovits_weights"] = os.getenv("GPT_SOVITS_SOVITS_WEIGHTS") or model_config.get("sovits_weights", "")
    model_config["gpt_weights"] = _compute_weight_fingerprint(model_config["gpt_weights"])
    model_config["sovits_weights"] = _compute_weight_fingerprint(model_config["sovits_weights"])
    default_reference = config.get("reference", {})
    emotion_references = config.get("emotion_refs", {})
    speed_factors = _emotion_speed_factors()
    reference_fingerprints = []
    for segment in parse_emotional_text(text):
        segment_speed_factor = (
            float(speed_factors.get(segment.emotion, 1.0))
            * float(speed_factor)
            * float(budget_overage)
        )
        reference = resolve_emotion_ref(segment.emotion, emotion_references, default_reference)
        reference_fingerprints.append(
            {
                "emotion": segment.emotion,
                **_compute_reference_fingerprint(reference, segment_speed_factor),
            }
        )
    encoded = json.dumps(
        {
            "model": model_config,
            "references": reference_fingerprints,
            "variant_id": variant_id,
            "profile_id": profile_id,
            "speed_factor": float(speed_factor),
            "budget_overage": float(budget_overage),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tts_runtime_fingerprint(config: dict, sample_rate_hz: int | None = None) -> dict:
    """从当前 TTS 运行时配置推导 engine/voice/preprocess 三个指纹。

    用于与 speech profile（§11.5）核对：任务单引用的 profile 指纹必须与
    Phase4 当前 TTS 运行指纹完全一致，否则禁止按任务单风险分级合成。
    """
    model_config = dict(config.get("model", {}))
    model_config["gpt_weights"] = os.getenv("GPT_SOVITS_GPT_WEIGHTS") or model_config.get("gpt_weights", "")
    model_config["sovits_weights"] = os.getenv("GPT_SOVITS_SOVITS_WEIGHTS") or model_config.get("sovits_weights", "")
    engine_payload = {
        "engine": "gpt-sovits",
        "weights": {
            key: _compute_weight_fingerprint(weight_path)
            for key, weight_path in model_config.items()
        },
    }
    reference_payload: dict[str, dict] = {}
    default_reference = config.get("reference", {})
    if isinstance(default_reference, dict) and default_reference:
        reference_payload["default"] = _compute_reference_fingerprint(default_reference, 1.0)
    emotion_references = config.get("emotion_refs", {})
    if isinstance(emotion_references, dict):
        for emotion in sorted(emotion_references):
            if isinstance(emotion_references[emotion], dict):
                reference_payload[emotion] = _compute_reference_fingerprint(
                    emotion_references[emotion], 1.0
                )
    preprocess_payload = {
        "policy": "phase4-pcm-policy-v1",
        "text_split_method": "cut5",
        "media_type": "wav",
        "streaming_mode": False,
        "emotion_tag_syntax": "[emotion]",
        "sample_rate_hz": sample_rate_hz,
    }

    def _compute_digest(payload: dict) -> str:
        encoded_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded_payload).hexdigest()

    return {
        "engine_fingerprint": _compute_digest(engine_payload),
        "voice_fingerprint": _compute_digest(reference_payload),
        "preprocess_fingerprint": _compute_digest(preprocess_payload),
    }


def _synthesize_bytes(
    config: dict,
    text: str,
    reference: dict,
    speed_factor: float = 1.0,
) -> bytes:
    """调用 GPT-SoVITS 的 /tts 接口，返回原始 WAV 字节，不落盘。"""
    api_url = _resolve_api_url(config)
    response = requests.post(
        f"{api_url}/tts",
        json=_build_tts_payload(text, reference, speed_factor=speed_factor),
        timeout=300,
    )
    response.raise_for_status()
    wav_bytes = response.content
    _validate_wav_bytes(wav_bytes)
    return wav_bytes


def synthesize_segment(config: dict, text: str, ref: dict, output_path: Path) -> Path:
    wav_bytes = _synthesize_bytes(config, text, ref)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(wav_bytes)
    return output_path


def synthesize(config: dict, text: str, output_path: Path) -> None:
    model_config = config.get("model", {})
    set_weights(
        _resolve_api_url(config),
        model_config.get("gpt_weights", ""),
        model_config.get("sovits_weights", ""),
    )
    synthesize_segment(config, text, config.get("reference", {}), output_path)
    print(f"audio written: {output_path}")


def synthesize_emotional(
    config: dict,
    text: str,
    output_path: Path,
    *,
    budget_overage: float = 1.0,
    speed_factor: float = 1.0,
) -> Path:
    """按情绪分段合成,全程在内存中拼接 PCM,最后一次写出 WAV。不生成中间段文件。

    budget_overage: Phase3b 产物的预算超支比例（output_chars / char_budget）。>1.0 时动态加速
    TTS 语速以容下更长的文本，上限 1.5x。
    speed_factor: 调用方显式指定的倍速（Phase4 v3 任务单选择算法使用），与情绪语速相乘；
    v2 单稿路径不传，行为保持不变。
    """
    from audio_service.emotion import parse_emotional_text, resolve_emotion_ref

    model_config = config.get("model", {})
    set_weights(
        _resolve_api_url(config),
        model_config.get("gpt_weights", ""),
        model_config.get("sovits_weights", ""),
    )

    default_reference = config.get("reference", {})
    emotion_references = config.get("emotion_refs", {})
    segments = parse_emotional_text(text)
    if not segments:
        raise ValueError("commentary text is empty or contains only emotion tags")

    # 一次性加载 tts_speed_factor，避免每段重复读取磁盘。
    speed_factors = _emotion_speed_factors()
    budget_speed_multiplier = max(1.0, min(1.5, budget_overage))

    # 按情绪批量请求，保持原始顺序。
    ordered_wav_bytes: list[bytes] = [b""] * len(segments)
    segment_indices_by_emotion: dict[str, list[int]] = {}
    for segment_index, segment in enumerate(segments):
        segment_indices_by_emotion.setdefault(segment.emotion, []).append(segment_index)

    for emotion, segment_indices in segment_indices_by_emotion.items():
        reference = resolve_emotion_ref(emotion, emotion_references, default_reference)
        segment_speed_factor = (
            float(speed_factors.get(emotion, 1.0))
            * budget_speed_multiplier
            * float(speed_factor)
        )
        for segment_index in segment_indices:
            ordered_wav_bytes[segment_index] = _synthesize_bytes(
                config,
                segments[segment_index].text,
                reference,
                speed_factor=segment_speed_factor,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(ordered_wav_bytes) == 1:
        output_path.write_bytes(ordered_wav_bytes[0])
    else:
        _concat_wav_bytes(ordered_wav_bytes, output_path)
    print(f"audio written: {output_path} ({len(segments)} segments, in-memory concat)")
    return output_path


def _concat_wav_bytes(wav_bytes_list: list[bytes], output_path: Path) -> None:
    """把内存中的多段 WAV 字节拼接成单个 WAV 文件，不使用临时文件。"""
    wav_params = None
    all_audio_frames: list[bytes] = []
    with wave.open(io.BytesIO(wav_bytes_list[0]), "rb") as first_wav:
        wav_params = first_wav.getparams()
        all_audio_frames.append(first_wav.readframes(first_wav.getnframes()))
    for wav_bytes in wav_bytes_list[1:]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getparams()[:3] != wav_params[:3]:  # 声道数、采样宽度和采样率
                raise RuntimeError(
                    f"WAV format mismatch: expected {wav_params[:3]}, got {wav_file.getparams()[:3]}"
                )
            all_audio_frames.append(wav_file.readframes(wav_file.getnframes()))

    with wave.open(str(output_path), "wb") as output_wav:
        output_wav.setparams(wav_params)
        for audio_frame_chunk in all_audio_frames:
            output_wav.writeframes(audio_frame_chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call GPT-SoVITS API to synthesize commentary audio.")
    parser.add_argument("--config", default="audio_service/gpt_sovits_runtime.yaml")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output/tts_demo.wav")
    parser.add_argument("--emotional", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = read_config(resolve_path(args.config))
        with FileLock(PROJECT_ROOT / "output" / ".sovits.lock"):
            if args.emotional:
                synthesize_emotional(config, args.text, resolve_path(args.output))
            else:
                synthesize(config, args.text, resolve_path(args.output))
    except Exception as exc:
        print(f"synthesis failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
