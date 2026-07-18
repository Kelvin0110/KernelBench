"""
NVIDIA OpenAI-compatible chat completions for dual-LLM setup (coder + summarizer).

Model resolution follows the same idea as call_llm_api_v3: short aliases map to
full integrate.api.nvidia.com model IDs. Extend NVIDIA_MODEL_ALIASES as needed.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Short name -> full NVIDIA model id (OpenAI-compatible path on NIM).
NVIDIA_MODEL_ALIASES: dict[str, str] = {
    "nemotron-ultra": "nvdev/nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "gpt-oss-120b": "nvdev/openai/gpt-oss-120b",
    "kimi-k2": "moonshotai/kimi-k2-thinking",
}

DEFAULT_NVIDIA_TIMEOUT_SEC = 60.0


def _get_nvidia_key() -> str:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Export it or add it to a .env file."
        )
    return key


def _get_nvidia_timeout_sec() -> float:
    raw = os.getenv("NVIDIA_API_TIMEOUT_SEC", str(DEFAULT_NVIDIA_TIMEOUT_SEC)).strip()
    try:
        parsed = float(raw)
        if parsed > 0:
            return parsed
    except ValueError:
        pass
    return DEFAULT_NVIDIA_TIMEOUT_SEC


def resolve_nvidia_model_id(spec: str) -> str:
    """
    Return the full NVIDIA model id. If `spec` is a known alias, expand it;
    otherwise treat `spec` as the full id (e.g. openai/gpt-oss-120b).
    """
    s = spec.strip()
    return NVIDIA_MODEL_ALIASES.get(s, s)


def get_tri_llm_model_ids() -> tuple[str, str, str]:
    """
    (coder_model_id, summarizer_model_id, extractor_model_id) for NVIDIA integrate API.

    Env:
      NVIDIA_CODER_MODEL (default: gpt-oss-120b)
      NVIDIA_SUMMARIZER_MODEL (default: gpt-oss-120b)
      NVIDIA_EXTRACTOR_MODEL (default: gpt-oss-120b)
    """
    coder = os.getenv("NVIDIA_CODER_MODEL", "gpt-oss-120b")
    summarizer = os.getenv("NVIDIA_SUMMARIZER_MODEL", "gpt-oss-120b")
    extractor = os.getenv("NVIDIA_EXTRACTOR_MODEL", "gpt-oss-120b")
    return (
        resolve_nvidia_model_id(coder),
        resolve_nvidia_model_id(summarizer),
        resolve_nvidia_model_id(extractor),
    )


def get_dual_llm_model_ids() -> tuple[str, str]:
    """
    (coder_model_id, summarizer_model_id) for NVIDIA integrate API.

    Env:
      NVIDIA_CODER_MODEL (default: gpt-oss-120b)
      NVIDIA_SUMMARIZER_MODEL (default: gpt-oss-120b)
    """
    coder_model_id, summarizer_model_id, _extractor_model_id = get_tri_llm_model_ids()
    return coder_model_id, summarizer_model_id


def _assistant_visible_text(message: Any) -> tuple[str | None, str | None]:
    """
    Some NVIDIA models (e.g. openai/gpt-oss-120b) return an empty `content` and
    put the user-visible reply in extended fields like `reasoning_content`.
    """
    raw = getattr(message, "content", None)
    if isinstance(raw, str) and raw.strip():
        return raw, "content"
    for attr in ("reasoning_content", "reasoning"):
        extra = getattr(message, attr, None)
        if isinstance(extra, str) and extra.strip():
            return extra, attr
    return None, None


def _assistant_message_fields(message: Any) -> dict[str, str | None]:
    """Return raw assistant message fields for downstream logging/inspection."""
    out: dict[str, str | None] = {}
    for name in ("content", "reasoning", "reasoning_content"):
        value = getattr(message, name, None)
        out[name] = value if isinstance(value, str) else None
    return out


def call_llm_nvidia(
    messages: list[dict[str, Any]],
    *,
    model_id: str,
    temperature: float = 0.4,
    max_tokens: int = 16384,
    timeout_sec: float | None = None,
) -> tuple[str | None, int]:
    """
    Single chat completion on NVIDIA integrate endpoint.

    Returns (message_content, completion_tokens_estimate).
    """
    request_timeout = timeout_sec if timeout_sec is not None else _get_nvidia_timeout_sec()
    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=_get_nvidia_key(),
        timeout=request_timeout,
    )

    params: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if "gpt-oss" in model_id.lower():
        params["reasoning_effort"] = "medium"

    response = client.chat.completions.create(**params)
    choice = response.choices[0]
    content, _field = _assistant_visible_text(choice.message)
    usage = response.usage
    tokens = getattr(usage, "completion_tokens", None) if usage else None
    if tokens is None and content:
        tokens = len(content.split())
    elif tokens is None:
        tokens = 0
    return content, int(tokens)


def call_llm_nvidia_with_meta(
    messages: list[dict[str, Any]],
    *,
    model_id: str,
    temperature: float = 0.4,
    max_tokens: int = 16384,
    timeout_sec: float | None = None,
) -> tuple[str | None, int, dict[str, Any]]:
    request_timeout = timeout_sec if timeout_sec is not None else _get_nvidia_timeout_sec()
    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=_get_nvidia_key(),
        timeout=request_timeout,
    )

    params: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if "gpt-oss" in model_id.lower():
        params["reasoning_effort"] = "medium"

    response = client.chat.completions.create(**params)
    choice = response.choices[0]
    content, field = _assistant_visible_text(choice.message)
    assistant_fields = _assistant_message_fields(choice.message)
    usage = response.usage
    tokens = getattr(usage, "completion_tokens", None) if usage else None
    if tokens is None and content:
        tokens = len(content.split())
    elif tokens is None:
        tokens = 0
    meta: dict[str, Any] = {
        "assistant_content_field": field or "none",
        "assistant_content": assistant_fields.get("content"),
        "assistant_reasoning": assistant_fields.get("reasoning") or assistant_fields.get("reasoning_content"),
        "model_id": model_id,
    }
    return content, int(tokens), meta


def call_coder(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.4,
    max_tokens: int = 16384,
    timeout_sec: float | None = None,
) -> tuple[str | None, int]:
    model_id, _ = get_dual_llm_model_ids()
    return call_llm_nvidia(
        messages,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


def call_coder_with_meta(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.4,
    max_tokens: int = 16384,
    timeout_sec: float | None = None,
) -> tuple[str | None, int, dict[str, Any]]:
    model_id, _ = get_dual_llm_model_ids()
    return call_llm_nvidia_with_meta(
        messages,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


def call_summarizer(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    timeout_sec: float | None = None,
) -> tuple[str | None, int]:
    _, model_id = get_dual_llm_model_ids()
    return call_llm_nvidia(
        messages,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


def call_summarizer_with_meta(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    timeout_sec: float | None = None,
) -> tuple[str | None, int, dict[str, Any]]:
    _, model_id = get_dual_llm_model_ids()
    return call_llm_nvidia_with_meta(
        messages,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


def call_extractor(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.1,
    max_tokens: int = 8192,
    timeout_sec: float | None = None,
    model_id: str | None = None,
) -> tuple[str | None, int]:
    _, _, extractor_model_id = get_tri_llm_model_ids()
    resolved_model_id = model_id or extractor_model_id
    return call_llm_nvidia(
        messages,
        model_id=resolved_model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


def call_extractor_with_meta(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.1,
    max_tokens: int = 8192,
    timeout_sec: float | None = None,
    model_id: str | None = None,
) -> tuple[str | None, int, dict[str, Any]]:
    _, _, extractor_model_id = get_tri_llm_model_ids()
    resolved_model_id = model_id or extractor_model_id
    return call_llm_nvidia_with_meta(
        messages,
        model_id=resolved_model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


if __name__ == "__main__":
    # Example usage
    content, tokens, meta = call_coder_with_meta(
        messages=[{"role": "user", "content": "Write a Python function that adds two numbers."}]
    )
    print("Content:", content)
    print("Tokens:", tokens)
    print("Meta:", meta)