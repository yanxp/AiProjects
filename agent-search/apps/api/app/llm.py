"""
LLM 客户端封装。

关键点：
- 使用 `openai` SDK，但 base_url 指向任何 OpenAI 兼容服务
  （DeepSeek 官方 API / vLLM / SGLang / OpenAI 本尊 都行），
  这样未来从 API 切到自托管 vLLM 只改环境变量，不改代码。
- 提供 `chat()` 一次性调用 和 `stream_chat()` 流式调用 两种形态。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .config import get_settings


def _client() -> AsyncOpenAI:
    """构造 OpenAI 兼容客户端；每次调用都现取，方便测试中 monkeypatch 配置。"""
    s = get_settings()
    return AsyncOpenAI(base_url=s.LLM_BASE_URL, api_key=s.LLM_API_KEY)


async def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
    response_format: dict | None = None,
) -> str:
    """一次性 chat completion。常用于 Planner / Reader / Reflector 这些要拿到完整 JSON 的节点。"""
    s = get_settings()
    kwargs: dict = {
        "model": model or s.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    # 某些后端（OpenAI / DeepSeek 新版）支持强制 JSON 输出，可以大幅减少解析错误
    if response_format:
        kwargs["response_format"] = response_format

    resp = await _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


async def stream_chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    """流式 chat completion。用于 Synthesizer 节点：一边生成答案 token 一边推给前端。"""
    s = get_settings()
    stream = await _client().chat.completions.create(
        model=model or s.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    async for chunk in stream:
        # chunk.choices 可能为空（某些 provider 的首个事件）；要做空值保护
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
