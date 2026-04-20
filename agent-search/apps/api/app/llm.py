"""
LLM 客户端封装。

关键点：
- 使用 `openai` SDK，但 base_url 指向任何 OpenAI 兼容服务
  （DeepSeek 官方 API / 火山 Ark 豆包 / vLLM / SGLang / OpenAI 本尊 都行），
  这样未来从 API 切到自托管 vLLM 只改环境变量，不改代码。
- 提供 `chat()` 一次性调用 和 `stream_chat()` 流式调用 两种形态。
- 内置指数退避重试，覆盖常见的瞬态错误（5xx / 429 / 连接抖动 / Ark endpoint 冷启动
  `ModelLoading`），避免 Agent 因为单次网络抖动或冷启动而整条管线挂掉。
  4xx（认证失败、模型不存在、请求体非法）**不**重试，没意义。
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from typing import Any, Callable, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from .config import get_settings

# 重试参数：首次失败后等 1s、2s、4s 各重试一次，最多 4 次调用。
# 对 Ark 冷启动（几十秒内可预热完）这个窗口基本够用；如果还是失败就让它抛。
_RETRY_ATTEMPTS = 4
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 8.0

# openai SDK 里被视为"瞬态"的异常族。遇到这些就退避重试。
_TRANSIENT_EXCS: tuple[type[BaseException], ...] = (
    InternalServerError,     # HTTP 5xx（含 Ark 的 ModelLoading）
    APIConnectionError,      # DNS / TCP / TLS 抖动
    APITimeoutError,         # 读超时
    RateLimitError,          # 429：退避后一般能过
)

T = TypeVar("T")


def _client() -> AsyncOpenAI:
    """构造 OpenAI 兼容客户端；每次调用都现取，方便测试中 monkeypatch 配置。"""
    s = get_settings()
    return AsyncOpenAI(base_url=s.LLM_BASE_URL, api_key=s.LLM_API_KEY)


async def _with_retry(fn: Callable[[], Any], label: str) -> Any:
    """对给定的 async 调用执行退避重试；仅重试瞬态错误，4xx 直接抛出。"""
    last_exc: BaseException | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return await fn()
        except _TRANSIENT_EXCS as exc:
            last_exc = exc
            if attempt >= _RETRY_ATTEMPTS:
                break
            # 退避时间：1s、2s、4s，上限 _RETRY_MAX_DELAY
            delay = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY)
            # 用 stderr 打印，避免污染 Synthesizer 的 stdout 流式输出
            print(
                f"[llm.{label}] transient error (attempt {attempt}/{_RETRY_ATTEMPTS}): "
                f"{type(exc).__name__}: {exc}; retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(delay)
    # 多次仍失败：重抛最后一次异常，让上层看到真实原因
    assert last_exc is not None
    raise last_exc


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

    async def _call():
        return await _client().chat.completions.create(**kwargs)

    resp = await _with_retry(_call, label="chat")
    return resp.choices[0].message.content or ""


async def stream_chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    """
    流式 chat completion。用于 Synthesizer 节点：一边生成答案 token 一边推给前端。

    注意：重试只覆盖"建立流"的阶段（首个 HTTP 响应前）。一旦开始读取 chunk、
    provider 中途断流，直接抛给上层；中途重试会让已经打印出去的 token 被重复，
    体验比挂掉更差。
    """
    s = get_settings()

    async def _open_stream():
        return await _client().chat.completions.create(
            model=model or s.LLM_MODEL,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

    stream = await _with_retry(_open_stream, label="stream_chat")
    async for chunk in stream:
        # chunk.choices 可能为空（某些 provider 的首个事件）；要做空值保护
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
