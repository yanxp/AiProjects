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

# 重试调度表（单位：秒）。
# 每个表代表一种错误类别的"等待序列"：第 N 次失败后睡 _SCHEDULES[...][N-1]，
# 序列用完还失败就抛给上层。表长 = 最多重试次数，总调用次数 = 表长 + 1。
_SCHEDULES: dict[str, list[float]] = {
    # 普通瞬态错误（网络抖动、一般 5xx、429）：10 秒级别就够，过久反而拖慢失败反馈
    "generic": [1.0, 2.0, 4.0],
    # Ark / 豆包 endpoint 冷启动返回 `ModelLoading`，实测通常 30-60s 才预热完。
    # 给 ~70s 的总窗口；如果这都不够就是 endpoint 有问题，让用户去控制台看
    "model_loading": [5.0, 10.0, 15.0, 20.0, 20.0],
}

# openai SDK 里被视为"瞬态"的异常族。遇到这些就退避重试。
_TRANSIENT_EXCS: tuple[type[BaseException], ...] = (
    InternalServerError,     # HTTP 5xx（含 Ark 的 ModelLoading）
    APIConnectionError,      # DNS / TCP / TLS 抖动
    APITimeoutError,         # 读超时
    RateLimitError,          # 429：退避后一般能过
)

T = TypeVar("T")


def _client() -> AsyncOpenAI:
    """构造 OpenAI 兼容客户端；每次调用都现取,方便测试中 monkeypatch 配置。"""
    s = get_settings()
    return AsyncOpenAI(base_url=s.LLM_BASE_URL, api_key=s.LLM_API_KEY)


def _classify(exc: BaseException) -> str:
    """
    根据异常内容选择退避表。
    目前只区分：Ark 的 `ModelLoading`（需长窗口） vs 其它瞬态错误（短窗口）。
    用 str(exc) 子串匹配而不是 exc.body["error"]["code"]，因为不同 provider 的
    错误体结构不一；字符串匹配对误判的代价只是"用了长表"，风险可接受。
    """
    if "ModelLoading" in str(exc):
        return "model_loading"
    return "generic"


async def _with_retry(fn: Callable[[], Any], label: str) -> Any:
    """对给定的 async 调用执行退避重试；仅重试瞬态错误，4xx 直接抛出。"""
    last_exc: BaseException | None = None
    # 允许在运行时根据第一次看到的异常动态切换调度表
    # （某些情况下首错和后续错的类别可能不同，比如连接抖动后 endpoint 开始 loading）
    schedule: list[float] = _SCHEDULES["generic"]
    max_attempts = len(schedule) + 1
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except _TRANSIENT_EXCS as exc:
            last_exc = exc
            # 根据异常实际类别动态选调度表；若已用更长的表就不降级
            category = _classify(exc)
            if category == "model_loading" and schedule is not _SCHEDULES["model_loading"]:
                schedule = _SCHEDULES["model_loading"]
                max_attempts = len(schedule) + 1
            if attempt >= max_attempts:
                break
            # 当前尝试号对应 schedule[attempt-1]
            delay = schedule[attempt - 1]
            # 用 stderr 打印，避免污染 Synthesizer 的 stdout 流式输出
            print(
                f"[llm.{label}] transient error (attempt {attempt}/{max_attempts}, "
                f"{category}): {type(exc).__name__}: {exc}; retrying in {delay:.1f}s",
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


async def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """
    批量嵌入。用于本地 RAG 的离线建库 / 运行时 query 编码。

    - 单次最大条数由 provider 控制（OpenAI 2048，DeepSeek 不支持 embeddings，
      豆包/Ark 建议 ≤100），调用方自行分批；这里只做最小封装。
    - 返回 list[list[float]]，与 input 顺序一一对应。
    - 使用和 chat 同一份退避重试，冷启动/瞬态错误下也能撑住。
    """
    s = get_settings()

    async def _call():
        resp = await _client().embeddings.create(
            model=model or s.LLM_EMBED_MODEL,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    return await _with_retry(_call, label="embed")


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
