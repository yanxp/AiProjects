"""
嵌入入口：根据 EMBED_BACKEND 分发到 API 还是本地 sentence-transformers。

- `api`：复用 llm.embed()，走 OpenAI 兼容协议（OpenAI / Ark embedding endpoint / vLLM）
- `local`：用 sentence-transformers 加载本地模型（默认 BAAI/bge-small-zh-v1.5）

调用方只要 `await embed(texts)`，不关心背后是谁。

设计选择：
- sentence-transformers 是**可选依赖**，只有在 EMBED_BACKEND=local 时才会 import；
  不用 RAG 或只用 API 的用户不需要装它（省 ~2GB torch + ~500MB 模型）。
- 本地模型单例缓存在进程内，避免每次 query 都重载。
- sentence-transformers 的 encode() 是同步 CPU/GPU bound，这里用 run_in_executor
  套一层避免阻塞事件循环；调用方的 await 语义不变。
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, List

from .config import get_settings


@lru_cache(maxsize=2)
def _get_local_model(name: str) -> Any:
    """
    懒加载本地 sentence-transformers 模型（单例）。
    首次调用会下载权重到 ~/.cache/huggingface/，可能耗时；之后走本地缓存。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            "EMBED_BACKEND=local 需要安装 sentence-transformers：\n"
            "    pip install sentence-transformers\n"
            "或 pip install -r apps/api/requirements.txt"
        ) from e
    return SentenceTransformer(name)


async def _embed_local(texts: List[str]) -> List[List[float]]:
    s = get_settings()
    model = _get_local_model(s.EMBED_LOCAL_MODEL)
    loop = asyncio.get_event_loop()
    # encode() 内部已会批处理 + L2 归一化（normalize_embeddings=True）
    # 放到默认 executor 避免把事件循环卡住
    vectors = await loop.run_in_executor(
        None,
        lambda: model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
    )
    return [list(map(float, v)) for v in vectors]


async def _embed_api(texts: List[str]) -> List[List[float]]:
    # 延迟 import 避免循环依赖
    from . import llm
    return await llm.embed(texts)


async def embed(texts: List[str]) -> List[List[float]]:
    """
    统一嵌入入口。根据 EMBED_BACKEND 选择后端。
    返回 list[list[float]]，顺序与 input 一致；调用方自行处理归一化/numpy 转换。
    """
    s = get_settings()
    backend = (s.EMBED_BACKEND or "api").lower()
    if backend == "local":
        return await _embed_local(texts)
    return await _embed_api(texts)
