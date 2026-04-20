"""
本地 RAG 检索（最小版：pickle + numpy 余弦相似度）。

设计目标：
- 零额外服务：单文件 pickle 存 (chunks, sources, vectors)，加载后驻留内存
- 零额外依赖：只用 numpy；没有 faiss / chromadb / sqlite-vec
- 够小够快：几千到几万块片段规模 OK；更大规模再换 chromadb/qdrant

索引格式（由 scripts/build_index.py 离线写出）：
    {
        "chunks":  list[str]         # 片段原文
        "sources": list[str]         # 来源路径（给引用和 UI 用）
        "vectors": np.ndarray (N, D) # 已 L2 归一化的嵌入
        "model":   str               # 构建时使用的 embedding 模型（供校验）
    }

运行时调用：
    hits = await local_rag.search("query...", top_k=5)
    # hits: list[dict(snippet, source, score)]
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .. import llm
from ..config import get_settings


@lru_cache(maxsize=1)
def _load_index() -> Optional[dict]:
    """
    懒加载索引文件。缓存一份避免每次 query 都读盘。
    索引不存在时返回 None，让调用方降级（不是抛异常，避免把整条 pipeline 拖挂）。
    """
    s = get_settings()
    path = Path(s.RAG_INDEX_PATH)
    if not path.exists():
        return None
    with path.open("rb") as f:
        idx = pickle.load(f)
    # 简单校验：必须含这三个字段
    if not all(k in idx for k in ("chunks", "sources", "vectors")):
        return None
    return idx


def is_available() -> bool:
    """给上层一个便捷判断：索引文件存在且结构合法。"""
    return _load_index() is not None


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """单条向量归一化，便于后续和已归一化的索引做点积 = 余弦相似度。"""
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


async def search(query: str, top_k: int = 5) -> list[dict]:
    """
    对 query 做嵌入，在索引里取 top_k 片段。
    返回 [{snippet, source, score}, ...]，按相似度降序。
    """
    idx = _load_index()
    if idx is None:
        return []

    # 调 embedding API 编码 query
    vectors_q = await llm.embed([query])
    qv = np.asarray(vectors_q[0], dtype="float32")
    qv = _l2_normalize(qv)

    mat: np.ndarray = idx["vectors"]  # 已经在建库阶段归一化了
    # 余弦相似度 = 归一化向量的点积
    scores = mat @ qv

    # 维度保护：top_k 不能超过索引长度
    k = min(top_k, scores.shape[0])
    # 用 argpartition 取前 k（O(N)），再对这 k 个做一次排序
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [
        {
            "snippet": idx["chunks"][i],
            "source": idx["sources"][i],
            "score": float(scores[i]),
        }
        for i in top_idx
    ]
