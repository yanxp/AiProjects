"""
Episodic 记忆（最小版：JSONL + numpy 余弦 + supersedes 指针）。

设计目标（和 retrieval/local_rag.py 一样的最小打法）：
- **零额外服务**：单文件 JSONL，一行一条 episode，append-only
- **零额外依赖**：只用 numpy + embeddings 复用现有 dispatcher
- **可审计可回滚**：不原地改行，"更新"通过 supersedes 指针覆盖旧行

一条 episode 的字段：
    {
      "id":            "ep_xxx",           # 唯一 id
      "ts":            1730000000.0,       # Unix 秒
      "query":         "...",              # 原始中文/英文 query
      "answer":        "...",              # 截断到 MEMORY_ANSWER_CAP 字
      "paper_ids":     ["W123", ...],
      "embed_backend": "api" | "local",    # 兼容检查用：换了后端要跳过
      "embed_model":   "...",              # 同上
      "embedding":     [512 floats],       # 已 L2 归一化
      "supersedes":    "ep_yyy" | (absent) # 指向被本条替换的旧 episode
    }

读路径（recall）：
    1. 加载全部行
    2. 过滤被 supersedes 指向的行（只剩"当前版"）
    3. 再按 TTL 过滤太老的
    4. 再按 backend/model 一致性过滤（换了嵌入后端的旧行没法比）
    5. 余下按余弦相似度取 top-k，低于 threshold 的丢掉

写路径（remember）：
    1. 嵌入新 query
    2. 和现有"当前版 + 同后端"算 cosine，取 top-1
    3. top-1 score > UPDATE_THR:
       - policy=supersede（默认）: 新行 supersedes 旧行
       - policy=append          : 两行都保留
       - policy=skip            : 不写
    4. 否则直接 append
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from . import embeddings
from .config import get_settings

# 截断 answer 存储长度，避免 JSONL 撑大（单条 >10KB 后 recall 加载会慢）
MEMORY_ANSWER_CAP = 4096


def _memory_path() -> Path:
    """允许 MEMORY_PATH 带 ~ 或相对路径。"""
    s = get_settings()
    return Path(s.MEMORY_PATH).expanduser()


def _current_backend_model() -> tuple[str, str]:
    """当前 embedding 后端 + 模型名，用于写入时打 tag、recall 时过滤。"""
    s = get_settings()
    backend = (s.EMBED_BACKEND or "api").lower()
    model = s.EMBED_LOCAL_MODEL if backend == "local" else s.LLM_EMBED_MODEL
    return backend, model


async def _embed_normalized(text: str) -> Optional[np.ndarray]:
    """
    嵌入 + 清洗 + L2 归一化。嵌入服务偶尔返 NaN / 空，这里统一兜底，
    拿到的要么是干净归一化向量，要么 None（调用方降级）。
    """
    if not text or not text.strip():
        return None
    vecs = await embeddings.embed([text])
    if not vecs:
        return None
    v = np.asarray(vecs[0], dtype="float32")
    if v.size == 0 or not np.isfinite(v).all():
        return None
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return None
    return v / n


def _load_all() -> list[dict]:
    """顺序读 JSONL，坏行跳过（崩过一次不影响其他 episode）。"""
    path = _memory_path()
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    # 单行坏不影响整库
                    continue
    except Exception:
        return []
    return out


def _active(episodes: list[dict]) -> list[dict]:
    """
    过滤掉被 supersedes 指向的"已作废版本"。
    同时 TTL 过滤（MEMORY_MAX_AGE_DAYS=0 关闭）。
    """
    s = get_settings()
    superseded_ids = {e["supersedes"] for e in episodes if e.get("supersedes")}
    now = time.time()
    ttl_sec = max(0, s.MEMORY_MAX_AGE_DAYS) * 86400 if s.MEMORY_MAX_AGE_DAYS else 0
    out: list[dict] = []
    for e in episodes:
        if e.get("id") in superseded_ids:
            continue
        if ttl_sec > 0:
            ts = e.get("ts", 0)
            if now - ts > ttl_sec:
                continue
        out.append(e)
    return out


def _compatible(e: dict) -> bool:
    """记录的 embedding 能不能和当前后端/模型做余弦。"""
    backend, model = _current_backend_model()
    return e.get("embed_backend") == backend and e.get("embed_model") == model


def _without_embedding(e: dict) -> dict:
    """对外暴露 episode 时不带原始向量（太大 + 没用）。"""
    out = dict(e)
    out.pop("embedding", None)
    return out


async def recall(
    query: str,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> list[dict]:
    """
    返回最相似的 k 条 episode（active + 同后端），score 降序，
    低于 threshold 的截断（可能返空列表）。
    不改任何状态，纯只读。
    """
    s = get_settings()
    top_k = k if k is not None else s.MEMORY_RECALL_K
    thr = threshold if threshold is not None else s.MEMORY_RECALL_THR
    if top_k <= 0:
        return []

    eps = _active(_load_all())
    usable = [e for e in eps if _compatible(e) and e.get("embedding")]
    if not usable:
        return []

    qv = await _embed_normalized(query)
    if qv is None:
        return []

    try:
        mat = np.asarray([e["embedding"] for e in usable], dtype="float32")
    except Exception:
        return []
    if mat.ndim != 2 or mat.shape[1] != qv.shape[0]:
        # 极端情况下（手写 JSONL / 历史数据脏）维度不齐，不要让整个 recall 挂掉
        return []

    scores = mat @ qv
    kk = min(top_k, scores.shape[0])
    idxs = np.argpartition(-scores, kk - 1)[:kk]
    idxs = idxs[np.argsort(-scores[idxs])]

    out: list[dict] = []
    for i in idxs:
        score = float(scores[i])
        if score < thr:
            break  # 已按分数降序，后面只会更低
        rec = _without_embedding(usable[i])
        rec["score"] = score
        out.append(rec)
    return out


async def remember(
    query: str,
    answer: str,
    paper_ids: Optional[list[str]] = None,
    *,
    policy: Optional[str] = None,
) -> Optional[dict]:
    """
    把本轮 Q/A 写入 JSONL。返回写入的 episode（不含 embedding），供上层 emit；
    如果根据策略决定不写（policy=skip 且命中更新阈值），返回 None。

    `policy` 可覆盖全局 MEMORY_UPDATE_POLICY，主要给 CLI / 测试用。
    """
    s = get_settings()
    qv = await _embed_normalized(query)
    if qv is None:
        # 嵌入拿不到就别写，下次再说；否则写进去的向量不可用，只会污染 recall
        return None

    # 找 top-1 是否达到"视为同一个问题"的阈值
    eps = _active(_load_all())
    usable = [e for e in eps if _compatible(e) and e.get("embedding")]
    supersedes: Optional[str] = None
    if usable:
        try:
            mat = np.asarray([e["embedding"] for e in usable], dtype="float32")
        except Exception:
            mat = None
        if (
            mat is not None
            and mat.ndim == 2
            and mat.shape[1] == qv.shape[0]
        ):
            scores = mat @ qv
            i = int(np.argmax(scores))
            top_score = float(scores[i])
            if top_score > s.MEMORY_UPDATE_THR:
                eff_policy = (policy or s.MEMORY_UPDATE_POLICY or "supersede").lower()
                if eff_policy == "skip":
                    return None
                if eff_policy == "supersede":
                    supersedes = usable[i].get("id")
                # eff_policy == "append": 保留 supersedes=None，新老两条都在

    backend, model = _current_backend_model()
    rec: dict = {
        "id": f"ep_{uuid.uuid4().hex[:12]}",
        "ts": time.time(),
        "query": query,
        # 截断大 answer 防 JSONL 膨胀
        "answer": (answer or "")[:MEMORY_ANSWER_CAP],
        "paper_ids": list(paper_ids or []),
        "embed_backend": backend,
        "embed_model": model,
        "embedding": qv.tolist(),
    }
    if supersedes:
        rec["supersedes"] = supersedes

    path = _memory_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # append 模式：崩了最多丢这一行，不会破坏已有记录
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return None

    return _without_embedding(rec)
