"""
离线建 RAG 索引脚本。

用法：
    # 准备好环境变量（和 demo.py 共用同一份 .env）：
    export LLM_BASE_URL=...
    export LLM_API_KEY=...
    export LLM_EMBED_MODEL=text-embedding-3-small    # 或 ep-xxx (Ark)
    export RAG_INDEX_PATH=rag_index.pkl              # 产物路径

    # 指向要索引的文档目录，跑脚本：
    python scripts/build_index.py ./docs

    # 默认递归扫 *.md / *.txt；加 --pdf 也收 *.pdf（需要 pypdf）
    python scripts/build_index.py ./docs --pdf

输出：单个 pickle 文件 (chunks, sources, vectors, model)，供运行时 local_rag.search 直接加载。

设计选择：
- 纯 Python，无向量库依赖
- 切块策略最简：按字符数固定 500 + overlap 50；对中文和英文都能 work
- 嵌入分批调（默认 64 条一批），走和 Agent 一样的 llm.embed（带退避重试）
"""

from __future__ import annotations

import argparse
import asyncio
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# 支持从仓库根或 agent-search/ 目录直接运行
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app import embeddings  # noqa: E402
from app.config import get_settings  # noqa: E402


def _iter_files(root: Path, include_pdf: bool) -> Iterable[Path]:
    """递归扫描支持的文本文件。默认 md/txt；可选 pdf。"""
    exts = {".md", ".txt", ".markdown"}
    if include_pdf:
        exts.add(".pdf")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _read_text(path: Path) -> str:
    """读取单个文件为纯文本。PDF 用 pypdf 简单抽文字，抽不到就返空。"""
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"[warn] pypdf 未安装，跳过 PDF: {path}", file=sys.stderr)
            return ""
        try:
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            print(f"[warn] PDF 解析失败 {path}: {e}", file=sys.stderr)
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[warn] 读文件失败 {path}: {e}", file=sys.stderr)
        return ""


def _chunk(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """
    固定字符切块，带 overlap 缓冲跨段语义。
    MVP 够用，要更精细可改成按段落 / 按标题切。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    out: list[str] = []
    step = max(size - overlap, 1)
    for i in range(0, len(text), step):
        piece = text[i : i + size].strip()
        if piece:
            out.append(piece)
    return out


async def _embed_all(chunks: list[str], batch: int = 64) -> np.ndarray:
    """分批调 embedding（API 或本地），拼成 (N, D) numpy 矩阵并 L2 归一化。"""
    vectors: list[list[float]] = []
    total = len(chunks)
    for i in range(0, total, batch):
        part = chunks[i : i + batch]
        print(f"[embed] {min(i + batch, total)}/{total}", flush=True)
        vs = await embeddings.embed(part)
        vectors.extend(vs)
    arr = np.asarray(vectors, dtype="float32")
    # 清洗：有些 provider 偶尔会返 NaN / Inf（空串、极短文本、内部抖动），
    # 留着不处理到运行时 matmul 会报 RuntimeWarning。一次性替换成 0。
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).any(axis=1).sum())
        print(f"[warn] {n_bad}/{total} 条向量含 NaN/Inf，已替换为 0", file=sys.stderr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # L2 归一化，运行时可直接点积 = 余弦相似度
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # 避免除零
    return arr / norms


async def _main_async(src: Path, include_pdf: bool) -> None:
    s = get_settings()
    files = list(_iter_files(src, include_pdf=include_pdf))
    if not files:
        print(f"[err] 目录 {src} 下没找到 md/txt/pdf 文件", file=sys.stderr)
        sys.exit(1)

    print(f"[scan] 扫到 {len(files)} 个文件")

    chunks: list[str] = []
    sources: list[str] = []
    for f in files:
        text = _read_text(f)
        for c in _chunk(text):
            chunks.append(c)
            sources.append(str(f))

    if not chunks:
        print("[err] 切块后没有可嵌入的文本", file=sys.stderr)
        sys.exit(1)

    # 统一大小写，和 embeddings.embed() 里的 .lower() 对齐；否则 EMBED_BACKEND=Local
    # 会路由到本地后端，但这里却把 API 模型名写进索引元数据，造成 model/backend 不匹配。
    backend = (s.EMBED_BACKEND or "api").lower()
    model_label = s.EMBED_LOCAL_MODEL if backend == "local" else s.LLM_EMBED_MODEL
    print(
        f"[chunk] 共 {len(chunks)} 个片段，开始嵌入（backend={backend}, model={model_label}）"
    )
    matrix = await _embed_all(chunks)

    out_path = Path(s.RAG_INDEX_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": chunks,
        "sources": sources,
        "vectors": matrix,
        "model": model_label,
        "backend": backend,
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[ok] 索引写入 {out_path.resolve()}  shape={matrix.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal RAG index (pickle + numpy).")
    parser.add_argument("source", help="要索引的目录（递归扫 md/txt，加 --pdf 也收 pdf）")
    parser.add_argument("--pdf", action="store_true", help="也索引 PDF（需要安装 pypdf）")
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists() or not src.is_dir():
        print(f"[err] 目录不存在：{src}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_main_async(src, include_pdf=args.pdf))


if __name__ == "__main__":
    main()
