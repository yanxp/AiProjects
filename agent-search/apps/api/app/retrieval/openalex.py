"""
OpenAlex 检索客户端。

为什么选 OpenAlex：
- 完全免费、无需 API key，覆盖 2.5 亿+ 论文
- 有官方的 "polite pool"：带 mailto 参数请求会获得更稳定的限流上限
- 有结构化字段（authors / concepts / cited_by_count / open_access / doi）

这里只实现 MVP 最必要的搜索接口：
- 输入：查询字符串、top_k
- 输出：List[Paper] （归一成我们自己的 schema）
"""

from __future__ import annotations

import httpx

from ..config import get_settings
from ..schemas import Paper

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """
    OpenAlex 出于版权把摘要存成 {word: [positions...]} 的"倒排索引"。
    需要我们自己拼回一段可读文本。
    """
    if not inverted_index:
        return None
    # 1) 先算出总长度；2) 按位置填词；3) join
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def _to_paper(work: dict) -> Paper:
    """把 OpenAlex 的 work 对象转换成我们内部的 Paper schema。"""
    # authors：OpenAlex 把作者放在 authorships[].author.display_name
    authors = [
        a.get("author", {}).get("display_name")
        for a in work.get("authorships", [])
        if a.get("author", {}).get("display_name")
    ]

    # 优先用 DOI 做稳定 ID；没有 DOI 就退回 OpenAlex 自带的 id URL
    doi = work.get("doi")
    paper_id = doi or work.get("id") or ""

    # Open Access 链接可能出现在 open_access.oa_url 或 primary_location.pdf_url
    oa = work.get("open_access") or {}
    pdf_url = oa.get("oa_url") or (work.get("primary_location") or {}).get("pdf_url")

    # venue 可能在 host_venue（老字段）或 primary_location.source（新字段），都要做 None 保护
    primary_loc = work.get("primary_location") or {}
    source_obj = primary_loc.get("source") or {}
    venue = (work.get("host_venue") or {}).get("display_name") or source_obj.get("display_name")

    return Paper(
        id=paper_id,
        title=work.get("title") or work.get("display_name") or "(untitled)",
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
        authors=authors,
        year=work.get("publication_year"),
        venue=venue,
        citations=work.get("cited_by_count"),
        url=work.get("id"),
        pdf_url=pdf_url,
        source="openalex",
    )


async def search(query: str, top_k: int = 10) -> list[Paper]:
    """
    调用 OpenAlex /works 搜索接口。

    参数说明：
      search：全文检索，OpenAlex 会在标题/摘要/全文上做匹配
      per_page：返回条数
      mailto：礼貌标识，用于 polite pool
      sort：按相关性（默认）即可；如需按引用数可加 sort=cited_by_count:desc
    """
    s = get_settings()
    params = {
        "search": query,
        "per_page": str(top_k),
        "mailto": s.OPENALEX_MAILTO,
    }
    # 超时设 20s：学术搜索 API 偶尔慢，太短容易误判失败
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(OPENALEX_WORKS_URL, params=params)
        r.raise_for_status()
        data = r.json()

    return [_to_paper(w) for w in data.get("results", [])]
