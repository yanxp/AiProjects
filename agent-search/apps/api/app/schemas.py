"""
共享数据结构（Pydantic 模型）。

同时作为：
- API 请求/响应的 schema
- Agent 内部状态的数据类型
- 给前端 TS 生成类型的来源（若后续加 openapi-codegen）
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """论文卡片的统一表示。不同数据源（OpenAlex/arXiv/S2）都归一到这个结构。"""

    id: str                              # 统一 ID，优先用 DOI，其次用数据源自带 ID
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None             # 会议/期刊名
    citations: int | None = None         # 引用数
    url: str | None = None               # 原文链接（landing page）
    pdf_url: str | None = None           # 可直接下载的 PDF（如 arXiv / OA）
    source: str = "openalex"             # 数据来源标识


class Evidence(BaseModel):
    """Reader 节点从论文里抽出的"证据片段"，用于 Synthesizer 合成答案时引用。"""

    paper_id: str
    snippet: str          # 关键摘录
    claim: str            # 这段证据支持的结论（由 LLM 概括）


class SearchRequest(BaseModel):
    """/search 接口请求体。"""

    query: str
    top_k: int | None = None   # 可选覆盖默认 top_k


class AgentEvent(BaseModel):
    """SSE 流式推送的事件统一结构。前端据此渲染 "思考过程"。"""

    type: str        # plan / retrieve / read / reflect / answer_delta / done / error
    payload: dict    # 具体负载（比如检索到的论文列表、token 增量等）
