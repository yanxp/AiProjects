"""
共享数据结构（Pydantic 模型）。

同时作为：
- API 请求/响应的 schema
- Agent 内部状态的数据类型
- 给前端 TS 生成类型的来源（若后续加 openapi-codegen）
"""

# 注意：为兼容 Python 3.9（用户环境），这里用 typing.Optional/List 代替 PEP 604 的 `X | None` 语法。
# Pydantic v2 会在类构造时解析注解，`str | None` 在 3.9 解释器阶段就会抛 TypeError。
from typing import List, Optional

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """论文卡片的统一表示。不同数据源（OpenAlex/arXiv/S2）都归一到这个结构。"""

    id: str                                  # 统一 ID，优先用 DOI，其次用数据源自带 ID
    title: str
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None              # 会议/期刊名
    citations: Optional[int] = None          # 引用数
    url: Optional[str] = None                # 原文链接（landing page）
    pdf_url: Optional[str] = None            # 可直接下载的 PDF（如 arXiv / OA）
    source: str = "openalex"                 # 数据来源标识


class Evidence(BaseModel):
    """Reader 节点从论文里抽出的"证据片段"，用于 Synthesizer 合成答案时引用。"""

    paper_id: str
    snippet: str          # 关键摘录
    claim: str            # 这段证据支持的结论（由 LLM 概括）


class SearchRequest(BaseModel):
    """/search 接口请求体。"""

    query: str
    top_k: Optional[int] = None   # 可选覆盖默认 top_k


class AgentEvent(BaseModel):
    """SSE 流式推送的事件统一结构。前端据此渲染 "思考过程"。"""

    type: str        # plan / retrieve / read / reflect / answer_delta / done / error
    payload: dict    # 具体负载（比如检索到的论文列表、token 增量等）
