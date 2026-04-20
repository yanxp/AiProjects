"""
Agent 全局状态。

LangGraph 把状态当作一个 TypedDict 在节点之间传递：
- 每个节点接收 state，返回"要合并到 state 的增量"
- 列表字段用 `Annotated[..., operator.add]` 标注为"累加合并"
- 其他字段默认"覆盖合并"
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..schemas import Evidence, Paper


class AgentState(TypedDict, total=False):
    # 用户原始问题
    query: str

    # Planner 产出的子查询（可能多轮迭代，每轮都往里追加，所以用累加合并）
    sub_queries: Annotated[list[str], operator.add]

    # Retriever 产出的候选论文（累加）
    candidates: Annotated[list[Paper], operator.add]

    # Reader 抽出的证据（累加）
    notes: Annotated[list[Evidence], operator.add]

    # Reflector 认为还缺的信息（每次覆盖）
    missing: list[str]

    # 当前循环步数，用于触发终止条件
    step: int

    # 最终答案（Synthesizer 流式写入后，在 done 时落到这里）
    answer: str
