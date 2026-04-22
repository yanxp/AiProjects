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

    # Reflector 认为还缺的信息（每次覆盖，仅用于展示 / 日志）
    missing: list[str]

    # Reflector 的显式判定：证据是否充足。
    # Router 据此决定是再搜一轮还是进入 Synthesizer，
    # 让"是否足够"这件事有唯一可信来源，避免 router 和 reflector 用不同条件而产生
    # 死循环（例如 LLM 返回 {"sufficient": true, "missing": [...]} 这种内部矛盾）。
    sufficient: bool

    # 当前循环步数，用于触发终止条件
    step: int

    # 最终答案（Synthesizer 流式写入后，在 done 时落到这里）
    answer: str

    # Memory recall 产出（memory.Episode 的 dict 形式，score 降序）。
    # Synthesizer 把它作为"相关历史问答"上下文注入 prompt。
    memory_hits: list[dict]

    # 用户显式请求强刷新（--memory-refresh）。memory_recall_node 读到后
    # 直接跳过 recall（memory_hits 保持空）；memory_write_node 仍会写本轮，
    # 按 MEMORY_UPDATE_POLICY 触发 supersede。用来绕开"查过的就当历史"。
    memory_refresh: bool
