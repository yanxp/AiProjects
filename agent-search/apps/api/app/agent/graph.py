"""
LangGraph 状态机装配。

流程：
    START → planner → retriever → reader → reflector → (loop or synthesizer) → END

反思路由：
- 若 reflector 判断"不够"且 step < MAX_STEPS，回到 retriever 继续搜。
- 否则走 synthesizer，生成最终答案。

为什么不用现成的 `create_react_agent`？
- 我们需要精细控制"先 plan、再搜、再读、再反思"这种学术场景特有的多阶段管线；
- ReAct 通用循环在论文搜索里容易反复调工具、浪费 token、且很难保证"引用溯源"。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Callable

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from .nodes import (
    planner_node,
    reader_node,
    reflector_node,
    retriever_node,
    synthesizer_node,
)
from .state import AgentState


def build_graph(emit: Callable[[str, dict], None]):
    """
    构建并编译 LangGraph。
    `emit` 是 SSE 事件推送器，会被闭包到每个节点里。
    """
    g = StateGraph(AgentState)

    # 注册节点：每个节点都是 async def，LangGraph 会自动识别
    g.add_node("planner", lambda s: planner_node(s, emit))
    g.add_node("retriever", lambda s: retriever_node(s, emit))
    g.add_node("reader", lambda s: reader_node(s, emit))
    g.add_node("reflector", lambda s: reflector_node(s, emit))
    g.add_node("synthesizer", lambda s: synthesizer_node(s, emit))

    # 线性边
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "reader")
    g.add_edge("reader", "reflector")

    # 条件边：反思后要么回到 retriever，要么进入 synthesizer
    def route_after_reflect(state: AgentState) -> str:
        s = get_settings()
        missing = state.get("missing") or []
        step = state.get("step", 0)
        # reflector 把新的缺失点写进 sub_queries 时，说明想再搜一轮
        if missing and step < s.AGENT_MAX_STEPS and s.AGENT_REFLECT:
            return "retriever"
        return "synthesizer"

    g.add_conditional_edges(
        "reflector",
        route_after_reflect,
        {"retriever": "retriever", "synthesizer": "synthesizer"},
    )
    g.add_edge("synthesizer", END)

    return g.compile()


async def run_agent(query: str, emit: Callable[[str, dict], None]) -> AgentState:
    """
    运行一次 agent。
    - emit 由上层（API 路由）提供，用于往 SSE 通道推事件。
    - 返回最终 state（含 answer、notes、candidates），上层决定如何发 "done" 事件。
    """
    graph = build_graph(emit)
    final_state: AgentState = await graph.ainvoke({"query": query, "step": 0})
    return final_state
