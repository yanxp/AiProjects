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

    # 注册节点。注意：LangGraph 通过 inspect 判断节点是同步还是 async。
    # 若用同步 lambda 包 async 函数（`lambda s: planner_node(s, emit)`），
    # LangGraph 只会把 lambda 的返回值当作 dict —— 但实际拿到的是 coroutine，
    # 随即抛 InvalidUpdateError: Expected dict, got <coroutine object ...>。
    # 所以这里显式用 async 闭包把 emit 绑上去，保持每个节点依然是 async def。
    async def _planner(s: AgentState) -> dict:
        return await planner_node(s, emit)

    async def _retriever(s: AgentState) -> dict:
        return await retriever_node(s, emit)

    async def _reader(s: AgentState) -> dict:
        return await reader_node(s, emit)

    async def _reflector(s: AgentState) -> dict:
        return await reflector_node(s, emit)

    async def _synthesizer(s: AgentState) -> dict:
        return await synthesizer_node(s, emit)

    g.add_node("planner", _planner)
    g.add_node("retriever", _retriever)
    g.add_node("reader", _reader)
    g.add_node("reflector", _reflector)
    g.add_node("synthesizer", _synthesizer)

    # 线性边
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "reader")
    g.add_edge("reader", "reflector")

    # 条件边：反思后要么回到 retriever，要么进入 synthesizer
    def route_after_reflect(state: AgentState) -> str:
        """
        唯一真相来源是 `sufficient`（reflector_node 写入）。
        早期版本错误地用 `missing` 非空作为回跳条件，遇到 LLM 返回
        {"sufficient": true, "missing": [...]} 这类自相矛盾的输出时，
        router 会回跳但 reflector 未注入新 sub_queries，导致 retriever
        用相同 query 反复检索、候选全部被去重、证据被 operator.add 累加，
        直到 step 撞上 AGENT_MAX_STEPS 才退出 —— 纯属浪费 token。
        现在 router 只看 sufficient，reflector 在"足够"时也会清空 missing。
        """
        s = get_settings()
        # 缺省 True：若 reflector 从未运行或解析失败，按"足够"处理，直接出答案
        sufficient = state.get("sufficient", True)
        step = state.get("step", 0)
        if not sufficient and step < s.AGENT_MAX_STEPS and s.AGENT_REFLECT:
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
