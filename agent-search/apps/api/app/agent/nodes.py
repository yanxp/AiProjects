"""
LangGraph 节点实现。

每个函数就是一个节点：
- 输入：当前 AgentState
- 输出：要合并回 state 的字段 dict（LangGraph 会按 state.py 里定义的合并策略处理）
- 同时通过 `emit()` 向 SSE 队列推送"思考过程"事件，让前端实时看到 Agent 在做什么。

节点之间通过 `graph.py` 里定义的边连接。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from ..config import get_settings
from ..retrieval import local_rag, openalex
from ..schemas import Evidence, Paper
from .. import llm, memory
from .prompts import (
    PLANNER_SYSTEM,
    READER_SYSTEM,
    REFLECTOR_SYSTEM,
    SYNTHESIZER_SYSTEM,
    reader_user_prompt,
    synthesizer_user_prompt,
)
from .state import AgentState

# Emitter 类型：节点可以往 SSE 通道推送一个事件（type + payload）
Emitter = Callable[[str, dict], None]


def _safe_json_loads(text: str) -> dict:
    """
    LLM 返回的 JSON 偶尔会带 ```json ... ``` 代码块包裹或前后多余文字。
    这里做一个容错解析，尽量挖出第一段合法 JSON。
    """
    text = text.strip()
    # 去掉 Markdown 代码块围栏
    if text.startswith("```"):
        text = text.strip("`")
        # 可能是 ```json\n{...}\n```
        if "\n" in text:
            text = text.split("\n", 1)[1]
        text = text.rstrip("`").strip()
    # 尝试直接解析；失败就尝试截第一个 { ... }
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# ========== Node 0: Memory Recall ==========
async def memory_recall_node(state: AgentState, emit: Emitter) -> dict:
    """
    在 Planner 之前跑一次，把"语义上相近的历史 Q/A"召回出来，
    写入 state["memory_hits"] 供后续节点（主要是 Synthesizer）作为上下文。

    三种情形都不阻塞主流程：
    - MEMORY_MODE=off             → 直接返回空（不触任何 I/O、不发事件）
    - state["memory_refresh"]=True → 用户要求强刷新，跳过 recall（写还是会写）
    - recall 失败/无命中           → memory_hits 写成空数组，上层 emit 便于调试
    """
    s = get_settings()
    mode = (s.MEMORY_MODE or "off").lower()
    if mode == "off":
        return {}

    # --memory-refresh：用户明确要"这次别用历史"，比如知道新论文出来了要刷新
    if state.get("memory_refresh"):
        emit("memory_hit", {"enabled": True, "refreshed": True, "hits": []})
        return {"memory_hits": []}

    try:
        hits = await memory.recall(state["query"])
    except Exception as e:
        # recall 只是上下文增强，任何异常都不应该让整条 pipeline 挂掉
        emit("memory_hit", {"enabled": True, "error": str(e), "hits": []})
        return {"memory_hits": []}

    emit(
        "memory_hit",
        {
            "enabled": True,
            "refreshed": False,
            "hits": [
                {
                    "id": h.get("id"),
                    "score": float(h.get("score", 0.0)),
                    "ts": h.get("ts"),
                    "query": h.get("query"),
                    # 预览截断：CLI / 事件流只要摘要，完整 answer 在 memory_hits 里
                    "answer_preview": (h.get("answer") or "")[:200],
                }
                for h in hits
            ],
        },
    )
    return {"memory_hits": hits}


# ========== Node 1: Planner ==========
async def planner_node(state: AgentState, emit: Emitter) -> dict:
    """
    读取用户 query，让 LLM 生成 3-5 个英文学术检索词。
    返回 {"sub_queries": [...], "step": +1}。
    """
    query = state["query"]
    # 小技巧：同时允许轻量模型替代主模型，省成本
    s = get_settings()
    raw = await llm.chat(
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": query},
        ],
        model=s.LLM_SMALL_MODEL or s.LLM_MODEL,
        temperature=0.1,
        # response_format 只有在 provider 明确支持时才开（LLM_JSON_MODE=true）。
        # Ark 部分 endpoint 不认这个参数，会回 5xx / 假 ModelLoading。
        response_format={"type": "json_object"} if s.LLM_JSON_MODE else None,
    )
    # LLM_JSON_MODE=false 时 LLM 可能返回非 JSON 文本（Ark 等 provider 不强制 JSON 输出），
    # 解析失败就降级成"只用原 query 检索"，不要让整条 pipeline 挂。
    try:
        data = _safe_json_loads(raw)
    except Exception:
        data = {}
    sub_queries = data.get("sub_queries", [])[:5] or [query]  # 兜底：至少用原问题

    emit("plan", {"sub_queries": sub_queries})
    # step 字段在 state 里是"覆盖合并"，这里手动 +1
    return {"sub_queries": sub_queries, "step": state.get("step", 0) + 1}


# ========== Node 2: Retriever ==========
async def retriever_node(state: AgentState, emit: Emitter) -> dict:
    """
    对每个子查询并发调用 OpenAlex；合并去重后按引用数粗排；保留 top_k。
    """
    s = get_settings()
    # 本轮要搜的 query：取 sub_queries 中"最近一次新增"的那批。
    # MVP 简化：直接拿 sub_queries 最后 N=5 条（反思补充的 query 会追加进来）。
    queries = state.get("sub_queries", [])[-5:]

    # 并发调 OpenAlex + 可选的本地 RAG。
    # 本地 RAG 只对"原始用户 query"搜一次（片段是你自己的文档，不需要多 query 展开）。
    tasks: list = [openalex.search(q, top_k=10) for q in queries]
    rag_task_idx = -1
    rag_enabled = s.RAG_ENABLED and local_rag.is_available()
    if rag_enabled:
        rag_task_idx = len(tasks)
        tasks.append(local_rag.search(state["query"], top_k=s.RAG_TOP_K))
    elif s.RAG_ENABLED:
        # 用户开了 RAG 但索引不可用（没建 / 路径不对 / pickle 坏），
        # 打一条事件让上层能看到，别以为没效果是 bug。
        emit("rag", {"enabled": True, "available": False, "hits": []})

    results_per_q = await asyncio.gather(*tasks, return_exceptions=True)

    # 合并 + 去重：以 paper.id（优先 DOI）作为唯一键
    seen: dict[str, Paper] = {}
    # 已有候选也要参与去重，避免反思轮把老论文又加进来
    for p in state.get("candidates", []):
        seen[p.id] = p
    new_papers: list[Paper] = []
    # 拆分 OpenAlex 结果和本地 RAG 结果
    openalex_results = results_per_q[: rag_task_idx] if rag_task_idx >= 0 else results_per_q
    for res in openalex_results:
        if isinstance(res, Exception):
            continue
        for p in res:
            if p.id and p.id not in seen:
                seen[p.id] = p
                new_papers.append(p)

    # 本地片段包装成 Paper，混进同一个候选池，复用后续 Reader/Reflector/Synthesizer
    if rag_task_idx >= 0:
        rag_res = results_per_q[rag_task_idx]
        # 单独 emit 一条 rag 事件，让 CLI / 前端能独立展示"本地检索到了什么"，
        # 不被 OpenAlex 的列表淹没。
        if isinstance(rag_res, Exception):
            emit(
                "rag",
                {
                    "enabled": True,
                    "available": True,
                    "error": str(rag_res),
                    "hits": [],
                },
            )
        else:
            emit(
                "rag",
                {
                    "enabled": True,
                    "available": True,
                    "hits": [
                        {
                            "score": float(h.get("score", 0.0)),
                            "source": h.get("source", ""),
                            # 预览截断：CLI 里列全文太长；完整 snippet 进 Paper.abstract
                            "preview": (h.get("snippet", "") or "")[:200],
                        }
                        for h in rag_res
                    ],
                },
            )
        if not isinstance(rag_res, Exception):
            for h in rag_res:
                source = h.get("source", "local")
                snippet = h.get("snippet", "")
                # 用 source + 内容哈希保证同一片段只进一次
                pid = f"local::{source}::{abs(hash(snippet)) & 0xffffffff:x}"
                if pid in seen:
                    continue
                p = Paper(
                    id=pid,
                    # title 用 source 本身即可；上层通过 source="local" 区分，demo.py
                    # 会自己加 [Local] 前缀，不在 title 里双重标注。
                    title=source,
                    abstract=snippet,       # Reader 只看 abstract，直接当摘要喂
                    authors=[],
                    year=None,
                    venue=None,
                    citations=None,         # 本地片段无引用数，留 None；粗排时会排后面
                    url=f"file://{source}",
                    pdf_url=None,
                    source="local",
                )
                seen[pid] = p
                new_papers.append(p)

    # 粗排：本地 RAG 片段永远排最前（你自己的文档最相关），OpenAlex 论文按引用数降序。
    # 这样 top_k 截断时不会把本地片段切掉 —— 它才是用户开 RAG 的目的。
    def _sort_key(p: Paper) -> tuple[int, int]:
        is_local = 1 if p.source == "local" else 0
        return (is_local, p.citations or 0)

    new_papers.sort(key=_sort_key, reverse=True)
    new_papers = new_papers[: s.AGENT_TOP_K]

    emit(
        "retrieve",
        {
            "queries": queries,
            "papers": [p.model_dump() for p in new_papers],
        },
    )
    return {"candidates": new_papers}


# ========== Node 3: Reader ==========
async def reader_node(state: AgentState, emit: Emitter) -> dict:
    """
    对候选论文逐篇调用 LLM 抽"证据卡"。
    只读标题+摘要（MVP 不下载 PDF，后续可加 GROBID 精读）。
    """
    query = state["query"]
    # 只读这一轮新增的候选（粗略实现：取最后 top_k 条）
    s = get_settings()
    papers: list[Paper] = state.get("candidates", [])[-s.AGENT_TOP_K :]

    async def _extract_one(paper: Paper) -> list[Evidence]:
        # 没摘要就跳过（Reader 依赖摘要做判断）
        if not paper.abstract:
            return []
        raw = await llm.chat(
            messages=[
                {"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": reader_user_prompt(query, paper)},
            ],
            model=s.LLM_SMALL_MODEL or s.LLM_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"} if s.LLM_JSON_MODE else None,
        )
        try:
            data = _safe_json_loads(raw)
        except Exception:
            return []
        if not data.get("relevant"):
            return []
        out = []
        for ev in data.get("evidences", []) or []:
            claim = (ev.get("claim") or "").strip()
            snippet = (ev.get("snippet") or "").strip()
            if claim and snippet:
                out.append(Evidence(paper_id=paper.id, claim=claim, snippet=snippet))
        return out

    # 并发抽取所有论文的证据
    all_evidences = await asyncio.gather(*[_extract_one(p) for p in papers])
    flat: list[Evidence] = [e for lst in all_evidences for e in lst]

    emit("read", {"evidences": [e.model_dump() for e in flat]})
    return {"notes": flat}


# ========== Node 4: Reflector ==========
async def reflector_node(state: AgentState, emit: Emitter) -> dict:
    """
    看当前证据够不够；不够就产出"还缺什么"，下一步回到 Planner 再搜。
    """
    s = get_settings()
    notes = state.get("notes", [])
    # 把证据压缩成紧凑列表丢给 LLM
    brief = "\n".join(f"- {e.claim}" for e in notes) or "(暂无)"
    raw = await llm.chat(
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {
                "role": "user",
                "content": f"用户问题：{state['query']}\n已有证据摘要：\n{brief}",
            },
        ],
        model=s.LLM_SMALL_MODEL or s.LLM_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"} if s.LLM_JSON_MODE else None,
    )
    try:
        data = _safe_json_loads(raw)
    except Exception:
        # 解析失败时默认"足够"，避免死循环
        data = {"sufficient": True, "missing": []}

    missing = data.get("missing") or []
    sufficient = bool(data.get("sufficient"))

    # 若 LLM 判"足够"，就把 missing 显式清空，避免 router 看到非空 missing 误判要回跳。
    # 这让 sufficient 成为是否回跳的唯一真相来源（见 graph.py 的 route_after_reflect）。
    if sufficient:
        missing = []

    emit("reflect", {"sufficient": sufficient, "missing": missing})

    # 把 sufficient 也写入 state，router 据此决定路由；miss 仍保留用于 UI 展示 / 日志
    out: dict = {
        "missing": missing,
        "sufficient": sufficient,
        "step": state.get("step", 0) + 1,
    }
    # 仅在"还不够"且未超上限时，把缺失点作为新 sub_queries 注入，回到 Retriever
    if not sufficient and state.get("step", 0) + 1 < s.AGENT_MAX_STEPS:
        out["sub_queries"] = missing[:3]
    return out


# ========== Node 5: Synthesizer（流式）==========
async def synthesizer_node(state: AgentState, emit: Emitter) -> dict:
    """
    流式生成最终答案。每个 token 都通过 emit("answer_delta", ...) 推给前端。
    """
    query = state["query"]
    notes = state.get("notes", [])

    # 若真的什么证据都没有，给个兜底回复，别让 LLM 空跑
    if not notes:
        msg = "抱歉，没有在可访问的学术数据库里找到与该问题直接相关的文献。"
        emit("answer_delta", {"delta": msg})
        return {"answer": msg}

    # 记忆召回结果作为补充上下文（可为空）；prompts.py 里会写入"独立区块"，
    # 并在 system prompt 里明确"不作为正文引用"。
    memory_hits = state.get("memory_hits") or []

    acc: list[str] = []
    async for delta in llm.stream_chat(
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {
                "role": "user",
                "content": synthesizer_user_prompt(query, notes, memory_hits),
            },
        ],
        temperature=0.3,
    ):
        acc.append(delta)
        emit("answer_delta", {"delta": delta})

    return {"answer": "".join(acc)}


# ========== Node 6: Memory Write ==========
async def memory_write_node(state: AgentState, emit: Emitter) -> dict:
    """
    Synthesizer 跑完后把本轮 Q/A 写回 JSONL 记忆。

    supersedes 判断和 policy 都在 memory.remember 内部处理，这里只负责：
    - MEMORY_MODE=off          → 跳过
    - 没有有效 answer           → 跳过（兜底文案也不值得存）
    - 任何 I/O / 嵌入异常       → 静默降级，不污染正常流程
    """
    s = get_settings()
    if (s.MEMORY_MODE or "off").lower() == "off":
        return {}

    answer = (state.get("answer") or "").strip()
    # Synthesizer 兜底文案，不值得记（记了会把下次真答案盖掉）
    if not answer or answer.startswith("抱歉，没有在可访问的"):
        return {}

    # 只记"主 answer 依赖的"论文 —— 即 Reader 抽到的 notes 对应 paper_ids 去重
    paper_ids: list[str] = []
    seen_ids: set[str] = set()
    for ev in state.get("notes", []) or []:
        pid = getattr(ev, "paper_id", None)
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            paper_ids.append(pid)

    try:
        rec = await memory.remember(state["query"], answer, paper_ids)
    except Exception as e:
        emit("memory_write", {"written": False, "error": str(e)})
        return {}

    if rec is None:
        # memory.remember 内部按 policy 决定"skip"，或嵌入失败等；
        # 这里也打一条事件，让 CLI / 调试能看到为啥没写成
        emit("memory_write", {"written": False, "reason": "skipped_by_policy_or_embed_fail"})
        return {}

    emit(
        "memory_write",
        {
            "written": True,
            "id": rec.get("id"),
            "ts": rec.get("ts"),
            "supersedes": rec.get("supersedes"),
            "paper_count": len(rec.get("paper_ids") or []),
        },
    )
    return {}
