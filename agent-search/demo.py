"""
Agent Search — 纯 Python CLI Demo
=================================

这是一个最小可运行的命令行版本，用来快速验证"学术 Agent Search"的端到端闭环：
    用户问题  →  LLM 拆解成学术检索词  →  OpenAlex 检索  →  LLM 阅读摘要抽证据
             →  LLM 判断是否够，不够再搜一轮  →  LLM 流式综合答案（带引用）

没有 Web UI、没有数据库、没有向量库、没有 docker —— 就跑这一个脚本。
跑通之后，再逐步接入 FastAPI + Next.js 前端 / Qdrant 向量检索 / PDF 全文精读。

运行方式
--------
1. 进入后端目录装依赖（推荐虚拟环境）：
       cd apps/api
       python -m venv .venv && source .venv/bin/activate
       pip install -r requirements.txt

2. 回到仓库根目录，准备 LLM 凭据（DeepSeek / OpenAI / 自托管 vLLM 任选其一）：
       export LLM_BASE_URL=https://api.deepseek.com/v1
       export LLM_API_KEY=sk-xxxxxxxx
       export LLM_MODEL=deepseek-chat
       export OPENALEX_MAILTO=you@example.com   # 可选，建议填

3. 跑 demo：
       python demo.py "few-shot object detection with meta-learning"
       # 或不带参数，进入交互式提问

输出会实时打印 Agent 每一步（Planner / Retrieve / Read / Reflect / Answer），
最后给出带 [n] 引用标号的答案和编号对应的论文列表。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 让这个脚本不需要 `pip install -e .` 也能 import 到 apps/api/app 模块。
# 原理：把 apps/api 加到 sys.path，这样 `from app.agent.graph import run_agent`
# 就能被解析到本仓库里的代码。
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

# 尝试自动加载仓库根目录的 .env（若用户习惯把 key 写在那里）。
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    # python-dotenv 不是硬依赖，没装也无所谓 —— 用户用 export 也行。
    pass

from app.agent.graph import run_agent  # noqa: E402  # 上面动态加了 sys.path
from app.schemas import Paper  # noqa: E402
from datetime import datetime  # noqa: E402


# ---------------------------------------------------------------------------
# 一点点 ANSI 颜色，让命令行输出更好读。
# 不想依赖 rich / colorama，直接用最朴素的转义序列，兼容绝大多数终端。
# ---------------------------------------------------------------------------
class C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"


def _header(text: str, color: str = C.CYAN) -> None:
    """打印一条醒目的分节标题。"""
    print(f"\n{color}{C.BOLD}▌ {text}{C.RESET}")


def _dim(text: str) -> str:
    return f"{C.DIM}{text}{C.RESET}"


# ---------------------------------------------------------------------------
# SSE "事件" 在 CLI 里直接打到 stdout。
# Agent 每个节点都会调 emit(type, payload)，我们按 type 分支格式化。
# ---------------------------------------------------------------------------
# 用一个 list 顺序收集 papers，后面打印引用列表时和答案里 [n] 对得上。
_papers_seen: dict[str, Paper] = {}
_papers_order: list[str] = []


def _register_paper(p: Paper) -> None:
    if p.id and p.id not in _papers_seen:
        _papers_seen[p.id] = p
        _papers_order.append(p.id)


def _ts_fmt(ts) -> str:
    """Unix 秒 → 本地时间 'YYYY-MM-DD HH:MM'；拿不到就返 '?'"""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def emit(event_type: str, payload: dict) -> None:
    """CLI 版的 emitter：把 agent 内部事件打到终端。"""
    if event_type == "memory_hit":
        _header("Memory — 相关历史问答", C.CYAN)
        if payload.get("refreshed"):
            print(_dim("  --memory-refresh: 本次跳过历史召回"))
            return
        err = payload.get("error")
        if err:
            print(f"{C.RED}  error:{C.RESET} {err}")
            return
        hits = payload.get("hits", [])
        if not hits:
            print(_dim("  （没有语义上相近的历史 Q/A）"))
            return
        for i, h in enumerate(hits, 1):
            score = h.get("score", 0.0)
            ts = _ts_fmt(h.get("ts"))
            q = (h.get("query") or "").strip()
            preview = (h.get("answer_preview") or "").replace("\n", " ")
            print(f"  {i}. score={score:.3f}  {_dim(ts)}")
            print(f"     Q: {q}")
            print(f"     A: {preview}")

    elif event_type == "memory_write":
        _header("Memory — 写入新记忆", C.CYAN)
        if not payload.get("written"):
            reason = payload.get("reason") or payload.get("error") or "skipped"
            print(_dim(f"  未写入（{reason}）"))
            return
        mid = payload.get("id") or "?"
        sup = payload.get("supersedes")
        pc = payload.get("paper_count", 0)
        tail = f" supersedes={sup}" if sup else " (new)"
        print(f"  id={mid}{tail}  refs={pc}")

    elif event_type == "plan":
        _header("Planner — 拆分为学术检索词")
        for i, q in enumerate(payload.get("sub_queries", []), 1):
            print(f"  {i}. {q}")

    elif event_type == "rag":
        # 单独列一块"本地 RAG"，和 OpenAlex 结果区分开
        _header("RAG — 本地文档检索", C.CYAN)
        if not payload.get("available", False):
            print(_dim("  索引不可用（没建 / 路径不对 / pickle 坏）"))
            return
        err = payload.get("error")
        if err:
            print(f"{C.RED}  error:{C.RESET} {err}")
            return
        hits = payload.get("hits", [])
        if not hits:
            print(_dim("  （本地没检索到相关片段）"))
            return
        for i, h in enumerate(hits, 1):
            score = h.get("score", 0.0)
            src = h.get("source", "")
            preview = (h.get("preview", "") or "").replace("\n", " ")
            print(f"  {i}. score={score:.3f}  {_dim(src)}")
            print(f"     {preview}")

    elif event_type == "retrieve":
        _header("Retriever — OpenAlex 检索结果", C.MAGENTA)
        qs = payload.get("queries", [])
        print(_dim(f"  本轮查询：{qs}"))
        papers_raw = payload.get("papers", [])
        for w in papers_raw:
            p = Paper(**w)
            _register_paper(p)
            cite = p.citations if p.citations is not None else "?"
            year = p.year or "?"
            title = (p.title or "")[:100]
            tag = "[Local] " if p.source == "local" else ""
            print(f"  - {tag}[{year}] ({cite} cites) {title}")

    elif event_type == "read":
        _header("Reader — 抽取证据", C.YELLOW)
        evs = payload.get("evidences", [])
        if not evs:
            print(_dim("  （本轮没抽到可用证据）"))
        for ev in evs:
            claim = ev.get("claim") or ""
            pid = ev.get("paper_id") or ""
            print(f"  • {claim}   {_dim(f'[{pid}]')}")

    elif event_type == "reflect":
        _header("Reflector — 证据是否充足？", C.MAGENTA)
        ok = payload.get("sufficient")
        print(f"  sufficient = {ok}")
        missing = payload.get("missing") or []
        if missing:
            print(_dim(f"  缺失点（会触发下一轮检索）：{missing}"))

    elif event_type == "answer_delta":
        # 流式打印最终答案的每个 token。第一次进入时先写个标题。
        if not getattr(emit, "_answer_started", False):
            _header("Synthesizer — 最终答案（流式）", C.GREEN)
            emit._answer_started = True  # type: ignore[attr-defined]
        sys.stdout.write(payload.get("delta", ""))
        sys.stdout.flush()

    elif event_type == "error":
        print(f"{C.RED}[ERROR]{C.RESET} {payload}")


def _print_references() -> None:
    """答案结束后，打印编号引用列表。"""
    if not _papers_order:
        return
    _header("References", C.CYAN)
    # Synthesizer 的提示词要求 [n] 从 1 起始、对应"证据"顺序；而证据顺序与 paper 顺序
    # 不必严格一致。这里简单按论文出现顺序列出，至少让用户能点到原文。
    for i, pid in enumerate(_papers_order, 1):
        p = _papers_seen[pid]
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        line1 = f"  [{i}] {p.title}"
        line2 = _dim(f"      {authors} · {p.year or '?'} · {p.venue or ''}")
        print(line1)
        print(line2)
        if p.url:
            print(_dim(f"      {p.url}"))


# ---------------------------------------------------------------------------
# 主入口：解析参数 → 跑 agent → 打印引用。
# ---------------------------------------------------------------------------
async def _run(query: str, memory_refresh: bool = False) -> None:
    # 必要环境检查：没配 LLM_API_KEY 时提前报错,比到调用时才 401 更友好。
    key = os.getenv("LLM_API_KEY", "")
    if not key or "placeholder" in key or key.endswith("your-key"):
        print(
            f"{C.RED}缺少 LLM_API_KEY。请 export LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 后重试。{C.RESET}"
        )
        sys.exit(2)

    print(f"{C.BOLD}Query:{C.RESET} {query}")
    try:
        final_state = await run_agent(query, emit, memory_refresh=memory_refresh)
    except Exception as e:
        # agent 里任何未捕获异常都在这里兜底，打印清晰的错误信息
        print(f"\n{C.RED}Agent 运行失败：{type(e).__name__}: {e}{C.RESET}")
        raise

    # answer_delta 是流式推的，最后补一个换行让终端整齐
    print()
    _print_references()

    # 打印一些运行概要，便于自查
    notes = final_state.get("notes", [])
    print(
        _dim(
            f"\n[done] steps={final_state.get('step', 0)} "
            f"papers={len(_papers_seen)} evidences={len(notes)}"
        )
    )


def main() -> None:
    # 命令行用法：
    #   python demo.py "your query"
    #   python demo.py --memory-refresh "your query"     # 强制刷新记忆
    #   python demo.py                                   # 交互模式
    # 交互模式下输入 "!refresh <query>" 相当于加了 --memory-refresh
    args = sys.argv[1:]
    one_shot_refresh = False
    if args and args[0] == "--memory-refresh":
        one_shot_refresh = True
        args = args[1:]

    if args:
        query = " ".join(args).strip()
        asyncio.run(_run(query, memory_refresh=one_shot_refresh))
        return

    print(
        "进入交互模式，输入空行或 Ctrl-D 退出。"
        "\n（在问题前加 '!refresh ' 会强制刷新记忆）"
    )
    while True:
        try:
            raw = input(f"\n{C.BOLD}?> {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            break
        refresh = False
        if raw.startswith("!refresh "):
            refresh = True
            raw = raw[len("!refresh ") :].strip()
            if not raw:
                continue
        # 每次提问都清空引用状态，避免跨问题串号
        _papers_seen.clear()
        _papers_order.clear()
        if hasattr(emit, "_answer_started"):
            delattr(emit, "_answer_started")
        asyncio.run(_run(raw, memory_refresh=refresh))


if __name__ == "__main__":
    main()
