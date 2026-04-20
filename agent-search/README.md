# Agent Search — 学术论文搜索 Agent (MVP 骨架)

一个面向学术场景的 Agent Search 最小可运行骨架：
- **后端**：FastAPI + LangGraph，调用 OpenAlex 检索论文、用 LLM 综合答案、SSE 流式返回
- **前端**：Next.js 14 + Tailwind，搜索框 → 流式展示 Agent 的每一步（Planner / Retriever / Synthesizer）+ 论文卡片
- **LLM**：默认走 OpenAI 兼容 API（可指向 DeepSeek / 自托管 vLLM / OpenAI），通过环境变量切换

## 流程总览

`Planner → Retriever (RAG + OpenAlex) → Reader → Reflector`，不够就回跳再搜一轮；够了或触达 `AGENT_MAX_STEPS` 走 Synthesizer 流式出答案。

```mermaid
flowchart TD
    U([用户输入 query])
    U --> P

    subgraph Plan["① Planner"]
      P[LLM 拆成 3-5 条英文<br/>sub_queries]
      P -->|emit plan| PE[("plan 事件")]
    end

    P --> R

    subgraph Retrieve["② Retriever"]
      direction TB
      R{RAG_ENABLED ?}
      R -- yes --> RAG[local_rag.search<br/>pickle + numpy 余弦]
      R -- yes --> OA1[openalex.search<br/>并发 N 条 sub_query]
      R -- no --> OA2[openalex.search<br/>并发 N 条 sub_query]
      RAG -->|emit rag| RE[("rag 事件<br/>分数/来源/预览")]
      RAG --> MERGE[合并 + 去重<br/>本地片段置顶]
      OA1 --> MERGE
      OA2 --> MERGE
      MERGE -->|emit retrieve| RVE[("retrieve 事件<br/>候选论文")]
    end

    MERGE --> RD

    subgraph Read["③ Reader"]
      RD[并发对每篇摘要问 LLM<br/>是否相关 + 抽证据句]
      RD -->|emit read| RDE[("read 事件<br/>Evidence 列表")]
    end

    RD --> RF

    subgraph Reflect["④ Reflector"]
      RF[LLM 判 sufficient?<br/>缺什么?]
      RF -->|emit reflect| RFE[("reflect 事件")]
    end

    RF --> DEC{sufficient<br/>或 step 达上限?}
    DEC -- 不够且还有轮次 --> P2[把 missing 作为新 sub_queries]
    P2 --> R

    DEC -- 够了/到上限 --> SY

    subgraph Synth["⑤ Synthesizer"]
      SY[LLM 流式生成<br/>带引用的答案]
      SY -->|emit answer_delta×N| SYE[("answer_delta 事件<br/>逐 token")]
    end

    SY --> OUT([带 References 的最终答案])

    classDef event fill:#fff7d6,stroke:#c7a600,stroke-width:1px,color:#000
    class PE,RE,RVE,RDE,RFE,SYE event
```

| 阶段 | LLM 调用 | 并发 | 代码 |
|---|---|---|---|
| Planner | 1 | — | `nodes.py::planner_node` |
| Retriever | 0 | `sub_queries` 数 + 本地 RAG 并发 | `nodes.py::retriever_node` |
| Reader | N（候选数） | N 并发 | `nodes.py::reader_node` |
| Reflector | 1 | — | `nodes.py::reflector_node` |
| Synthesizer | 1（流式） | — | `nodes.py::synthesizer_node` |
| 回跳条件 | — | `not sufficient and step < MAX_STEPS` | `agent/graph.py::route_after_reflect` |

单轮无反思 ≈ `N + 3` 次 LLM；反思一次再加 `N + 1`；默认 `AGENT_MAX_STEPS=4`。

## 快速开始

### 1. 准备 LLM 凭据
复制 `.env.example` → `.env`，填入：
```
LLM_BASE_URL=https://api.deepseek.com/v1       # 或 http://vllm:8000/v1  或  https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat                        # 或 Qwen/Qwen2.5-72B-Instruct-AWQ / gpt-4o-mini
```

### 2. 启动
```bash
docker compose up --build
```

- 后端：http://localhost:8000 （docs 在 `/docs`）
- 前端：http://localhost:3000

### 3. 本地开发（不走 docker）
```bash
# 后端
cd apps/api
uv sync   # 或 pip install -e .
uvicorn app.main:app --reload --port 8000

# 前端
cd apps/web
pnpm install
pnpm dev
```

## 目录
```
apps/
  api/          FastAPI + LangGraph
  web/          Next.js 14 前端
scripts/        开发脚本
docker-compose.yml
```

## 本地 RAG（可选，最小版）

在 OpenAlex 旁边加一条"本地文档检索"支路，让 Agent 同时能搜你自己的笔记 / PDF / markdown：

```bash
# 1. 把文档丢进 docs/ （支持 md / txt，加 --pdf 也能吃 PDF）
mkdir docs && cp ~/notes/*.md docs/

# 2. 建索引（写出 rag_index.pkl）
python scripts/build_index.py ./docs

# 3. 打开 RAG
export RAG_ENABLED=true

# 4. 选嵌入后端：
#  (A) 走 API：OpenAI / Ark embedding endpoint / vLLM 起的 bge-m3
export EMBED_BACKEND=api
export LLM_EMBED_MODEL=text-embedding-3-small   # 或 Ark 的 ep-xxx
#  (B) 本地 sentence-transformers（零外部调用，首次会下权重到 ~/.cache/huggingface/）
pip install sentence-transformers
export EMBED_BACKEND=local
export EMBED_LOCAL_MODEL=BAAI/bge-small-zh-v1.5   # 中英双语 ~100MB

# 5. 跑 demo；Retriever 会把本地片段和 OpenAlex 结果混进同一个候选池
python demo.py "..."
```

跑起来会看到一个独立的 RAG 分区，和 OpenAlex 的候选列表分开展示：

```
▌ Planner — 拆分为学术检索词
  1. few-shot object detection meta-learning
  ...

▌ RAG — 本地文档检索
  1. score=0.782  /path/to/notes/meta-rcnn.md
     Meta R-CNN introduces a class-attentive vector to re-weight ROI features ...
  2. score=0.691  /path/to/notes/fsod-survey.md
     ...

▌ Retriever — OpenAlex 检索结果
  - [Local] [?] (? cites) /path/to/notes/meta-rcnn.md
  - [2023] (142 cites) Meta R-CNN: Towards General Solver for ...
  ...
```

> **DeepSeek 用户注意**：DeepSeek 不提供 embeddings API。把 `EMBED_BACKEND=local` 最省事。

> **Ark/豆包 用户注意**：chat 和 embedding 需要**两个独立的 endpoint**（在控制台创建 embedding 类型 endpoint），填在 `LLM_EMBED_MODEL`。不想开就切 local。

> **切换后端必须重建索引**：api 和 local 的嵌入维度不同（典型 1536 vs 512），旧 pickle shape 对不上会静默降级成空命中。重跑 `python scripts/build_index.py ./docs` 就行。

实现细节：`pickle + numpy` 点积余弦，无额外服务和依赖。建索引时片段被 L2 归一化，运行时 query 编码后直接 `vectors @ qv` 取 top-k。NaN/Inf 在建库和加载两处都会被 `np.nan_to_num` 清洗，避免 matmul 触发 `RuntimeWarning`。见 [`apps/api/app/retrieval/local_rag.py`](apps/api/app/retrieval/local_rag.py) 和 [`scripts/build_index.py`](scripts/build_index.py)。

## CLI 事件一览

每个阶段都会 `emit(event_type, payload)`，CLI（`demo.py`）按类型分支渲染，未来接 SSE 前端也走一样的协议：

| event_type | 时机 | payload 关键字段 |
|---|---|---|
| `plan` | Planner 出 sub_queries 后 | `sub_queries: list[str]` |
| `rag` | Retriever 启用 RAG 时（含索引不可用 / 异常分支） | `enabled, available, hits[{score,source,preview}], error?` |
| `retrieve` | Retriever 合并去重粗排后 | `queries, papers[...]`（包含 `source="local"` 标记） |
| `read` | Reader 完成一轮 | `evidences[{paper_id, claim, snippet}]` |
| `reflect` | Reflector 完成一轮 | `sufficient: bool, missing: list[str]` |
| `answer_delta` | Synthesizer 流式输出每个 token | `delta: str` |
| `error` | 任意节点 catch 到异常 | `message: str` |

## 下一步扩展
- 接入 Semantic Scholar / arXiv / Unpaywall
- 加入 Qdrant + bge-m3 做语义检索与 Rerank
- GROBID/MinerU 解析 PDF，做单篇追问
- Langfuse 接入做可观测性
- 评估集 + RAGAS 回归
