# Agent Search — 学术论文搜索 Agent (MVP 骨架)

一个面向学术场景的 Agent Search 最小可运行骨架：
- **后端**：FastAPI + LangGraph，调用 OpenAlex 检索论文、用 LLM 综合答案、SSE 流式返回
- **前端**：Next.js 14 + Tailwind，搜索框 → 流式展示 Agent 的每一步（Planner / Retriever / Synthesizer）+ 论文卡片
- **LLM**：默认走 OpenAI 兼容 API（可指向 DeepSeek / 自托管 vLLM / OpenAI），通过环境变量切换

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

## 下一步扩展
- 接入 Semantic Scholar / arXiv / Unpaywall
- 加入 Qdrant + bge-m3 做语义检索与 Rerank
- GROBID/MinerU 解析 PDF，做单篇追问
- Langfuse 接入做可观测性
- 评估集 + RAGAS 回归
