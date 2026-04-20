"""
全局配置模块。

使用 pydantic-settings 从环境变量 / .env 加载配置，
集中管理，避免在业务代码里到处 os.getenv。
"""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM 配置（OpenAI 兼容协议） ---------------------------------
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"  # LLM 服务地址
    LLM_API_KEY: str = "sk-placeholder"                # API Key
    LLM_MODEL: str = "deepseek-chat"                   # 模型名

    # 轻量模型（用于查询改写、便宜的判断）；不设则复用主模型。
    # 用 Optional[str] 而不是 `str | None`，以兼容 Python 3.9（PEP 604 union 在 3.10+ 才支持）。
    LLM_SMALL_MODEL: Optional[str] = None

    # 嵌入后端 ---------------------------------------------------
    # "api"   → 走 OpenAI 兼容的 embeddings API（用 LLM_EMBED_MODEL）
    # "local" → 本地 sentence-transformers，零外部调用（用 EMBED_LOCAL_MODEL）
    # DeepSeek 不提供 embeddings，Ark 需要单独开 embedding endpoint；不想折腾就用 "local"。
    EMBED_BACKEND: str = "api"

    # API 嵌入模型名（EMBED_BACKEND=api 时生效）
    # OpenAI: text-embedding-3-small / text-embedding-3-large
    # 豆包/Ark: 创建 embedding 类型的 endpoint，填 ep-xxx
    # 自托管：vLLM/TEI 启动 bge-m3 之类
    LLM_EMBED_MODEL: str = "text-embedding-3-small"

    # 本地嵌入模型（EMBED_BACKEND=local 时生效；HuggingFace 路径）
    # - bge-small-zh-v1.5：中英都能用，512 维，~100MB，首次用会下载
    # - bge-m3：多语言+多粒度，~2GB，质量更好但大
    # - paraphrase-multilingual-MiniLM-L12-v2：多语言小模型，~420MB
    EMBED_LOCAL_MODEL: str = "BAAI/bge-small-zh-v1.5"

    # 本地 RAG ----------------------------------------------------
    RAG_ENABLED: bool = False             # 默认关，开后 Retriever 会并发查本地 pickle 索引
    RAG_INDEX_PATH: str = "rag_index.pkl" # 相对 demo.py 所在目录；build_index.py 也写到这里
    RAG_TOP_K: int = 5                    # 每次注入多少条本地片段到候选池

    # 记忆（episodic）---------------------------------------------
    # MEMORY_MODE:
    #   off    → 完全关闭（默认）；不读不写
    #   recall → 每次请求前 recall 相关历史 Q/A 注入给 Synthesizer；
    #            Synthesizer 跑完后 remember 新一条（带 supersedes 更新逻辑）
    # 未来可能加 cache 档（相似度超阈值直接返历史 answer，跳过整个 Agent），MVP 先不做。
    MEMORY_MODE: str = "off"

    # JSONL 存储路径；~ 会被展开。单文件 append-only，崩了最多丢最后一行。
    MEMORY_PATH: str = "~/.agent-search/memory.jsonl"

    # recall 阶段：取 top-K、score 低于 threshold 的丢弃
    MEMORY_RECALL_K: int = 3
    MEMORY_RECALL_THR: float = 0.75

    # remember 阶段："视为同一个问题"的余弦阈值；超过则按 UPDATE_POLICY 处理
    MEMORY_UPDATE_THR: float = 0.92

    # 同一问题的更新策略：
    #   supersede → 新行打 supersedes 指针，recall 只看新的（默认）
    #   append    → 新老两条都保留（便于手动对比 / diff）
    #   skip      → 已有就不写（省空间）
    MEMORY_UPDATE_POLICY: str = "supersede"

    # 超过多少天的 episode 不再参与 recall（依然保留在文件里，仅不召回）。
    # 0 表示不启用 TTL。
    MEMORY_MAX_AGE_DAYS: int = 0

    # 是否在 Planner/Reader/Reflector 这些节点上启用 OpenAI 的
    # `response_format={"type":"json_object"}` 强制 JSON 输出。
    # 默认 False，因为：
    # - OpenAI / DeepSeek 支持得很好 → 可以手动开 true 提升 JSON 稳定性；
    # - 火山方舟 Ark（豆包）部分 endpoint 不认这个参数，会回 5xx / 假 `ModelLoading`；
    # - vLLM / SGLang 自托管按版本参差，开了会踩兼容坑。
    # 关掉时依赖 lenient JSON 解析（从 ```json``` 块 / 首个 {} 段中抓），实测够用。
    LLM_JSON_MODE: bool = False

    # Agent 控制参数 ----------------------------------------------
    AGENT_MAX_STEPS: int = 4   # 最大循环步数，防止死循环
    AGENT_TOP_K: int = 8       # 每次检索保留多少篇
    AGENT_REFLECT: bool = True # 是否启用反思（不够就再搜一轮）

    # OpenAlex ----------------------------------------------------
    # OpenAlex 免费，但带 mailto 查询会进入 "polite pool"，限流更宽松
    OPENALEX_MAILTO: str = "anon@example.com"

    # CORS --------------------------------------------------------
    # 前端开发时是 http://localhost:3000；生产可以改成你自己的域名
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),  # 支持在 api 目录或仓库根目录放 .env
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """带缓存的单例 getter；FastAPI 的依赖注入或手动调用都走这里。"""
    return Settings()
