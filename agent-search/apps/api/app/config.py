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
