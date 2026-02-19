"""Configuration settings for the ingestion pipeline."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Pipeline configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from .env
    )

    # Qdrant settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = None  # Alternative: full URL like http://localhost:6333
    COLLECTION_NAME: str = "genshin_story_v3"

    # Embedding settings (BGE-M3: dense 1024 + sparse lexical)
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 64
    DEVICE: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Hybrid Search
    HYBRID_SEARCH_ENABLED: bool = True
    HYBRID_QUOTA_DENSE: int = 6
    HYBRID_QUOTA_BM25: int = 6
    HYBRID_QUOTA_LEXICAL: int = 6

    # BM25
    BM25_K1: float = 1.2
    BM25_B: float = 0.75
    BM25_IDF_PATH: Path = Path(".cache/vector/bm25_idf.json")
    BM25_USER_DICT_PATH: Path = Path(".cache/vector/jieba_user_dict.txt")

    # Reranker settings
    RERANKER_MODEL: str = "jinaai/jina-reranker-v2-base-multilingual"
    RERANKER_TOP_K: int = 5

    # LLM settings - Three model categories
    # REASONING_MODEL: 主 Agent 推理、工具编排 (需要强推理能力)
    REASONING_MODEL: Optional[str] = None
    # GRADER_MODEL: 答案质量评分、Query 改写 (快速模型)
    GRADER_MODEL: Optional[str] = None
    # DATA_MODEL: 结构化输出 (KG/Event 提取)
    DATA_MODEL: Optional[str] = None

    # API Key settings — multi-provider
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # Backward compatibility aliases (deprecated)
    LLM_MODEL: Optional[str] = None
    GEMINI_MODEL: Optional[str] = None

    # Chunking settings
    MAX_CHUNK_SIZE: int = 1500
    MIN_CHUNK_SIZE: int = 200
    CHUNK_OVERLAP: int = 100

    # Region — controls which config/regions/{region}/ to load
    ACTIVE_REGION: str = "nodkrai"

    # Data paths
    DATA_DIR: Path = Path("Data")

    # Incremental indexing
    VECTOR_TRACKING_FILE: Path = Path(".cache/vector/tracking.json")

    # ── Agent workflow 参数 (env prefix: AGENT_) ─────────────────
    AGENT_MAX_ITERATIONS: int = 2
    AGENT_MAX_TOOL_CALLS: int = 4
    AGENT_MAX_EVIDENCE: int = 6
    AGENT_FAITHFULNESS_THRESHOLD: float = 0.7
    AGENT_COMPLETENESS_THRESHOLD: float = 0.5
    AGENT_RELEVANCE_THRESHOLD: float = 0.7
    AGENT_TOOL_OUTPUT_TRUNCATE: int = 8000
    AGENT_GRADER_TOOL_OUTPUT_LIMIT: int = 6000
    # ── Agent v4 参数 ─────────────────────────────────────────────
    AGENT_V4_MAX_ITERATIONS: int = 6
    AGENT_V4_MAX_TOOL_CALLS: int = 20
    AGENT_V4_MAX_TOOLS_PER_ITER: int = 5  # 单轮最多执行的工具调用数
    AGENT_V4_WALL_TIMEOUT: int = 200  # 秒，单次 graph invoke 的墙钟超时
    AGENT_THINKING_LEVEL: str = "high"  # Gemini 3: "low" | "medium" | "high"
    ABLATION_MODE: str = "full"  # "full" | "vector_only"

    @model_validator(mode="after")
    def resolve_settings(self):
        """Resolve settings with fallbacks and aliases."""
        # Parse QDRANT_URL into host and port if provided
        if self.QDRANT_URL:
            url = self.QDRANT_URL.replace("http://", "").replace("https://", "")
            if ":" in url:
                host, port = url.split(":")
                self.QDRANT_HOST = host
                self.QDRANT_PORT = int(port)
            else:
                self.QDRANT_HOST = url

        # Resolve GOOGLE_API_KEY: prefer GOOGLE_API_KEY, fallback to GEMINI_API_KEY
        if not self.GOOGLE_API_KEY:
            self.GOOGLE_API_KEY = self.GEMINI_API_KEY

        # Resolve REASONING_MODEL (main agent)
        if not self.REASONING_MODEL:
            # Fallback chain: LLM_MODEL -> GEMINI_MODEL -> default
            self.REASONING_MODEL = self.LLM_MODEL or self.GEMINI_MODEL or "gemini-2.5-flash"

        # Resolve GRADER_MODEL (fast model for grading/refinement)
        if not self.GRADER_MODEL:
            self.GRADER_MODEL = "gemini-2.5-flash"

        # Resolve DATA_MODEL (structured output for extraction)
        if not self.DATA_MODEL:
            self.DATA_MODEL = "gemini-2.5-flash"

        # Backward compatibility: LLM_MODEL points to REASONING_MODEL
        if not self.LLM_MODEL:
            self.LLM_MODEL = self.REASONING_MODEL

        return self


# Global settings instance
settings = Settings()
