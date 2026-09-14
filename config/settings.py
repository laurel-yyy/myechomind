"""Centralized runtime configuration loaded from environment / .env file.

All modules import the singleton `settings` object from `config` and read
their knobs from it; they never touch os.environ directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM ----
    llm_mode: Literal["mock", "real"] = "mock"
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # ---- Embedding ----
    embedding_mode: Literal["mock", "sentence-transformers"] = "mock"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ---- Redis ----
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # ---- ChromaDB ----
    chroma_mode: Literal["local", "http"] = "http"
    chroma_path: str = "./data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_kb: str = "knowledge_base"
    chroma_collection_episodic: str = "episodic"
    chroma_collection_profile: str = "user_profile"

    # ---- App ----
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ---- Working memory & RAG ----
    working_memory_max: int = 20
    working_memory_ttl: int = 86_400
    conversation_summary_ttl: int = 604_800

    rag_topk: int = 5
    rag_rewrite_n: int = 3
    rag_rerank_enabled: bool = True

    # ---- Tool governance (MCP layer) ----
    tool_cache_ttl: int = 300
    tool_circuit_fail_threshold: int = 5
    tool_circuit_cooldown_sec: int = 60
    tool_timeout_sec: int = 10

    # ---- Monitor ----
    monitor_success_rate_threshold: float = 0.90
    monitor_latency_ms_threshold: int = 3000
    monitor_anomaly_zscore: float = 2.5

    # ---- Paths ----
    skills_dir: str = "./skills"
    skills_reload_interval_sec: int = 30
    skills_max_prompt_chars: int = 12_000
    eval_baseline_path: str = "./data/eval/baseline.json"
    logs_dir: str = "./logs"

    # -------- Derived helpers --------
    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent


settings = Settings()
