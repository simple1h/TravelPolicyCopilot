from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BackendMode = Literal["strict", "demo_fallback", "test"]


class Settings(BaseSettings):
    app_name: str = "Travel Policy Copilot"
    app_env: str = "dev"
    app_timezone: str = "Asia/Yekaterinburg"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    llm_provider: str = "router.ai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://routerai.ru/api/v1"
    llm_api_key: str = ""
    llm_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_max_tokens: int = Field(default=800, ge=32, le=8192)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_input_cost_per_1m_tokens_usd: float = Field(default=0.15, ge=0.0)
    llm_output_cost_per_1m_tokens_usd: float = Field(default=0.60, ge=0.0)

    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/travelcopilot"
    backend_mode: BackendMode = "strict"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "travel_policy_chunks"
    agent_api_url: str = "http://localhost:8000"
    mock_travel_api_url: str = "http://localhost:8081"
    embedding_provider: str = "router.ai"
    embedding_base_url: str = "https://routerai.ru/api/v1"
    embedding_model: str = "baai/bge-m3"
    embedding_dimension: int = Field(default=1024, ge=32, le=4096)
    embedding_api_key: str = ""
    embedding_timeout_seconds: float = Field(default=8.0, ge=0.5, le=60.0)
    embedding_max_retries: int = Field(default=0, ge=0, le=10)
    embedding_input_cost_per_1m_tokens_usd: float = Field(default=0.02, ge=0.0)
    rag_collection_version: str = "v1"
    rag_corpus_version: str = "v1"
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    request_timeout_seconds: float = Field(default=8.0, ge=1.0, le=60.0)
    max_user_chars: int = Field(default=4000, ge=100, le=20000)
    max_agent_steps: int = Field(default=12, ge=1, le=100)
    rag_chunk_size: int = Field(default=800, ge=200, le=2000)
    rag_chunk_overlap: int = Field(default=150, ge=20, le=400)
    rag_embedding_size: int = Field(default=96, ge=32, le=2048)
    rag_top_k: int = Field(default=12, ge=1, le=100)
    rag_top_n: int = Field(default=5, ge=1, le=20)
    rag_local_index_path: str = "data/rag/policy_index.json"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()


def get_backend_mode() -> BackendMode:
    return settings.backend_mode


def is_strict_mode() -> bool:
    return get_backend_mode() == "strict"


def is_demo_fallback_mode() -> bool:
    return get_backend_mode() == "demo_fallback"
