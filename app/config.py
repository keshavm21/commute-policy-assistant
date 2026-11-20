"""All tunable settings in one place, read from environment variables (or .env)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Paths ---
    policies_dir: Path = PROJECT_ROOT / "data" / "policies"
    users_file: Path = PROJECT_ROOT / "data" / "users.json"
    var_dir: Path = PROJECT_ROOT / "var"  # runtime state: Chroma files + SQLite registry

    # --- JWT (mock issuer and verifier share one HS256 secret) ---
    jwt_secret: str = "dev-only-insecure-secret-change-me"
    jwt_issuer: str = "commute-mock-issuer"
    jwt_audience: str = "commute-policy-assistant"
    jwt_ttl_minutes: int = 60

    # --- LLM ---
    # "auto": use the real LLM if an API key is set, otherwise the offline fake.
    llm_provider: Literal["auto", "openai_compatible", "fake"] = "auto"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    # Chosen by measurement (eval/REPORT.md): same accuracy as gpt-oss-120b on the eval set,
    # about half the latency, and no deadline timeouts in 96 calls.
    llm_model: str = "qwen/qwen3.8-27b"
    llm_max_tokens: int = 300
    # Only for models that accept it (e.g. gpt-oss: "low" keeps hidden thinking short,
    # which keeps latency down and leaves room in max_tokens). Empty = don't send it.
    llm_reasoning_effort: str = ""

    # --- Budgets from the brief ---
    max_llm_calls: int = 2  # per question
    max_chunks: int = 3  # cited/sent to the LLM per question
    request_deadline_s: float = 2.7  # whole /ask pipeline; p95 target is 3 s
    retrieval_timeout_s: float = 1.0
    min_time_for_retry_s: float = 1.0  # only retry the LLM if this much time is left

    # --- Retrieval ---
    # Vector is the default: hybrid (BM25 + RRF) was measured and lowered hit@1 on the
    # eval set (see eval/REPORT.md), so it stays available but off.
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    # Cosine distance (0 = identical). If even the best chunk is farther than this,
    # we answer "not covered" without calling the LLM. It is deliberately loose: on the
    # eval set the hardest answerable question sits at 0.56, and some unanswerable ones
    # sit closer than that, so the gate only catches clearly off-topic questions and the
    # LLM judges the rest. A wrongly gated answerable question can never recover.
    relevance_max_distance: float = 0.65

    @property
    def chroma_path(self) -> Path:
        return self.var_dir / "chroma"

    @property
    def registry_path(self) -> Path:
        return self.var_dir / "registry.sqlite3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
