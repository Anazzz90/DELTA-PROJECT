"""
config/settings.py
==================
Central configuration loader for DMARS.

Reads all environment variables from the .env file (via python-dotenv / pydantic-settings).
Every other module imports from here — never reads os.environ directly.

Usage:
    from config.settings import settings
    print(settings.openai_api_key)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # Silently ignore unknown env vars
    )

    # ── LLM API Keys ──────────────────────────────────────────────────────────
    openai_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""  # Phase 3 only
    google_api_key: str = ""
    siliconflow_api_key: str = ""
    firecrawl_api_key: str = ""

    # ── Local LLMs ────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # ── Database ──────────────────────────────────────────────────────────────
    # Phase 1: SQLite | Phase 2+: PostgreSQL (swap DATABASE_URL in .env)
    database_url: str = "sqlite+aiosqlite:///./dmars.db"

    # ── Application ───────────────────────────────────────────────────────────
    env: str = "development"           # development | production
    log_level: str = "INFO"
    active_prompt_version: str = "v1"

    # ── Scoring ───────────────────────────────────────────────────────────────
    cache_similarity_threshold: float = 0.92

    # ── Budget Alerts ─────────────────────────────────────────────────────────
    daily_cost_alert_usd: float = 5.00
    per_query_cost_alert_usd: float = 0.10

    # ── Redis (Phase 2+) ──────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Firecrawl (Checkpoint 15a) ────────────────────────────────────────────
    firecrawl_rate_limit_per_min: int = 10

    # ── LangFuse (Phase 2+) ───────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Computed Properties ───────────────────────────────────────────────────
    @property
    def prompts_dir(self) -> Path:
        """Returns the absolute path to the active prompt version directory."""
        return Path(__file__).parent.parent / "prompts" / self.active_prompt_version

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


# Single shared instance — import this everywhere
settings = Settings()
