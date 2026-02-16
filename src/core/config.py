"""
Центральная конфигурация приложения.

Все параметры читаются из переменных окружения (или .env файла).
Единая плоская модель — pydantic-settings гарантированно подхватывает
каждую переменную из .env без проблем с вложенностью.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Все настройки приложения в одной модели.

    Каждое поле соответствует переменной окружения (или строке в .env).
    Например: postgres_host → POSTGRES_HOST в .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── PostgreSQL ──────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "log_clustering"
    postgres_user: str = "app"
    postgres_password: str = "secret"

    # ── Redis ───────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379

    # ── Application ─────────────────────────────
    log_level: str = "INFO"
    accepted_log_levels: str = "ERROR,WARN"
    clustering_interval_seconds: int = 300
    default_period: str = "24h"
    available_periods: list[str] = Field(
        default=["1h", "6h", "24h", "3d", "7d"],
    )
    log_retention_days: int = 30

    # ── ML / SBERT ──────────────────────────────
    sbert_model_name: str = "all-MiniLM-L6-v2"

    # ── UMAP ────────────────────────────────────
    umap_n_components: int = 10
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    # ── HDBSCAN ─────────────────────────────────
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: int = 2
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"

    # ── Пороги кластеризации ────────────────────
    min_templates_for_clustering: int = 5
    skip_umap_threshold: int = 50

    # ── Вычисляемые свойства ────────────────────

    @property
    def postgres_dsn(self) -> str:
        """DSN для asyncpg (async). SSL отключён для Docker-контейнера."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?ssl=disable"
        )

    @property
    def postgres_sync_dsn(self) -> str:
        """DSN для синхронных операций (Alembic)."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def accepted_levels_list(self) -> list[str]:
        return [level.strip().upper() for level in self.accepted_log_levels.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Синглтон настроек. Кэшируется при первом вызове."""
    return Settings()