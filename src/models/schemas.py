"""
Pydantic-схемы для валидации входных данных и формирования ответов API.

Отделены от ORM-моделей для соблюдения принципа разделения ответственности.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Входные данные: приём логов
# ──────────────────────────────────────────────


class LogEntry(BaseModel):
    """Одна запись лога, приходящая от внешнего сервиса."""

    timestamp: datetime
    level: str = Field(..., examples=["ERROR", "WARN"])
    message: str = Field(..., min_length=1, max_length=10_000)
    microservice: str = Field(..., min_length=1, max_length=128)
    host: str = Field(default="unknown", max_length=128)
    stacktrace: str | None = Field(default=None, max_length=50_000)

    @field_validator("level")
    @classmethod
    def normalize_level(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("microservice")
    @classmethod
    def normalize_microservice(cls, v: str) -> str:
        return v.strip().lower()


class IngestRequest(BaseModel):
    """Тело запроса POST /ingest — батч логов."""

    logs: list[LogEntry] = Field(..., min_length=1, max_length=10_000)


# ──────────────────────────────────────────────
# Ответы API: приём логов
# ──────────────────────────────────────────────


class ServiceIngestStats(BaseModel):
    """Статистика приёма для одного микросервиса."""

    processed: int
    new_templates: int
    filtered_out: int


class IngestResponse(BaseModel):
    """Ответ на POST /ingest."""

    status: str = "ok"
    total_processed: int
    total_filtered_out: int
    by_service: dict[str, ServiceIngestStats]


# ──────────────────────────────────────────────
# Ответы API: результаты кластеризации
# ──────────────────────────────────────────────


class TemplateInfo(BaseModel):
    """Информация об одном лог-шаблоне внутри кластера."""

    template_text: str
    log_count: int
    stacktrace_pattern: str | None = None


class ClusterDetail(BaseModel):
    """Детальная информация об одном кластере."""

    cluster_id: int
    size: int
    percentage: float
    keywords: list[str]
    top_templates: list[TemplateInfo]
    level_distribution: dict[str, float]
    host_distribution: dict[str, int]
    time_trend: str = "stable"


class UnclusteredInfo(BaseModel):
    """Информация о некластеризованных (шумовых) событиях."""

    size: int
    percentage: float
    templates: list[TemplateInfo] = Field(default_factory=list)


class ClusteringResult(BaseModel):
    """Полный результат кластеризации для одного сервиса и периода."""

    microservice: str
    period: str
    computed_at: datetime
    data_up_to: datetime
    total_logs: int
    unique_templates: int
    num_clusters: int
    noise_ratio: float
    silhouette_score: float | None = None
    clusters: list[ClusterDetail]
    unclustered: UnclusteredInfo


# ──────────────────────────────────────────────
# Ответы API: сервисы и статус
# ──────────────────────────────────────────────


class ServiceInfo(BaseModel):
    """Информация об одном микросервисе в системе."""

    microservice: str
    total_logs: int
    total_templates: int
    last_ingest: datetime | None = None
    last_clustering: datetime | None = None


class SystemStatus(BaseModel):
    """Общее состояние системы."""

    services: list[ServiceInfo]
    redis_connected: bool
    postgres_connected: bool
    sbert_model: str
    uptime_seconds: float