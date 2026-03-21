"""
SQLAlchemy ORM-модели для PostgreSQL.

Три основных таблицы:
- logs: сырые лог-записи (растёт быстро)
- templates: уникальные шаблоны логов (растёт медленно)
- cluster_results: результаты периодической кластеризации
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""

    pass


class LogRecord(Base):
    """
    Таблица логов — основное хранилище.

    Каждая строка — один принятый лог-запись.
    Быстро растёт, индексируется по (microservice, timestamp).
    """

    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    microservice: Mapped[str] = mapped_column(String(128), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False)
    raw_message: Mapped[str] = mapped_column(Text, nullable=False)
    host: Mapped[str] = mapped_column(String(128), nullable=False, default="unknown")
    stacktrace: Mapped[str | None] = mapped_column(Text, nullable=True)
    template_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("templates.id", ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (
        Index("ix_logs_service_timestamp", "microservice", "timestamp"),
        Index("ix_logs_template_id", "template_id"),
    )


class Template(Base):
    """
    Таблица шаблонов — результат парсинга Drain3.

    Растёт медленно: из миллионов логов обычно сотни уникальных шаблонов.
    Эмбеддинг хранится как бинарные данные (сериализованный numpy-массив).
    """

    __tablename__ = "templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    microservice: Mapped[str] = mapped_column(String(128), nullable=False)
    drain_cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    template_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    stacktrace_pattern: Mapped[str | None] = mapped_column(String(256), nullable=True)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    log_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    __table_args__ = (
        Index(
            "uq_templates_service_drain",
            "microservice",
            "drain_cluster_id",
            unique=True,
        ),
    )


class ClusterResult(Base):
    """
    Таблица результатов кластеризации.

    Каждая строка — один снимок кластеризации для конкретного
    микросервиса и периода. Результат хранится как JSONB.
    """

    __tablename__ = "cluster_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    microservice: Mapped[str] = mapped_column(String(128), nullable=False)
    period: Mapped[str] = mapped_column(String(16), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    num_clusters: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    noise_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    silhouette_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_data: Mapped[dict] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        Index("ix_cluster_results_lookup", "microservice", "period", "computed_at"),
    )