"""
Точка входа FastAPI-приложения.

Lifespan управляет жизненным циклом:
- При старте: подключение к БД, Redis, загрузка SBERT-модели.
- При остановке: корректное закрытие соединений.
"""

import asyncio
import time
from contextlib import asynccontextmanager

import redis as sync_redis
from fastapi import FastAPI

from src.api.v1.router import api_v1_router
from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.db.session import close_db, init_db
from src.services.embedding import EmbeddingService
from src.services.ingestion import IngestionService
from src.services.log_parser import LogParserService

logger = get_logger(__name__)

MAX_RETRIES = 15
RETRY_DELAY_SECONDS = 2


async def _wait_for_postgres() -> None:
    """Ожидание готовности PostgreSQL с повторными попытками."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await init_db()
            return
        except Exception as exc:
            if attempt == MAX_RETRIES:
                logger.error("postgres_unavailable", attempts=attempt, error=str(exc))
                raise
            logger.warning(
                "postgres_not_ready",
                attempt=attempt,
                max_retries=MAX_RETRIES,
                retry_in=RETRY_DELAY_SECONDS,
            )
            await asyncio.sleep(RETRY_DELAY_SECONDS)


def _wait_for_redis(host: str, port: int) -> sync_redis.Redis:
    """Ожидание готовности Redis с повторными попытками."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = sync_redis.Redis(
                host=host, port=port, decode_responses=False,
            )
            client.ping()
            return client
        except Exception as exc:
            if attempt == MAX_RETRIES:
                logger.error("redis_unavailable", attempts=attempt, error=str(exc))
                raise
            logger.warning(
                "redis_not_ready",
                attempt=attempt,
                max_retries=MAX_RETRIES,
                retry_in=RETRY_DELAY_SECONDS,
            )
            time.sleep(RETRY_DELAY_SECONDS)

    # unreachable, но для типизации
    raise RuntimeError("Redis connection failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Жизненный цикл приложения.

    Startup:
        1. Настройка логирования.
        2. Подключение к PostgreSQL (с retry), создание таблиц.
        3. Подключение к Redis (с retry).
        4. Загрузка SBERT-модели.
        5. Инициализация сервисов.

    Shutdown:
        1. Закрытие соединений.
    """
    setup_logging()
    settings = get_settings()

    logger.info("app_starting", log_level=settings.log_level)

    # ── PostgreSQL (с retry) ──
    await _wait_for_postgres()
    logger.info("postgres_connected", dsn=settings.postgres_dsn.split("@")[-1])

    # ── Redis (с retry) ──
    redis_client = _wait_for_redis(settings.redis_host, settings.redis_port)
    logger.info("redis_connected", host=settings.redis_host)

    # ── SBERT ──
    embedding_service = EmbeddingService(redis_client)
    embedding_service.load_model()

    # ── Сервисы ──
    parser_service = LogParserService(redis_client)
    ingestion_service = IngestionService(parser_service, embedding_service, redis_client)

    # Сохраняем в app.state для dependency injection
    app.state.redis_client = redis_client
    app.state.embedding_service = embedding_service
    app.state.parser_service = parser_service
    app.state.ingestion_service = ingestion_service
    app.state.startup_time = time.time()

    logger.info(
        "app_ready",
        accepted_levels=settings.accepted_levels_list,
        sbert_model=settings.sbert_model_name,
        clustering_interval=settings.clustering_interval_seconds,
    )

    yield

    # ── Shutdown ──
    redis_client.close()
    await close_db()
    logger.info("app_stopped")


app = FastAPI(
    title="Log Event Clustering System",
    description=(
        "Система кластеризации лог-событий на основе семантического сходства. "
        "Принимает логи от микросервисов, парсит через Drain3, "
        "векторизует через Sentence Transformers, "
        "кластеризует через UMAP + HDBSCAN."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_v1_router)


@app.get("/health", tags=["system"])
async def health_check():
    """Простая проверка доступности сервиса."""
    return {"status": "ok"}