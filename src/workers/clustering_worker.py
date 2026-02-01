"""
ARQ Worker — выполняет фоновые задачи кластеризации.

Запускается отдельным процессом: arq src.workers.clustering_worker.WorkerSettings
"""

import redis as sync_redis
from arq import cron
from arq.connections import RedisSettings

from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.db.session import get_session
from src.services.clustering import ClusteringService

logger = get_logger(__name__)


async def run_clustering(ctx: dict) -> None:
    """
    Задача: запустить кластеризацию для всех 'dirty' микросервисов.

    Вызывается периодически планировщиком ARQ.
    """
    settings = get_settings()
    redis_client: sync_redis.Redis = ctx["redis_sync"]
    clustering_service: ClusteringService = ctx["clustering_service"]

    # Ищем все 'dirty' сервисы
    dirty_keys = redis_client.keys("dirty:*")
    if not dirty_keys:
        logger.debug("clustering_no_dirty_services")
        return

    services = [key.decode().split(":", 1)[1] for key in dirty_keys]
    logger.info("clustering_triggered", services=services)

    for microservice in services:
        for period in settings.available_periods:
            try:
                async with get_session() as session:
                    result = await clustering_service.cluster(
                        session, microservice, period
                    )
                    if result:
                        logger.info(
                            "clustering_done",
                            microservice=microservice,
                            period=period,
                            clusters=result["num_clusters"],
                        )
            except Exception:
                logger.exception(
                    "clustering_failed",
                    microservice=microservice,
                    period=period,
                )

        # Снимаем флаг 'dirty'
        redis_client.delete(f"dirty:{microservice}")


async def startup(ctx: dict) -> None:
    """Инициализация при старте воркера."""
    setup_logging()
    settings = get_settings()

    # Синхронный Redis-клиент для Drain3 и кэша
    redis_client = sync_redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=False,
    )

    ctx["redis_sync"] = redis_client
    ctx["clustering_service"] = ClusteringService(redis_client)

    logger.info("worker_started")


async def shutdown(ctx: dict) -> None:
    """Очистка при остановке воркера."""
    if "redis_sync" in ctx:
        ctx["redis_sync"].close()
    logger.info("worker_stopped")


class WorkerSettings:
    """Конфигурация ARQ-воркера."""

    functions = [run_clustering]
    on_startup = startup
    on_shutdown = shutdown

    # Периодический запуск кластеризации
    cron_jobs = [
        cron(
            run_clustering,
            minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55},
            run_at_startup=True,
        ),
    ]

    # Подключение к Redis для ARQ
    _s = get_settings()
    redis_settings = RedisSettings(
        host=_s.redis_host,
        port=_s.redis_port,
    )