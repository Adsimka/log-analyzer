"""
ARQ Worker — выполняет фоновые задачи кластеризации.

Запускается отдельным процессом: arq src.workers.clustering_worker.WorkerSettings
"""

from datetime import datetime, timedelta, timezone

import redis as sync_redis
from arq import cron
from arq.connections import RedisSettings
from sqlalchemy import delete

from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.db.models import ClusterResult, LogRecord
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

    # Ищем все 'dirty' сервисы через SCAN (не блокирует Redis)
    dirty_keys = []
    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor, match="dirty:*", count=100)
        dirty_keys.extend(keys)
        if cursor == 0:
            break

    if not dirty_keys:
        logger.debug("clustering_no_dirty_services")
        return

    services = [key.decode().split(":", 1)[1] for key in dirty_keys]
    logger.info("clustering_triggered", services=services)

    for microservice in services:
        all_periods_ok = True
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
                all_periods_ok = False
                logger.exception(
                    "clustering_failed",
                    microservice=microservice,
                    period=period,
                )

        # Снимаем флаг 'dirty' только если все периоды обработаны успешно
        if all_periods_ok:
            redis_client.delete(f"dirty:{microservice}")
        else:
            logger.warning(
                "dirty_flag_kept",
                microservice=microservice,
                reason="some periods failed, will retry next cycle",
            )


async def run_log_retention(ctx: dict) -> None:
    """
    Удалить старые логи и результаты кластеризации за пределами retention.

    Вызывается раз в час планировщиком ARQ.
    """
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.log_retention_days)

    try:
        async with get_session() as session:
            # Удаляем старые логи
            log_result = await session.execute(
                delete(LogRecord).where(LogRecord.timestamp < cutoff)
            )
            deleted_logs = log_result.rowcount

            # Удаляем старые результаты кластеризации
            cluster_result = await session.execute(
                delete(ClusterResult).where(ClusterResult.computed_at < cutoff)
            )
            deleted_results = cluster_result.rowcount

        if deleted_logs > 0 or deleted_results > 0:
            logger.info(
                "retention_cleanup_done",
                deleted_logs=deleted_logs,
                deleted_cluster_results=deleted_results,
                retention_days=settings.log_retention_days,
            )
    except Exception:
        logger.exception("retention_cleanup_failed")


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

    functions = [run_clustering, run_log_retention]
    on_startup = startup
    on_shutdown = shutdown

    # Периодический запуск кластеризации и очистки
    cron_jobs = [
        cron(
            run_clustering,
            minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55},
            run_at_startup=True,
        ),
        cron(
            run_log_retention,
            hour={3},
            minute={0},
            run_at_startup=False,
        ),
    ]

    # Подключение к Redis для ARQ
    _s = get_settings()
    redis_settings = RedisSettings(
        host=_s.redis_host,
        port=_s.redis_port,
    )