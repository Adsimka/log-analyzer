"""
Эндпоинты для получения результатов кластеризации и статуса системы.
"""

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.dependencies import get_db_session, get_redis_client, get_startup_time
from src.core.config import get_settings
from src.db.models import ClusterResult, LogRecord, Template
from src.models.schemas import ClusteringResult, ServiceInfo, SystemStatus

router = APIRouter(tags=["results"])


@router.get(
    "/services",
    response_model=list[ServiceInfo],
    summary="Список микросервисов",
    description="Возвращает все микросервисы, от которых получены логи.",
)
async def list_services(
    session: AsyncSession = Depends(get_db_session),
) -> list[ServiceInfo]:
    # Статистика по логам
    logs_stmt = (
        select(
            LogRecord.microservice,
            func.count(LogRecord.id).label("total_logs"),
            func.max(LogRecord.timestamp).label("last_ingest"),
        )
        .group_by(LogRecord.microservice)
    )
    logs_result = await session.execute(logs_stmt)
    logs_data = {row.microservice: row for row in logs_result}

    # Статистика по шаблонам
    tmpl_stmt = (
        select(
            Template.microservice,
            func.count(Template.id).label("total_templates"),
        )
        .group_by(Template.microservice)
    )
    tmpl_result = await session.execute(tmpl_stmt)
    tmpl_data = {row.microservice: row.total_templates for row in tmpl_result}

    # Последняя кластеризация
    cluster_stmt = (
        select(
            ClusterResult.microservice,
            func.max(ClusterResult.computed_at).label("last_clustering"),
        )
        .group_by(ClusterResult.microservice)
    )
    cluster_result = await session.execute(cluster_stmt)
    cluster_data = {row.microservice: row.last_clustering for row in cluster_result}

    services = []
    for microservice, log_info in logs_data.items():
        services.append(
            ServiceInfo(
                microservice=microservice,
                total_logs=log_info.total_logs,
                total_templates=tmpl_data.get(microservice, 0),
                last_ingest=log_info.last_ingest,
                last_clustering=cluster_data.get(microservice),
            )
        )

    return services


@router.get(
    "/results/{microservice}",
    summary="Результаты кластеризации",
    description=(
        "Возвращает последний результат кластеризации для микросервиса. "
        "Если fresh=true, ставит задачу на пересчёт."
    ),
)
async def get_results(
    microservice: str,
    period: str = Query(default="24h", pattern=r"^\d+(h|d)$"),
    fresh: bool = Query(default=False),
    redis_client=Depends(get_redis_client),
    session: AsyncSession = Depends(get_db_session),
):
    settings = get_settings()

    if period not in settings.available_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Недопустимый период. Допустимые: {settings.available_periods}",
        )

    # Проверяем кэш Redis
    cache_key = f"cluster:latest:{microservice}:{period}"
    cached = redis_client.get(cache_key)

    if cached and not fresh:
        return json.loads(cached)

    # Если нет кэша — берём из БД (последний результат)
    stmt = (
        select(ClusterResult.result_data)
        .where(
            ClusterResult.microservice == microservice,
            ClusterResult.period == period,
        )
        .order_by(ClusterResult.computed_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Результаты кластеризации для '{microservice}' "
                f"за период '{period}' не найдены. "
                f"Дождитесь следующего цикла кластеризации."
            ),
        )

    # Кэшируем
    redis_client.set(cache_key, json.dumps(row, default=str), ex=3600)

    return row


@router.get(
    "/results/{microservice}/{cluster_id}",
    summary="Детали кластера",
    description="Возвращает подробную информацию о конкретном кластере.",
)
async def get_cluster_detail(
    microservice: str,
    cluster_id: int,
    period: str = Query(default="24h", pattern=r"^\d+(h|d)$"),
    redis_client=Depends(get_redis_client),
    session: AsyncSession = Depends(get_db_session),
):
    # Получаем полный результат
    cache_key = f"cluster:latest:{microservice}:{period}"
    cached = redis_client.get(cache_key)

    result_data = None
    if cached:
        result_data = json.loads(cached)
    else:
        stmt = (
            select(ClusterResult.result_data)
            .where(
                ClusterResult.microservice == microservice,
                ClusterResult.period == period,
            )
            .order_by(ClusterResult.computed_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        result_data = result.scalar_one_or_none()

    if result_data is None:
        raise HTTPException(status_code=404, detail="Результаты не найдены.")

    # Ищем нужный кластер
    clusters = result_data.get("clusters", [])
    for cluster in clusters:
        if cluster.get("cluster_id") == cluster_id:
            return cluster

    raise HTTPException(
        status_code=404,
        detail=f"Кластер {cluster_id} не найден в результатах.",
    )


@router.get(
    "/status",
    response_model=SystemStatus,
    summary="Состояние системы",
    description="Проверка здоровья и общая статистика.",
)
async def get_status(
    session: AsyncSession = Depends(get_db_session),
    redis_client=Depends(get_redis_client),
    startup_time: float = Depends(get_startup_time),
) -> SystemStatus:
    settings = get_settings()

    # Проверяем Redis
    redis_ok = False
    try:
        redis_ok = redis_client.ping()
    except Exception:
        pass

    # Проверяем PostgreSQL
    pg_ok = False
    try:
        await session.execute(select(func.now()))
        pg_ok = True
    except Exception:
        pass

    # Собираем информацию по сервисам
    services = await list_services(session)

    return SystemStatus(
        services=services,
        redis_connected=redis_ok,
        postgres_connected=pg_ok,
        sbert_model=settings.sbert_model_name,
        uptime_seconds=round(time.time() - startup_time, 1),
    )