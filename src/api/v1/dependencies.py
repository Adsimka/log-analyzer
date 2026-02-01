"""
Зависимости (Dependency Injection) для FastAPI-эндпоинтов.

Все сервисы инициализируются в lifespan приложения (src/main.py)
и хранятся в app.state. Здесь мы извлекаем их для инъекции в роутеры.
"""

from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session_factory
from src.services.ingestion import IngestionService


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Получить сессию БД для одного запроса."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_redis_client(request: Request):
    """Получить синхронный Redis-клиент."""
    return request.app.state.redis_client


def get_ingestion_service(request: Request) -> IngestionService:
    """Получить сервис приёма логов."""
    return request.app.state.ingestion_service


def get_startup_time(request: Request) -> float:
    """Получить время старта приложения."""
    return request.app.state.startup_time