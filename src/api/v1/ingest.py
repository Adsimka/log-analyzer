"""
Эндпоинт приёма логов.

POST /api/v1/ingest — принимает батч логов,
обрабатывает и возвращает статистику.
"""

from fastapi import APIRouter, Depends

from src.api.v1.dependencies import get_ingestion_service, get_db_session
from src.models.schemas import IngestRequest, IngestResponse
from src.services.ingestion import IngestionService

router = APIRouter(tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Приём батча логов",
    description=(
        "Принимает массив лог-записей от внешних сервисов. "
        "Логи фильтруются по уровню, парсятся, векторизуются "
        "и сохраняются в хранилище."
    ),
)
async def ingest_logs(
    request: IngestRequest,
    ingestion: IngestionService = Depends(get_ingestion_service),
    session=Depends(get_db_session),
) -> IngestResponse:
    return await ingestion.process_batch(request, session)