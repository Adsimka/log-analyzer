"""
Сервис приёма (ingestion) логов.

Оркестрирует полный пайплайн обработки входящего батча:
фильтрация → группировка → парсинг → векторизация → запись в БД.
"""

from collections import defaultdict

from sqlalchemy import case, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.logging import get_logger
from src.db.models import LogRecord, Template
from src.models.schemas import IngestRequest, IngestResponse, LogEntry, ServiceIngestStats
from src.services.embedding import EmbeddingService
from src.services.log_parser import LogParserService

logger = get_logger(__name__)


class IngestionService:
    """
    Координирует приём и обработку батча логов.

    Один инстанс на всё приложение, получает зависимости при создании.
    """

    def __init__(
        self,
        parser: LogParserService,
        embedder: EmbeddingService,
        redis_client,
    ) -> None:
        self._parser = parser
        self._embedder = embedder
        self._redis = redis_client
        self._settings = get_settings()

    async def process_batch(
        self, request: IngestRequest, session: AsyncSession
    ) -> IngestResponse:
        """
        Обработать входящий батч логов.

        1. Фильтрует по level.
        2. Группирует по microservice.
        3. Для каждой группы: парсит, векторизует, записывает в БД.
        4. Помечает сервисы как 'dirty' для кластеризации.
        """
        accepted_levels = self._settings.accepted_levels_list

        # Шаг 1: Фильтрация
        accepted: list[LogEntry] = []
        filtered_out = 0

        for log_entry in request.logs:
            if log_entry.level in accepted_levels:
                accepted.append(log_entry)
            else:
                filtered_out += 1

        if not accepted:
            return IngestResponse(
                total_processed=0,
                total_filtered_out=filtered_out,
                by_service={},
            )

        # Шаг 2: Группировка по microservice
        grouped: dict[str, list[LogEntry]] = defaultdict(list)
        for log_entry in accepted:
            grouped[log_entry.microservice].append(log_entry)

        # Шаг 3: Обработка каждой группы
        by_service: dict[str, ServiceIngestStats] = {}

        for microservice, logs in grouped.items():
            stats = await self._process_service_group(microservice, logs, session)
            by_service[microservice] = stats

            # Помечаем сервис как 'dirty' — нужна перекластеризация
            self._redis.set(f"dirty:{microservice}", "1")

        total_processed = sum(s.processed for s in by_service.values())

        logger.info(
            "batch_processed",
            total_processed=total_processed,
            total_filtered_out=filtered_out,
            services=list(by_service.keys()),
        )

        return IngestResponse(
            total_processed=total_processed,
            total_filtered_out=filtered_out,
            by_service=by_service,
        )

    async def _process_service_group(
        self,
        microservice: str,
        logs: list[LogEntry],
        session: AsyncSession,
    ) -> ServiceIngestStats:
        """
        Обработать группу логов одного микросервиса.

        Парсит каждый лог → определяет новые шаблоны → векторизует →
        upsert шаблонов → batch insert логов.
        """
        # Парсинг через Drain3
        messages = [(log.message, log.stacktrace) for log in logs]
        parsed_results = self._parser.parse_batch(microservice, messages)

        # Собираем новые шаблоны для векторизации
        new_templates: dict[int, tuple[str, str | None]] = {}
        template_counts: dict[int, int] = defaultdict(int)

        for parsed in parsed_results:
            template_counts[parsed.drain_cluster_id] += 1
            if parsed.is_new_template and parsed.drain_cluster_id not in new_templates:
                new_templates[parsed.drain_cluster_id] = (
                    parsed.template_text,
                    parsed.stacktrace_pattern,
                )

        # Векторизация новых шаблонов
        new_embeddings: dict[int, bytes] = {}
        if new_templates:
            templates_to_encode = [
                (drain_id, text) for drain_id, (text, _) in new_templates.items()
            ]
            encoded = self._embedder.encode_and_cache(microservice, templates_to_encode)
            new_embeddings = {
                drain_id: EmbeddingService.serialize_embedding(emb)
                for drain_id, emb in encoded.items()
            }

        # Upsert шаблонов в PostgreSQL
        template_id_map = await self._upsert_templates(
            session,
            microservice,
            parsed_results,
            new_templates,
            new_embeddings,
            template_counts,
        )

        # Batch insert логов
        log_records = [
            LogRecord(
                microservice=microservice,
                timestamp=log_entry.timestamp,
                level=log_entry.level,
                raw_message=log_entry.message,
                host=log_entry.host,
                stacktrace=log_entry.stacktrace,
                template_id=template_id_map.get(parsed.drain_cluster_id),
            )
            for log_entry, parsed in zip(logs, parsed_results)
        ]
        session.add_all(log_records)

        return ServiceIngestStats(
            processed=len(logs),
            new_templates=len(new_templates),
            filtered_out=0,
        )

    async def _upsert_templates(
        self,
        session: AsyncSession,
        microservice: str,
        parsed_results: list,
        new_templates: dict[int, tuple[str, str | None]],
        new_embeddings: dict[int, bytes],
        template_counts: dict[int, int],
    ) -> dict[int, int]:
        """
        Upsert шаблонов: вставить новые, обновить счётчики существующих.

        Returns:
            Маппинг {drain_cluster_id: template_db_id}.
        """
        all_drain_ids = set(p.drain_cluster_id for p in parsed_results)

        # Колонки уникального индекса для ON CONFLICT
        conflict_columns = [Template.microservice, Template.drain_cluster_id]

        # Batch upsert: собираем все значения и выполняем одним запросом
        values_list = []
        for drain_id in all_drain_ids:
            count = template_counts[drain_id]
            if drain_id in new_templates:
                text_val, st_pattern = new_templates[drain_id]
                embedding_bytes = new_embeddings.get(drain_id)
                values_list.append({
                    "microservice": microservice,
                    "drain_cluster_id": drain_id,
                    "template_text": text_val,
                    "embedding": embedding_bytes,
                    "stacktrace_pattern": st_pattern,
                    "log_count": count,
                })
            else:
                values_list.append({
                    "microservice": microservice,
                    "drain_cluster_id": drain_id,
                    "template_text": "",
                    "embedding": None,
                    "stacktrace_pattern": None,
                    "log_count": count,
                })

        if values_list:
            stmt = pg_insert(Template).values(values_list)
            stmt = stmt.on_conflict_do_update(
                index_elements=conflict_columns,
                set_={
                    "template_text": case(
                        (stmt.excluded.template_text != "", stmt.excluded.template_text),
                        else_=Template.template_text,
                    ),
                    "embedding": func.COALESCE(
                        stmt.excluded.embedding, Template.embedding
                    ),
                    "stacktrace_pattern": func.COALESCE(
                        stmt.excluded.stacktrace_pattern, Template.stacktrace_pattern
                    ),
                    "log_count": Template.log_count + stmt.excluded.log_count,
                },
            )
            await session.execute(stmt)

        await session.flush()

        result = await session.execute(
            select(Template.drain_cluster_id, Template.id).where(
                Template.microservice == microservice,
                Template.drain_cluster_id.in_(all_drain_ids),
            )
        )
        return dict(result.all())