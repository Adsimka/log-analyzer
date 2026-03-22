"""
Сервис парсинга логов через Drain3.

Управляет отдельными Drain3-инстансами для каждого микросервиса.
Состояние каждого парсера сохраняется в Redis для персистентности.
"""

import re
from dataclasses import dataclass

from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.persistence_handler import PersistenceHandler
from drain3.template_miner_config import TemplateMinerConfig

from src.core.logging import get_logger

logger = get_logger(__name__)

# Регулярка для извлечения класса и метода из stacktrace
_STACKTRACE_PATTERN = re.compile(
    r"at\s+([\w$.]+)\.([\w<>]+)\s*\("
)

# Маскирующие правила — порядок важен (более специфичные первыми)
_MASKING_RULES = [
    (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "IP"),
    (r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "UUID"),
    (r"0x[0-9a-fA-F]+", "HEX"),
    (r"(?<=:)\d+", "PORT"),
    (r"(/[a-zA-Z0-9._\-]+){2,}", "PATH"),
    (r"\b\d+\b", "NUM"),
]


@dataclass
class ParsedLog:
    """Результат парсинга одного лог-сообщения."""

    drain_cluster_id: int
    template_text: str
    is_new_template: bool
    stacktrace_pattern: str | None


class RedisDrain3Persistence(PersistenceHandler):
    """
    Адаптер персистентности Drain3 для Redis.

    Сохраняет и загружает состояние дерева разбора Drain3
    из Redis по ключу, уникальному для каждого микросервиса.
    """

    def __init__(self, redis_client, microservice: str) -> None:
        self._redis = redis_client
        self._key = f"drain3:state:{microservice}"

    def save_state(self, state: bytes) -> None:
        self._redis.set(self._key, state)

    def load_state(self) -> bytes | None:
        return self._redis.get(self._key)


class LogParserService:
    """
    Управляет Drain3-парсерами для всех микросервисов.

    Один инстанс LogParserService живёт на всё время работы приложения.
    Для каждого микросервиса создаётся отдельный TemplateMiner
    со своим деревом разбора и Redis-персистентностью.
    """

    def __init__(self, redis_client) -> None:
        self._redis = redis_client
        self._miners: dict[str, TemplateMiner] = {}

    def _create_miner(self, microservice: str) -> TemplateMiner:
        """Создать новый Drain3 TemplateMiner для микросервиса."""
        config = TemplateMinerConfig()
        config.drain_sim_th = 0.4
        config.drain_depth = 4
        config.drain_max_children = 100
        config.drain_max_clusters = 1024
        config.snapshot_interval_minutes = 5

        # Маскирующие инструкции — Drain3 ожидает список MaskingInstruction
        config.masking_instructions = [
            MaskingInstruction(pattern, mask_with)
            for pattern, mask_with in _MASKING_RULES
        ]

        persistence = RedisDrain3Persistence(self._redis, microservice)
        miner = TemplateMiner(persistence_handler=persistence, config=config)

        logger.info(
            "drain3_miner_created",
            microservice=microservice,
            restored_clusters=len(miner.drain.clusters) if miner.drain else 0,
        )

        return miner

    def _get_miner(self, microservice: str) -> TemplateMiner:
        """Получить или создать Drain3 miner для микросервиса."""
        if microservice not in self._miners:
            self._miners[microservice] = self._create_miner(microservice)
        return self._miners[microservice]

    def parse(self, microservice: str, message: str, stacktrace: str | None = None) -> ParsedLog:
        """
        Распарсить одно лог-сообщение.

        Args:
            microservice: имя микросервиса
            message: текст лог-сообщения
            stacktrace: stacktrace (опционально)

        Returns:
            ParsedLog с ID шаблона, текстом и флагом новизны.
        """
        miner = self._get_miner(microservice)
        result = miner.add_log_message(message)

        is_new = result["change_type"] in ("cluster_created", "cluster_template_changed")

        # Извлекаем паттерн stacktrace
        stacktrace_pattern = None
        if stacktrace:
            stacktrace_pattern = self._extract_stacktrace_pattern(stacktrace)

        return ParsedLog(
            drain_cluster_id=result["cluster_id"],
            template_text=result["template_mined"],
            is_new_template=is_new,
            stacktrace_pattern=stacktrace_pattern,
        )

    def parse_batch(
        self,
        microservice: str,
        messages: list[tuple[str, str | None]],
    ) -> list[ParsedLog]:
        """
        Распарсить батч сообщений одного микросервиса.

        Args:
            microservice: имя микросервиса
            messages: список кортежей (message, stacktrace)

        Returns:
            Список ParsedLog в том же порядке.
        """
        return [
            self.parse(microservice, message, stacktrace)
            for message, stacktrace in messages
        ]

    @staticmethod
    def _extract_stacktrace_pattern(stacktrace: str) -> str | None:
        """
        Извлечь ключевой паттерн из stacktrace.

        Из 'at DatabaseConnection.connect(DatabaseConnection.java:132)'
        извлекает 'DatabaseConnection.connect'.
        """
        match = _STACKTRACE_PATTERN.search(stacktrace)
        if match:
            class_name = match.group(1).split(".")[-1]  # Берём только имя класса
            method_name = match.group(2)
            return f"{class_name}.{method_name}"
        return None