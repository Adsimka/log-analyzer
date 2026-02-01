"""
Сервис семантической векторизации лог-шаблонов.

Использует Sentence Transformers для получения эмбеддингов.
Кэширует эмбеддинги в Redis — повторная векторизация одного и того же
шаблона не требует вызова модели.
"""

import io

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Управляет загрузкой SBERT-модели и кэшированием эмбеддингов.

    Модель загружается один раз при инициализации и живёт
    в памяти на всё время работы приложения.
    """

    def __init__(self, redis_client) -> None:
        self._redis = redis_client
        self._model: SentenceTransformer | None = None
        self._model_name = get_settings().sbert_model_name
        self._embedding_dim: int | None = None

    def load_model(self) -> None:
        """
        Загрузить SBERT-модель в память.

        Вызывается один раз при старте приложения.
        """
        logger.info("sbert_loading", model=self._model_name)

        self._model = SentenceTransformer(self._model_name)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

        logger.info(
            "sbert_loaded",
            model=self._model_name,
            embedding_dim=self._embedding_dim,
        )

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError("Модель не загружена. Вызовите load_model() сначала.")
        return self._embedding_dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Закодировать список текстов в эмбеддинги.

        Args:
            texts: список строк для векторизации.

        Returns:
            numpy-массив размером (len(texts), embedding_dim).
        """
        if self._model is None:
            raise RuntimeError("Модель не загружена. Вызовите load_model() сначала.")

        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-нормализация для cosine similarity
        )
        return embeddings

    def get_cached_embedding(self, microservice: str, drain_cluster_id: int) -> np.ndarray | None:
        """
        Получить эмбеддинг из Redis-кэша.

        Returns:
            numpy-массив или None, если не найден.
        """
        key = f"template:embedding:{microservice}:{drain_cluster_id}"
        data = self._redis.get(key)
        if data is None:
            return None
        return self._deserialize_embedding(data)

    def cache_embedding(
        self, microservice: str, drain_cluster_id: int, embedding: np.ndarray
    ) -> None:
        """Сохранить эмбеддинг в Redis-кэш."""
        key = f"template:embedding:{microservice}:{drain_cluster_id}"
        self._redis.set(key, self._serialize_embedding(embedding))

    def encode_and_cache(
        self,
        microservice: str,
        templates: list[tuple[int, str]],
    ) -> dict[int, np.ndarray]:
        """
        Закодировать новые шаблоны и сохранить в кэш.

        Args:
            microservice: имя микросервиса
            templates: список кортежей (drain_cluster_id, template_text)

        Returns:
            Словарь {drain_cluster_id: embedding}.
        """
        if not templates:
            return {}

        drain_ids = [t[0] for t in templates]
        texts = [t[1] for t in templates]

        embeddings = self.encode(texts)
        result: dict[int, np.ndarray] = {}

        for drain_id, embedding in zip(drain_ids, embeddings):
            self.cache_embedding(microservice, drain_id, embedding)
            result[drain_id] = embedding

        logger.info(
            "templates_encoded",
            microservice=microservice,
            count=len(templates),
        )

        return result

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Сериализовать numpy-массив в bytes для хранения."""
        buffer = io.BytesIO()
        np.save(buffer, embedding)
        return buffer.getvalue()

    @staticmethod
    def _deserialize_embedding(data: bytes) -> np.ndarray:
        """Десериализовать bytes обратно в numpy-массив."""
        buffer = io.BytesIO(data)
        return np.load(buffer)

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Публичный метод сериализации (для записи в PostgreSQL)."""
        buffer = io.BytesIO()
        np.save(buffer, embedding)
        return buffer.getvalue()

    @staticmethod
    def deserialize_embedding(data: bytes) -> np.ndarray:
        """Публичный метод десериализации (для чтения из PostgreSQL)."""
        buffer = io.BytesIO(data)
        return np.load(buffer)