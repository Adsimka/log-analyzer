"""
Сервис кластеризации лог-шаблонов.

Выполняет полный пайплайн:
1. Извлечение шаблонов и эмбеддингов из БД за заданный период.
2. UMAP — снижение размерности.
3. HDBSCAN — кластеризация.
4. c-TF-IDF — извлечение ключевых слов.
5. Формирование и сохранение результата.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import hdbscan
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from umap import UMAP

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.utils import parse_period
from src.db.models import ClusterResult, LogRecord, Template
from src.services.embedding import EmbeddingService

logger = get_logger(__name__)


class ClusteringService:
    """
    Выполняет кластеризацию шаблонов для одного микросервиса и периода.
    """

    def __init__(self, redis_client) -> None:
        self._redis = redis_client
        self._settings = get_settings()

    async def cluster(
        self,
        session: AsyncSession,
        microservice: str,
        period: str,
    ) -> dict | None:
        """
        Выполнить полный цикл кластеризации.

        Args:
            session: сессия БД
            microservice: имя микросервиса
            period: строка периода ('1h', '24h', '3d', ...)

        Returns:
            dict с результатом или None, если недостаточно данных.
        """
        s = self._settings
        delta = parse_period(period)

        # ──────────────────────────────────
        # Шаг 1: Извлечение данных из БД
        # ──────────────────────────────────
        # Определяем cutoff от последнего лога сервиса, а не от now(),
        # чтобы корректно работать с историческими датасетами (BGL 2005 и т.п.)
        latest_ts = await self._get_latest_timestamp(session, microservice)
        if latest_ts is None:
            logger.info(
                "clustering_skipped_no_logs",
                microservice=microservice,
                period=period,
            )
            return None
        cutoff = latest_ts - delta

        template_data = await self._fetch_template_data(session, microservice, cutoff)

        if len(template_data) < s.min_templates_for_clustering:
            logger.info(
                "clustering_skipped_insufficient_data",
                microservice=microservice,
                period=period,
                templates=len(template_data),
            )
            return None

        # Разбираем данные
        template_ids: list[int] = []
        template_texts: list[str] = []
        embeddings_list: list[np.ndarray] = []
        log_counts: list[int] = []
        error_counts: list[int] = []
        host_maps: list[dict[str, int]] = []
        stacktrace_patterns: list[str | None] = []
        hourly_counts_list: list[list[int]] = []

        for row in template_data:
            if row["embedding"] is None:
                continue

            template_ids.append(row["template_id"])
            template_texts.append(row["template_text"])
            embeddings_list.append(
                EmbeddingService.deserialize_embedding(row["embedding"])
            )
            log_counts.append(row["log_count"])
            error_counts.append(row["error_count"])
            host_maps.append(row["host_distribution"])
            stacktrace_patterns.append(row["stacktrace_pattern"])
            hourly_counts_list.append(row.get("hourly_counts", []))

        if len(embeddings_list) < s.min_templates_for_clustering:
            logger.info(
                "clustering_skipped_no_embeddings",
                microservice=microservice,
                period=period,
                templates_total=len(template_data),
                templates_with_embedding=len(embeddings_list),
                min_required=s.min_templates_for_clustering,
            )
            return None

        embeddings = np.array(embeddings_list)
        total_logs = sum(log_counts)

        logger.info(
            "clustering_started",
            microservice=microservice,
            period=period,
            templates=len(template_ids),
            total_logs=total_logs,
        )

        # ──────────────────────────────────
        # Шаг 2: UMAP (опционально)
        # ──────────────────────────────────
        umap_applied = False
        n_templates = len(embeddings)
        if n_templates > s.skip_umap_threshold:
            # n_neighbors масштабируется с размером датасета:
            # слишком большое значение сглаживает локальную структуру.
            adaptive_n_neighbors = min(
                s.umap_n_neighbors,
                max(5, int(np.sqrt(n_templates))),
                n_templates - 1,
            )
            reducer = UMAP(
                n_components=min(s.umap_n_components, n_templates - 2),
                n_neighbors=adaptive_n_neighbors,
                min_dist=s.umap_min_dist,
                metric=s.umap_metric,
                random_state=42,
            )
            reduced = reducer.fit_transform(embeddings)
            umap_applied = True
        else:
            reduced = embeddings

        # ──────────────────────────────────
        # Шаг 3: HDBSCAN
        # ──────────────────────────────────
        # После UMAP евклидова метрика оптимальна (низкоразмерное пространство).
        # Без UMAP используем cosine — SBERT-эмбеддинги оптимизированы под неё.
        effective_metric = s.hdbscan_metric if umap_applied else "cosine"
        hdbscan_kwargs = dict(
            min_cluster_size=s.hdbscan_min_cluster_size,
            min_samples=s.hdbscan_min_samples,
            metric=effective_metric,
            cluster_selection_method=s.hdbscan_cluster_selection_method,
        )
        if s.hdbscan_cluster_selection_epsilon > 0:
            hdbscan_kwargs["cluster_selection_epsilon"] = (
                s.hdbscan_cluster_selection_epsilon
            )
        clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
        labels = clusterer.fit_predict(reduced)

        # ──────────────────────────────────
        # Шаг 4: Оценка качества
        # ──────────────────────────────────
        unique_labels = set(labels)
        num_clusters = len(unique_labels - {-1})
        noise_count = int(np.sum(labels == -1))
        noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0.0

        # DBCV — первичная метрика для density-based кластеризации.
        # Silhouette — вторичная (для сравнимости с бенчмарками).
        dbcv_score = None
        try:
            validity = getattr(clusterer, "relative_validity_", None)
            if validity is not None and np.isfinite(validity):
                dbcv_score = float(validity)
        except Exception:
            pass

        score = None
        if num_clusters >= 2 and noise_count < len(labels):
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > num_clusters:
                try:
                    score = float(
                        silhouette_score(
                            reduced[non_noise_mask],
                            labels[non_noise_mask],
                            metric=effective_metric,
                        )
                    )
                except ValueError:
                    score = None

        # ──────────────────────────────────
        # Шаг 5: c-TF-IDF и формирование кластеров
        # ──────────────────────────────────
        clusters_detail = []

        for cluster_id in sorted(unique_labels - {-1}):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            cluster_info = self._build_cluster_info(
                cluster_id=int(cluster_id),
                indices=cluster_indices,
                template_texts=template_texts,
                log_counts=log_counts,
                error_counts=error_counts,
                host_maps=host_maps,
                stacktrace_patterns=stacktrace_patterns,
                hourly_counts_list=hourly_counts_list,
                total_logs=total_logs,
                all_template_texts=template_texts,
                all_labels=labels,
            )
            clusters_detail.append(cluster_info)

        # Некластеризованные (noise)
        noise_indices = np.where(labels == -1)[0]
        unclustered = self._build_unclustered_info(
            indices=noise_indices,
            template_texts=template_texts,
            log_counts=log_counts,
            stacktrace_patterns=stacktrace_patterns,
            total_logs=total_logs,
        )

        # ──────────────────────────────────
        # Шаг 6: Сборка результата
        # ──────────────────────────────────
        now = datetime.now(timezone.utc)

        result = {
            "microservice": microservice,
            "period": period,
            "computed_at": now.isoformat(),
            "data_up_to": latest_ts.isoformat(),
            "total_logs": total_logs,
            "unique_templates": len(template_ids),
            "num_clusters": num_clusters,
            "noise_ratio": round(noise_ratio, 4),
            "silhouette_score": round(score, 4) if score is not None else None,
            "dbcv_score": round(dbcv_score, 4) if dbcv_score is not None else None,
            "clusters": clusters_detail,
            "unclustered": unclustered,
        }

        # ──────────────────────────────────
        # Шаг 7: Сохранение
        # ──────────────────────────────────
        await self._save_result(session, microservice, period, result)

        # Кэш в Redis
        cache_key = f"cluster:latest:{microservice}:{period}"
        self._redis.set(cache_key, json.dumps(result, default=str), ex=3600)

        logger.info(
            "clustering_completed",
            microservice=microservice,
            period=period,
            num_clusters=num_clusters,
            noise_ratio=round(noise_ratio, 4),
            silhouette_score=round(score, 4) if score else None,
            dbcv_score=round(dbcv_score, 4) if dbcv_score else None,
        )

        return result

    @staticmethod
    async def _get_latest_timestamp(
        session: AsyncSession,
        microservice: str,
    ) -> datetime | None:
        """Получить timestamp самого свежего лога для микросервиса."""
        result = await session.execute(
            select(func.max(LogRecord.timestamp)).where(
                LogRecord.microservice == microservice
            )
        )
        return result.scalar_one_or_none()

    async def _fetch_template_data(
        self,
        session: AsyncSession,
        microservice: str,
        cutoff: datetime,
    ) -> list[dict]:
        """
        Получить агрегированные данные шаблонов за период.

        Для каждого шаблона: эмбеддинг, количество логов,
        распределение по level и host.

        Используем два простых запроса вместо одного сложного:
        1. Основная агрегация (count, error/warn).
        2. Host-распределение отдельно.
        """
        # Запрос 1: основная агрегация по шаблонам
        main_stmt = text("""
            SELECT
                t.id AS template_id,
                t.template_text,
                t.embedding,
                t.stacktrace_pattern,
                COUNT(l.id) AS log_count,
                COUNT(l.id) AS error_count
            FROM templates t
            JOIN logs l ON l.template_id = t.id
            WHERE l.microservice = :microservice
              AND l.timestamp > :cutoff
            GROUP BY t.id, t.template_text, t.embedding, t.stacktrace_pattern
        """)

        main_result = await session.execute(
            main_stmt, {"microservice": microservice, "cutoff": cutoff}
        )
        main_rows = main_result.mappings().all()

        if not main_rows:
            return []

        # Запрос 2: host-распределение по шаблонам
        host_stmt = text("""
            SELECT
                l.template_id,
                l.host,
                COUNT(*) AS cnt
            FROM logs l
            WHERE l.microservice = :microservice
              AND l.timestamp > :cutoff
              AND l.template_id IS NOT NULL
            GROUP BY l.template_id, l.host
        """)

        host_result = await session.execute(
            host_stmt, {"microservice": microservice, "cutoff": cutoff}
        )

        # Собираем host-распределение в словарь {template_id: {host: count}}
        host_dist: dict[int, dict[str, int]] = defaultdict(dict)
        for row in host_result.mappings():
            host_dist[row["template_id"]][row["host"]] = row["cnt"]

        # Запрос 3: почасовая статистика по шаблонам для time_trend
        hourly_stmt = text("""
            SELECT
                l.template_id,
                date_trunc('hour', l.timestamp) AS hour,
                COUNT(*) AS cnt
            FROM logs l
            WHERE l.microservice = :microservice
              AND l.timestamp > :cutoff
              AND l.template_id IS NOT NULL
            GROUP BY l.template_id, date_trunc('hour', l.timestamp)
            ORDER BY l.template_id, hour
        """)

        hourly_result = await session.execute(
            hourly_stmt, {"microservice": microservice, "cutoff": cutoff}
        )

        # Собираем {template_id: {hour: count}}
        hourly_data: dict[int, dict[datetime, int]] = defaultdict(dict)
        for row in hourly_result.mappings():
            hourly_data[row["template_id"]][row["hour"]] = row["cnt"]

        # Объединяем
        result = []
        for row in main_rows:
            tid = row["template_id"]

            # Строим упорядоченный список почасовых счётчиков
            hourly_map = hourly_data.get(tid, {})
            if hourly_map:
                hours_sorted = sorted(hourly_map.keys())
                hourly_counts = [hourly_map[h] for h in hours_sorted]
            else:
                hourly_counts = []

            result.append({
                "template_id": tid,
                "template_text": row["template_text"],
                "embedding": row["embedding"],
                "stacktrace_pattern": row["stacktrace_pattern"],
                "log_count": row["log_count"],
                "error_count": row["error_count"],
                "host_distribution": host_dist.get(tid, {}),
                "hourly_counts": hourly_counts,
            })

        return result

    def _build_cluster_info(
        self,
        cluster_id: int,
        indices: np.ndarray,
        template_texts: list[str],
        log_counts: list[int],
        error_counts: list[int],
        host_maps: list[dict],
        stacktrace_patterns: list[str | None],
        hourly_counts_list: list[list[int]],
        total_logs: int,
        all_template_texts: list[str],
        all_labels: np.ndarray,
    ) -> dict:
        """Построить описание одного кластера."""
        cluster_logs = sum(log_counts[i] for i in indices)
        percentage = (cluster_logs / total_logs * 100) if total_logs > 0 else 0

        # Топ шаблонов по частоте
        templates_with_counts = sorted(
            [
                {
                    "template_text": template_texts[i],
                    "log_count": log_counts[i],
                    "stacktrace_pattern": stacktrace_patterns[i],
                }
                for i in indices
            ],
            key=lambda x: x["log_count"],
            reverse=True,
        )

        # Level distribution — система работает только с ERROR
        total_errors = sum(error_counts[i] for i in indices)
        level_dist = {"ERROR": 1.0} if total_errors > 0 else {}

        # Host distribution
        merged_hosts: Counter = Counter()
        for i in indices:
            if isinstance(host_maps[i], dict):
                for host, count in host_maps[i].items():
                    merged_hosts[host] += count

        # Keywords через c-TF-IDF
        keywords = self._extract_keywords(
            cluster_id, all_template_texts, all_labels
        )

        # Time trend: агрегируем hourly_counts по шаблонам кластера
        merged_hourly: Counter = Counter()
        for i in indices:
            for hour_idx, cnt in enumerate(hourly_counts_list[i]):
                merged_hourly[hour_idx] += cnt
        if merged_hourly:
            max_hour = max(merged_hourly.keys())
            aggregated_hourly = [merged_hourly.get(h, 0) for h in range(max_hour + 1)]
        else:
            aggregated_hourly = []
        time_trend = self._compute_time_trend(aggregated_hourly)

        return {
            "cluster_id": cluster_id,
            "size": cluster_logs,
            "percentage": round(percentage, 2),
            "keywords": keywords,
            "top_templates": templates_with_counts[:5],
            "level_distribution": level_dist,
            "host_distribution": dict(merged_hosts.most_common(10)),
            "time_trend": time_trend,
        }

    def _build_unclustered_info(
        self,
        indices: np.ndarray,
        template_texts: list[str],
        log_counts: list[int],
        stacktrace_patterns: list[str | None],
        total_logs: int,
    ) -> dict:
        """Построить описание некластеризованных шаблонов."""
        noise_logs = sum(log_counts[i] for i in indices)
        percentage = (noise_logs / total_logs * 100) if total_logs > 0 else 0

        templates = [
            {
                "template_text": template_texts[i],
                "log_count": log_counts[i],
                "stacktrace_pattern": stacktrace_patterns[i],
            }
            for i in indices
        ]

        return {
            "size": noise_logs,
            "percentage": round(percentage, 2),
            "templates": sorted(templates, key=lambda x: x["log_count"], reverse=True)[
                :10
            ],
        }

    @staticmethod
    def _compute_time_trend(
        hourly_counts: list[int],
    ) -> str:
        """
        Определить тренд по почасовым счётчикам логов.

        Делит период на две половины и сравнивает средние.
        Returns: 'increasing', 'decreasing', 'stable', или 'spike'.
        """
        if not hourly_counts or len(hourly_counts) < 2:
            return "stable"

        mid = len(hourly_counts) // 2
        first_half = hourly_counts[:mid] or [0]
        second_half = hourly_counts[mid:] or [0]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        # Проверяем на spike: последний час резко выше среднего
        total_avg = sum(hourly_counts) / len(hourly_counts)
        if total_avg > 0 and hourly_counts[-1] > total_avg * 3:
            return "spike"

        if avg_first == 0 and avg_second == 0:
            return "stable"

        base = max(avg_first, avg_second, 1)
        ratio = (avg_second - avg_first) / base

        if ratio > 0.25:
            return "increasing"
        elif ratio < -0.25:
            return "decreasing"
        return "stable"

    @staticmethod
    def _extract_keywords(
        target_cluster_id: int,
        all_texts: list[str],
        all_labels: np.ndarray,
        top_n: int = 10,
    ) -> list[str]:
        """
        Извлечь ключевые слова кластера через c-TF-IDF.

        Объединяем шаблоны каждого кластера в один «документ»,
        строим TF-IDF и берём топ-N слов целевого кластера.
        """
        # Собираем «документы» по кластерам
        cluster_docs: dict[int, str] = defaultdict(str)
        for text_val, label in zip(all_texts, all_labels):
            cluster_docs[int(label)] += " " + text_val

        if target_cluster_id not in cluster_docs:
            return []

        # Все документы кластеров (кроме noise=-1)
        cluster_ids_sorted = sorted(
            cid for cid in cluster_docs if cid != -1
        )
        if target_cluster_id not in cluster_ids_sorted:
            return []

        documents = [cluster_docs[cid] for cid in cluster_ids_sorted]
        target_idx = cluster_ids_sorted.index(target_cluster_id)

        if len(documents) < 1:
            return []

        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words="english",
                token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b",
            )
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()

            scores = tfidf_matrix[target_idx].toarray().flatten()
            top_indices = scores.argsort()[::-1][:top_n]

            return [feature_names[i] for i in top_indices if scores[i] > 0]
        except ValueError:
            return []

    async def _save_result(
        self,
        session: AsyncSession,
        microservice: str,
        period: str,
        result: dict,
    ) -> None:
        """Сохранить результат кластеризации в PostgreSQL."""
        record = ClusterResult(
            microservice=microservice,
            period=period,
            num_clusters=result["num_clusters"],
            noise_ratio=result["noise_ratio"],
            silhouette_score=result.get("silhouette_score"),
            result_data=result,
        )
        session.add(record)
        await session.flush()