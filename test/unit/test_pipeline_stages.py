"""
Unit-тесты для каждого этапа ML-пайплайна кластеризации.

Тестирует Drain3, SBERT, HDBSCAN и полный E2E pipeline
на небольшом фиксированном наборе из 21 лога (3 группы по 7).
Работает полностью офлайн — без Docker, Redis, PostgreSQL.
"""

import numpy as np
import pytest

# ─── Тестовые данные: 3 семантические группы по 7 сообщений ───

GROUP_DB = [
    "Database connection to 192.168.1.10 failed after 5000ms timeout",
    "Database connection to 10.0.0.5 failed after 3000ms timeout",
    "Unable to establish connection to PostgreSQL at 172.16.0.1:5432",
    "Connection to database server 10.0.0.3 refused",
    "TCP connection to database host 192.168.1.20:5432 timed out after 8000ms",
    "Failed to acquire database connection from pool within 4000ms",
    "Lost connection to database server at 10.0.1.15 during query",
]

GROUP_AUTH = [
    "Authentication failed for user admin from IP 203.0.113.5",
    "Invalid credentials for user root, attempt 3 of 5",
    "Authentication token expired for user service-account-1",
    "Login attempt failed for user john@example.com from 198.51.100.1",
    "Access denied: invalid password for user deploy-bot",
    "Failed authentication: user analyst account is locked after 5 failed attempts",
    "JWT token validation failed for user api-client-2, token expired",
]

GROUP_TIMEOUT = [
    "Request to payment-service timed out after 30000ms",
    "API call to order-service exceeded timeout of 15000ms",
    "HTTP request to inventory-api failed: read timeout after 20000ms",
    "Upstream service notification-service did not respond within 10000ms",
    "Service call to shipping-api timed out after 25000ms",
    "Request timeout: billing-service did not respond in 12000ms",
    "Connection to external API gateway timed out after 45000ms",
]

ALL_MESSAGES = GROUP_DB + GROUP_AUTH + GROUP_TIMEOUT
# Ground truth: 0=DB, 1=AUTH, 2=TIMEOUT
GT_LABELS = [0] * 7 + [1] * 7 + [2] * 7


# ════════════════════════════════════════════════════════════
#  ЭТАП 1: Drain3 — нормализация и шаблонизация
# ════════════════════════════════════════════════════════════


class TestDrain3Parsing:
    """Тесты парсинга логов через Drain3."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Создать LogParserService с mock Redis (без реального подключения)."""
        from unittest.mock import MagicMock
        from src.services.log_parser import LogParserService

        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # нет сохранённого состояния
        self.parser = LogParserService(mock_redis)
        self.service = "test-service"

        # Парсим все сообщения
        messages = [(msg, None) for msg in ALL_MESSAGES]
        self.parsed = self.parser.parse_batch(self.service, messages)

    def test_all_templates_non_empty(self):
        """Каждый шаблон должен быть непустой строкой."""
        for p in self.parsed:
            assert p.template_text, f"Пустой шаблон для cluster_id={p.drain_cluster_id}"

    def test_ip_addresses_masked(self):
        """IP-адреса должны быть замаскированы."""
        for p in self.parsed:
            assert "192.168" not in p.template_text
            assert "10.0.0" not in p.template_text
            assert "172.16" not in p.template_text

    def test_numbers_masked(self):
        """Числа (таймауты, порты) должны быть замаскированы."""
        for p in self.parsed:
            # Конкретные числа из тестовых данных не должны присутствовать
            assert "5000" not in p.template_text
            assert "3000" not in p.template_text
            assert "30000" not in p.template_text

    def test_reasonable_template_count(self):
        """Количество уникальных шаблонов: от 3 (идеально) до 21 (без объединения)."""
        unique_ids = set(p.drain_cluster_id for p in self.parsed)
        assert 3 <= len(unique_ids) <= 21

    def test_similar_logs_share_template(self):
        """Похожие логи из одной группы должны частично объединяться."""
        db_ids = set(self.parsed[i].drain_cluster_id for i in range(7))
        # Хотя бы 2 лога DB-группы должны иметь общий шаблон
        assert len(db_ids) < 7, "Все DB-логи получили уникальные шаблоны, ожидалось объединение"

    def test_different_groups_different_templates(self):
        """Шаблоны разных групп не должны пересекаться."""
        db_ids = set(self.parsed[i].drain_cluster_id for i in range(7))
        auth_ids = set(self.parsed[i].drain_cluster_id for i in range(7, 14))
        timeout_ids = set(self.parsed[i].drain_cluster_id for i in range(14, 21))

        assert not db_ids & auth_ids, "DB и AUTH шаблоны пересекаются"
        assert not db_ids & timeout_ids, "DB и TIMEOUT шаблоны пересекаются"
        assert not auth_ids & timeout_ids, "AUTH и TIMEOUT шаблоны пересекаются"


# ════════════════════════════════════════════════════════════
#  ЭТАП 2: SBERT — семантическая векторизация
# ════════════════════════════════════════════════════════════


class TestSBERTEmbedding:
    """Тесты семантической векторизации через Sentence-BERT."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Загрузить SBERT и вычислить эмбеддинги для шаблонов."""
        from unittest.mock import MagicMock
        from src.services.embedding import EmbeddingService

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        self.embedder = EmbeddingService(mock_redis)
        self.embedder.load_model()

        # Используем репрезентативные шаблоны (не сырые сообщения)
        self.db_texts = [
            "Database connection to <IP> failed after <NUM>ms timeout",
            "Connection to database server <IP> refused",
            "TCP connection to database host <IP>:<PORT> timed out after <NUM>ms",
        ]
        self.auth_texts = [
            "Authentication failed for user <*> from IP <IP>",
            "Invalid credentials for user <*>, attempt <NUM> of <NUM>",
            "Access denied: invalid password for user <*>",
        ]
        self.timeout_texts = [
            "Request to <*> timed out after <NUM>ms",
            "API call to <*> exceeded timeout of <NUM>ms",
            "HTTP request to <*> failed: read timeout after <NUM>ms",
        ]

        all_texts = self.db_texts + self.auth_texts + self.timeout_texts
        self.embeddings = self.embedder.encode(all_texts)

    def test_embedding_dimension(self):
        """Размерность эмбеддинга должна быть 384."""
        assert self.embeddings.shape[1] == 384

    def test_l2_normalized(self):
        """Каждый вектор должен быть L2-нормализован (||v|| ≈ 1.0)."""
        norms = np.linalg.norm(self.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_intra_group_similarity_higher(self):
        """Средняя внутригрупповая похожесть должна быть выше межгрупповой."""
        db_emb = self.embeddings[:3]
        auth_emb = self.embeddings[3:6]
        timeout_emb = self.embeddings[6:9]

        # Внутригрупповые
        intra_db = np.mean(db_emb @ db_emb.T)
        intra_auth = np.mean(auth_emb @ auth_emb.T)
        intra_timeout = np.mean(timeout_emb @ timeout_emb.T)
        avg_intra = (intra_db + intra_auth + intra_timeout) / 3

        # Межгрупповые
        inter_db_auth = np.mean(db_emb @ auth_emb.T)
        inter_db_timeout = np.mean(db_emb @ timeout_emb.T)
        inter_auth_timeout = np.mean(auth_emb @ timeout_emb.T)
        avg_inter = (inter_db_auth + inter_db_timeout + inter_auth_timeout) / 3

        assert avg_intra > avg_inter, (
            f"Intra-group similarity ({avg_intra:.3f}) должна быть выше "
            f"inter-group ({avg_inter:.3f})"
        )

    def test_minimum_intra_group_similarity(self):
        """Минимальная внутригрупповая similarity > 0.4."""
        for i, group in enumerate([
            self.embeddings[:3], self.embeddings[3:6], self.embeddings[6:9]
        ]):
            sim_matrix = group @ group.T
            # Убираем диагональ
            np.fill_diagonal(sim_matrix, 0)
            min_sim = sim_matrix[sim_matrix > 0].min()
            assert min_sim > 0.4, (
                f"Группа {i}: min similarity = {min_sim:.3f}, ожидается > 0.4"
            )

    def test_significant_difference(self):
        """Разница между intra и inter должна быть значимой (> 0.05)."""
        db_emb = self.embeddings[:3]
        auth_emb = self.embeddings[3:6]

        intra = np.mean(db_emb @ db_emb.T)
        inter = np.mean(db_emb @ auth_emb.T)

        assert intra - inter > 0.05


# ════════════════════════════════════════════════════════════
#  ЭТАП 3: HDBSCAN — кластеризация
# ════════════════════════════════════════════════════════════


class TestHDBSCANClustering:
    """Тесты кластеризации HDBSCAN на SBERT-эмбеддингах."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Подготовить эмбеддинги и запустить HDBSCAN."""
        import hdbscan
        from unittest.mock import MagicMock
        from src.services.embedding import EmbeddingService

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        embedder = EmbeddingService(mock_redis)
        embedder.load_model()

        # Шаблоны (по 4 от каждой группы)
        texts = [
            "Database connection to <IP> failed after <NUM>ms timeout",
            "Connection to database server <IP> refused",
            "TCP connection to database host <IP>:<PORT> timed out",
            "Failed to acquire database connection from pool within <NUM>ms",
            "Authentication failed for user <*> from IP <IP>",
            "Invalid credentials for user <*> attempt <NUM> of <NUM>",
            "Access denied invalid password for user <*>",
            "JWT token validation failed for user <*> token expired",
            "Request to <*> timed out after <NUM>ms",
            "API call to <*> exceeded timeout of <NUM>ms",
            "HTTP request to <*> failed read timeout after <NUM>ms",
            "Service call to <*> timed out after <NUM>ms",
        ]
        self.gt_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        embeddings = embedder.encode(texts)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        self.labels = clusterer.fit_predict(embeddings)
        self.num_clusters = len(set(self.labels) - {-1})

    def test_cluster_count_range(self):
        """Количество кластеров должно быть от 2 до 8."""
        assert 2 <= self.num_clusters <= 8, f"Кластеров: {self.num_clusters}"

    def test_noise_ratio_acceptable(self):
        """Доля шума должна быть < 30%."""
        noise = np.sum(self.labels == -1) / len(self.labels)
        assert noise < 0.30, f"Noise ratio: {noise:.2%}"

    def test_majority_groups_consistent(self):
        """Хотя бы 2 из 3 GT-групп должны быть согласованы (>60% в одном кластере)."""
        consistent = 0
        for gt_id in [0, 1, 2]:
            mask = self.gt_labels == gt_id
            group_labels = self.labels[mask]
            group_labels = group_labels[group_labels != -1]

            if len(group_labels) == 0:
                continue

            from collections import Counter
            most_common_count = Counter(group_labels).most_common(1)[0][1]
            if most_common_count / len(group_labels) >= 0.6:
                consistent += 1

        assert consistent >= 2, f"Только {consistent}/3 групп согласованы"

    def test_different_gt_groups_separated(self):
        """Разные GT-группы должны попадать преимущественно в разные кластеры."""
        from collections import Counter

        dominant_clusters = []
        for gt_id in [0, 1, 2]:
            mask = self.gt_labels == gt_id
            group_labels = self.labels[mask]
            group_labels = group_labels[group_labels != -1]
            if len(group_labels) > 0:
                dominant = Counter(group_labels).most_common(1)[0][0]
                dominant_clusters.append(dominant)

        # Хотя бы 2 группы должны иметь разные доминирующие кластеры
        assert len(set(dominant_clusters)) >= 2

    def test_purity_above_threshold(self):
        """Purity кластеризации > 0.7."""
        non_noise = self.labels != -1
        if not np.any(non_noise):
            pytest.skip("Все точки — шум")

        from collections import Counter
        total_correct = 0
        for cluster_id in set(self.labels[non_noise]):
            mask = self.labels == cluster_id
            gt_in_cluster = self.gt_labels[mask]
            most_common = Counter(gt_in_cluster.tolist()).most_common(1)[0][1]
            total_correct += most_common

        purity = total_correct / np.sum(non_noise)
        assert purity > 0.7, f"Purity = {purity:.3f}, ожидается > 0.7"


# ════════════════════════════════════════════════════════════
#  ЭТАП 4: c-TF-IDF — извлечение ключевых слов
# ════════════════════════════════════════════════════════════


class TestKeywordExtraction:
    """Тесты извлечения ключевых слов через c-TF-IDF."""

    def test_keywords_for_cluster(self):
        """Ключевые слова должны быть релевантны кластеру."""
        from src.services.clustering import ClusteringService

        texts = [
            "Database connection to <IP> failed after <NUM>ms timeout",
            "Connection to database server <IP> refused",
            "Authentication failed for user <*> from IP <IP>",
            "Invalid credentials for user <*>",
        ]
        labels = np.array([0, 0, 1, 1])

        keywords_db = ClusteringService._extract_keywords(0, texts, labels)
        keywords_auth = ClusteringService._extract_keywords(1, texts, labels)

        assert len(keywords_db) > 0, "Нет ключевых слов для DB-кластера"
        assert len(keywords_auth) > 0, "Нет ключевых слов для AUTH-кластера"

        # DB-кластер должен содержать слова про базу данных
        db_words = set(w.lower() for w in keywords_db)
        assert db_words & {"database", "connection", "server"}, (
            f"DB keywords не содержат ожидаемых слов: {keywords_db}"
        )

    def test_empty_cluster_returns_empty(self):
        """Для несуществующего кластера — пустой список."""
        from src.services.clustering import ClusteringService

        texts = ["test message one", "test message two"]
        labels = np.array([0, 0])

        keywords = ClusteringService._extract_keywords(99, texts, labels)
        assert keywords == []


# ════════════════════════════════════════════════════════════
#  ЭТАП 5: Time Trend — анализ трендов
# ════════════════════════════════════════════════════════════


class TestTimeTrend:
    """Тесты анализа временных трендов."""

    def test_increasing_trend(self):
        from src.services.clustering import ClusteringService
        counts = [1, 2, 3, 5, 8, 12, 15, 20]
        assert ClusteringService._compute_time_trend(counts) == "increasing"

    def test_decreasing_trend(self):
        from src.services.clustering import ClusteringService
        counts = [20, 15, 12, 8, 5, 3, 2, 1]
        assert ClusteringService._compute_time_trend(counts) == "decreasing"

    def test_stable_trend(self):
        from src.services.clustering import ClusteringService
        counts = [10, 10, 10, 10, 10, 10]
        assert ClusteringService._compute_time_trend(counts) == "stable"

    def test_spike_trend(self):
        from src.services.clustering import ClusteringService
        counts = [5, 5, 5, 5, 5, 5, 5, 100]
        assert ClusteringService._compute_time_trend(counts) == "spike"

    def test_empty_returns_stable(self):
        from src.services.clustering import ClusteringService
        assert ClusteringService._compute_time_trend([]) == "stable"


# ════════════════════════════════════════════════════════════
#  FULL E2E Pipeline (офлайн)
# ════════════════════════════════════════════════════════════


class TestFullPipelineE2E:
    """Полный E2E-тест: Drain3 → SBERT → HDBSCAN."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Прогнать весь pipeline офлайн."""
        import hdbscan
        from unittest.mock import MagicMock
        from src.services.log_parser import LogParserService
        from src.services.embedding import EmbeddingService

        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        # Drain3
        parser = LogParserService(mock_redis)
        messages = [(msg, None) for msg in ALL_MESSAGES]
        parsed = parser.parse_batch("e2e-test", messages)

        # Дедупликация шаблонов
        unique_templates: dict[int, str] = {}
        for p in parsed:
            unique_templates[p.drain_cluster_id] = p.template_text

        self.num_templates = len(unique_templates)

        # SBERT
        embedder = EmbeddingService(mock_redis)
        embedder.load_model()
        texts = list(unique_templates.values())
        embeddings = embedder.encode(texts)
        self.embeddings = embeddings

        # HDBSCAN
        if len(embeddings) >= 2:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=2,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            self.labels = clusterer.fit_predict(embeddings)
        else:
            self.labels = np.array([-1] * len(embeddings))

        self.num_clusters = len(set(self.labels) - {-1})

    def test_drain3_produces_templates(self):
        """Drain3 должен создать >= 3 уникальных шаблонов."""
        assert self.num_templates >= 3

    def test_sbert_creates_embeddings(self):
        """SBERT должен создать эмбеддинги для всех шаблонов."""
        assert self.embeddings.shape[0] == self.num_templates
        assert self.embeddings.shape[1] == 384

    def test_hdbscan_finds_clusters(self):
        """HDBSCAN должен найти >= 2 кластеров."""
        assert self.num_clusters >= 2, f"Найдено кластеров: {self.num_clusters}"

    def test_noise_not_dominant(self):
        """Шум не должен доминировать (< 50%)."""
        if len(self.labels) == 0:
            pytest.skip("Нет данных")
        noise_ratio = np.sum(self.labels == -1) / len(self.labels)
        assert noise_ratio < 0.5, f"Слишком много шума: {noise_ratio:.2%}"


# ════════════════════════════════════════════════════════════
#  Вспомогательные тесты: сериализация эмбеддингов
# ════════════════════════════════════════════════════════════


class TestEmbeddingSerialization:
    """Тесты сериализации/десериализации numpy-массивов."""

    def test_roundtrip(self):
        """Сериализация → десериализация должна быть идентичной."""
        from src.services.embedding import EmbeddingService

        original = np.random.rand(384).astype(np.float32)
        serialized = EmbeddingService.serialize_embedding(original)
        deserialized = EmbeddingService.deserialize_embedding(serialized)

        np.testing.assert_array_almost_equal(original, deserialized)

    def test_serialized_is_bytes(self):
        """Сериализованный эмбеддинг должен быть bytes."""
        from src.services.embedding import EmbeddingService

        emb = np.zeros(384, dtype=np.float32)
        result = EmbeddingService.serialize_embedding(emb)
        assert isinstance(result, bytes)
