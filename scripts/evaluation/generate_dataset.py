#!/usr/bin/env python3
"""
Генератор синтетического датасета с ground truth разметкой.

Каждый лог привязан к семантическому кластеру (ground_truth_cluster).
Это позволяет объективно оценить качество кластеризации через
внешние метрики: ARI, NMI, V-measure, Homogeneity, Completeness.

Генерирует ~5000 логов от одного микросервиса (для чистоты эксперимента)
с известной кластерной структурой.

Использование:
    python scripts/evaluation/generate_dataset.py
    python scripts/evaluation/generate_dataset.py --count 10000 --noise 0.05
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ═══════════════════════════════════════════════════════
#  GROUND TRUTH КЛАСТЕРЫ
#
#  Каждый кластер — семантическая группа ошибок.
#  Шаблоны внутри одного кластера описывают одну и ту же
#  проблему, но разными словами / с разными деталями.
#  Система должна объединить их в один кластер.
# ═══════════════════════════════════════════════════════

GROUND_TRUTH_CLUSTERS = {
    # ─── Кластер 0: Проблемы подключения к базе данных ───
    "db_connection": {
        "id": 0,
        "description": "Database connectivity issues",
        "templates": [
            "Database connection to {ip} failed after {ms}ms timeout",
            "Unable to establish connection to PostgreSQL at {ip}:{port}",
            "Connection to database server {ip} refused",
            "Database connection pool exhausted, {num} active connections",
            "Lost connection to MySQL server at {ip} during query",
            "TCP connection to database host {ip}:{port} timed out after {ms}ms",
            "Failed to acquire database connection from pool within {ms}ms",
            "Database server at {ip} is not accepting connections",
            "Connection reset by database server {ip} during transaction",
            "FATAL: too many connections for database at {ip}:{port}",
        ],
        "weight": 0.18,
        "stacktraces": [
            "at com.app.db.ConnectionPool.acquire(ConnectionPool.java:{line})",
            "at com.app.db.DatabaseDriver.connect(DatabaseDriver.java:{line})",
            "at com.app.repository.BaseRepository.getConnection(BaseRepository.java:{line})",
        ],
    },

    # ─── Кластер 1: Аутентификация и авторизация ───
    "auth_failure": {
        "id": 1,
        "description": "Authentication and authorization failures",
        "templates": [
            "Authentication failed for user {user}: invalid credentials",
            "Login attempt rejected for user {user} from {ip}",
            "Invalid password provided for account {user}",
            "User {user} authentication denied: account locked after {num} attempts",
            "Access denied for user {user}: insufficient permissions",
            "Authorization check failed for user {user} on resource /api/{path}",
            "Bearer token validation failed: signature mismatch",
            "JWT token for user {user} has expired",
            "OAuth2 access token rejected: invalid scope",
            "SAML assertion for user {user} is not valid",
        ],
        "weight": 0.15,
        "stacktraces": [
            "at com.app.auth.AuthenticationService.authenticate(AuthenticationService.java:{line})",
            "at com.app.auth.TokenValidator.validate(TokenValidator.java:{line})",
            "at com.app.security.AccessControl.checkPermission(AccessControl.java:{line})",
        ],
    },

    # ─── Кластер 2: Таймауты HTTP / внешних API ───
    "api_timeout": {
        "id": 2,
        "description": "External API and HTTP timeouts",
        "templates": [
            "HTTP request to {url} timed out after {ms}ms",
            "Timeout waiting for response from external service at {url}",
            "API call to {url} exceeded deadline of {ms}ms",
            "Connection to upstream service {url} timed out",
            "Read timeout after {ms}ms waiting for response from {url}",
            "Gateway timeout: upstream server {url} did not respond within {ms}ms",
            "HTTP client timeout: {url} took longer than {ms}ms",
            "External API request to {url} failed: socket timeout after {ms}ms",
            "Service call to {url} aborted due to timeout ({ms}ms)",
            "Request to partner API at {url} timed out after {ms}ms",
        ],
        "weight": 0.14,
        "stacktraces": [
            "at com.app.http.HttpClient.execute(HttpClient.java:{line})",
            "at com.app.gateway.ApiGateway.sendRequest(ApiGateway.java:{line})",
            "at com.app.client.ExternalServiceClient.call(ExternalServiceClient.java:{line})",
        ],
    },

    # ─── Кластер 3: Ошибки обработки платежей ───
    "payment_error": {
        "id": 3,
        "description": "Payment processing errors",
        "templates": [
            "Payment processing failed for order {uuid}: insufficient funds",
            "Transaction {uuid} declined by payment gateway: card expired",
            "Failed to charge card ending in {num} for amount {num}",
            "Payment gateway returned error for transaction {uuid}: fraud detected",
            "Refund processing failed for order {uuid}: original transaction not found",
            "Currency conversion failed for payment {uuid}: unsupported pair {currency}",
            "Payment authorization for order {uuid} timed out at gateway",
            "Duplicate payment detected for order {uuid}",
            "Payment processor returned decline code {num} for transaction {uuid}",
            "Unable to process payment for order {uuid}: gateway unavailable",
        ],
        "weight": 0.12,
        "stacktraces": [
            "at com.app.payment.PaymentProcessor.process(PaymentProcessor.java:{line})",
            "at com.app.payment.TransactionService.charge(TransactionService.java:{line})",
            "at com.app.payment.GatewayClient.authorize(GatewayClient.java:{line})",
        ],
    },

    # ─── Кластер 4: Ошибки сериализации / данных ───
    "data_error": {
        "id": 4,
        "description": "Data serialization and validation errors",
        "templates": [
            "Failed to deserialize JSON payload: unexpected token at position {num}",
            "Data validation failed: field {field} is required but missing",
            "Invalid input format: expected {type} but received {type}",
            "JSON parsing error in request body: malformed UTF-8 at byte {num}",
            "Schema validation failed: {field} does not match pattern",
            "Cannot deserialize value of type {type} from String {field}",
            "Request body contains invalid field: {field} exceeds max length {num}",
            "Data integrity error: duplicate key value for field {field}",
            "Serialization error: circular reference detected in object {field}",
            "XML parsing failed: unexpected element {field} at line {num}",
        ],
        "weight": 0.10,
        "stacktraces": [
            "at com.app.serializer.JsonParser.parse(JsonParser.java:{line})",
            "at com.app.validation.InputValidator.validate(InputValidator.java:{line})",
            "at com.app.data.DataTransformer.deserialize(DataTransformer.java:{line})",
        ],
    },

    # ─── Кластер 5: Проблемы с памятью / ресурсами ───
    "resource_error": {
        "id": 5,
        "description": "Memory and resource exhaustion",
        "templates": [
            "OutOfMemoryError: Java heap space exceeded {num}MB limit",
            "Thread pool exhausted: {num} threads active, queue full",
            "Memory usage critical: {num}% of available heap consumed",
            "GC overhead limit exceeded: spent {num}% of time in garbage collection",
            "Unable to allocate {num}MB of memory for request processing",
            "Resource limit reached: maximum {num} file descriptors in use",
            "CPU usage spike detected: {num}% utilization on worker thread pool",
            "Disk space critically low: only {num}MB remaining on /data volume",
            "Connection pool limit reached: {num} connections active, max {num} allowed",
            "Rate limit exceeded: {num} requests per second from {ip}",
        ],
        "weight": 0.08,
        "stacktraces": [
            "at com.app.runtime.MemoryManager.allocate(MemoryManager.java:{line})",
            "at com.app.pool.ThreadPoolExecutor.execute(ThreadPoolExecutor.java:{line})",
            "at com.app.monitor.ResourceMonitor.check(ResourceMonitor.java:{line})",
        ],
    },

    # ─── Кластер 6: Ошибки очереди сообщений ───
    "queue_error": {
        "id": 6,
        "description": "Message queue and event processing errors",
        "templates": [
            "Failed to publish message to Kafka topic {topic}: broker unavailable",
            "Message consumption failed on queue {topic}: deserialization error",
            "RabbitMQ connection to {ip}:{port} lost during message delivery",
            "Dead letter queue overflow: {num} unprocessed messages on {topic}",
            "Message delivery timeout on Kafka topic {topic} after {ms}ms",
            "Failed to acknowledge message on queue {topic}: channel closed",
            "Event processing failed for message {uuid} on topic {topic}",
            "Queue {topic} consumer lag exceeded {num} messages",
            "Message broker at {ip} returned error: partition {num} unavailable",
            "Failed to commit offset for consumer group on topic {topic}",
        ],
        "weight": 0.09,
        "stacktraces": [
            "at com.app.messaging.KafkaProducer.send(KafkaProducer.java:{line})",
            "at com.app.messaging.QueueConsumer.process(QueueConsumer.java:{line})",
            "at com.app.events.EventDispatcher.dispatch(EventDispatcher.java:{line})",
        ],
    },

    # ─── Кластер 7: Ошибки кэширования (Redis) ───
    "cache_error": {
        "id": 7,
        "description": "Cache and Redis errors",
        "templates": [
            "Redis connection to {ip}:{port} refused",
            "Cache lookup failed for key {field}: connection timeout after {ms}ms",
            "Redis cluster node at {ip} unreachable",
            "Cache invalidation failed: unable to delete key {field}",
            "Redis command timeout after {ms}ms for operation GET on key {field}",
            "Cache deserialization error: corrupted data for key {field}",
            "Redis sentinel failover in progress, {ip} is new master",
            "Memcached connection to {ip}:{port} failed",
            "Cache write failed: maximum memory {num}MB exceeded",
            "Redis replication lag detected: {num}ms behind master at {ip}",
        ],
        "weight": 0.07,
        "stacktraces": [
            "at com.app.cache.RedisClient.execute(RedisClient.java:{line})",
            "at com.app.cache.CacheManager.get(CacheManager.java:{line})",
            "at com.app.cache.DistributedCache.invalidate(DistributedCache.java:{line})",
        ],
    },

    # ─── Кластер 8: Ошибки файловой системы / хранилища ───
    "storage_error": {
        "id": 8,
        "description": "File system and object storage errors",
        "templates": [
            "Failed to upload file {field} to S3 bucket: access denied",
            "File not found: /data/uploads/{field} does not exist",
            "Storage write failed: insufficient disk space on volume /data",
            "S3 GetObject failed for key {field}: NoSuchKey",
            "File upload to cloud storage timed out after {ms}ms for {field}",
            "Permission denied when writing to /var/log/{field}",
            "Object storage request to {url} returned HTTP {num}",
            "Failed to read configuration file /etc/app/{field}: file corrupted",
            "Cloud storage bucket {field} is not accessible from current region",
            "File system operation failed: too many open files ({num} open)",
        ],
        "weight": 0.07,
        "stacktraces": [
            "at com.app.storage.S3Client.upload(S3Client.java:{line})",
            "at com.app.storage.FileManager.read(FileManager.java:{line})",
            "at com.app.storage.ObjectStore.get(ObjectStore.java:{line})",
        ],
    },
}


# ═══════════════════════════════════════════════════════
#  ГЕНЕРАЦИЯ
# ═══════════════════════════════════════════════════════

def _random_ip() -> str:
    return f"{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def _random_uuid() -> str:
    return "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}".format(
        random.randint(0, 0xFFFFFFFF),
        random.randint(0, 0xFFFF),
        random.randint(0, 0xFFFF),
        random.randint(0, 0xFFFF),
        random.randint(0, 0xFFFFFFFFFFFF),
    )


FIELDS = ["user_id", "email", "order_id", "session_id", "request_id", "config.yaml", "data.json"]
TYPES = ["Integer", "String", "Boolean", "Date", "Object", "Array"]
URLS = [
    "https://api.payment-provider.com/v1/charge",
    "https://inventory-service.internal:8080/api/stock",
    "https://notification-service.internal:8443/send",
    "https://partner.example.com/api/v2/orders",
    "https://analytics.internal:9090/events",
]
TOPICS = ["orders.created", "payments.processed", "users.updated", "events.dlq", "notifications.send"]
CURRENCIES = ["USD/BTC", "EUR/CNY", "GBP/JPY"]
USERS = ["alice", "bob", "charlie", "admin", "service-account", "john.doe", "jane.smith"]


def _fill_template(template: str) -> str:
    """Заполнить шаблон реалистичными случайными значениями."""
    result = template
    while "{ip}" in result:
        result = result.replace("{ip}", _random_ip(), 1)
    while "{uuid}" in result:
        result = result.replace("{uuid}", _random_uuid(), 1)
    while "{ms}" in result:
        result = result.replace("{ms}", str(random.randint(100, 30000)), 1)
    while "{port}" in result:
        result = result.replace("{port}", str(random.choice([3306, 5432, 6379, 8080, 8443, 9200])), 1)
    while "{num}" in result:
        result = result.replace("{num}", str(random.randint(1, 9999)), 1)
    while "{user}" in result:
        result = result.replace("{user}", random.choice(USERS), 1)
    while "{url}" in result:
        result = result.replace("{url}", random.choice(URLS), 1)
    while "{field}" in result:
        result = result.replace("{field}", random.choice(FIELDS), 1)
    while "{type}" in result:
        result = result.replace("{type}", random.choice(TYPES), 1)
    while "{topic}" in result:
        result = result.replace("{topic}", random.choice(TOPICS), 1)
    while "{line}" in result:
        result = result.replace("{line}", str(random.randint(10, 500)), 1)
    while "{currency}" in result:
        result = result.replace("{currency}", random.choice(CURRENCIES), 1)
    while "{path}" in result:
        result = result.replace("{path}", random.choice(["users", "orders", "payments", "admin"]), 1)
    return result


def generate_dataset(
    count: int = 5000,
    noise_ratio: float = 0.03,
    microservice: str = "eval-service",
) -> tuple[list[dict], list[dict]]:
    """
    Генерирует датасет с ground truth разметкой.

    Args:
        count: общее количество логов
        noise_ratio: доля "шумовых" логов (не принадлежат ни одному кластеру)
        microservice: имя микросервиса

    Returns:
        (logs_for_api, ground_truth_records)
        - logs_for_api: список логов для отправки в API
        - ground_truth_records: список с gt-разметкой
    """
    now = datetime.now(timezone.utc)
    hosts = [f"eval-host-{i}" for i in range(1, 6)]

    # Вычисляем количество логов на кластер по весам
    noise_count = int(count * noise_ratio)
    normal_count = count - noise_count

    cluster_names = list(GROUND_TRUTH_CLUSTERS.keys())
    weights = [GROUND_TRUTH_CLUSTERS[name]["weight"] for name in cluster_names]
    total_weight = sum(weights)
    weights_norm = [w / total_weight for w in weights]

    # Распределяем логи по кластерам
    cluster_counts = [int(normal_count * w) for w in weights_norm]
    # Добираем остаток
    remainder = normal_count - sum(cluster_counts)
    for i in range(remainder):
        cluster_counts[i % len(cluster_counts)] += 1

    logs_api = []
    ground_truth = []

    # Генерируем логи для каждого кластера
    for cluster_name, cluster_count in zip(cluster_names, cluster_counts):
        cluster_def = GROUND_TRUTH_CLUSTERS[cluster_name]

        for _ in range(cluster_count):
            template = random.choice(cluster_def["templates"])
            message = _fill_template(template)

            level = "ERROR"
            offset = timedelta(seconds=random.randint(0, 3600))
            timestamp = now - offset
            host = random.choice(hosts)

            stacktrace = None
            if cluster_def["stacktraces"] and random.random() < 0.7:
                stacktrace = _fill_template(random.choice(cluster_def["stacktraces"]))

            log_entry = {
                "timestamp": timestamp.isoformat(),
                "level": level,
                "message": message,
                "microservice": microservice,
                "host": host,
            }
            if stacktrace:
                log_entry["stacktrace"] = stacktrace

            gt_entry = {
                "message": message,
                "ground_truth_cluster_id": cluster_def["id"],
                "ground_truth_cluster_name": cluster_name,
                "template_base": template,
            }

            logs_api.append(log_entry)
            ground_truth.append(gt_entry)

    # Генерируем шумовые логи (уникальные, не из кластеров)
    noise_templates = [
        "Unexpected error code {num} in module {field}",
        "Unknown exception during scheduled maintenance task",
        "Configuration mismatch detected in deployment {uuid}",
        "Temporary network glitch at {ip}: packet loss {num}%",
        "Sporadic timeout in background health check to {url}",
        "Debug assertion failed at line {num} in module {field}",
        "Edge case triggered: negative value {num} in counter {field}",
        "Rare encoding error in legacy module processing {field}",
    ]

    for _ in range(noise_count):
        template = random.choice(noise_templates)
        message = _fill_template(template)
        level = "ERROR"
        offset = timedelta(seconds=random.randint(0, 3600))
        timestamp = now - offset

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "level": level,
            "message": message,
            "microservice": microservice,
            "host": random.choice(hosts),
        }

        gt_entry = {
            "message": message,
            "ground_truth_cluster_id": -1,  # noise
            "ground_truth_cluster_name": "noise",
            "template_base": template,
        }

        logs_api.append(log_entry)
        ground_truth.append(gt_entry)

    # Перемешиваем (сохраняя соответствие индексов)
    combined = list(zip(logs_api, ground_truth))
    random.shuffle(combined)
    logs_api, ground_truth = zip(*combined)

    return list(logs_api), list(ground_truth)


def main():
    parser = argparse.ArgumentParser(description="Генерация датасета с ground truth")
    parser.add_argument("--count", type=int, default=5000, help="Количество логов")
    parser.add_argument("--noise", type=float, default=0.03, help="Доля шумовых логов")
    parser.add_argument("--service", default="eval-service", help="Имя микросервиса")
    parser.add_argument("--output-dir", default="scripts/evaluation/data", help="Директория для выходных файлов")
    args = parser.parse_args()

    print(f"Генерация {args.count} логов (шум: {args.noise:.0%})...")
    logs, ground_truth = generate_dataset(args.count, args.noise, args.service)

    # Создаём директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем
    logs_path = output_dir / "eval_logs.json"
    gt_path = output_dir / "ground_truth.json"

    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump({"logs": logs}, f, indent=2, ensure_ascii=False)

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    # Статистика
    from collections import Counter
    cluster_dist = Counter(gt["ground_truth_cluster_name"] for gt in ground_truth)

    print(f"\nСохранено:")
    print(f"  Логи:         {logs_path} ({len(logs)} записей)")
    print(f"  Ground truth: {gt_path}")
    print(f"\nРаспределение по кластерам:")
    for name, cnt in sorted(cluster_dist.items(), key=lambda x: -x[1]):
        pct = cnt / len(ground_truth) * 100
        print(f"  {name:20s}: {cnt:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()