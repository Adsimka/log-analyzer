#!/usr/bin/env python3
"""
Генератор тестовых логов для проверки системы.

Использование:
    python scripts/send_test_logs.py                    # 200 логов, 2 сервиса
    python scripts/send_test_logs.py --count 1000       # 1000 логов
    python scripts/send_test_logs.py --url http://host:8000  # другой хост
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError

# ──────────────────────────────────────────────
# Шаблоны реалистичных лог-сообщений
# ──────────────────────────────────────────────

PAYMENT_ERRORS = [
    "Database connection to 192.168.1.{ip} failed after {ms}ms timeout",
    "Payment processing failed for order {uuid}: insufficient funds",
    "Timeout waiting for payment gateway response on port {port}",
    "Failed to serialize transaction {uuid} to database",
    "Connection pool exhausted: max connections {num} reached",
    "SSL handshake failed with payment provider at 10.0.{ip}.{ip2}",
    "Duplicate transaction detected: order {uuid}",
    "Currency conversion error: unsupported currency pair",
    "Rate limiter triggered: {num} requests per second exceeded",
    "Failed to decrypt payment token: invalid key format",
]

AUTH_ERRORS = [
    "Authentication failed for user {user}: invalid credentials",
    "JWT token expired for session {uuid}",
    "LDAP connection to 10.0.1.{ip} timed out after {ms}ms",
    "OAuth2 callback failed: state mismatch for provider google",
    "Brute force detection: {num} failed attempts from 192.168.{ip}.{ip2}",
    "Session store unavailable: Redis connection refused on port {port}",
    "MFA verification failed: TOTP code expired",
    "SAML assertion validation failed: signature mismatch",
    "API key {uuid} revoked but still in use",
    "Password hash migration error: unsupported algorithm bcrypt_v{num}",
]

ORDER_ERRORS = [
    "Inventory check failed for product {uuid}: service unavailable",
    "Order {uuid} stuck in processing state for {num} minutes",
    "Shipping calculation error: invalid postal code format",
    "Email notification failed for order {uuid}: SMTP connection refused",
    "Elasticsearch index order_events unavailable on node 10.0.2.{ip}",
    "Cart merge conflict: concurrent modification for user {user}",
    "Discount code validation timeout after {ms}ms",
    "Order export to ERP system failed: connection reset by peer",
    "Price calculation overflow for bulk order: {num} items",
    "Webhook delivery failed to https://partner.example.com/hook: HTTP {status}",
]

SERVICE_CONFIGS = {
    "payment-service": {
        "templates": PAYMENT_ERRORS,
        "hosts": ["pay-host-1", "pay-host-2", "pay-host-3"],
        "stacktraces": [
            "at com.app.payment.DatabaseConnection.connect(DatabaseConnection.java:132)",
            "at com.app.payment.TransactionProcessor.process(TransactionProcessor.java:87)",
            "at com.app.payment.GatewayClient.send(GatewayClient.java:201)",
            None,
        ],
    },
    "auth-service": {
        "templates": AUTH_ERRORS,
        "hosts": ["auth-host-1", "auth-host-2"],
        "stacktraces": [
            "at com.app.auth.TokenValidator.validate(TokenValidator.java:56)",
            "at com.app.auth.SessionManager.create(SessionManager.java:143)",
            None,
            None,
        ],
    },
    "order-service": {
        "templates": ORDER_ERRORS,
        "hosts": ["order-host-1", "order-host-2", "order-host-3", "order-host-4"],
        "stacktraces": [
            "at com.app.order.InventoryClient.check(InventoryClient.java:78)",
            "at com.app.order.OrderStateMachine.transition(OrderStateMachine.java:215)",
            None,
            None,
            None,
        ],
    },
}


def _random_uuid() -> str:
    return "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}".format(
        random.randint(0, 0xFFFFFFFF),
        random.randint(0, 0xFFFF),
        random.randint(0, 0xFFFF),
        random.randint(0, 0xFFFF),
        random.randint(0, 0xFFFFFFFFFFFF),
    )


def _fill_template(template: str) -> str:
    """Заполнить шаблон случайными значениями."""
    return (
        template
        .replace("{ip}", str(random.randint(1, 254)))
        .replace("{ip2}", str(random.randint(1, 254)))
        .replace("{ms}", str(random.randint(100, 30000)))
        .replace("{port}", str(random.choice([3306, 5432, 6379, 8080, 8443, 9200])))
        .replace("{uuid}", _random_uuid())
        .replace("{num}", str(random.randint(1, 9999)))
        .replace("{user}", random.choice(["alice", "bob", "charlie", "admin", "service-account"]))
        .replace("{status}", str(random.choice([500, 502, 503, 504, 429])))
    )


def generate_logs(count: int, services: list[str] | None = None) -> list[dict]:
    """Сгенерировать батч тестовых логов."""
    if services is None:
        services = list(SERVICE_CONFIGS.keys())

    now = datetime.now(timezone.utc)
    logs = []

    for _ in range(count):
        service_name = random.choice(services)
        config = SERVICE_CONFIGS[service_name]

        template = random.choice(config["templates"])
        message = _fill_template(template)

        level = "ERROR"

        # Время в пределах последнего часа
        offset = timedelta(seconds=random.randint(0, 3600))
        timestamp = now - offset

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "level": level,
            "message": message,
            "microservice": service_name,
            "host": random.choice(config["hosts"]),
        }

        stacktrace = random.choice(config["stacktraces"])
        if stacktrace:
            log_entry["stacktrace"] = stacktrace

        logs.append(log_entry)

    return logs


def send_batch(url: str, logs: list[dict]) -> dict:
    """Отправить батч логов на сервер."""
    payload = json.dumps({"logs": logs}).encode("utf-8")

    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Отправка тестовых логов")
    parser.add_argument("--url", default="http://localhost:8000/api/v1/ingest")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--services",
        nargs="+",
        default=None,
        choices=list(SERVICE_CONFIGS.keys()),
    )
    args = parser.parse_args()

    print(f"Генерация {args.count} логов...")
    all_logs = generate_logs(args.count, args.services)

    # Отправляем батчами
    sent = 0
    for i in range(0, len(all_logs), args.batch_size):
        batch = all_logs[i : i + args.batch_size]
        try:
            result = send_batch(args.url, batch)
            sent += len(batch)
            print(
                f"  Батч {i // args.batch_size + 1}: "
                f"отправлено {len(batch)}, "
                f"обработано {result.get('total_processed', '?')}, "
                f"отфильтровано {result.get('total_filtered_out', '?')}"
            )
        except URLError as e:
            print(f"  Ошибка подключения: {e}", file=sys.stderr)
            print(f"  Убедитесь, что сервер запущен: docker compose up", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  Ошибка: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nГотово! Отправлено {sent} логов.")
    print("Кластеризация произойдёт автоматически в течение 5 минут.")
    print(f"Проверьте результаты: curl {args.url.replace('/ingest', '/services')}")


if __name__ == "__main__":
    main()