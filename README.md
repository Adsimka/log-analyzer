# Log Event Clustering System

Система кластеризации лог-событий на основе семантического сходства.

## Быстрый старт

### 1. Запуск

```bash
docker compose up --build
```

При первом запуске:
- Скачивается SBERT-модель (~80 МБ) — может занять 1–2 минуты.
- Автоматически создаются таблицы в PostgreSQL.
- Модель кэшируется в Docker volume — повторные запуски быстрые.

Дождитесь в логах строки:
```
app_ready  accepted_levels=['ERROR', 'WARN']  sbert_model=all-MiniLM-L6-v2
```

### 2. Проверка работоспособности

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/api/v1/status
```

### 3. Отправка тестовых логов

```bash
python scripts/send_test_logs.py
# или с параметрами:
python scripts/send_test_logs.py --count 500 --batch-size 100
```

### 4. Просмотр результатов

Кластеризация запускается автоматически каждые 5 минут.

```bash
# Список сервисов
curl http://localhost:8000/api/v1/services

# Результаты кластеризации
curl "http://localhost:8000/api/v1/results/payment-service?period=1h"

# Детали конкретного кластера
curl "http://localhost:8000/api/v1/results/payment-service/0?period=1h"
```

### 5. Остановка

```bash
docker compose down           # остановить
docker compose down -v        # остановить + удалить данные
```

## Архитектура

```
┌─────────────┐     ┌──────────────────────────────────────────┐
│  Сервисы    │     │           Docker Compose                 │
│  (логи)     │────→│  ┌─────────┐  ┌───────┐  ┌───────────┐   │
│             │     │  │ FastAPI │──│ Redis │──│ PostgreSQL│   │
│             │     │  └────┬────┘  └───────┘  └───────────┘   │
│             │     │       │       ┌──────────┐               │
│             │     │       └───────│  Worker  │               │
│             │     │               └──────────┘               │
│             │     └──────────────────────────────────────────┘
└─────────────┘
```

## Стек

| Компонент | Технология | Роль |
|---|---|---|
| API | FastAPI | Приём батчей, выдача результатов |
| Парсинг | Drain3 | Логи → шаблоны |
| Векторизация | Sentence Transformers | Шаблоны → эмбеддинги |
| Снижение размерности | UMAP | Подготовка к кластеризации |
| Кластеризация | HDBSCAN | Группировка по семантике |
| Интерпретация | c-TF-IDF | Ключевые слова кластеров |
| Хранилище | PostgreSQL | Логи, шаблоны, результаты |
| Кэш / брокер | Redis | Эмбеддинги, Drain3 state, задачи |
| Фоновые задачи | ARQ | Периодическая кластеризация |

## API

| Метод | Эндпоинт | Описание |
|---|---|---|
| POST | `/api/v1/ingest` | Приём батча логов |
| GET | `/api/v1/services` | Список сервисов с данными |
| GET | `/api/v1/results/{service}?period=24h` | Результаты кластеризации |
| GET | `/api/v1/results/{service}/{cluster_id}?period=24h` | Детали кластера |
| GET | `/api/v1/status` | Состояние системы |
| GET | `/health` | Health check |

## Конфигурация

Все параметры настраиваются через `.env`:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `SBERT_MODEL_NAME` | `all-MiniLM-L6-v2` | SBERT-модель |
| `ACCEPTED_LOG_LEVELS` | `ERROR,WARN` | Принимаемые уровни |
| `CLUSTERING_INTERVAL_SECONDS` | `300` | Интервал кластеризации |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `3` | Мин. размер кластера |
| `UMAP_N_COMPONENTS` | `10` | Размерность после UMAP |