#!/usr/bin/env python3
"""
Загрузчик реальных лог-датасетов с Kaggle для тестирования системы кластеризации.

Поддерживаемые датасеты (проверенные, доступные на Kaggle):

1. LogHub BGL (BlueGene/L supercomputer, labeled)
   - https://www.kaggle.com/datasets/omduggineni/loghub-bgl-log-data
   - kaggle datasets download -d omduggineni/loghub-bgl-log-data

2. LogHub HDFS (Hadoop Distributed File System)
   - https://www.kaggle.com/datasets/omduggineni/loghub-hadoop-distributed-file-system-log-data
   - kaggle datasets download -d omduggineni/loghub-hadoop-distributed-file-system-log-data

3. LogHub Thunderbird (supercomputer, labeled)
   - https://www.kaggle.com/datasets/omduggineni/loghub-mozilla-thunderbird-log-data
   - kaggle datasets download -d omduggineni/loghub-mozilla-thunderbird-log-data

4. Structured BGL Logs CSV (готовый CSV)
   - https://www.kaggle.com/datasets/ayush2222/structured-bgl-logs-csv
   - kaggle datasets download -d ayush2222/structured-bgl-logs-csv

5. Server Logs (Apache-формат)
   - https://www.kaggle.com/datasets/vishnu0399/server-logs
   - kaggle datasets download -d vishnu0399/server-logs

6. Произвольные текстовые лог-файлы
   - Каждая строка — одно лог-сообщение

Также доступны напрямую с Zenodo (без Kaggle-аккаунта):
   wget https://zenodo.org/record/3227177/files/BGL.tar.gz
   wget https://zenodo.org/record/3227177/files/HDFS_1.tar.gz

Использование:
    # 1. Скачать датасет
    kaggle datasets download -d omduggineni/loghub-bgl-log-data -p data/ --unzip

    # 2. Загрузить в систему (по умолчанию — только ERROR)
    python scripts/load_kaggle_logs.py --input data/BGL.log_structured.csv \\
        --format loghub --service bgl-supercomputer --count 5000

    # 3. Или из текстового файла
    python scripts/load_kaggle_logs.py --input data/BGL.log \\
        --format thunderbird --service bgl --count 10000

    # 4. Dry-run для проверки парсинга
    python scripts/load_kaggle_logs.py --input data/BGL.log_structured.csv \\
        --format loghub --service bgl --dry-run

    # 5. Сохранить GT для оценки (если есть EventId/EventTemplate в CSV)
    python scripts/load_kaggle_logs.py --input data/BGL.log_structured.csv \\
        --format loghub --service bgl --save-ground-truth
"""

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ─── Парсеры для разных форматов ───

# Регулярное выражение для типичных лог-строк:
# "2024-01-01 12:00:00 ERROR [ServiceName] Message here"
_LOG_LINE_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"(?:(?P<level>TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL|SEVERE)\s+)?"
    r"(?:\[(?P<component>[^\]]+)\]\s*)?"
    r"(?P<message>.+)$"
)

# Thunderbird/BGL формат: "- AlertID Date Time ... Level ... Content"
_THUNDERBIRD_PATTERN = re.compile(
    r"^(?P<label>[-\w]+)\s+"
    r"(?P<id>\d+)\s+"
    r"(?P<date>\d{4}\.\d{2}\.\d{2})\s+"
    r"(?P<node>\S+)\s+"
    r"(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}[-T]\d{2}:\d{2}:\d{2})\s+"
    r".*?\s+(?P<level>INFO|WARNING|ERROR|FATAL|SEVERE|CRITICAL)\s+"
    r"(?P<message>.+)$"
)


def parse_loghub_csv(filepath: Path, count: int, errors_only: bool, save_gt: bool):
    """
    Парсить CSV из Loghub: колонки Content, Level/Label, EventId.

    BGL формат:
    LineId,Label,Timestamp,Date,Node,Time,NodeRepeat,Type,Component,Level,Content,EventId,EventTemplate

    В BGL колонка Label определяет аномалию:
      "-" = нормальное сообщение, всё остальное = alert/error.
    Колонка Level (INFO/FATAL) — это severity ОС-компонента, не уровень ошибки.
    """
    logs = []
    ground_truth = {}

    with open(filepath, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        # Определяем имена колонок (разные датасеты — разные колонки)
        fieldnames = reader.fieldnames or []
        content_col = _find_column(fieldnames, ["Content", "content", "Message", "message", "Log", "log_message"])
        label_col = _find_column(fieldnames, ["Label", "label"])
        level_col = _find_column(fieldnames, ["Level", "level", "Severity", "severity"])
        event_id_col = _find_column(fieldnames, ["EventId", "eventid", "event_id", "EventType", "event_type"])
        template_col = _find_column(fieldnames, ["EventTemplate", "event_template", "Template", "template"])
        date_col = _find_column(fieldnames, ["Time", "time", "Date", "date", "Timestamp", "timestamp"])
        component_col = _find_column(fieldnames, ["Component", "component", "Source", "source"])
        node_col = _find_column(fieldnames, ["Node", "node", "NodeRepeat"])

        if not content_col:
            print(f"ОШИБКА: Не найдена колонка с текстом лога. Доступные: {fieldnames}")
            sys.exit(1)

        # Определяем стратегию фильтрации:
        # Если есть Label (BGL/Thunderbird) — фильтруем по label ("-" = нет ошибки)
        # Иначе — фильтруем по Level
        use_label_strategy = label_col is not None

        if use_label_strategy:
            print(f"Обнаружена колонка Label — фильтрация по alert-меткам (BGL/Thunderbird формат)")
        else:
            print(f"Колонка Label не найдена — фильтрация по Level")

        for row_idx, row in enumerate(reader):
            message = row.get(content_col, "").strip()
            if not message:
                continue

            # Определяем level на основе стратегии
            if use_label_strategy:
                raw_label = row.get(label_col, "").strip()
                if raw_label == "-":
                    # Нормальное сообщение — не ошибка
                    if errors_only:
                        continue
                    level = "INFO"
                else:
                    # Alert — это ошибка
                    level = "ERROR"
            else:
                # Fallback: фильтрация по Level
                raw_level = row.get(level_col, "").strip().upper() if level_col else "ERROR"
                if raw_level in ("ERROR", "FATAL", "CRITICAL", "SEVERE", "ERR", "EMERG", "ALERT"):
                    level = "ERROR"
                elif raw_level in ("WARN", "WARNING"):
                    if errors_only:
                        continue
                    level = "WARN"
                elif raw_level in ("INFO", "DEBUG", "TRACE", "NOTICE"):
                    if errors_only:
                        continue
                    level = "INFO"
                else:
                    level = "ERROR"

            if errors_only and level != "ERROR":
                continue

            # Timestamp — пробуем Time (точный), потом Date
            ts = None
            if date_col:
                ts = _parse_timestamp(row.get(date_col, ""))
            if ts is None:
                ts = datetime.now(timezone.utc) - timedelta(
                    seconds=random.randint(0, 86400)
                )

            # Host: Node (BGL) или Component
            host = "unknown"
            if node_col:
                host = row.get(node_col, "unknown") or "unknown"
            elif component_col:
                host = row.get(component_col, "unknown") or "unknown"

            logs.append({
                "timestamp": ts.isoformat(),
                "level": level,
                "message": message[:10000],
                "microservice": None,  # будет установлен позже
                "host": host[:128] if host else "unknown",
                "stacktrace": None,
            })

            # Ground truth
            if save_gt and event_id_col:
                event_id = row.get(event_id_col, "")
                ground_truth[row_idx] = event_id

            if len(logs) >= count:
                break

    return logs, ground_truth


def parse_text_logs(filepath: Path, count: int, errors_only: bool):
    """Парсить плоский текстовый лог-файл (по строке на сообщение)."""
    logs = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = _LOG_LINE_PATTERN.match(line)
            if match:
                ts = _parse_timestamp(match.group("timestamp"))
                level = _normalize_level(match.group("level") or "ERROR")
                message = match.group("message")
                host = match.group("component") or "unknown"
            else:
                ts = datetime.now(timezone.utc) - timedelta(
                    seconds=random.randint(0, 86400)
                )
                level = _guess_level(line)
                message = line
                host = "unknown"

            if errors_only and level != "ERROR":
                continue

            logs.append({
                "timestamp": ts.isoformat() if ts else datetime.now(timezone.utc).isoformat(),
                "level": level,
                "message": message[:10000],
                "microservice": None,
                "host": host[:128],
                "stacktrace": None,
            })

            if len(logs) >= count:
                break

    return logs, {}


def parse_thunderbird(filepath: Path, count: int, errors_only: bool):
    """Парсить логи Thunderbird/BGL (суперкомпьютерные)."""
    logs = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = _THUNDERBIRD_PATTERN.match(line)
            if match:
                level = _normalize_level(match.group("level"))
                message = match.group("message")
                ts = _parse_timestamp(match.group("timestamp"))
                host = match.group("node")
            else:
                # Fallback: разбиваем на токены
                parts = line.split(None, 9)
                if len(parts) >= 10:
                    message = parts[-1]
                    level = _guess_level(line)
                    ts = None
                    host = parts[3] if len(parts) > 3 else "unknown"
                else:
                    message = line
                    level = _guess_level(line)
                    ts = None
                    host = "unknown"

            if errors_only and level != "ERROR":
                continue

            if ts is None:
                ts = datetime.now(timezone.utc) - timedelta(
                    seconds=random.randint(0, 86400)
                )

            logs.append({
                "timestamp": ts.isoformat(),
                "level": level,
                "message": message[:10000],
                "microservice": None,
                "host": host[:128],
                "stacktrace": None,
            })

            if len(logs) >= count:
                break

    return logs, {}


# ─── Утилиты ───


def _find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    """Найти колонку по списку возможных имён."""
    fieldnames_lower = {f.lower(): f for f in fieldnames}
    for candidate in candidates:
        if candidate.lower() in fieldnames_lower:
            return fieldnames_lower[candidate.lower()]
    return None


def _parse_timestamp(raw: str) -> datetime | None:
    """Попытаться распарсить timestamp из строки."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d-%H.%M.%S.%f",  # BGL формат: 2005-06-03-15.42.50.675872
        "%Y-%m-%d-%H.%M.%S",     # BGL без микросекунд
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _normalize_level(raw: str) -> str:
    """Нормализовать level. Система принимает только ERROR."""
    raw = (raw or "").upper().strip()
    if raw in ("ERROR", "FATAL", "CRITICAL", "SEVERE", "ERR", "EMERG", "ALERT"):
        return "ERROR"
    if raw in ("WARN", "WARNING"):
        return "WARN"
    return "INFO"


def _guess_level(line: str) -> str:
    """Угадать level из текста лога."""
    upper = line.upper()
    if any(kw in upper for kw in ("ERROR", "FATAL", "CRITICAL", "EXCEPTION", "FAIL")):
        return "ERROR"
    if any(kw in upper for kw in ("WARN", "WARNING")):
        return "WARN"
    return "INFO"


def send_to_api(logs: list[dict], url: str, batch_size: int):
    """Отправить логи в систему кластеризации по батчам."""
    total_sent = 0
    total_batches = (len(logs) + batch_size - 1) // batch_size

    for i in range(0, len(logs), batch_size):
        batch = logs[i:i + batch_size]
        batch_num = i // batch_size + 1

        try:
            response = requests.post(
                f"{url}/api/v1/ingest",
                json={"logs": batch},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            total_sent += result.get("total_processed", 0)
            print(
                f"  Batch {batch_num}/{total_batches}: "
                f"processed={result.get('total_processed', 0)}, "
                f"filtered_out={result.get('total_filtered_out', 0)}"
            )
        except requests.exceptions.RequestException as e:
            print(f"  Batch {batch_num}/{total_batches}: ОШИБКА — {e}")

    return total_sent


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка реальных лог-датасетов для тестирования кластеризации"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Путь к файлу с логами (CSV, TXT, LOG)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["loghub", "text", "thunderbird", "auto"],
        default="auto",
        help="Формат входных данных (default: auto-detect)",
    )
    parser.add_argument(
        "--service", "-s",
        default="kaggle-logs",
        help="Имя микросервиса для загрузки (default: kaggle-logs)",
    )
    parser.add_argument(
        "--url", "-u",
        default="http://localhost:8000",
        help="URL системы кластеризации (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5000,
        help="Максимальное количество логов для загрузки (default: 5000)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1000,
        help="Размер батча для отправки (default: 1000)",
    )
    parser.add_argument(
        "--all-levels",
        action="store_true",
        help="Загружать все уровни (по умолчанию — только ERROR)",
    )
    parser.add_argument(
        "--save-ground-truth",
        action="store_true",
        help="Сохранить ground truth разметку (для Loghub CSV с EventId)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только спарсить и показать статистику, не отправлять в API",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Сохранить спарсенные логи в JSON (для отладки)",
    )

    args = parser.parse_args()
    filepath = Path(args.input)

    if not filepath.exists():
        print(f"ОШИБКА: Файл не найден: {filepath}")
        sys.exit(1)

    # Определяем формат
    fmt = args.format
    if fmt == "auto":
        suffix = filepath.suffix.lower()
        if suffix == ".csv":
            fmt = "loghub"
        elif suffix in (".log", ".txt"):
            # Пробуем определить по содержимому
            with open(filepath, encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
            if _THUNDERBIRD_PATTERN.match(first_line):
                fmt = "thunderbird"
            else:
                fmt = "text"
        else:
            fmt = "text"

    print(f"Формат: {fmt}")
    print(f"Файл: {filepath}")
    print(f"Сервис: {args.service}")
    print(f"Макс. логов: {args.count}")
    print()

    # Парсинг
    ground_truth = {}
    if fmt == "loghub":
        logs, ground_truth = parse_loghub_csv(
            filepath, args.count, not args.all_levels, args.save_ground_truth
        )
    elif fmt == "thunderbird":
        logs, ground_truth = parse_thunderbird(filepath, args.count, not args.all_levels)
    else:
        logs, ground_truth = parse_text_logs(filepath, args.count, not args.all_levels)

    # Устанавливаем microservice
    for log in logs:
        log["microservice"] = args.service

    print(f"Спарсено логов: {len(logs)}")

    # Статистика
    levels = {}
    for log in logs:
        levels[log["level"]] = levels.get(log["level"], 0) + 1
    print(f"По уровням: {levels}")

    hosts = set(log["host"] for log in logs)
    print(f"Уникальных хостов: {len(hosts)}")

    if ground_truth:
        unique_events = len(set(ground_truth.values()))
        print(f"Ground truth событий: {unique_events}")

    # Сохранение в файл
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"logs": logs}, f, ensure_ascii=False, indent=2)
        print(f"\nЛоги сохранены в: {output_path}")

    if ground_truth and args.save_ground_truth:
        gt_path = filepath.parent / f"{filepath.stem}_ground_truth.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, ensure_ascii=False, indent=2)
        print(f"Ground truth сохранён в: {gt_path}")

    # Отправка
    if args.dry_run:
        print("\n[DRY RUN] Логи не отправлены в API.")
        # Показать примеры
        print("\nПримеры логов:")
        for log in logs[:5]:
            print(f"  [{log['level']}] {log['message'][:100]}")
        return

    print(f"\nОтправка в {args.url} батчами по {args.batch_size}...")
    total = send_to_api(logs, args.url, args.batch_size)
    print(f"\nГотово! Отправлено: {total} логов")
    print(
        f"Дождитесь кластеризации (~5 минут) и проверьте результаты:\n"
        f"  curl {args.url}/api/v1/results/{args.service}?period=24h | python -m json.tool"
    )


if __name__ == "__main__":
    main()
