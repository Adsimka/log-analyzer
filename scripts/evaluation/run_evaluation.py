#!/usr/bin/env python3
"""
Полный evaluation pipeline: отправка → кластеризация → метрики качества.

Вычисляет внешние метрики кластеризации, сравнивая результат системы
с ground truth разметкой:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - V-measure (Homogeneity + Completeness)
  - Purity
  - Silhouette Score (из результатов системы)

Использование:
    # 1. Сгенерировать датасет
    python scripts/evaluation/generate_dataset.py --count 5000

    # 2. Запустить evaluation (отправит логи, дождётся кластеризации, посчитает метрики)
    python scripts/evaluation/run_evaluation.py

    # Или только посчитать метрики (если логи уже отправлены и кластеризация прошла)
    python scripts/evaluation/run_evaluation.py --skip-ingest --skip-wait
"""

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ═══════════════════════════════════════════════════════
#  ШАГ 1: Отправка логов в систему
# ═══════════════════════════════════════════════════════

def send_logs(api_url: str, logs: list[dict], batch_size: int = 200) -> None:
    """Отправить логи батчами."""
    total = len(logs)
    sent = 0

    for i in range(0, total, batch_size):
        batch = logs[i : i + batch_size]
        payload = json.dumps({"logs": batch}).encode("utf-8")

        request = Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                sent += result.get("total_processed", len(batch))
                print(f"  Батч {i // batch_size + 1}: "
                      f"отправлено {len(batch)}, "
                      f"принято {result.get('total_processed', '?')}")
        except URLError as e:
            print(f"  ОШИБКА: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"  Итого отправлено: {sent} из {total}")


# ═══════════════════════════════════════════════════════
#  ШАГ 2: Ожидание кластеризации
# ═══════════════════════════════════════════════════════

def wait_for_clustering(
    base_url: str,
    microservice: str,
    period: str = "1h",
    max_wait: int = 420,
    poll_interval: int = 15,
) -> dict:
    """
    Ждём завершения кластеризации, опрашивая API.

    Returns:
        dict с результатом кластеризации.
    """
    results_url = f"{base_url}/api/v1/results/{microservice}?period={period}"
    start = time.time()

    while time.time() - start < max_wait:
        try:
            request = Request(results_url, method="GET")
            with urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data and data.get("num_clusters", 0) > 0:
                    return data
        except HTTPError as e:
            if e.code == 404:
                pass  # ещё не готово
            else:
                print(f"  HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        except Exception:
            pass

        elapsed = int(time.time() - start)
        print(f"  Ожидание кластеризации... ({elapsed}s / {max_wait}s)")
        time.sleep(poll_interval)

    print("ОШИБКА: Тайм-аут ожидания кластеризации.", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════
#  ШАГ 3: Сопоставление ground truth ↔ результаты
# ═══════════════════════════════════════════════════════

def build_label_mapping(
    ground_truth: list[dict],
    clustering_result: dict,
) -> tuple[list[int], list[int]]:
    """
    Сопоставить ground truth с результатами системы.

    Каждый лог в ground truth знает свой gt_cluster_id.
    Каждый кластер в результатах содержит top_templates.
    Мы назначаем каждому gt-логу predicted_cluster через
    ближайший шаблон.

    Returns:
        (gt_labels, predicted_labels) — два параллельных списка.
    """
    # Строим маппинг: шаблон → predicted cluster_id
    template_to_pred_cluster = {}
    for cluster in clustering_result.get("clusters", []):
        cid = cluster["cluster_id"]
        for tmpl in cluster.get("top_templates", []):
            template_to_pred_cluster[tmpl["template_text"]] = cid

    # Шаблоны unclustered = -1
    unclustered = clustering_result.get("unclustered", {})
    for tmpl in unclustered.get("templates", []):
        template_to_pred_cluster[tmpl["template_text"]] = -1

    gt_labels = []
    pred_labels = []
    matched = 0
    unmatched = 0

    for gt in ground_truth:
        gt_label = gt["ground_truth_cluster_id"]
        message = gt["message"]

        # Ищем лучшее совпадение шаблона по подстроке
        best_match = None
        best_score = 0

        for tmpl_text, pred_cid in template_to_pred_cluster.items():
            score = _template_similarity(message, tmpl_text)
            if score > best_score:
                best_score = score
                best_match = pred_cid

        if best_match is not None and best_score > 0.3:
            gt_labels.append(gt_label)
            pred_labels.append(best_match)
            matched += 1
        else:
            # Не удалось сопоставить — пропускаем
            unmatched += 1

    print(f"  Сопоставлено: {matched}, пропущено: {unmatched}")
    return gt_labels, pred_labels


def _template_similarity(message: str, template: str) -> float:
    """
    Простая оценка похожести сообщения и шаблона.
    Используем Jaccard similarity по словам.
    """
    # Заменяем mask-токены на пустоту для сравнения
    template_clean = template
    for mask in ["<IP>", "<NUM>", "<PORT>", "<UUID>", "<HEX>"]:
        template_clean = template_clean.replace(mask, "")

    msg_words = set(message.lower().split())
    tmpl_words = set(template_clean.lower().split())

    if not tmpl_words:
        return 0.0

    intersection = msg_words & tmpl_words
    union = msg_words | tmpl_words

    return len(intersection) / len(union) if union else 0.0


# ═══════════════════════════════════════════════════════
#  ШАГ 4: Вычисление метрик
# ═══════════════════════════════════════════════════════

def compute_metrics(
    gt_labels: list[int],
    pred_labels: list[int],
    clustering_result: dict,
) -> dict:
    """
    Вычислить все метрики качества кластеризации.

    Внешние метрики (требуют ground truth):
      - ARI  (Adjusted Rand Index)    [-1, 1], 1 = идеально
      - NMI  (Normalized MI)          [0, 1],  1 = идеально
      - V-measure                     [0, 1],  1 = идеально
        - Homogeneity                 [0, 1]
        - Completeness                [0, 1]
      - Purity                        [0, 1],  1 = идеально

    Внутренние метрики (из результатов системы):
      - Silhouette Score
      - Noise ratio
    """
    n = len(gt_labels)
    assert n == len(pred_labels), "Длины не совпадают"

    # ── ARI (Adjusted Rand Index) ──
    ari = _adjusted_rand_index(gt_labels, pred_labels)

    # ── NMI (Normalized Mutual Information) ──
    nmi = _normalized_mutual_info(gt_labels, pred_labels)

    # ── Homogeneity, Completeness, V-measure ──
    homogeneity = _homogeneity(gt_labels, pred_labels)
    completeness = _completeness(gt_labels, pred_labels)
    v_measure = (
        2 * homogeneity * completeness / (homogeneity + completeness)
        if (homogeneity + completeness) > 0
        else 0.0
    )

    # ── Purity ──
    purity = _purity(gt_labels, pred_labels)

    # ── Из результатов системы ──
    silhouette = clustering_result.get("silhouette_score")
    noise_ratio = clustering_result.get("noise_ratio", 0.0)
    num_clusters = clustering_result.get("num_clusters", 0)

    return {
        "n_samples": n,
        "n_gt_clusters": len(set(gt_labels) - {-1}),
        "n_pred_clusters": num_clusters,
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "homogeneity": round(homogeneity, 4),
        "completeness": round(completeness, 4),
        "v_measure": round(v_measure, 4),
        "purity": round(purity, 4),
        "silhouette_score": round(silhouette, 4) if silhouette is not None else None,
        "noise_ratio": round(noise_ratio, 4),
    }


# ── Реализации метрик (без sklearn) ──────────────────

def _contingency_matrix(gt: list[int], pred: list[int]) -> dict:
    """Таблица сопряжённости."""
    matrix = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gt, pred):
        matrix[g][p] += 1
    return matrix


def _adjusted_rand_index(gt: list[int], pred: list[int]) -> float:
    """ARI по формуле Hubert & Arabie."""
    n = len(gt)
    contingency = _contingency_matrix(gt, pred)

    # Суммы по строкам и столбцам
    row_sums = {}
    col_sums = defaultdict(int)
    for g, cols in contingency.items():
        row_sums[g] = sum(cols.values())
        for p, count in cols.items():
            col_sums[p] += count

    def comb2(x):
        return x * (x - 1) / 2

    # Сумма C(n_ij, 2)
    sum_comb = sum(comb2(count) for cols in contingency.values() for count in cols.values())
    sum_row_comb = sum(comb2(s) for s in row_sums.values())
    sum_col_comb = sum(comb2(s) for s in col_sums.values())
    total_comb = comb2(n)

    expected = sum_row_comb * sum_col_comb / total_comb if total_comb > 0 else 0
    max_index = (sum_row_comb + sum_col_comb) / 2
    denominator = max_index - expected

    if denominator == 0:
        return 1.0 if sum_comb == expected else 0.0

    return (sum_comb - expected) / denominator


def _entropy(labels: list[int]) -> float:
    """Энтропия распределения меток."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)


def _mutual_info(gt: list[int], pred: list[int]) -> float:
    """Взаимная информация."""
    n = len(gt)
    contingency = _contingency_matrix(gt, pred)

    gt_counts = Counter(gt)
    pred_counts = Counter(pred)

    mi = 0.0
    for g, cols in contingency.items():
        for p, n_ij in cols.items():
            if n_ij == 0:
                continue
            mi += (n_ij / n) * math.log((n * n_ij) / (gt_counts[g] * pred_counts[p]))

    return mi


def _normalized_mutual_info(gt: list[int], pred: list[int]) -> float:
    """NMI (среднее геометрическое)."""
    mi = _mutual_info(gt, pred)
    h_gt = _entropy(gt)
    h_pred = _entropy(pred)

    denom = math.sqrt(h_gt * h_pred) if h_gt * h_pred > 0 else 0
    return mi / denom if denom > 0 else 0.0


def _conditional_entropy(gt: list[int], pred: list[int]) -> float:
    """H(gt | pred)."""
    n = len(gt)
    contingency = _contingency_matrix(gt, pred)
    pred_counts = Counter(pred)

    h = 0.0
    for p, p_count in pred_counts.items():
        for g in contingency:
            n_ij = contingency[g].get(p, 0)
            if n_ij > 0:
                h -= (n_ij / n) * math.log(n_ij / p_count)
    return h


def _homogeneity(gt: list[int], pred: list[int]) -> float:
    """Однородность: каждый predicted-кластер содержит только один gt-кластер."""
    h_gt = _entropy(gt)
    if h_gt == 0:
        return 1.0
    h_gt_given_pred = _conditional_entropy(gt, pred)
    return 1.0 - h_gt_given_pred / h_gt


def _completeness(gt: list[int], pred: list[int]) -> float:
    """Полнота: все элементы gt-кластера попали в один predicted-кластер."""
    h_pred = _entropy(pred)
    if h_pred == 0:
        return 1.0
    h_pred_given_gt = _conditional_entropy(pred, gt)
    return 1.0 - h_pred_given_gt / h_pred


def _purity(gt: list[int], pred: list[int]) -> float:
    """Чистота: доля правильно назначенных элементов."""
    n = len(gt)
    contingency = _contingency_matrix(gt, pred)

    # Для каждого predicted-кластера берём максимальный gt-кластер
    pred_clusters = defaultdict(list)
    for i, p in enumerate(pred):
        pred_clusters[p].append(gt[i])

    correct = 0
    for p, gt_in_cluster in pred_clusters.items():
        most_common = Counter(gt_in_cluster).most_common(1)[0][1]
        correct += most_common

    return correct / n if n > 0 else 0.0


# ═══════════════════════════════════════════════════════
#  ШАГ 5: Confusion matrix (кластер × кластер)
# ═══════════════════════════════════════════════════════

def print_confusion_summary(gt_labels: list[int], pred_labels: list[int]) -> None:
    """Выводит таблицу: для каждого GT-кластера → в какие predicted попал."""
    gt_names = {
        0: "db_connection",
        1: "auth_failure",
        2: "api_timeout",
        3: "payment_error",
        4: "data_error",
        5: "resource_error",
        6: "queue_error",
        7: "cache_error",
        8: "storage_error",
        -1: "noise",
    }

    gt_to_pred = defaultdict(list)
    for g, p in zip(gt_labels, pred_labels):
        gt_to_pred[g].append(p)

    print("\n  Confusion Summary (GT cluster → predicted clusters):")
    print("  " + "=" * 65)

    for gt_id in sorted(gt_to_pred.keys()):
        pred_list = gt_to_pred[gt_id]
        dist = Counter(pred_list)
        total = len(pred_list)
        name = gt_names.get(gt_id, f"cluster_{gt_id}")

        # Основной predicted-кластер
        main_pred, main_count = dist.most_common(1)[0]
        main_pct = main_count / total * 100

        other = [(p, c) for p, c in dist.most_common() if p != main_pred]
        other_str = ", ".join(f"c{p}:{c}" for p, c in other[:3])

        print(f"  GT {gt_id} ({name:16s}): "
              f"→ pred c{main_pred} ({main_pct:5.1f}% of {total})"
              f"{'  + ' + other_str if other_str else ''}")


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluation pipeline")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--data-dir", default="scripts/evaluation/data")
    parser.add_argument("--period", default="1h")
    parser.add_argument("--service", default="eval-service")
    parser.add_argument("--skip-ingest", action="store_true", help="Пропустить отправку логов")
    parser.add_argument("--skip-wait", action="store_true", help="Пропустить ожидание")
    parser.add_argument("--max-wait", type=int, default=420, help="Макс. ожидание (сек)")
    parser.add_argument("--output", default=None, help="Файл для сохранения метрик (JSON)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    logs_path = data_dir / "eval_logs.json"
    gt_path = data_dir / "ground_truth.json"

    # Проверяем наличие данных
    if not logs_path.exists() or not gt_path.exists():
        print("Датасет не найден. Сначала сгенерируйте:")
        print("  python scripts/evaluation/generate_dataset.py")
        sys.exit(1)

    with open(logs_path, encoding="utf-8") as f:
        logs_data = json.load(f)
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    logs = logs_data["logs"]
    print(f"Загружено {len(logs)} логов, {len(ground_truth)} GT-записей")

    # ── Шаг 1: Отправка ──
    if not args.skip_ingest:
        print(f"\n{'='*50}")
        print("ШАГ 1: Отправка логов в систему")
        print(f"{'='*50}")
        send_logs(f"{args.base_url}/api/v1/ingest", logs)
    else:
        print("\n[Пропуск отправки логов]")

    # ── Шаг 2: Ожидание ──
    if not args.skip_wait:
        print(f"\n{'='*50}")
        print("ШАГ 2: Ожидание кластеризации")
        print(f"{'='*50}")
        print(f"  (Worker запускается каждые 5 мин, макс. ожидание: {args.max_wait}s)")
        clustering_result = wait_for_clustering(
            args.base_url, args.service, args.period, args.max_wait
        )
    else:
        print("\n[Получение результатов без ожидания]")
        url = f"{args.base_url}/api/v1/results/{args.service}?period={args.period}"
        try:
            with urlopen(Request(url), timeout=10) as resp:
                clustering_result = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            print(f"ОШИБКА: Результаты не найдены ({e.code})", file=sys.stderr)
            sys.exit(1)

    print(f"\n  Получен результат:")
    print(f"    Кластеров: {clustering_result.get('num_clusters', 0)}")
    print(f"    Шаблонов:  {clustering_result.get('unique_templates', 0)}")
    print(f"    Silhouette: {clustering_result.get('silhouette_score', 'N/A')}")
    print(f"    Noise:      {clustering_result.get('noise_ratio', 'N/A')}")

    # ── Шаг 3: Сопоставление ──
    print(f"\n{'='*50}")
    print("ШАГ 3: Сопоставление GT ↔ результаты")
    print(f"{'='*50}")
    gt_labels, pred_labels = build_label_mapping(ground_truth, clustering_result)

    if len(gt_labels) < 10:
        print("ОШИБКА: Слишком мало сопоставленных записей.", file=sys.stderr)
        sys.exit(1)

    # ── Шаг 4: Метрики ──
    print(f"\n{'='*50}")
    print("ШАГ 4: Метрики качества кластеризации")
    print(f"{'='*50}")

    metrics = compute_metrics(gt_labels, pred_labels, clustering_result)

    print(f"""
  ┌────────────────────────────────────────────────┐
  │          РЕЗУЛЬТАТЫ EVALUATION                 │
  ├────────────────────────────────────────────────┤
  │  Samples matched:    {metrics['n_samples']:>5d}                    │
  │  GT clusters:        {metrics['n_gt_clusters']:>5d}                    │
  │  Predicted clusters: {metrics['n_pred_clusters']:>5d}                    │
  ├────────────────────────────────────────────────┤
  │  ВНЕШНИЕ МЕТРИКИ (сравнение с ground truth):   │
  │                                                │
  │  ARI (Adjusted Rand Index):  {metrics['ari']:>7.4f}            │
  │  NMI (Normalized MI):        {metrics['nmi']:>7.4f}            │
  │  V-measure:                  {metrics['v_measure']:>7.4f}            │
  │    ├─ Homogeneity:           {metrics['homogeneity']:>7.4f}            │
  │    └─ Completeness:          {metrics['completeness']:>7.4f}            │
  │  Purity:                     {metrics['purity']:>7.4f}            │
  ├────────────────────────────────────────────────┤
  │  ВНУТРЕННИЕ МЕТРИКИ (из системы):              │
  │                                                │
  │  Silhouette Score:           {str(metrics['silhouette_score']):>7s}            │
  │  Noise Ratio:                {metrics['noise_ratio']:>7.4f}            │
  └────────────────────────────────────────────────┘
""")

    # Интерпретация
    print("  Интерпретация:")
    ari = metrics["ari"]
    if ari > 0.7:
        print("    ARI > 0.7  → Отличное качество кластеризации")
    elif ari > 0.4:
        print("    ARI > 0.4  → Хорошее качество кластеризации")
    elif ari > 0.2:
        print("    ARI > 0.2  → Умеренное качество, есть перемешивание кластеров")
    else:
        print("    ARI < 0.2  → Низкое качество, кластеры плохо соответствуют GT")

    # ── Шаг 5: Confusion ──
    print_confusion_summary(gt_labels, pred_labels)

    # ── Сохранение ──
    output_path = args.output or str(data_dir / "eval_metrics.json")
    metrics["clustering_result_summary"] = {
        "num_clusters": clustering_result.get("num_clusters"),
        "unique_templates": clustering_result.get("unique_templates"),
        "total_logs": clustering_result.get("total_logs"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  Метрики сохранены: {output_path}")
    print(f"  Результат кластеризации: {args.base_url}/api/v1/results/{args.service}?period={args.period}")


if __name__ == "__main__":
    main()