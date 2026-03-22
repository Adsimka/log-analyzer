#!/usr/bin/env python3
"""
Offline Grid Search по параметрам UMAP + HDBSCAN.

Работает без запущенного сервера — напрямую:
  1. Берёт eval-логи из ground truth.
  2. Прогоняет через Drain3 + SBERT (одноразово).
  3. Перебирает комбинации UMAP/HDBSCAN параметров.
  4. Считает ARI/NMI/V-measure для каждой комбинации.
  5. Выводит ТОП-10 лучших и рекомендацию.

Использование:
    # Сначала сгенерировать датасет
    python scripts/evaluation/generate_dataset.py --count 5000

    # Запустить grid search
    python scripts/evaluation/grid_search.py

Требования (pip install):
    sentence-transformers drain3 hdbscan umap-learn numpy
"""

import itertools
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════
#  МЕТРИКИ (копия из run_evaluation.py)
# ═══════════════════════════════════════════════════════

def _contingency_matrix(gt, pred):
    matrix = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gt, pred):
        matrix[g][p] += 1
    return matrix


def adjusted_rand_index(gt, pred):
    n = len(gt)
    contingency = _contingency_matrix(gt, pred)
    row_sums, col_sums = {}, defaultdict(int)
    for g, cols in contingency.items():
        row_sums[g] = sum(cols.values())
        for p, count in cols.items():
            col_sums[p] += count
    comb2 = lambda x: x * (x - 1) / 2
    sum_comb = sum(comb2(c) for cols in contingency.values() for c in cols.values())
    sum_row = sum(comb2(s) for s in row_sums.values())
    sum_col = sum(comb2(s) for s in col_sums.values())
    total = comb2(n)
    expected = sum_row * sum_col / total if total > 0 else 0
    max_idx = (sum_row + sum_col) / 2
    denom = max_idx - expected
    if denom == 0:
        return 1.0 if sum_comb == expected else 0.0
    return (sum_comb - expected) / denom


def _entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)


def _mutual_info(gt, pred):
    n = len(gt)
    contingency = _contingency_matrix(gt, pred)
    gt_counts, pred_counts = Counter(gt), Counter(pred)
    mi = 0.0
    for g, cols in contingency.items():
        for p, n_ij in cols.items():
            if n_ij > 0:
                mi += (n_ij / n) * math.log((n * n_ij) / (gt_counts[g] * pred_counts[p]))
    return mi


def normalized_mutual_info(gt, pred):
    mi = _mutual_info(gt, pred)
    h_gt, h_pred = _entropy(gt), _entropy(pred)
    denom = math.sqrt(h_gt * h_pred) if h_gt * h_pred > 0 else 0
    return mi / denom if denom > 0 else 0.0


def _conditional_entropy(gt, pred):
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


def v_measure(gt, pred):
    h_gt, h_pred = _entropy(gt), _entropy(pred)
    h_gt_p = _conditional_entropy(gt, pred)
    h_pred_g = _conditional_entropy(pred, gt)
    hom = 1.0 - h_gt_p / h_gt if h_gt > 0 else 1.0
    com = 1.0 - h_pred_g / h_pred if h_pred > 0 else 1.0
    if hom + com == 0:
        return 0.0, 0.0, 0.0
    vm = 2 * hom * com / (hom + com)
    return vm, hom, com


def purity(gt, pred):
    n = len(gt)
    pred_clusters = defaultdict(list)
    for i, p in enumerate(pred):
        pred_clusters[p].append(gt[i])
    correct = sum(Counter(g).most_common(1)[0][1] for g in pred_clusters.values())
    return correct / n if n > 0 else 0.0


# ═══════════════════════════════════════════════════════
#  DRAIN3 ПАРСИНГ
# ═══════════════════════════════════════════════════════

def parse_with_drain3(messages: list[str]) -> tuple[list[int], dict[int, str]]:
    """
    Парсит все сообщения через Drain3.

    Returns:
        (cluster_ids, templates) — id кластера для каждого лога
        и маппинг cluster_id → template_text.
    """
    from drain3 import TemplateMiner
    from drain3.masking import MaskingInstruction
    from drain3.template_miner_config import TemplateMinerConfig

    config = TemplateMinerConfig()
    config.drain_sim_th = 0.3
    config.drain_depth = 4
    config.drain_max_children = 100
    config.drain_max_clusters = 1024

    config.masking_instructions = [
        MaskingInstruction(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "IP"),
        MaskingInstruction(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            "UUID",
        ),
        MaskingInstruction(r"0x[0-9a-fA-F]+", "HEX"),
        MaskingInstruction(r"(?<=:)\d+", "PORT"),
        MaskingInstruction(r"\b\d+\b", "NUM"),
    ]

    miner = TemplateMiner(config=config)
    cluster_ids = []
    templates = {}

    for msg in messages:
        result = miner.add_log_message(msg)
        cid = result["cluster_id"]
        cluster_ids.append(cid)
        templates[cid] = result["template_mined"]

    print(f"  Drain3: {len(messages)} логов → {len(templates)} шаблонов")
    return cluster_ids, templates


# ═══════════════════════════════════════════════════════
#  SBERT ЭМБЕДДИНГИ
# ═══════════════════════════════════════════════════════

def _preprocess_template(text: str) -> str:
    """Заменить Drain3-плейсхолдеры на осмысленные слова для SBERT."""
    replacements = [
        ("<IP>", "ADDRESS"),
        ("<UUID>", "IDENTIFIER"),
        ("<HEX>", "HEXVALUE"),
        ("<PORT>", "PORT"),
        ("<PATH>", "FILEPATH"),
        ("<NUM>", "NUMBER"),
        ("<*>", "PARAM"),
    ]
    for placeholder, replacement in replacements:
        text = text.replace(placeholder, replacement)
    text = re.sub(r"(\bPARAM\b(?:\s+\bPARAM\b)+)", "PARAM", text)
    return " ".join(text.split())


def compute_embeddings(
    templates: dict[int, str],
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[int, np.ndarray]:
    """Вычислить SBERT-эмбеддинги для всех шаблонов."""
    from sentence_transformers import SentenceTransformer

    print(f"  Загрузка SBERT модели: {model_name}...")
    model = SentenceTransformer(model_name)

    ids = list(templates.keys())
    texts = [_preprocess_template(templates[i]) for i in ids]

    print(f"  Кодирование {len(texts)} шаблонов (с preprocessing)...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    return {cid: vec for cid, vec in zip(ids, vectors)}


# ═══════════════════════════════════════════════════════
#  КЛАСТЕРИЗАЦИЯ (одна комбинация параметров)
# ═══════════════════════════════════════════════════════

@dataclass
class ParamSet:
    """Одна комбинация параметров."""
    umap_n_components: int
    umap_n_neighbors: int
    umap_min_dist: float
    umap_metric: str
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    hdbscan_metric: str
    hdbscan_method: str  # eom / leaf
    skip_umap: bool = False


def cluster_templates(
    embeddings: np.ndarray,
    params: ParamSet,
) -> np.ndarray:
    """Запустить UMAP + HDBSCAN с заданными параметрами."""
    import hdbscan as hdbscan_lib
    from umap import UMAP

    data = embeddings
    n = len(data)

    # UMAP
    if not params.skip_umap and n > params.umap_n_components + 2:
        adaptive_nn = min(
            params.umap_n_neighbors,
            max(5, int(np.sqrt(n))),
            n - 1,
        )
        reducer = UMAP(
            n_components=min(params.umap_n_components, n - 2),
            n_neighbors=adaptive_nn,
            min_dist=params.umap_min_dist,
            metric=params.umap_metric,
            random_state=42,
        )
        data = reducer.fit_transform(data)

    # HDBSCAN — cosine для высокоразмерных, euclidean после UMAP
    effective_metric = params.hdbscan_metric
    if params.skip_umap and effective_metric == "euclidean":
        effective_metric = "cosine"

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=params.hdbscan_min_cluster_size,
        min_samples=params.hdbscan_min_samples,
        metric=effective_metric,
        cluster_selection_method=params.hdbscan_method,
    )
    labels = clusterer.fit_predict(data)
    return labels


# ═══════════════════════════════════════════════════════
#  СОПОСТАВЛЕНИЕ ШАБЛОН → GT КЛАСТЕР
# ═══════════════════════════════════════════════════════

def build_template_gt_mapping(
    messages: list[str],
    drain_cluster_ids: list[int],
    gt_cluster_ids: list[int],
) -> dict[int, int]:
    """
    Для каждого Drain3-шаблона определяем ground truth кластер
    через голосование: какой GT-кластер чаще всего встречается
    среди логов с этим шаблоном.

    Returns:
        {drain_cluster_id: gt_cluster_id}
    """
    votes = defaultdict(list)
    for drain_id, gt_id in zip(drain_cluster_ids, gt_cluster_ids):
        votes[drain_id].append(gt_id)

    mapping = {}
    for drain_id, gt_list in votes.items():
        most_common = Counter(gt_list).most_common(1)[0][0]
        mapping[drain_id] = most_common

    return mapping


def evaluate_params(
    params: ParamSet,
    template_embeddings: np.ndarray,
    template_ids: list[int],
    template_gt: dict[int, int],
    template_counts: dict[int, int],
) -> dict:
    """
    Запустить кластеризацию с данными параметрами и вычислить метрики.

    Метрики считаются на уровне ЛОГОВ (с учётом весов шаблонов),
    а не на уровне шаблонов.
    """
    try:
        labels = cluster_templates(template_embeddings, params)
    except Exception as e:
        return {"error": str(e)}

    # Разворачиваем шаблон-уровень → лог-уровень (учитываем log_count)
    gt_labels = []
    pred_labels = []

    for i, tid in enumerate(template_ids):
        gt_cluster = template_gt.get(tid, -1)
        pred_cluster = int(labels[i])
        count = template_counts.get(tid, 1)

        gt_labels.extend([gt_cluster] * count)
        pred_labels.extend([pred_cluster] * count)

    # Метрики
    ari = adjusted_rand_index(gt_labels, pred_labels)
    nmi = normalized_mutual_info(gt_labels, pred_labels)
    vm, hom, com = v_measure(gt_labels, pred_labels)
    pur = purity(gt_labels, pred_labels)

    n_clusters = len(set(labels) - {-1})
    noise = int(np.sum(labels == -1))
    noise_ratio = noise / len(labels) if len(labels) > 0 else 0

    return {
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "v_measure": round(vm, 4),
        "homogeneity": round(hom, 4),
        "completeness": round(com, 4),
        "purity": round(pur, 4),
        "n_clusters": n_clusters,
        "noise_ratio": round(noise_ratio, 4),
        "noise_templates": noise,
    }


# ═══════════════════════════════════════════════════════
#  GRID SEARCH
# ═══════════════════════════════════════════════════════

# Пространство параметров для перебора
PARAM_GRID = {
    "umap_n_components": [5, 8, 10, 15],
    "umap_n_neighbors": [5, 8, 10, 15],
    "umap_min_dist": [0.0, 0.01, 0.05, 0.1],
    "umap_metric": ["cosine"],
    "hdbscan_min_cluster_size": [2, 3, 5, 8],
    "hdbscan_min_samples": [1, 2, 3],
    "hdbscan_metric": ["euclidean"],
    "hdbscan_method": ["eom", "leaf"],
}

# + вариант без UMAP (прямой HDBSCAN на SBERT-векторах)
DIRECT_HDBSCAN_GRID = {
    "hdbscan_min_cluster_size": [2, 3, 5, 8],
    "hdbscan_min_samples": [1, 2, 3],
    "hdbscan_metric": ["cosine"],
    "hdbscan_method": ["eom", "leaf"],
}


def generate_param_sets() -> list[ParamSet]:
    """Сгенерировать все комбинации параметров."""
    param_sets = []

    # С UMAP
    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*PARAM_GRID.values()):
        kv = dict(zip(keys, values))
        param_sets.append(ParamSet(
            umap_n_components=kv["umap_n_components"],
            umap_n_neighbors=kv["umap_n_neighbors"],
            umap_min_dist=kv["umap_min_dist"],
            umap_metric=kv["umap_metric"],
            hdbscan_min_cluster_size=kv["hdbscan_min_cluster_size"],
            hdbscan_min_samples=kv["hdbscan_min_samples"],
            hdbscan_metric=kv["hdbscan_metric"],
            hdbscan_method=kv["hdbscan_method"],
            skip_umap=False,
        ))

    # Без UMAP
    keys2 = list(DIRECT_HDBSCAN_GRID.keys())
    for values in itertools.product(*DIRECT_HDBSCAN_GRID.values()):
        kv = dict(zip(keys2, values))
        param_sets.append(ParamSet(
            umap_n_components=0,
            umap_n_neighbors=0,
            umap_min_dist=0.0,
            umap_metric="cosine",
            hdbscan_min_cluster_size=kv["hdbscan_min_cluster_size"],
            hdbscan_min_samples=kv["hdbscan_min_samples"],
            hdbscan_metric=kv["hdbscan_metric"],
            hdbscan_method=kv["hdbscan_method"],
            skip_umap=True,
        ))

    return param_sets


def main():
    data_dir = Path("scripts/evaluation/data")
    gt_path = data_dir / "ground_truth.json"
    logs_path = data_dir / "eval_logs.json"

    if not gt_path.exists():
        print("Сначала сгенерируйте датасет:")
        print("  python scripts/evaluation/generate_dataset.py --count 5000")
        sys.exit(1)

    with open(logs_path, encoding="utf-8") as f:
        logs = json.load(f)["logs"]
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    messages = [log["message"] for log in logs]
    gt_ids = [gt["ground_truth_cluster_id"] for gt in ground_truth]

    print("=" * 60)
    print("  OFFLINE GRID SEARCH: UMAP + HDBSCAN")
    print("=" * 60)

    # Шаг 1: Drain3
    print("\n[1/3] Drain3 парсинг...")
    drain_ids, templates = parse_with_drain3(messages)

    # Шаг 2: SBERT
    print("\n[2/3] SBERT эмбеддинги...")
    embeddings_map = compute_embeddings(templates)

    # Подготовка данных для кластеризации
    template_ids = sorted(templates.keys())
    template_embeddings = np.array([embeddings_map[tid] for tid in template_ids])

    # GT-маппинг и счётчики
    template_gt = build_template_gt_mapping(messages, drain_ids, gt_ids)
    template_counts = Counter(drain_ids)

    print(f"\n  Шаблонов для кластеризации: {len(template_ids)}")
    print(f"  GT кластеров: {len(set(gt_ids) - {-1})}")

    # Шаг 3: Grid Search
    param_sets = generate_param_sets()
    total = len(param_sets)
    print(f"\n[3/3] Grid Search: {total} комбинаций параметров")
    print("-" * 60)

    results = []
    for i, params in enumerate(param_sets):
        metrics = evaluate_params(
            params, template_embeddings, template_ids,
            template_gt, template_counts,
        )

        if "error" in metrics:
            continue

        # Фильтруем нереалистичные результаты
        if metrics["n_clusters"] < 3 or metrics["n_clusters"] > 30:
            continue

        result = {
            "params": {
                "skip_umap": params.skip_umap,
                "umap_n_components": params.umap_n_components,
                "umap_n_neighbors": params.umap_n_neighbors,
                "umap_min_dist": params.umap_min_dist,
                "hdbscan_min_cluster_size": params.hdbscan_min_cluster_size,
                "hdbscan_min_samples": params.hdbscan_min_samples,
                "hdbscan_metric": params.hdbscan_metric,
                "hdbscan_method": params.hdbscan_method,
            },
            "metrics": metrics,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Прогресс: {i + 1}/{total} "
                  f"(найдено {len(results)} валидных)")

    # Сортируем по ARI
    results.sort(key=lambda r: r["metrics"]["ari"], reverse=True)

    print(f"\n{'=' * 60}")
    print(f"  РЕЗУЛЬТАТЫ: {len(results)} валидных комбинаций")
    print(f"{'=' * 60}")

    # ТОП-15
    print(f"\n  {'─' * 56}")
    print(f"  {'Rank':>4s}  {'ARI':>6s}  {'NMI':>6s}  {'V-m':>6s}  "
          f"{'Pur':>6s}  {'#Cl':>3s}  {'Noise':>5s}  Config")
    print(f"  {'─' * 56}")

    for rank, r in enumerate(results[:15], 1):
        m = r["metrics"]
        p = r["params"]

        if p["skip_umap"]:
            cfg = f"noUMAP hdb({p['hdbscan_min_cluster_size']},{p['hdbscan_min_samples']},{p['hdbscan_method']},{p['hdbscan_metric']})"
        else:
            cfg = (f"umap({p['umap_n_components']},{p['umap_n_neighbors']},{p['umap_min_dist']}) "
                   f"hdb({p['hdbscan_min_cluster_size']},{p['hdbscan_min_samples']},{p['hdbscan_method']})")

        print(f"  {rank:>4d}  {m['ari']:>6.4f}  {m['nmi']:>6.4f}  "
              f"{m['v_measure']:>6.4f}  {m['purity']:>6.4f}  "
              f"{m['n_clusters']:>3d}  {m['noise_ratio']:>5.2f}  {cfg}")

    # Лучший результат
    best = results[0]
    bm = best["metrics"]
    bp = best["params"]

    print(f"\n{'=' * 60}")
    print(f"  ЛУЧШАЯ КОМБИНАЦИЯ")
    print(f"{'=' * 60}")
    print(f"  ARI:          {bm['ari']:.4f}")
    print(f"  NMI:          {bm['nmi']:.4f}")
    print(f"  V-measure:    {bm['v_measure']:.4f}")
    print(f"  Purity:       {bm['purity']:.4f}")
    print(f"  Кластеров:    {bm['n_clusters']}")
    print(f"  Noise ratio:  {bm['noise_ratio']:.4f}")
    print()

    if bp["skip_umap"]:
        print("  UMAP:         отключён (прямой HDBSCAN)")
    else:
        print(f"  UMAP:         n_components={bp['umap_n_components']}, "
              f"n_neighbors={bp['umap_n_neighbors']}, min_dist={bp['umap_min_dist']}")

    print(f"  HDBSCAN:      min_cluster_size={bp['hdbscan_min_cluster_size']}, "
          f"min_samples={bp['hdbscan_min_samples']}, "
          f"method={bp['hdbscan_method']}, metric={bp['hdbscan_metric']}")

    print(f"\n  Для применения в .env:")
    if not bp["skip_umap"]:
        print(f"    UMAP_N_COMPONENTS={bp['umap_n_components']}")
        print(f"    UMAP_N_NEIGHBORS={bp['umap_n_neighbors']}")
        print(f"    UMAP_MIN_DIST={bp['umap_min_dist']}")
    else:
        print(f"    # Установите SKIP_UMAP_THRESHOLD=99999 чтобы отключить UMAP")
    print(f"    HDBSCAN_MIN_CLUSTER_SIZE={bp['hdbscan_min_cluster_size']}")
    print(f"    HDBSCAN_MIN_SAMPLES={bp['hdbscan_min_samples']}")
    print(f"    HDBSCAN_CLUSTER_SELECTION_METHOD={bp['hdbscan_method']}")
    if bp["skip_umap"]:
        print(f"    HDBSCAN_METRIC={bp['hdbscan_metric']}")

    # Сохраняем
    output_path = data_dir / "grid_search_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_combinations": total,
            "valid_results": len(results),
            "best": best,
            "top_15": results[:15],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Полные результаты: {output_path}")


if __name__ == "__main__":
    main()