"""Evaluate raw, PCA-whitened, and ABTT SemJI candidates (paper Table 7)."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from experiments.embedding_geometry import (
    apply_transform,
    fit_transform,
    l2_matrix,
    spectral_diagnostics,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    database_url_env: str
    default_database_url: str
    left_table: str
    right_table: str
    left_id: str
    right_id: str
    left_join_field: str
    right_join_field: str
    workload_root: str


SPECS = {
    "imdb": DatasetSpec(
        "imdb", "IMDB_DATABASE_URL", "postgresql://localhost/imdb",
        "imdb_t1_hnsw", "imdb_t2_hnsw", "id", "id",
        "actor_director_emb", "actor_director_emb", "imdb_data",
    ),
    "molecule": DatasetSpec(
        "molecule", "MOLECULE_DATABASE_URL", "postgresql://localhost/molecule",
        "facts_50k_hnsw", "paper_hnsw", "fact_id", "id",
        "fact_text_emb", "abstract_emb", "molecule_data",
    ),
}


@dataclass
class ReservoirSample:
    spec: DatasetSpec
    left_rows: list[dict[str, Any]]
    right_rows: list[dict[str, Any]]
    left_join: np.ndarray
    right_join: np.ndarray

    def rows_for(self, logical_table: str) -> list[dict[str, Any]]:
        return self.left_rows if logical_table.lower() == self.spec.left_table.removesuffix("_hnsw") else self.right_rows


def load_reservoir(
    spec: DatasetSpec,
    database_url: str,
    sample_size: int,
    sample_seed: int,
) -> ReservoirSample:
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.rows import dict_row

    def fetch(conn, table: str, id_column: str) -> list[dict[str, Any]]:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f'SELECT * FROM "{table}" ORDER BY md5("{id_column}"::text || %s) LIMIT %s',
                (str(sample_seed), sample_size),
            )
            return [dict(row) for row in cur.fetchall()]

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        left_rows = fetch(conn, spec.left_table, spec.left_id)
        right_rows = fetch(conn, spec.right_table, spec.right_id)
    if len(left_rows) != sample_size or len(right_rows) != sample_size:
        raise ValueError(
            f"requested {sample_size} rows but loaded {len(left_rows)} x {len(right_rows)}"
        )
    left_join = np.stack([
        np.asarray(row[spec.left_join_field], dtype=np.float32) for row in left_rows
    ])
    right_join = np.stack([
        np.asarray(row[spec.right_join_field], dtype=np.float32) for row in right_rows
    ])
    return ReservoirSample(spec, left_rows, right_rows, left_join, right_join)


def _predicate_value_matches(actual: Any, operator: str, expected: Any) -> bool:
    if actual is None:
        return False
    if operator == "in":
        return actual in expected
    if operator == "@>":
        if isinstance(actual, str):
            actual_values = {actual}
        else:
            actual_values = set(actual)
        return set(expected).issubset(actual_values)
    if operator == ">=":
        return actual >= expected
    if operator == "<=":
        return actual <= expected
    if operator == ">":
        return actual > expected
    if operator == "<":
        return actual < expected
    if operator in {"=", "=="}:
        return actual == expected
    if operator in {"!=", "<>"}:
        return actual != expected
    raise ValueError(f"unsupported predicate operator: {operator}")


def predicate_mask(
    rows: Sequence[dict[str, Any]], predicates: Sequence[dict[str, Any]]
) -> np.ndarray:
    mask = np.ones(len(rows), dtype=bool)
    for predicate in predicates:
        mask &= np.asarray([
            _predicate_value_matches(
                row.get(predicate["attribute"]), predicate["operator"], predicate["value"]
            )
            for row in rows
        ])
    return mask


def _bit_array(value: Any, dimensions: int) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=bool)
    text = str(value)
    if text.startswith("B'") and text.endswith("'"):
        text = text[2:-1]
    bits = np.fromiter((character == "1" for character in text), dtype=bool)
    if len(bits) != dimensions:
        raise ValueError(f"expected {dimensions} bits, got {len(bits)}")
    return bits


def jaccard_distances(rows: Sequence[dict[str, Any]], field: str, query: Sequence[int]) -> np.ndarray:
    query_bits = np.asarray(query, dtype=bool)
    values = np.stack([_bit_array(row[field], len(query_bits)) for row in rows])
    intersection = np.sum(values & query_bits, axis=1)
    union = np.sum(values | query_bits, axis=1)
    return 1.0 - np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)


def semantic_distances(rows: Sequence[dict[str, Any]], signal: dict[str, Any]) -> np.ndarray:
    if signal.get("metric") == "jaccard":
        return jaccard_distances(rows, signal["field"], signal["query_embed"])
    values = np.stack([np.asarray(row[signal["field"]], dtype=np.float32) for row in rows])
    query = np.asarray(signal["query_embed"], dtype=np.float32)
    return np.linalg.norm(values - query, axis=1)


def _top_pair_ids(
    scores: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    count: int,
    candidate_mask: np.ndarray | None = None,
) -> set[tuple[int, int]]:
    values = np.asarray(scores, dtype=float)
    if candidate_mask is not None:
        values = np.where(candidate_mask, values, np.inf)
    flat = values.ravel()
    finite = np.flatnonzero(np.isfinite(flat))
    if not len(finite):
        return set()
    selected_count = min(count, len(finite))
    selected = finite[np.argpartition(flat[finite], selected_count - 1)[:selected_count]]
    selected = selected[np.argsort(flat[selected], kind="stable")]
    width = values.shape[1]
    return {
        (int(left_indices[index // width]), int(right_indices[index % width]))
        for index in selected
    }


def evaluate_query(
    sample: ReservoirSample,
    query: dict[str, Any],
    join_distances: np.ndarray,
    materialization_radius: float,
) -> tuple[float, float]:
    left_logical = sample.spec.left_table.removesuffix("_hnsw")
    right_logical = sample.spec.right_table.removesuffix("_hnsw")
    left_predicates = [
        predicate for predicate in query.get("predicates", [])
        if predicate["table"].lower() == left_logical
    ]
    right_predicates = [
        predicate for predicate in query.get("predicates", [])
        if predicate["table"].lower() == right_logical
    ]
    left_indices = np.flatnonzero(predicate_mask(sample.left_rows, left_predicates))
    right_indices = np.flatnonzero(predicate_mask(sample.right_rows, right_predicates))
    if not len(left_indices) or not len(right_indices):
        return 1.0, 1.0

    left_score = np.zeros(len(sample.left_rows), dtype=float)
    right_score = np.zeros(len(sample.right_rows), dtype=float)
    join_weight = 0.0
    signals = query["scoring"]["signals"]
    weights = query["scoring"]["weights"]
    for signal, weight in zip(signals, weights):
        if signal["type"] == "join_distance":
            join_weight += float(weight)
        elif signal["table"].lower() == left_logical:
            left_score += float(weight) * semantic_distances(sample.left_rows, signal)
        elif signal["table"].lower() == right_logical:
            right_score += float(weight) * semantic_distances(sample.right_rows, signal)
        else:
            raise ValueError(f"signal table is not an endpoint: {signal['table']!r}")

    distance_subset = join_distances[np.ix_(left_indices, right_indices)]
    scores = (
        left_score[left_indices, None]
        + right_score[None, right_indices]
        + join_weight * distance_subset
    )
    expected = _top_pair_ids(scores, left_indices, right_indices, int(query.get("K", 20)))
    actual = _top_pair_ids(
        scores,
        left_indices,
        right_indices,
        int(query.get("K", 20)),
        distance_subset <= materialization_radius,
    )
    intersection = len(expected & actual)
    recall = intersection / len(expected) if expected else 1.0
    precision = intersection / len(actual) if actual else (1.0 if not expected else 0.0)
    return recall, precision


def calibrated_radius(distances: np.ndarray) -> float:
    return float(distances.mean() - 3.0 * distances.std())


def evaluate_sample(
    sample: ReservoirSample,
    queries: Sequence[dict[str, Any]],
    *,
    pca_shrinkage: float = 0.1,
    abtt_components: int = 1,
) -> dict[str, Any]:
    raw_distances = l2_matrix(sample.left_join, sample.right_join).astype(np.float32)
    combined = np.vstack([sample.left_join, sample.right_join])

    abtt = fit_transform(combined, method="abtt", top_components=abtt_components)
    transformed = apply_transform(combined, abtt)
    split = len(sample.left_join)
    abtt_distances = l2_matrix(transformed[:split], transformed[split:]).astype(np.float32)
    abtt_radius = calibrated_radius(abtt_distances)
    abtt_fraction = float(np.mean(abtt_distances <= abtt_radius))
    raw_control_radius = float(np.quantile(raw_distances, abtt_fraction))

    pca = fit_transform(combined, method="pca", shrinkage=pca_shrinkage)
    transformed = apply_transform(combined, pca)
    pca_distances = l2_matrix(transformed[:split], transformed[split:]).astype(np.float32)
    raw_radius = calibrated_radius(raw_distances)
    pca_radius = calibrated_radius(pca_distances)

    arms: dict[str, tuple[np.ndarray, Callable[[dict[str, Any]], float]]] = {
        "raw_workload": (raw_distances, lambda query: float(query["join"]["distance_threshold"])),
        "raw_calibrated": (raw_distances, lambda _query: raw_radius),
        "pca": (pca_distances, lambda _query: pca_radius),
        "abtt": (abtt_distances, lambda _query: abtt_radius),
        "raw_radius_control": (raw_distances, lambda _query: raw_control_radius),
    }
    results: dict[str, Any] = {}
    for name, (distances, radius_for_query) in arms.items():
        metrics = np.asarray([
            evaluate_query(sample, query, distances, radius_for_query(query))
            for query in queries
        ])
        fractions = [float(np.mean(distances <= radius_for_query(query))) for query in queries]
        results[name] = {
            "recall": metrics[:, 0],
            "precision": metrics[:, 1],
            "pair_fraction": float(np.mean(fractions)),
        }
    results["diagnostics"] = spectral_diagnostics(combined)
    return results


def hierarchical_bootstrap(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Two-stage bootstrap over endpoint samples, then workload queries."""
    rng = np.random.default_rng(seed)
    sample_count, query_count = values.shape
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected_samples = rng.integers(0, sample_count, size=sample_count)
        sample_means = []
        for sample_index in selected_samples:
            selected_queries = rng.integers(0, query_count, size=query_count)
            sample_means.append(float(values[sample_index, selected_queries].mean()))
        estimates[draw] = float(np.mean(sample_means))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def load_queries(root: Path, spec: DatasetSpec) -> list[dict[str, Any]]:
    queries = []
    for workload in ("w7", "w8"):
        path = root / spec.workload_root / "workload" / workload / f"{workload}_queries_100.json"
        queries.extend(json.loads(path.read_text(encoding="utf-8"))["queries"])
    return queries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    spec = SPECS[args.dataset]
    root = Path(__file__).resolve().parents[1]
    queries = load_queries(root, spec)
    database_url = args.database_url or os.environ.get(spec.database_url_env, spec.default_database_url)
    sample_results = [
        evaluate_sample(load_reservoir(spec, database_url, args.sample_size, seed), queries)
        for seed in args.sample_seeds
    ]

    output: dict[str, Any] = {
        "dataset": args.dataset,
        "sample_size": args.sample_size,
        "sample_seeds": args.sample_seeds,
        "queries": len(queries),
        "arms": {},
        "diagnostics": [result["diagnostics"] for result in sample_results],
    }
    for arm in ("raw_workload", "raw_calibrated", "pca", "abtt", "raw_radius_control"):
        recall = np.stack([result[arm]["recall"] for result in sample_results])
        precision = np.stack([result[arm]["precision"] for result in sample_results])
        output["arms"][arm] = {
            "recall": float(recall.mean()),
            "recall_ci": hierarchical_bootstrap(recall, draws=args.bootstrap_draws, seed=0),
            "precision": float(precision.mean()),
            "precision_ci": hierarchical_bootstrap(precision, draws=args.bootstrap_draws, seed=1),
            "pair_fraction": float(np.mean([result[arm]["pair_fraction"] for result in sample_results])),
        }
    output_path = args.output or Path(f"results/semji_robustness_{args.dataset}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())