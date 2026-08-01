"""Reproduce the SemJI capacity-doubling experiment and Figure 7."""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.embedding_geometry import l2_matrix


@dataclass(frozen=True)
class EvolutionConfig:
    k_target: int = 3
    overflow_tolerance: float = 0.5
    monitor_window: int = 100
    async_delay: int = 20
    maximum_pairs_fraction: float = 0.05
    minimum_pairs_fraction: float = 0.002
    capacity_multiplier: float = 2.0
    cache_latency_ms: float = 0.066
    fallback_latency_ms: float = 7.67


class RunningMoments:
    def __init__(self, initial: np.ndarray | None = None) -> None:
        self.count = 0
        self.mean = 0.0
        self.sum_squared_delta = 0.0
        if initial is not None:
            self.add(initial)

    def add(self, values: np.ndarray) -> None:
        for value in np.asarray(values, dtype=float).ravel():
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.sum_squared_delta += delta * (value - self.mean)

    @property
    def standard_deviation(self) -> float:
        return float(np.sqrt(self.sum_squared_delta / max(self.count - 1, 1)))


@dataclass(frozen=True)
class PairOracle:
    sorted_distances: np.ndarray
    population_pairs: int

    def fraction(self, radius: float) -> float:
        return float(
            np.searchsorted(self.sorted_distances, radius, side="right")
            / len(self.sorted_distances)
        )

    def count(self, radius: float) -> int:
        return int(round(self.fraction(radius) * self.population_pairs))

    def radius(self, target_pairs: int) -> float:
        fraction = np.clip(target_pairs / self.population_pairs, 0.0, 1.0)
        index = min(
            len(self.sorted_distances) - 1,
            int(np.ceil(fraction * len(self.sorted_distances))),
        )
        return float(self.sorted_distances[index])


def simulate_evolution(
    requested_radii: np.ndarray,
    diagnostic_samples: np.ndarray,
    heldout_requested_radii: np.ndarray,
    warmup_distances: np.ndarray,
    pair_oracle: PairOracle,
    order: Sequence[int],
    config: EvolutionConfig,
) -> dict[str, np.ndarray | int]:
    """Simulate asynchronous pair-capacity growth for one query ordering."""
    moments = RunningMoments(warmup_distances)
    radius = float(warmup_distances.mean() - 3.0 * warmup_distances.std())
    maximum_pairs = int(config.maximum_pairs_fraction * pair_oracle.population_pairs)
    minimum_pairs = int(config.minimum_pairs_fraction * pair_oracle.population_pairs)
    overflow_window: deque[float] = deque(maxlen=config.monitor_window)
    pending_step: int | None = None
    expansions = 0

    trace: dict[str, list[float]] = {
        "mean": [],
        "standard_deviation": [],
        "radius": [],
        "pairs": [],
        "overflow": [],
        "heldout_overflow_rate": [],
        "latency_ms": [],
    }

    for step, query_index_value in enumerate(order):
        query_index = int(query_index_value)
        moments.add(diagnostic_samples[query_index])
        overflow = float(requested_radii[query_index] > radius)
        overflow_window.append(overflow)

        if (
            pending_step is None
            and len(overflow_window) == config.monitor_window
            and float(np.mean(overflow_window)) > config.overflow_tolerance
            and pair_oracle.count(radius) < maximum_pairs
        ):
            pending_step = step + config.async_delay

        if pending_step is not None and step >= pending_step:
            current_pairs = pair_oracle.count(radius)
            target_pairs = min(
                maximum_pairs,
                max(int(config.capacity_multiplier * current_pairs), minimum_pairs),
            )
            next_radius = pair_oracle.radius(target_pairs)
            if pair_oracle.count(next_radius) <= maximum_pairs and next_radius > radius:
                radius = next_radius
                expansions += 1
            pending_step = None
            overflow_window.clear()

        trace["mean"].append(moments.mean)
        trace["standard_deviation"].append(moments.standard_deviation)
        trace["radius"].append(radius)
        trace["pairs"].append(float(pair_oracle.count(radius)))
        trace["overflow"].append(overflow)
        trace["heldout_overflow_rate"].append(
            float(np.mean(heldout_requested_radii > radius))
        )
        trace["latency_ms"].append(
            config.fallback_latency_ms if overflow else config.cache_latency_ms
        )

    return {**{key: np.asarray(values) for key, values in trace.items()}, "expansions": expansions}


def _sample_rows(
    database_url: str,
    table: str,
    id_column: str,
    embedding_column: str,
    count: int,
    seed: int,
) -> np.ndarray:
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT "{embedding_column}" FROM "{table}" '
                f'WHERE "{embedding_column}" IS NOT NULL '
                f'ORDER BY md5("{id_column}"::text || %s) LIMIT %s',
                (str(seed), count),
            )
            rows = cur.fetchall()
    return np.asarray([np.asarray(row[0], dtype=np.float32) for row in rows])


def prepare_experiment(
    partners: np.ndarray,
    seed_pool: np.ndarray,
    *,
    query_count: int,
    sample_count: int,
    warmup_pairs: int,
    oracle_pairs: int,
    heldout_count: int,
    k_target: int,
    seed: int,
) -> dict[str, np.ndarray | PairOracle | float]:
    rng = np.random.default_rng(seed)
    evaluation_size = min(2000, len(partners))
    evaluation = partners[rng.choice(len(partners), evaluation_size, replace=False)]
    means = l2_matrix(seed_pool, evaluation).mean(axis=1)
    order = np.argsort(means)
    lower = int(len(seed_pool) * 0.025)
    upper = int(len(seed_pool) * 0.975) - 1
    positions = np.linspace(lower, upper, query_count).round().astype(int)
    query_indices = order[positions]
    queries = seed_pool[query_indices]
    all_distances = l2_matrix(queries, partners).astype(np.float32)

    diagnostic_indices = np.stack(
        [rng.choice(len(partners), sample_count, replace=False) for _ in queries]
    )
    diagnostic_samples = np.take_along_axis(all_distances, diagnostic_indices, axis=1)
    requested = np.partition(all_distances, k_target - 1, axis=1)[:, k_target - 1]

    test_indices = rng.choice(len(seed_pool), heldout_count, replace=False)
    test_distances = l2_matrix(seed_pool[test_indices], partners)
    heldout_requested = np.partition(test_distances, k_target - 1, axis=1)[:, k_target - 1]

    left = rng.integers(0, len(partners), size=warmup_pairs)
    right = rng.integers(0, len(partners), size=warmup_pairs)
    warmup = np.linalg.norm(partners[left] - partners[right], axis=1)
    left = rng.integers(0, len(partners), size=oracle_pairs)
    right = rng.integers(0, len(partners), size=oracle_pairs)
    oracle_distances = np.sort(np.linalg.norm(partners[left] - partners[right], axis=1))
    population_pairs = len(partners) * (len(partners) - 1) // 2
    oracle = PairOracle(oracle_distances, population_pairs)
    return {
        "means": means[query_indices],
        "requested_radii": requested,
        "diagnostic_samples": diagnostic_samples,
        "heldout_requested_radii": heldout_requested,
        "warmup_distances": warmup,
        "pair_oracle": oracle,
        "initial_radius": float(warmup.mean() - 3.0 * warmup.std()),
    }


def _trace_to_json(trace: dict[str, np.ndarray | int]) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in trace.items()
    }


def render_figure(
    ascending: dict[str, np.ndarray | int],
    random_traces: list[dict[str, np.ndarray | int]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    def band(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.stack([np.asarray(trace[key]) for trace in random_traces])
        return values.mean(axis=0), values.min(axis=0), values.max(axis=0)

    x = np.arange(1, len(np.asarray(ascending["radius"])) + 1)
    figure, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    specifications = [
        ("mean", "A. Running mean estimate", "mean"),
        ("standard_deviation", "B. Running std estimate", "std"),
        ("radius", "C. SemJI threshold", "radius"),
        ("pairs", "D. Cache pairs", "pairs"),
        ("heldout_overflow_rate", "E. Held-out base-engine completion rate", "rate"),
        ("latency_ms", "F. Cumulative workload latency", "latency (ms)"),
    ]
    for axis, (key, title, label) in zip(axes.flat, specifications):
        ascending_values = np.asarray(ascending[key])
        mean, low, high = band(key)
        if key == "latency_ms":
            ascending_values = np.cumsum(ascending_values)
            stacked = np.stack([np.cumsum(np.asarray(trace[key])) for trace in random_traces])
            mean, low, high = stacked.mean(0), stacked.min(0), stacked.max(0)
        axis.fill_between(x, low, high, color="#9e8fc7", alpha=0.45)
        axis.plot(x, ascending_values, color="#1b9e77", label="ascending")
        axis.plot(x, mean, color="#5e3c99", label="random")
        axis.set_title(title)
        axis.set_ylabel(label)
    axes[1, 1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}K"))
    for axis in axes[-1]:
        axis.set_xlabel("query index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("MOLECULE_DATABASE_URL", "postgresql://localhost/molecule"),
    )
    parser.add_argument("--partner-table", default="paper_hnsw")
    parser.add_argument("--partner-id", default="id")
    parser.add_argument("--partner-embedding", default="abstract_emb")
    parser.add_argument("--seed-table", default="facts_50k_hnsw")
    parser.add_argument("--seed-id", default="fact_id")
    parser.add_argument("--seed-embedding", default="fact_text_emb")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--random-orders", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/semji_evolution.json"))
    parser.add_argument("--figure", type=Path, default=Path("results/semji_evolution.png"))
    args = parser.parse_args(argv)

    partners = _sample_rows(
        args.database_url, args.partner_table, args.partner_id,
        args.partner_embedding, args.sample_size, 0,
    )
    seeds = _sample_rows(
        args.database_url, args.seed_table, args.seed_id,
        args.seed_embedding, args.sample_size, 1,
    )
    config = EvolutionConfig()
    prepared = prepare_experiment(
        partners,
        seeds,
        query_count=args.queries,
        sample_count=50,
        warmup_pairs=2000,
        oracle_pairs=50000,
        heldout_count=500,
        k_target=config.k_target,
        seed=args.seed,
    )
    ascending_order = np.argsort(np.asarray(prepared["means"]))
    simulation_args = (
        np.asarray(prepared["requested_radii"]),
        np.asarray(prepared["diagnostic_samples"]),
        np.asarray(prepared["heldout_requested_radii"]),
        np.asarray(prepared["warmup_distances"]),
        prepared["pair_oracle"],
    )
    ascending = simulate_evolution(*simulation_args, ascending_order, config)
    random_traces = [
        simulate_evolution(
            *simulation_args,
            np.random.default_rng(seed).permutation(args.queries),
            config,
        )
        for seed in range(args.random_orders)
    ]
    payload = {
        "config": config.__dict__,
        "initial_radius": prepared["initial_radius"],
        "ascending": _trace_to_json(ascending),
        "random": [_trace_to_json(trace) for trace in random_traces],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    render_figure(ascending, random_traces, args.figure)
    print(json.dumps({
        "initial_radius": prepared["initial_radius"],
        "final_radius": float(np.asarray(ascending["radius"])[-1]),
        "final_pairs": int(np.asarray(ascending["pairs"])[-1]),
        "expansions": ascending["expansions"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())