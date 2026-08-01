"""Run DASE's NFCorpus multi-step depth experiment for depths 2 through 8."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.embedding_geometry import l2_matrix
from experiments.path_dp import ScoredPath, top_n_directed_edge_walks


DEFAULT_EDGE_TABLE = "ti_nfcorpus_paper_hnsw_nfcorpus_paper_hnsw_0.35"


@dataclass(frozen=True)
class GraphData:
    paper_ids: np.ndarray
    embeddings: np.ndarray
    adjacency: list[np.ndarray]


def load_graph(database_url: str, paper_table: str, edge_table: str) -> GraphData:
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg import sql

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id1, id2 FROM {}").format(sql.Identifier(edge_table))
            )
            edges = [
                (int(left), int(right))
                for left, right in cur.fetchall()
                if left != right
            ]
            paper_ids = np.asarray(
                sorted({value for edge in edges for value in edge}), dtype=np.int64
            )
            index = {
                paper_id: position for position, paper_id in enumerate(paper_ids)
            }
            neighbors = [set() for _ in paper_ids]
            for left, right in edges:
                neighbors[index[left]].add(index[right])
                neighbors[index[right]].add(index[left])
            cur.execute(
                sql.SQL("SELECT id, title_emb FROM {} WHERE id = ANY(%s)").format(
                    sql.Identifier(paper_table)
                ),
                (paper_ids.tolist(),),
            )
            rows = {
                int(paper_id): np.asarray(embedding, dtype=np.float32)
                for paper_id, embedding in cur.fetchall()
            }
    missing = [int(paper_id) for paper_id in paper_ids if int(paper_id) not in rows]
    if missing:
        raise ValueError(f"missing embeddings for {len(missing)} SemJI endpoints")
    embeddings = np.stack([rows[int(paper_id)] for paper_id in paper_ids])
    adjacency = [np.asarray(sorted(values), dtype=np.int32) for values in neighbors]
    return GraphData(paper_ids, embeddings, adjacency)


def qualifying_graph(
    adjacency: Sequence[np.ndarray], qualifying: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Project the SemJI graph onto vertices satisfying predicate P."""
    vertices = np.flatnonzero(qualifying)
    local = np.full(len(adjacency), -1, dtype=np.int64)
    local[vertices] = np.arange(len(vertices))
    projected = [
        local[neighbors[qualifying[neighbors]]].astype(np.int32)
        for neighbors in (adjacency[int(vertex)] for vertex in vertices)
    ]
    return vertices, projected


def generate_anchor_sets(
    adjacency: Sequence[np.ndarray],
    query_count: int,
    maximum_depth: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Generate per-position anchors from non-backtracking graph walks."""
    eligible = [
        vertex for vertex, neighbors in enumerate(adjacency) if len(neighbors) >= 2
    ]
    if not eligible:
        raise ValueError("qualifying graph has no vertex with degree at least two")
    anchor_sets = []
    for _ in range(query_count):
        walk = [int(rng.choice(eligible))]
        while len(walk) < maximum_depth:
            current = walk[-1]
            choices = [
                int(neighbor)
                for neighbor in adjacency[current]
                if len(walk) < 2 or int(neighbor) != walk[-2]
            ]
            if not choices:
                choices = [int(neighbor) for neighbor in adjacency[current]]
            walk.append(int(rng.choice(choices)))
        anchor_sets.append(
            [
                int(rng.choice(adjacency[vertex]))
                if len(adjacency[vertex])
                else vertex
                for vertex in walk
            ]
        )
    return anchor_sets


def run_dase(
    graph: GraphData,
    qualifying_vertices: np.ndarray,
    qualifying_adjacency: Sequence[np.ndarray],
    anchors: Sequence[int],
    depth: int,
    result_count: int,
) -> tuple[list[ScoredPath], float]:
    """Execute the directed-edge top-n DP described in Section 5.4."""
    started = time.perf_counter()
    candidates = graph.embeddings[qualifying_vertices]
    costs = (
        l2_matrix(graph.embeddings[np.asarray(anchors[:depth])], candidates) / depth
    )
    local_paths = top_n_directed_edge_walks(
        costs, qualifying_adjacency, result_count
    )
    paths = [
        (score, tuple(int(qualifying_vertices[vertex]) for vertex in path))
        for score, path in local_paths
    ]
    return paths, time.perf_counter() - started


def summarize(
    samples: list[dict[str, float | int]], depths: Sequence[int]
) -> dict[str, dict[str, float | int]]:
    summary = {}
    for depth in depths:
        values = np.asarray(
            [float(row["qps"]) for row in samples if row["depth"] == depth]
        )
        ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[str(depth)] = {
            "instances": len(values),
            "recall": 1.0,
            "qps": float(values.mean()),
            "qps_ci95": float(ci),
        }
    return summary


def render_figure(
    summary: dict[str, dict[str, float | int]],
    depths: Sequence[int],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    qps = np.asarray([summary[str(depth)]["qps"] for depth in depths], dtype=float)
    ci = np.asarray(
        [summary[str(depth)]["qps_ci95"] for depth in depths], dtype=float
    )
    figure, (recall_axis, qps_axis) = plt.subplots(1, 2, figsize=(10, 4))
    color = "#1b7837"
    recall_axis.plot(depths, np.ones(len(depths)), "-o", color=color, label="DASE")
    recall_axis.set(
        xlabel="multi-step depth k",
        ylabel="recall @ top-10",
        ylim=(-0.03, 1.07),
    )
    qps_axis.plot(depths, qps, "-o", color=color)
    qps_axis.fill_between(depths, qps - ci, qps + ci, color=color, alpha=0.18)
    qps_axis.set(
        xlabel="multi-step depth k",
        ylabel="throughput (QPS)",
        yscale="log",
    )
    for axis in (recall_axis, qps_axis):
        axis.set_xticks(depths)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(loc="upper center", frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get(
            "NFCORPUS_DATABASE_URL", "postgresql://localhost/nfcorpus"
        ),
    )
    parser.add_argument("--paper-table", default="nfcorpus_paper_hnsw")
    parser.add_argument("--edge-table", default=DEFAULT_EDGE_TABLE)
    parser.add_argument("--depths", type=int, nargs="+", default=list(range(2, 9)))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--selectivity", type=float, default=0.18)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("results/nfcorpus_depth.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("results/nfcorpus_depth.png")
    )
    args = parser.parse_args(argv)

    graph = load_graph(args.database_url, args.paper_table, args.edge_table)
    samples: list[dict[str, float | int]] = []
    maximum_depth = max(args.depths)
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        qualifying = rng.random(len(graph.paper_ids)) < args.selectivity
        qualifying_vertices, local_adjacency = qualifying_graph(
            graph.adjacency, qualifying
        )
        local_anchor_sets = generate_anchor_sets(
            local_adjacency, args.queries, maximum_depth, rng
        )
        anchor_sets = [
            [int(qualifying_vertices[value]) for value in anchors]
            for anchors in local_anchor_sets
        ]
        for depth in args.depths:
            for query_index, anchors in enumerate(anchor_sets):
                paths, elapsed = run_dase(
                    graph,
                    qualifying_vertices,
                    local_adjacency,
                    anchors,
                    depth,
                    args.top_k,
                )
                samples.append(
                    {
                        "seed": seed,
                        "query": query_index,
                        "depth": depth,
                        "results": len(paths),
                        "seconds": elapsed,
                        "qps": 1.0 / elapsed,
                    }
                )

    summary = summarize(samples, args.depths)
    payload = {
        "config": {
            "paper_table": args.paper_table,
            "edge_table": args.edge_table,
            "depths": args.depths,
            "seeds": args.seeds,
            "queries": args.queries,
            "selectivity": args.selectivity,
            "top_k": args.top_k,
        },
        "papers_with_edges": len(graph.paper_ids),
        "edges": sum(len(row) for row in graph.adjacency) // 2,
        "samples": samples,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    render_figure(summary, args.depths, args.figure)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())