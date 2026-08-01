"""Top-n bounded-length walks for the NFCorpus multi-step workload."""
from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

ScoredPath = tuple[float, tuple[int, ...]]
State = tuple[int, int]


def _keep_top_n(candidates: Iterable[ScoredPath], count: int) -> list[ScoredPath]:
    return heapq.nsmallest(count, candidates, key=lambda item: (item[0], item[1]))


def top_n_directed_edge_walks(
    position_costs: np.ndarray,
    adjacency: Sequence[Sequence[int] | np.ndarray],
    count: int,
) -> list[ScoredPath]:
    """Return the lowest-cost walks with no immediate backtracking.

    ``position_costs[i, v]`` is the cost of placing vertex ``v`` at position
    ``i``. A partial walk is summarized by ``(previous, current)``; retaining
    the best ``count`` paths per directed-edge state is sufficient for the
    global top-``count`` result because all future transitions depend only on
    that state.
    """
    costs = np.asarray(position_costs, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("position_costs must have shape (depth, vertices)")
    depth, vertex_count = costs.shape
    if depth < 1 or vertex_count != len(adjacency):
        raise ValueError("cost dimensions do not match adjacency")
    if count < 1:
        raise ValueError("count must be positive")

    states: dict[State, list[ScoredPath]] = {
        (-1, vertex): [(float(costs[0, vertex]), (vertex,))]
        for vertex in range(vertex_count)
        if np.isfinite(costs[0, vertex])
    }

    for position in range(1, depth):
        candidates: dict[State, list[ScoredPath]] = defaultdict(list)
        for (previous, current), paths in states.items():
            for neighbor_value in adjacency[current]:
                neighbor = int(neighbor_value)
                if neighbor == previous or not np.isfinite(costs[position, neighbor]):
                    continue
                state = (current, neighbor)
                added_cost = float(costs[position, neighbor])
                candidates[state].extend(
                    (score + added_cost, path + (neighbor,)) for score, path in paths
                )
        states = {
            state: _keep_top_n(paths, count)
            for state, paths in candidates.items()
        }
        if not states:
            break

    if depth == 1:
        complete = [item for paths in states.values() for item in paths]
    elif not states or len(next(iter(states.values()))[0][1]) != depth:
        return []
    else:
        complete = [item for paths in states.values() for item in paths]
    return _keep_top_n(complete, count)


def brute_force_walks(
    position_costs: np.ndarray,
    adjacency: Sequence[Sequence[int] | np.ndarray],
    count: int,
) -> list[ScoredPath]:
    """Small-graph oracle used to verify :func:`top_n_directed_edge_walks`."""
    costs = np.asarray(position_costs, dtype=np.float64)
    depth, vertex_count = costs.shape
    complete: list[ScoredPath] = []

    def visit(path: tuple[int, ...], score: float) -> None:
        position = len(path)
        if position == depth:
            complete.append((score, path))
            return
        current = path[-1]
        previous = path[-2] if len(path) > 1 else -1
        for neighbor_value in adjacency[current]:
            neighbor = int(neighbor_value)
            if neighbor == previous or not np.isfinite(costs[position, neighbor]):
                continue
            visit(path + (neighbor,), score + float(costs[position, neighbor]))

    for vertex in range(vertex_count):
        if np.isfinite(costs[0, vertex]):
            visit((vertex,), float(costs[0, vertex]))
    return _keep_top_n(complete, count)