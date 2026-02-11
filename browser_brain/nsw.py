from __future__ import annotations

import heapq
from typing import Dict, List

import numpy as np


Graph = Dict[int, List[int]]


def cosine_distance(point: np.ndarray, docs: np.ndarray) -> np.ndarray:
    sims = docs @ point
    return 1.0 - sims


def create_sw_graph(
    data: np.ndarray,
    num_candidates_for_choice_long: int = 10,
    num_edges_long: int = 5,
    num_candidates_for_choice_short: int = 10,
    num_edges_short: int = 5,
    seed: int = 42,
) -> Graph:
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    graph: Graph = {i: [] for i in range(n)}

    for i in range(n):
        # short edges
        short_pool_size = min(n, max(num_candidates_for_choice_short, num_edges_short + 1))
        short_candidates = rng.choice(n, size=short_pool_size, replace=False)
        short_candidates = short_candidates[short_candidates != i]
        if len(short_candidates) > 0:
            dists = cosine_distance(data[i], data[short_candidates])
            best_idx = np.argsort(dists)[:num_edges_short]
            short_neighbors = short_candidates[best_idx].tolist()
        else:
            short_neighbors = []

        # long edges
        long_pool_size = min(n, max(num_candidates_for_choice_long, num_edges_long + 1))
        long_candidates = rng.choice(n, size=long_pool_size, replace=False)
        long_candidates = [x for x in long_candidates.tolist() if x != i]
        rng.shuffle(long_candidates)
        long_neighbors = long_candidates[:num_edges_long]

        merged = []
        seen = {i}
        for nb in short_neighbors + long_neighbors:
            if nb not in seen:
                merged.append(nb)
                seen.add(nb)

        graph[i] = merged

    # symmetrize
    for i, neighbors in list(graph.items()):
        for j in neighbors:
            if i not in graph[j]:
                graph[j].append(i)

    return graph


def nsw_search(
    query_vector: np.ndarray,
    data: np.ndarray,
    graph_edges: Graph,
    search_k: int = 5,
    num_entry_points: int = 5,
    max_visits: int = 200,
    seed: int = 7,
) -> list[int]:
    n = data.shape[0]
    if n == 0:
        return []

    query = query_vector.reshape(-1)
    rng = np.random.default_rng(seed)
    entries = rng.choice(n, size=min(num_entry_points, n), replace=False).tolist()

    best_heap: list[tuple[float, int]] = []
    visited = set()
    frontier: list[tuple[float, int]] = []

    for entry in entries:
        dist = float(cosine_distance(query, data[entry : entry + 1])[0])
        heapq.heappush(frontier, (dist, entry))

    visits = 0
    while frontier and visits < max_visits:
        cur_dist, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        visits += 1

        heapq.heappush(best_heap, (-cur_dist, node))
        if len(best_heap) > search_k * 4:
            heapq.heappop(best_heap)

        for nb in graph_edges.get(node, []):
            if nb in visited:
                continue
            nb_dist = float(cosine_distance(query, data[nb : nb + 1])[0])
            heapq.heappush(frontier, (nb_dist, nb))

    candidates = [node for _, node in best_heap]
    if not candidates:
        return []

    candidate_dists = cosine_distance(query, data[candidates])
    order = np.argsort(candidate_dists)[:search_k]
    return [int(candidates[i]) for i in order]
