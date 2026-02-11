from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embeddings import encode_texts, get_embedder
from .models import ChunkRecord
from .nsw import nsw_search


def load_index(index_dir: Path) -> tuple[np.ndarray, dict[int, list[int]], list[ChunkRecord], dict]:
    embeddings = np.load(index_dir / "embeddings.npy")
    graph_raw = json.loads((index_dir / "graph.json").read_text(encoding="utf-8"))
    graph = {int(k): [int(v) for v in vals] for k, vals in graph_raw.items()}
    chunks = _read_chunks(index_dir / "chunks.jsonl")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return embeddings, graph, chunks, meta


def semantic_search(
    query: str,
    index_dir: Path,
    top_k: int = 5,
    model_name: str | None = None,
) -> list[dict]:
    embeddings, graph, chunks, meta = load_index(index_dir)
    chosen_model = model_name or meta.get("model_name", "all-MiniLM-L6-v2")
    embedder = get_embedder(chosen_model)
    query_vec = encode_texts(embedder, [query])[0]

    idxs = nsw_search(
        query_vector=query_vec,
        data=embeddings,
        graph_edges=graph,
        search_k=top_k,
    )

    results = []
    for rank, idx in enumerate(idxs, start=1):
        c = chunks[idx]
        score = float(np.dot(query_vec, embeddings[idx]))
        results.append(
            {
                "rank": rank,
                "chunk_id": c.chunk_id,
                "score": score,
                "url": c.url,
                "title": c.title,
                "visited_at": c.visited_at,
                "text": c.text,
                "browser": c.browser,
            }
        )

    return sorted(results, key=lambda x: x["score"], reverse=True)


def _read_chunks(path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(ChunkRecord.from_dict(json.loads(line)))
    return chunks
