from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .browser_history import load_history
from .embeddings import encode_texts, get_embedder
from .models import ChunkRecord
from .nsw import create_sw_graph
from .preprocessing import records_to_chunks


def build_index(
    browser: str,
    output_dir: Path,
    history_path: Path | None = None,
    limit: int | None = None,
    chunk_size: int = 260,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_history(browser=browser, path=history_path, limit=limit)
    chunks = records_to_chunks(records, chunk_size_tokens=chunk_size)
    texts = [c.text for c in chunks]

    embedder = get_embedder(model_name=model_name)
    embeddings = encode_texts(embedder, texts) if texts else np.zeros((0, 384), dtype=np.float32)

    graph = create_sw_graph(
        data=embeddings,
        num_candidates_for_choice_long=10,
        num_edges_long=5,
        num_candidates_for_choice_short=10,
        num_edges_short=5,
    ) if len(chunks) else {}

    np.save(output_dir / "embeddings.npy", embeddings)
    _write_chunks(output_dir / "chunks.jsonl", chunks)
    (output_dir / "graph.json").write_text(json.dumps({str(k): v for k, v in graph.items()}, ensure_ascii=False, indent=2))
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "browser": browser,
                "num_records": len(records),
                "num_chunks": len(chunks),
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
                "model_name": model_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return {
        "records": len(records),
        "chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
    }


def _write_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
