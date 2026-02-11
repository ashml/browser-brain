from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


class FallbackHasherEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: Iterable[str], normalize_embeddings: bool = True) -> np.ndarray:
        texts = list(texts)
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.split():
                h = hashlib.sha1(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if (h[4] % 2 == 0) else -1.0
                vecs[i, idx] += sign
        if normalize_embeddings:
            vecs = l2_normalize(vecs)
        return vecs


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception:
        return FallbackHasherEmbedder(dim=384)


def encode_texts(embedder, texts: list[str]) -> np.ndarray:
    vectors = embedder.encode(texts, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    return l2_normalize(vectors)
