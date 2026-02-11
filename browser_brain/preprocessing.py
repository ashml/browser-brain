from __future__ import annotations

import html
import re
from collections import OrderedDict

from .models import ChunkRecord, HistoryRecord

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def chunk_text(text: str, chunk_size_tokens: int = 260) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    return [" ".join(tokens[i : i + chunk_size_tokens]) for i in range(0, len(tokens), chunk_size_tokens)]


def records_to_chunks(records: list[HistoryRecord], chunk_size_tokens: int = 260) -> list[ChunkRecord]:
    uniq = OrderedDict()
    chunk_id = 0

    for rec in records:
        base_text = clean_text(f"{rec.title} {rec.snippet} {rec.url}")
        for part in chunk_text(base_text, chunk_size_tokens=chunk_size_tokens):
            normalized = part.lower().strip()
            if not normalized:
                continue
            if normalized in uniq:
                continue
            chunk = ChunkRecord(
                chunk_id=chunk_id,
                url=rec.url,
                title=rec.title,
                text=part,
                visited_at=rec.visited_at,
                browser=rec.browser,
            )
            uniq[normalized] = chunk
            chunk_id += 1

    return list(uniq.values())
