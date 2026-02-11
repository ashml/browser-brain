from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class HistoryRecord:
    url: str
    title: str
    snippet: str
    visited_at: str
    browser: str


@dataclass
class ChunkRecord:
    chunk_id: int
    url: str
    title: str
    text: str
    visited_at: str
    browser: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ChunkRecord":
        return ChunkRecord(**data)
