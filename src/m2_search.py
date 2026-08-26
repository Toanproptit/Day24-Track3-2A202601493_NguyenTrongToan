from __future__ import annotations

"""Dependency-free lexical search adapters used when the Day 18 stack is absent."""

from dataclasses import dataclass
import re


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\wÀ-ỹ]+", text.lower()) if len(token) > 1}


class _Search:
    def __init__(self, *args, **kwargs):
        self._chunks: list[dict] = []

    def index(self, chunks: list[dict], collection: str | None = None):
        self._chunks = list(chunks or [])
        return len(self._chunks)

    def search(self, query: str, top_k: int = 20, collection: str | None = None):
        query_tokens = _tokens(query)
        scored = []
        for item in self._chunks:
            text = item.get("text", "")
            text_tokens = _tokens(text)
            overlap = len(query_tokens & text_tokens)
            score = overlap / len(query_tokens) if query_tokens else 0.0
            if overlap:
                score += min(0.01, overlap / max(len(text_tokens), 1))
            scored.append(SearchResult(text, score, dict(item.get("metadata", {}))))
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:max(0, top_k)]


class DenseSearch(_Search):
    pass


class HybridSearch(_Search):
    pass

