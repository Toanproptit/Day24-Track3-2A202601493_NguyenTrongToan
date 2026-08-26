from __future__ import annotations

"""Small reranker interface compatible with the Day 18 setup script."""

from dataclasses import dataclass
import re


@dataclass
class RerankedResult:
    text: str
    score: float
    metadata: dict


class CrossEncoderReranker:
    def __init__(self, *args, **kwargs):
        pass

    def rerank(self, query: str, documents: list, top_k: int = 3) -> list[RerankedResult]:
        q_tokens = set(re.findall(r"[\wÀ-ỹ]+", (query or "").lower()))
        ranked = []
        for item in documents or []:
            text = item.text if hasattr(item, "text") else item.get("text", "")
            metadata = item.metadata if hasattr(item, "metadata") else item.get("metadata", {})
            base_score = item.score if hasattr(item, "score") else item.get("score", 0.0)
            tokens = set(re.findall(r"[\wÀ-ỹ]+", text.lower()))
            overlap = len(q_tokens & tokens) / len(q_tokens) if q_tokens else 0.0
            ranked.append(RerankedResult(text, float(base_score) + overlap, dict(metadata)))
        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[:max(0, top_k)]

