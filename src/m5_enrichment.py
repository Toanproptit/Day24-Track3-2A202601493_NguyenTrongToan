from __future__ import annotations

"""Offline-safe enrichment adapter."""

from dataclasses import dataclass


@dataclass
class EnrichedChunk:
    enriched_text: str
    auto_metadata: dict


def enrich_chunks(chunks: list[dict], *args, **kwargs) -> list[EnrichedChunk]:
    return [
        EnrichedChunk(str(item.get("text", "")), dict(item.get("metadata", {})))
        for item in chunks or []
    ]

