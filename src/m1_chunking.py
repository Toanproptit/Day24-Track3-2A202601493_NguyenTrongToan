from __future__ import annotations

"""Lightweight document loading and hierarchical chunking for the lab pipeline."""

from dataclasses import dataclass
from pathlib import Path
import re

from config import DATA_DIR


@dataclass
class Chunk:
    text: str
    metadata: dict
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    documents = []
    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append({
            "text": text,
            "metadata": {"source": path.name, "title": path.stem},
        })
    return documents


def _paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "")]
    return [part for part in paragraphs if part]


def chunk_basic(text: str, metadata: dict | None = None) -> list[Chunk]:
    base = dict(metadata or {})
    return [Chunk(part, base.copy()) for part in _paragraphs(text)]


def chunk_hierarchical(
    text: str,
    metadata: dict | None = None,
    parent_size: int = 2048,
    child_size: int = 256,
) -> tuple[list[Chunk], list[Chunk]]:
    """Return parent and child chunks using character windows."""
    source = dict(metadata or {})
    parents, children = [], []
    paragraphs = _paragraphs(text)
    if not paragraphs:
        return parents, children
    buffer, parent_index = [], 0
    for paragraph in paragraphs:
        if buffer and len("\n\n".join(buffer)) + len(paragraph) + 2 > parent_size:
            parent_text = "\n\n".join(buffer)
            parent_id = f"{source.get('source', 'document')}::p{parent_index}"
            parent = Chunk(parent_text, source.copy(), parent_id)
            parents.append(parent)
            for offset in range(0, len(parent_text), child_size):
                child_text = parent_text[offset:offset + child_size].strip()
                if child_text:
                    children.append(Chunk(child_text, source.copy(), parent_id))
            buffer, parent_index = [], parent_index + 1
        buffer.append(paragraph)
    if buffer:
        parent_text = "\n\n".join(buffer)
        parent_id = f"{source.get('source', 'document')}::p{parent_index}"
        parents.append(Chunk(parent_text, source.copy(), parent_id))
        for offset in range(0, len(parent_text), child_size):
            child_text = parent_text[offset:offset + child_size].strip()
            if child_text:
                children.append(Chunk(child_text, source.copy(), parent_id))
    return parents, children

