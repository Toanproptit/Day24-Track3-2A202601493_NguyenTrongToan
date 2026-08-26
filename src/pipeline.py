from __future__ import annotations

"""Minimal RAG pipeline facade used by setup scripts and local smoke tests."""

from .m1_chunking import load_documents, chunk_hierarchical
from .m2_search import HybridSearch
from .m3_rerank import CrossEncoderReranker


def build_pipeline():
    chunks = []
    for document in load_documents():
        _, children = chunk_hierarchical(document["text"], document["metadata"])
        chunks.extend({"text": item.text, "metadata": item.metadata} for item in children)
    search = HybridSearch()
    search.index(chunks)
    return search, CrossEncoderReranker()


def answer_question(question: str, search=None, reranker=None, top_k: int = 3) -> dict:
    if search is None:
        search, reranker = build_pipeline()
    results = search.search(question, top_k=20)
    documents = [{"text": result.text, "score": result.score, "metadata": result.metadata}
                 for result in results]
    ranked = reranker.rerank(question, documents, top_k=top_k) if reranker else results[:top_k]
    contexts = [result.text for result in ranked]
    return {"answer": contexts[0] if contexts else "Không tìm thấy thông tin.",
            "contexts": contexts}

