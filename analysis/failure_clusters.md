# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Trọng Toàn  
**Ngày chạy lab:** 2026-08-26

## 1. Aggregate RAGAS Scores

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1.0000 | 1.0000 | 1.0000 |
| answer_relevancy | 0.6374 | 0.5181 | 0.7129 |
| context_precision | 0.2179 | 0.1824 | 0.1594 |
| context_recall | 0.8379 | 0.6246 | 0.4859 |
| **avg_score** | **0.6733** | **0.5813** | **0.5896** |

## 2. Bottom 10 Questions

| Rank | Distribution | ID | Average | Worst metric |
|---:|---|---:|---:|---|
| 1 | adversarial | 50 | 0.4241 | context_precision |
| 2 | multi_hop | 40 | 0.4358 | context_precision |
| 3 | multi_hop | 33 | 0.4848 | context_precision |
| 4 | multi_hop | 22 | 0.4858 | context_precision |
| 5 | multi_hop | 37 | 0.4898 | context_precision |
| 6 | multi_hop | 34 | 0.5247 | context_precision |
| 7 | factual | 5 | 0.5277 | context_precision |
| 8 | adversarial | 49 | 0.5312 | context_precision |
| 9 | multi_hop | 21 | 0.5386 | context_precision |
| 10 | adversarial | 41 | 0.5409 | context_precision |

The lowest case is the personal VPN/version-policy conflict. It needs metadata-aware selection of the current policy rather than simply matching general VPN terms.

## 3. Failure Cluster Matrix

| Worst metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 0 | 0 | 0 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 20 | 20 | 10 | 50 |
| context_recall | 0 | 0 | 0 | 0 |

## 4. Dominant Failure Analysis

**Dominant distribution:** factual by count (tied with multi_hop)  
**Dominant metric:** context_precision

All 50 records identify context precision as their weakest metric. This indicates that the lexical fallback retrieves broad policy paragraphs that contain related terms but also include unrelated facts. Multi-hop questions additionally lose recall when the answer needs two documents, while adversarial questions expose version conflicts. The next improvement should add source/version metadata filtering and a stronger reranker, then rerun the 50-question evaluation.

## 5. Suggested Fixes

| Metric | Root cause | Suggested fix |
|---|---|---|
| faithfulness | No material failure in this run | Keep answer generation grounded in retrieved contexts |
| context_recall | Multi-hop facts span multiple documents | Increase candidate pool and add query decomposition |
| context_precision | Related but irrelevant chunks are returned | Add metadata filters and cross-encoder reranking |
| answer_relevancy | Multi-hop prompts contain several intents | Use structured answer prompts and intent-aware retrieval |

## 6. Adversarial Distribution

The adversarial average score (0.5896) is below factual (0.6733), as expected. The VPN version conflict and old-vs-current leave-policy questions appear in the bottom 10. This confirms that current-version metadata and explicit conflict resolution are important production safeguards.

