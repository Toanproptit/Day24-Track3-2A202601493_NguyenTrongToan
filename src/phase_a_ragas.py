from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved → {path}")


# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(
                f"Unknown distribution {distribution!r}; expected factual, multi_hop, or adversarial"
            )
        groups[distribution].append(item)
    return groups


def _value(source: Any, name: str, default: float = 0.0) -> float:
    """Read a metric from either a RAGAS object or a plain mapping."""
    if isinstance(source, dict):
        value = source.get(name, default)
    else:
        value = getattr(source, name, default)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _fallback_metrics(answer: str, contexts: list[str], ground_truth: str,
                      question: str) -> dict[str, float]:
    """Small deterministic fallback used when the optional RAGAS stack is absent."""
    def tokens(text: str) -> set[str]:
        import re
        return {t for t in re.findall(r"[\wÀ-ỹ]+", (text or "").lower()) if len(t) > 1}

    answer_tokens = tokens(answer)
    truth_tokens = tokens(ground_truth)
    question_tokens = tokens(question)
    context_tokens = tokens(" ".join(contexts))

    def overlap(left: set[str], right: set[str]) -> float:
        return len(left & right) / len(left) if left else 0.0

    return {
        "faithfulness": overlap(answer_tokens, context_tokens),
        "answer_relevancy": overlap(question_tokens, answer_tokens),
        "context_precision": overlap(context_tokens, truth_tokens),
        "context_recall": overlap(truth_tokens, context_tokens),
    }


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Gợi ý — import từ Day 18 của bạn:
        from src.m4_eval import evaluate_ragas

    Steps:
        1. Extract questions, answers, contexts, ground_truths từ answers list
        2. Gọi evaluate_ragas() từ m4_eval.py
        3. Kết hợp kết quả với distribution info từ answers list
        4. Return list[RagasResult]
    """
    if not answers:
        return []

    questions = [a.get("question", "") for a in answers]
    answer_texts = [a.get("answer", "") for a in answers]
    contexts = [a.get("contexts", []) or [] for a in answers]
    ground_truths = [a.get("ground_truth", "") for a in answers]

    raw: Any = None
    try:
        from src.m4_eval import evaluate_ragas
        raw = evaluate_ragas(questions, answer_texts, contexts, ground_truths)
    except Exception as exc:
        print(f"Warning: RAGAS unavailable ({exc}); using deterministic metrics.")

    if isinstance(raw, dict):
        per_question = raw.get("per_question", [])
    else:
        per_question = getattr(raw, "per_question", []) if raw is not None else []
    per_question = list(per_question or [])

    results: list[RagasResult] = []
    for index, item in enumerate(answers):
        fallback = _fallback_metrics(
            answer_texts[index], contexts[index], ground_truths[index], questions[index]
        )
        metrics = {
            name: _value(per_question[index], name, fallback[name])
            if index < len(per_question) else fallback[name]
            for name in DIAGNOSTIC_TREE
        }
        results.append(RagasResult(
            question_id=int(item.get("id", index + 1)),
            distribution=item.get("distribution", "factual"),
            question=questions[index], answer=answer_texts[index],
            contexts=list(contexts[index]), ground_truth=ground_truths[index],
            faithfulness=metrics["faithfulness"],
            answer_relevancy=metrics["answer_relevancy"],
            context_precision=metrics["context_precision"],
            context_recall=metrics["context_recall"],
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    output = []
    for rank, result in enumerate(sorted(results, key=lambda r: r.avg_score)[:10], 1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({
            "rank": rank,
            "question_id": result.question_id,
            "distribution": result.distribution,
            "question": result.question,
            "avg_score": round(result.avg_score, 4),
            "worst_metric": result.worst_metric,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Mục tiêu: tìm ra distribution nào hay bị failure nhất và metric nào yếu nhất.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    distributions = ["factual", "multi_hop", "adversarial"]
    matrix = {
        metric: {distribution: 0 for distribution in distributions}
        for metric in DIAGNOSTIC_TREE
    }
    for result in results:
        if result.worst_metric in matrix and result.distribution in distributions:
            matrix[result.worst_metric][result.distribution] += 1

    dominant_dist = max(
        distributions,
        key=lambda distribution: sum(matrix[metric][distribution] for metric in matrix),
        default="factual",
    )
    dominant_metric = max(
        matrix,
        key=lambda metric: sum(matrix[metric].values()),
        default="faithfulness",
    )
    insight = (
        f"Distribution '{dominant_dist}' có nhiều failure nhất; "
        f"'{dominant_metric}' là metric yếu chủ đạo. "
        f"Gợi ý cải thiện: {DIAGNOSTIC_TREE[dominant_metric][1]}."
    )
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
    else:
        print("⚠️  No results — implement run_ragas_50q() first.")
