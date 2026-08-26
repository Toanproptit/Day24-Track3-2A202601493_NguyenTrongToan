from __future__ import annotations

"""RAG evaluation interface with a deterministic offline implementation."""

from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path

from config import TEST_SET_PATH


@dataclass
class PerQuestionResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\wÀ-ỹ]+", (text or "").lower()) if len(token) > 1}


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left) if left else 0.0


def _metrics(question: str, answer: str, contexts: list[str], truth: str) -> PerQuestionResult:
    answer_tokens = _tokens(answer)
    question_tokens = _tokens(question)
    context_tokens = _tokens(" ".join(contexts or []))
    truth_tokens = _tokens(truth)
    return PerQuestionResult(
        faithfulness=_overlap(answer_tokens, context_tokens),
        answer_relevancy=_overlap(question_tokens, answer_tokens),
        context_precision=_overlap(context_tokens, truth_tokens),
        context_recall=_overlap(truth_tokens, context_tokens),
    )


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question = [
        _metrics(question, answer, ctx, truth)
        for question, answer, ctx, truth in zip(questions, answers, contexts, ground_truths)
    ]
    result = {"per_question": per_question}
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        values = [getattr(item, name) for item in per_question]
        result[name] = sum(values) / len(values) if values else 0.0
    return result


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_report(results: dict, questions: list, path: str = "ragas_report.json") -> None:
    serializable = dict(results)
    serializable["per_question"] = [
        asdict(item) if hasattr(item, "__dataclass_fields__") else item
        for item in serializable.get("per_question", [])
    ]
    Path(path).write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
