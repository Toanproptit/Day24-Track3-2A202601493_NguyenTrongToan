from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    def tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[\wÀ-ỹ]+", (text or "").lower()) if len(t) > 1}

    def heuristic_score(answer: str) -> float:
        q_tokens = tokens(question)
        a_tokens = tokens(answer)
        topical = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.0
        has_content = 1.0 if a_tokens else 0.0
        # Reward concrete policy answers without allowing verbosity alone to win.
        concrete = 0.12 if re.search(r"\d|%|VNĐ|VND|ngày|tháng|được|không", answer or "", re.I) else 0.0
        concise = max(0.0, 1.0 - max(0, len(answer or "") - 500) / 1500)
        return max(0.0, min(1.0, 0.55 * topical + 0.25 * has_content + concrete + 0.20 * concise))

    def fallback() -> dict:
        score_a = heuristic_score(answer_a)
        score_b = heuristic_score(answer_b)
        if abs(score_a - score_b) < 0.05:
            winner = "tie"
            reasoning = "Hai câu trả lời có chất lượng tương đương theo các tín hiệu nội dung có sẵn."
        elif score_a > score_b:
            winner = "A"
            reasoning = "Answer A phù hợp với câu hỏi và có thông tin cụ thể hơn."
        else:
            winner = "B"
            reasoning = "Answer B phù hợp với câu hỏi và có thông tin cụ thể hơn."
        return {"winner": winner, "reasoning": reasoning,
                "scores": {"A": round(score_a, 4), "B": round(score_b, 4)}}

    # Use the real judge only when credentials are present.  Tests and local CI
    # remain deterministic and do not make an accidental network request.
    if not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
        return fallback()
    prompt = (
        "Bạn là expert đánh giá chất lượng câu trả lời RAG.\n"
        f"Câu hỏi: {question}\n\nAnswer A:\n{answer_a}\n\nAnswer B:\n{answer_b}\n\n"
        "Chấm accuracy, completeness và conciseness. Chỉ trả JSON với winner (A/B/tie), "
        "reasoning và scores là số 0-1."
    )
    try:
        from openai import OpenAI
        response = OpenAI().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        winner = str(parsed.get("winner", "tie")).strip().upper()
        winner = winner if winner in {"A", "B", "TIE"} else "TIE"
        scores = parsed.get("scores", {}) or {}
        score_a = scores.get("A", scores.get("a", 0.0))
        score_b = scores.get("B", scores.get("b", 0.0))
        try:
            score_a = max(0.0, min(1.0, float(score_a)))
            score_b = max(0.0, min(1.0, float(score_b)))
        except (TypeError, ValueError):
            return fallback()
        return {
            "winner": "tie" if winner == "TIE" else winner,
            "reasoning": str(parsed.get("reasoning") or "LLM đã so sánh accuracy, completeness và conciseness."),
            "scores": {"A": score_a, "B": score_b},
        }
    except Exception as exc:
        # A judge failure must not break the complete evaluation report.
        result = fallback()
        result["reasoning"] += f" (fallback: {type(exc).__name__})"
        return result


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass1 = pass1.get("winner", "tie")
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")
    final = winner_pass1 if winner_pass1 == winner_pass2 else "tie"
    scores1 = pass1.get("scores", {}) or {}
    scores2_raw = pass2_raw.get("scores", {}) or {}
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=winner_pass1, winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=str(pass1.get("reasoning", "")),
        reasoning_pass2=str(pass2_raw.get("reasoning", "")),
        position_consistent=(winner_pass1 == winner_pass2),
        scores_pass1={"A": float(scores1.get("A", 0.0)), "B": float(scores1.get("B", 0.0))},
        scores_pass2={"A": float(scores2_raw.get("B", 0.0)), "B": float(scores2_raw.get("A", 0.0))},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    n = len(judge_labels)
    if n == 0:
        return 0.0
    observed = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    categories = set(judge_labels) | set(human_labels)
    expected = sum(
        (judge_labels.count(category) / n) * (human_labels.count(category) / n)
        for category in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return max(-1.0, min(1.0, (observed - expected) / (1.0 - expected)))


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    position_bias_count = sum(1 for result in judge_results if not result.position_consistent)
    position_bias_rate = position_bias_count / total if total else 0.0
    a_wins_a_longer = sum(
        1 for result in judge_results
        if result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
    )
    b_wins_b_longer = sum(
        1 for result in judge_results
        if result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
    )
    decisive = sum(1 for result in judge_results if result.final_winner != "tie")
    verbosity_count = a_wins_a_longer + b_wins_b_longer
    verbosity_bias = verbosity_count / decisive if decisive else 0.0
    interpretation = (
        "Position bias cao — nên dùng swap-and-average."
        if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_phase_b_report(path: str = "reports/judge_results.json") -> dict:
    """Run the judge on ten labelled examples and persist a JSON report."""
    answers_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "answers_50q.json")
    answers_by_id = {}
    if os.path.exists(answers_path):
        with open(answers_path, encoding="utf-8") as handle:
            answers_by_id = {item.get("id"): item for item in json.load(handle)}
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as handle:
        human_data = json.load(handle)

    results = []
    for item in human_data:
        generated = answers_by_id.get(item.get("question_id"), {})
        answer_a = item.get("model_answer") or generated.get("answer", "")
        answer_b = generated.get("ground_truth", "")
        if not answer_b:
            answer_b = "Câu trả lời chuẩn theo chính sách nội bộ chưa được cung cấp."
        results.append(swap_and_average(item.get("question", ""), answer_a, answer_b))

    judge_labels = [1 if result.final_winner == "A" else 0 for result in results]
    human_labels = [int(item.get("human_label", 0)) for item in human_data]
    report = {
        "total_judged": len(results),
        "results": [asdict(result) for result in results],
        "judge_labels": judge_labels,
        "human_labels": human_labels,
        "cohen_kappa": round(cohen_kappa(judge_labels, human_labels), 4),
        "bias": bias_report(results),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {path}")
    return report


if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    # In production: run judge on the same 10 questions to get judge_labels
    judge_labels = [0] * len(human_labels)  # placeholder — replace with real judge output
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ (placeholder): {kappa:.3f}")

    # --- Bias report ---
    bias = bias_report([result])
    print(f"\nBias report: {bias}")
    generate_phase_b_report()
