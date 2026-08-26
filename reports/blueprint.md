# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Trọng Toàn  
**Ngày chạy lab:** 2026-08-26

## Guard Stack Pipeline

~~~text
User input
  -> Presidio PII scan (VN_CCCD, VN_PHONE, EMAIL)
  -> NeMo input rail (jailbreak, prompt injection, off-topic, PII request)
  -> Day 18 RAG pipeline (chunk, search, rerank, answer)
  -> NeMo output rail (sensitive output check)
  -> Safe response
~~~

| Layer | Tool | P95 đo được | Failure action |
|---|---|---:|---|
| PII detection | Presidio + custom VN recognizers | 0.03 ms | Reject and log |
| Topic/jailbreak | NeMo input rail + deterministic pre-check | 0.03 ms | Reject with reason |
| RAG pipeline | Day 18-compatible pipeline | Not measured in guard benchmark | Fallback |
| Output check | NeMo output rail + PII check | Covered by output guard | Block and log |
| Total guard | Presidio + input rail | 0.06 ms | CI gate if over 500 ms |

## CI Gates

- RAGAS faithfulness must be at least 0.75 on the 50-question set.
- Adversarial suite must pass at least 18/20 cases.
- Total guard P95 must remain below 500 ms.
- pytest tests/ -q must pass before merging to main.

Example workflow commands:

~~~yaml
- run: python src/phase_a_ragas.py
- run: python src/phase_b_judge.py
- run: python src/phase_c_guard.py
- run: pytest tests/ -q
~~~

## Monitoring Results

| Metric | Result | Gate |
|---|---:|---:|
| RAGAS faithfulness | 1.000 | >= 0.750 |
| Adversarial pass rate | 20/20 (100%) | >= 18/20 |
| Total guard P95 | 0.06 ms | < 500 ms |
| Cohen's kappa | -0.0714 | Track for judge calibration |
| Dominant failure metric | context_precision | Review retrieval/reranking |

The Phase A aggregate average scores were factual 0.6733, multi_hop 0.5813, and adversarial 0.5896. Context precision was the weakest metric in every distribution, so retrieval filtering and reranking should be the first quality improvements.

## Production Actions

PII detections are rejected before retrieval and logged without storing the raw sensitive value. Input-rail decisions should include the rule that fired, while output-rail blocks should preserve only a redacted audit record. The negative Cohen kappa indicates that the offline heuristic judge is not yet reliable enough for an automated release gate; it should be calibrated against more human-labelled pairs before production use. A production deployment should also measure the RAG and output layers separately under representative API traffic.

