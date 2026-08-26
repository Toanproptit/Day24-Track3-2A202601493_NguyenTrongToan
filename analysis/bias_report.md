# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Trọng Toàn  
**Ngày chạy lab:** 2026-08-26  
**Judge:** gpt-4o-mini when configured; deterministic fallback in this offline run

## 1. Evaluation Summary

Ten labelled examples were evaluated with pairwise judging and swap-and-average. The final judge labels were compared with human_labels_10q.json.

| Metric | Result |
|---|---:|
| Total judged | 10 |
| Cohen's kappa | -0.0714 |
| Position bias rate | 0.0% (0/10) |
| Verbosity bias | 75.0% (6/8 decisive cases) |
| Decisive cases | 8 |

## 2. Swap-and-Average

Both passes are converted back to the original A/B space before comparison. A case is position-consistent only when both passes select the same original answer. The observed rate was 100%, so no position-order instability appeared in this small offline sample.

## 3. Cohen's Kappa

The kappa value of -0.0714 is below zero, meaning agreement was slightly worse than chance for this run. It does not meet the substantial-agreement target of 0.6. The result is a warning against using this fallback judge as an automatic quality gate; more human-labelled examples and a calibrated rubric are needed.

## 4. Verbosity Bias

In decisive cases, the selected answer was also the longer answer in 6 of 8 cases. This 75% rate suggests the judge may reward extra detail even when correctness is not established. The next prompt version should score factual support first, cap length-related preferences, and include adversarial examples where a concise answer is the correct one.

## 5. Production Recommendation

Keep swap-and-average enabled because it makes positional instability observable. Require a minimum kappa on a fixed human-labelled calibration set, log both pass results, and send ties or low-confidence disagreements to human review. Do not treat verbosity as a proxy for completeness.

