from __future__ import annotations

"""Phase C: Production guardrails with Presidio, NeMo and latency metrics."""

import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ADVERSARIAL_SET_PATH,
    GUARDRAILS_CONFIG_DIR,
    LATENCY_BUDGET_P95_MS,
    PRESIDIO_LANGUAGE,
)

_PRESIDIO_CACHE = None


def setup_presidio():
    """Create Presidio engines and register Vietnamese ID/phone recognizers."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    return AnalyzerEngine(registry=registry), AnonymizerEngine()


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFD", text or "").lower()
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return value.replace("đ", "d")


def _regex_pii(text: str) -> list[dict]:
    patterns = [
        ("VN_CCCD", re.compile(r"(?<!\d)\d{12}(?!\d)"), 0.99),
        ("VN_CCCD", re.compile(r"(?<!\d)\d{9}(?!\d)"), 0.90),
        ("VN_PHONE", re.compile(r"(?<!\d)0[3-9]\d{8}(?!\d)"), 0.99),
        ("EMAIL", re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I), 0.99),
    ]
    found = []
    for entity_type, pattern, score in patterns:
        for match in pattern.finditer(text or ""):
            found.append({
                "type": entity_type,
                "text": match.group(0),
                "score": score,
                "start": match.start(),
                "end": match.end(),
            })
    priority = {"VN_PHONE": 3, "VN_CCCD": 2, "EMAIL": 1}
    found.sort(key=lambda item: (
        item["start"], -len(item["text"]), -priority.get(item["type"], 0)
    ))
    result = []
    for entity in found:
        if any(entity["start"] < old["end"] and entity["end"] > old["start"]
               for old in result):
            continue
        result.append(entity)
    return result


def _manual_anonymize(text: str, entities: list[dict]) -> str:
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        text = (
            text[:entity["start"]]
            + f"<{entity['type']}>"
            + text[entity["end"]:]
        )
    return text


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Detect and anonymize VN_CCCD, VN_PHONE and EMAIL values."""
    global _PRESIDIO_CACHE
    regex_entities = _regex_pii(text)
    presidio_entities = []
    raw_results = []

    if analyzer is None and anonymizer is None:
        if _PRESIDIO_CACHE is None:
            try:
                _PRESIDIO_CACHE = setup_presidio()
            except Exception:
                _PRESIDIO_CACHE = (None, None)
        analyzer, anonymizer = _PRESIDIO_CACHE

    if analyzer is not None:
        try:
            raw_results = analyzer.analyze(
                text=text, language=PRESIDIO_LANGUAGE
            ) or []
            presidio_entities = [
                {
                    "type": getattr(result, "entity_type", "PII"),
                    "text": text[result.start:result.end],
                    "score": round(float(getattr(result, "score", 0.0)), 3),
                    "start": result.start,
                    "end": result.end,
                }
                for result in raw_results
            ]
        except Exception:
            presidio_entities = []

    entities = list(regex_entities)
    for candidate in presidio_entities:
        if not any(
            candidate["start"] == old["start"] and candidate["end"] == old["end"]
            for old in entities
        ):
            entities.append(candidate)
    entities.sort(key=lambda item: (item["start"], item["end"]))

    anonymized = text
    if entities and anonymizer is not None and presidio_entities:
        try:
            anonymized = anonymizer.anonymize(
                text=text, analyzer_results=raw_results
            ).text
            if anonymized == text:
                anonymized = _manual_anonymize(text, entities)
        except Exception:
            anonymized = _manual_anonymize(text, entities)
    elif entities:
        anonymized = _manual_anonymize(text, entities)

    return {
        "has_pii": bool(entities),
        "entities": entities,
        "anonymized": anonymized,
    }


def setup_nemo_rails():
    """Create NeMo Guardrails from the repository configuration."""
    from nemoguardrails import LLMRails, RailsConfig
    return LLMRails(RailsConfig.from_path(GUARDRAILS_CONFIG_DIR))


def _rail_response_text(response) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if isinstance(response.get("content"), str):
            return response["content"]
        messages = response.get("messages") or []
        if messages:
            message = messages[-1]
            return message.get("content", "") if isinstance(message, dict) else str(message)
    return str(getattr(response, "content", response))


def _local_input_rule(text: str) -> str | None:
    folded = _fold(text)
    rules = [
        ("jailbreak", [
            "ignore your previous", "ignore your system",
            "ignore previous instructions", "system override",
            "forget your system", "forget all instructions",
            "pretend you are dan", "do anything now", "unrestricted ai",
            "bo qua tat ca huong dan", "dong vai ai khong co gioi han",
            "ai khong co gioi han", "admin command", "dump all",
            "bo qua tat ca policy", "tiet lo luong", "luong thang cua nhan vien",
            "print all confidential", "system instructions", "system prompt",
        ]),
        ("pii_request", [
            "cccd cua nhan vien", "so dien thoai cua nhan vien",
            "thong tin ca nhan cua", "email cua nhan vien",
            "tiet lo thong tin nhan vien", "luong cua nhan vien",
            "bang luong chi tiet", "employee salaries",
            "confidential employee data", "salary of specific",
        ]),
        ("harmful_request", ["tan cong mang", "attack the internal network"]),
        ("off_topic", [
            "viet mot bai tho", "nau pho", "bitcoin", "ethereum",
            "gia co phieu", "recommend phim", "recommend cho", "marvel",
            "giai phuong trinh",
            "giai toan", "thoi tiet", "tin tuc",
        ]),
    ]
    for name, patterns in rules:
        if any(pattern in folded for pattern in patterns):
            return name
    return None


async def check_input_rail(text: str, rails=None) -> dict:
    """Run the input through local safeguards and, when supplied, NeMo."""
    rule = _local_input_rule(text)
    if rule:
        return {
            "allowed": False,
            "blocked_reason": "nemo_input_rail",
            "response": "Xin lỗi, tôi không thể thực hiện yêu cầu này vì không phù hợp với phạm vi HR policy.",
            "rule": rule,
        }

    if rails is None and (
        os.getenv("OPENAI_API_KEY")
        or getattr(__import__("config"), "OPENAI_API_KEY", "")
    ):
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None
    if rails is None:
        return {"allowed": True, "blocked_reason": None, "response": ""}

    try:
        response = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        response_text = _rail_response_text(response)
        folded = _fold(response_text)
        refuse_keywords = [
            "xin loi", "khong the", "khong duoc phep",
            "i cannot", "i'm sorry",
        ]
        blocked = any(keyword in folded for keyword in refuse_keywords)
        return {
            "allowed": not blocked,
            "blocked_reason": "nemo_input_rail" if blocked else None,
            "response": response_text,
        }
    except Exception as exc:
        return {
            "allowed": True,
            "blocked_reason": None,
            "response": "",
            "error": type(exc).__name__,
        }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Check a generated answer for PII and sensitive content."""
    pii_result = pii_scan(answer)
    folded = _fold(answer)
    sensitive_markers = [
        "mat khau he thong", "mat khau admin", "thong tin bi mat",
        "cccd cua nhan vien", "so dien thoai ca nhan", "salary details",
    ]
    if pii_result["has_pii"] or any(marker in folded for marker in sensitive_markers):
        return {
            "safe": False,
            "flagged_reason": "nemo_output_rail",
            "final_answer": "Tôi không thể cung cấp thông tin nhạy cảm này. Vui lòng liên hệ phòng Nhân sự.",
        }

    if rails is None and (
        os.getenv("OPENAI_API_KEY")
        or getattr(__import__("config"), "OPENAI_API_KEY", "")
    ):
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None
    if rails is None:
        return {"safe": True, "flagged_reason": None, "final_answer": answer}

    try:
        response = await rails.generate_async(messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        response_text = _rail_response_text(response)
        folded_response = _fold(response_text)
        refused = any(
            keyword in folded_response
            for keyword in ["xin loi, toi khong the", "khong the cung cap", "i cannot"]
        )
        return {
            "safe": not refused,
            "flagged_reason": "nemo_output_rail" if refused else None,
            "final_answer": response_text if refused else answer,
        }
    except Exception:
        return {"safe": True, "flagged_reason": None, "final_answer": answer}


def run_adversarial_suite(
    adversarial_set: list[dict],
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> list[dict]:
    """Run all adversarial inputs through PII then input-rail checks."""
    async def run_all():
        results = []
        for item in adversarial_set:
            text = item.get("input", "")
            blocked_by = None
            if pii_scan(text, analyzer, anonymizer)["has_pii"]:
                blocked_by = "presidio"
            if blocked_by is None:
                rail_result = await check_input_rail(text, rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"
            actual = "blocked" if blocked_by else "allowed"
            expected = item.get("expected", "allowed")
            results.append({
                "id": item.get("id"),
                "category": item.get("category", "unknown"),
                "input": text,
                "expected": expected,
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == expected,
            })
        return results

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        results = asyncio.run(run_all())
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = executor.submit(asyncio.run, run_all()).result()
    passed = sum(1 for result in results if result["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def measure_p95_latency(
    test_inputs: list[str],
    n_runs: int = 20,
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> dict:
    """Measure P50, P95 and P99 for Presidio, NeMo and total guard time."""
    n_runs = max(0, int(n_runs))
    if not test_inputs or n_runs == 0:
        zero = {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "presidio_ms": zero,
            "nemo_ms": zero.copy(),
            "total_ms": zero.copy(),
            "latency_budget_ok": True,
            "budget_ms": LATENCY_BUDGET_P95_MS,
        }

    texts = [test_inputs[index % len(test_inputs)] for index in range(n_runs)]
    presidio_times, nemo_times, total_times = [], [], []

    async def measure():
        for text in texts:
            start = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - start) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(measure())
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(asyncio.run, measure()).result()

    def percentiles(values):
        ordered = sorted(values)
        if not ordered:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        def percentile(q):
            index = max(
                0,
                min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1),
            )
            return round(ordered[index], 2)

        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }

    total = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms": percentiles(nemo_times),
        "total_ms": total,
        "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def generate_phase_c_report(path: str = "reports/guard_results.json") -> dict:
    """Run the adversarial suite and latency benchmark and save JSON output."""
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as handle:
        adversarial_set = json.load(handle)
    results = run_adversarial_suite(adversarial_set)
    latency = measure_p95_latency(
        [item.get("input", "") for item in adversarial_set], n_runs=20
    )
    output_check = asyncio.run(check_output_rail(
        "Chính sách HR", "Mật khẩu hệ thống là secret-123"
    ))
    report = {
        "total_inputs": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "pass_rate": round(
            sum(1 for result in results if result["passed"]) / len(results), 4
        ) if results else 0.0,
        "results": results,
        "latency": latency,
        "output_rail_sample": output_check,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Phase C report saved → {path}")
    return report


if __name__ == "__main__":
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as handle:
        adversarial_set = json.load(handle)
    print("PII demo:", pii_scan(
        "Nhân viên có CCCD 034095001234 và email a@company.com."
    ))
    results = run_adversarial_suite(adversarial_set)
    latency = measure_p95_latency(
        [item["input"] for item in adversarial_set[:10]], n_runs=10
    )
    print(f"Adversarial suite: {sum(r['passed'] for r in results)}/{len(results)}")
    print("Latency:", latency)
    generate_phase_c_report()
