"""Validate benchmark gold and calculate evidence retrieval metrics."""
import math
import re
import statistics
from typing import Any

from .models import DocumentPage


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def get_gold_evidence(question: dict[str, Any]) -> list[dict[str, str]]:
    """Read canonical benchmark gold fields or the legacy multi-evidence form."""
    golds = question.get("gold")
    if golds:
        return golds
    source_id = question.get("gold_source_id")
    page = question.get("gold_page")
    quote = question.get("evidence_quote")
    if source_id is None or page is None or not isinstance(quote, str):
        return []
    return [{
        "document_id": str(source_id),
        "page": str(page),
        "evidence_quote": quote,
    }]


def _metadata_matches(actual: Any, expected: Any) -> bool:
    return expected in actual if isinstance(actual, list) else actual == expected


def validate_benchmark(questions: list[dict[str, Any]], pages: list[DocumentPage]) -> None:
    """Fail before retrieval if splits, filters, or exact gold evidence are invalid."""
    if len(questions) < 20:
        raise ValueError(f"Benchmark has {len(questions)} questions; at least 20 are required")
    ids = [question.get("id") for question in questions]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("Benchmark question IDs must be present and unique")
    splits = {split: sum(question.get("split") == split for question in questions) for split in ("dev", "test")}
    if any(question.get("split") not in splits for question in questions):
        raise ValueError("Benchmark split must be either 'dev' or 'test'")
    if splits["dev"] < 15 or splits["test"] < 5:
        raise ValueError(f"Benchmark split must have at least 15 dev and 5 test questions; found {splits}")
    filtered = [question for question in questions if question.get("filter")]
    if len(filtered) < 3 or not any(question["split"] == "test" for question in filtered):
        raise ValueError("At least 3 questions need metadata filters, including at least 1 test question")

    page_lookup = {(page.document_id, page.page): page for page in pages}
    seen_questions: set[str] = set()
    for question in questions:
        question_id = question["id"]
        question_text = question.get("question")
        expected_answer = question.get("expected_answer")
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"Question {question_id} needs a non-empty question")
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise ValueError(f"Question {question_id} needs a non-empty expected answer")
        normalized_question = _normalized(question_text)
        if normalized_question in seen_questions:
            raise ValueError(f"Question {question_id} duplicates another benchmark question")
        seen_questions.add(normalized_question)
        if _normalized(expected_answer) in normalized_question:
            raise ValueError(f"Question {question_id} leaks its expected answer into the query")

        canonical_fields = ("gold_source_id", "gold_page", "evidence_quote")
        has_canonical = any(field in question for field in canonical_fields)
        if has_canonical and not all(field in question for field in canonical_fields):
            raise ValueError(f"Question {question_id} needs all canonical gold fields")
        golds = get_gold_evidence(question)
        if not golds:
            raise ValueError(f"Question {question_id} has no gold evidence")
        if has_canonical and question.get("gold"):
            first_gold = golds[0]
            canonical_gold = (
                str(question["gold_source_id"]),
                str(question["gold_page"]),
                question["evidence_quote"],
            )
            legacy_gold = (
                str(first_gold.get("document_id")),
                str(first_gold.get("page")),
                first_gold.get("evidence_quote"),
            )
            if canonical_gold != legacy_gold:
                raise ValueError(f"Question {question_id} has conflicting gold fields")

        filter_values = question.get("filter", {})
        if not isinstance(filter_values, dict):
            raise ValueError(f"Question {question_id} filter must be an object")
        for key, value in filter_values.items():
            if not isinstance(key, str) or not isinstance(value, str) or not value:
                raise ValueError(f"Question {question_id} filter keys and values must be non-empty strings")
            if value.casefold() not in question_text.casefold():
                raise ValueError(f"Question {question_id} does not provide filter value {value!r} in its query")

        for gold in golds:
            document_id = str(gold.get("document_id", ""))
            page_id = str(gold.get("page", ""))
            source_page = page_lookup.get((document_id, page_id))
            if source_page is None:
                raise ValueError(f"Question {question_id} points to an unknown source unit")
            quote = _normalized(gold.get("evidence_quote", ""))
            if not quote or quote not in _normalized(source_page.text):
                raise ValueError(f"Question {question_id} has evidence not present in its gold unit")
            if document_id.casefold() in question_text.casefold():
                raise ValueError(f"Question {question_id} includes its gold source ID")
            if re.search(rf"(?<!\w){re.escape(page_id)}(?!\w)", question_text):
                raise ValueError(f"Question {question_id} includes its gold page ID")
            if any(value == document_id or value == page_id for value in filter_values.values()):
                raise ValueError(f"Question {question_id} includes its gold source/page in the filter")
        if filter_values and not any(
            all(_metadata_matches(page.metadata.get(key), value) for key, value in filter_values.items())
            for page in pages
        ):
            raise ValueError(f"Question {question_id} uses a filter value absent from the corpus")


def validate_hit_provenance(
    hit: dict[str, Any],
    page_lookup: dict[tuple[str, str], DocumentPage],
) -> dict[str, bool]:
    """Check that returned payload metadata and text map back to the source page."""
    source_page = page_lookup.get((str(hit.get("document_id")), str(hit.get("page"))))
    metadata = hit.get("metadata", {})
    page_exists = source_page is not None
    source_matches = page_exists and hit.get("source") == source_page.source
    start = metadata.get("source_offset_start")
    end = metadata.get("source_offset_end")
    offsets_valid = (
        page_exists
        and isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start <= end <= len(source_page.text)
    )
    text_matches = offsets_valid and source_page.text[start:end] == hit.get("text", "")
    metadata_matches = page_exists and all(
        metadata.get(key) == value for key, value in source_page.metadata.items()
    )
    return {
        "source_page_exists": bool(page_exists),
        "source_matches": bool(source_matches),
        "offsets_valid": bool(offsets_valid),
        "text_matches_source": bool(text_matches),
        "metadata_matches_source": bool(metadata_matches),
        "is_correct": bool(page_exists and source_matches and offsets_valid and text_matches and metadata_matches),
    }


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """A hit requires a gold source/page and the complete verified evidence quote."""
    golds = get_gold_evidence(question)
    if not golds:
        raise ValueError(f"Question {question.get('id')} has no gold evidence")
    gold_keys = {(gold["document_id"], str(gold["page"])) for gold in golds}

    def contains_gold(hit: dict[str, Any]) -> bool:
        if (hit.get("document_id"), str(hit.get("page"))) not in gold_keys:
            return False
        chunk_text = _normalized(hit.get("text", ""))
        return any(
            (gold["document_id"], str(gold["page"])) == (hit.get("document_id"), str(hit.get("page")))
            and _normalized(gold["evidence_quote"]) in chunk_text
            for gold in golds
        )

    hit_flags = [contains_gold(hit) for hit in hits]
    return {
        "hit_at_1": bool(hit_flags and hit_flags[0]),
        "evidence_hit_at_3": any(hit_flags[:3]),
        "gold_matches": hit_flags[:3],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval, latency, filter, and source-provenance metrics."""
    summary: dict[str, Any] = {}
    keys = sorted({(trace["strategy"], trace["split"]) for trace in traces})
    for strategy, split in keys:
        rows = [trace for trace in traces if trace["strategy"] == strategy and trace["split"] == split]
        summary.setdefault(strategy, {})[split] = {
            "hit_at_1": {
                "numerator": sum(bool(row["evaluation"]["hit_at_1"]) for row in rows),
                "denominator": len(rows),
            },
            "evidence_hit_at_3": {
                "numerator": sum(bool(row["evaluation"]["evidence_hit_at_3"]) for row in rows),
                "denominator": len(rows),
            },
            "questions": len(rows),
        }
        latencies = [
            row["retrieval_seconds"]
            for row in rows
            if isinstance(row.get("retrieval_seconds"), (int, float))
            and row["retrieval_seconds"] >= 0
        ]
        if latencies:
            summary[strategy][split]["retrieval_latency_seconds"] = {
                "count": len(latencies),
                "mean": round(statistics.mean(latencies), 6),
                "median": round(statistics.median(latencies), 6),
                "min": round(min(latencies), 6),
                "max": round(max(latencies), 6),
            }

        latency_trials = [trial for row in rows for trial in row.get("latency_trials", [])]
        if latency_trials:
            fields = {
                "query_embedding": "query_embedding_seconds",
                "qdrant_search": "qdrant_search_seconds",
                "end_to_end_retrieval": "end_to_end_retrieval_seconds",
            }
            latency_components = {}
            for name, field in fields.items():
                values = sorted(float(trial[field]) for trial in latency_trials)
                p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
                latency_components[name] = {
                    "trial_count": len(values),
                    "query_count": len(rows),
                    "mean": round(statistics.mean(values), 6),
                    "median": round(statistics.median(values), 6),
                    "p95": round(values[p95_index], 6),
                    "min": round(values[0], 6),
                    "max": round(values[-1], 6),
                    "unit": "seconds",
                }
            summary[strategy][split]["latency_components"] = latency_components

        filter_rows = [row for row in rows if row.get("filter")]
        if filter_rows:
            compliant = sum(
                all(
                    _metadata_matches(hit.get("metadata", {}).get(key), value)
                    for key, value in row["filter"].items()
                )
                for row in filter_rows
                for hit in row["hits"]
            )
            total = sum(len(row["hits"]) for row in filter_rows)
            summary[strategy][split]["filter_compliance"] = {
                "numerator": compliant,
                "denominator": total,
                "filtered_questions": len(filter_rows),
                "queries_with_results": sum(bool(row["hits"]) for row in filter_rows),
                "empty_result_queries": sum(not row["hits"] for row in filter_rows),
            }

        provenance_flags = [
            hit["metadata_validation"]["is_correct"]
            for row in rows
            for hit in row["hits"]
            if "metadata_validation" in hit
        ]
        if provenance_flags:
            summary[strategy][split]["metadata_correctness"] = {
                "numerator": sum(provenance_flags),
                "denominator": len(provenance_flags),
            }
    return summary
