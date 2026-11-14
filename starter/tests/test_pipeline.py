import unittest

from starter.src.chunking import by_paragraph, fixed_size
from starter.src.evaluation import evaluate_query, summarize, validate_hit_provenance
from starter.src.ingestion import content_units, normalize_text
from starter.src.models import DocumentPage
from starter.src.store import QdrantVectorStore


class IngestionTests(unittest.TestCase):
    def test_normalization_preserves_paragraph_boundaries_and_unicode(self):
        self.assertEqual(normalize_text("Cafe\u0301 \t test\r\n\r\n next"), "Café test\n\nnext")

    def test_content_units_meet_minimum_and_keep_section_trace(self):
        paragraph = "A sufficiently long sentence about a source passage. " * 35
        source = (
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "CHAPTER I\n\n" + paragraph + "\n\n"
            "CHAPTER II\n\n" + paragraph + "\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***"
        )
        pages = content_units("test", "test.txt", source, min_chars=1500, max_chars=2500)
        self.assertGreaterEqual(len(pages), 2)
        self.assertTrue(all(len(page.text) >= 1500 for page in pages))
        self.assertTrue(all(page.metadata["section"] for page in pages))

    def test_chunkers_preserve_source_metadata(self):
        page = DocumentPage("book", "book.txt", "unit-1", "First paragraph.\n\nSecond paragraph.", {"section": "I"})
        fixed = fixed_size(page, size=12, overlap=3)
        paragraphs = by_paragraph(page, max_size=30)
        self.assertGreater(len(fixed), 1)
        self.assertEqual([chunk.text for chunk in paragraphs], ["First paragraph.", "Second paragraph."])
        self.assertTrue(all(chunk.page == "unit-1" and chunk.metadata["section"] == "I" for chunk in fixed + paragraphs))

    def test_chunker_rejects_invalid_overlap(self):
        page = DocumentPage("book", "book.txt", "unit-1", "text")
        with self.assertRaises(ValueError):
            fixed_size(page, size=10, overlap=10)

    def test_fixed_size_does_not_emit_a_redundant_tail(self):
        page = DocumentPage("book", "book.txt", "unit-1", "x" * 18)
        chunks = fixed_size(page, size=10, overlap=2)
        self.assertEqual([len(chunk.text) for chunk in chunks], [10, 10])
        self.assertEqual(chunks[-1].metadata["source_offset_end"], len(page.text))

    def test_long_paragraph_split_avoids_tiny_tail_and_preserves_source_text(self):
        text = "word " * 240 + "tail"
        page = DocumentPage("book", "book.txt", "unit-1", text)
        chunks = by_paragraph(page, max_size=1200)
        self.assertTrue(all(100 <= len(chunk.text) <= 1200 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertEqual(
            [(chunk.metadata["source_offset_start"], chunk.metadata["source_offset_end"]) for chunk in chunks],
            [(0, len(chunks[0].text)), (len(chunks[0].text), len(text))],
        )

    def test_long_paragraph_prefers_sentence_boundaries(self):
        text = "The report states a clear fact. " * 49 + "The report states a clear fact."
        page = DocumentPage("book", "book.txt", "unit-1", text)
        chunks = by_paragraph(page, max_size=1200)
        self.assertTrue(all(chunk.text.endswith(". ") for chunk in chunks[:-1]))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)


class EvaluationTests(unittest.TestCase):
    def test_canonical_gold_fields_are_evaluable(self):
        question = {
            "id": "q1",
            "gold_source_id": "book",
            "gold_page": "unit-1",
            "evidence_quote": "complete evidence quote",
        }
        hit = {"document_id": "book", "page": "unit-1", "text": "A complete evidence quote."}
        self.assertTrue(evaluate_query(question, [hit])["evidence_hit_at_3"])

    def test_evidence_hit_requires_both_gold_location_and_complete_quote(self):
        question = {
            "id": "q1",
            "gold": [{"document_id": "book", "page": "unit-1", "evidence_quote": "complete evidence quote"}],
        }
        wrong_location = {"document_id": "book", "page": "unit-2", "text": "complete evidence quote"}
        partial = {"document_id": "book", "page": "unit-1", "text": "complete evidence"}
        correct = {"document_id": "book", "page": "unit-1", "text": "A complete evidence quote."}
        self.assertFalse(evaluate_query(question, [wrong_location, partial])["evidence_hit_at_3"])
        self.assertTrue(evaluate_query(question, [partial, correct])["evidence_hit_at_3"])
        self.assertFalse(evaluate_query(question, [partial, correct])["hit_at_1"])

    def test_hit_provenance_checks_source_text_and_metadata(self):
        page = DocumentPage("book", "book.pdf", "1", "source chunk", {"section": "I"})
        hit = {
            "document_id": "book",
            "source": "book.pdf",
            "page": "1",
            "text": "source chunk",
            "metadata": {
                "section": "I",
                "source_offset_start": 0,
                "source_offset_end": 12,
            },
        }
        page_lookup = {("book", "1"): page}
        self.assertTrue(validate_hit_provenance(hit, page_lookup)["is_correct"])
        hit["text"] = "different text"
        self.assertFalse(validate_hit_provenance(hit, page_lookup)["is_correct"])

    def test_collection_prefix_keeps_new_run_separate(self):
        self.assertEqual(
            QdrantVectorStore._collection_name("A", "day06_benchmark_v3"),
            "day06_benchmark_v3_a",
        )

    def test_summary_reports_numerators_and_filter_compliance(self):
        traces = [
            {
                "strategy": "C",
                "split": "test",
                "filter": {"category": "Romance"},
                "evaluation": {"hit_at_1": True, "evidence_hit_at_3": True},
                "hits": [{"metadata": {"category": "Romance"}}, {"metadata": {"category": "Romance"}}],
            }
        ]
        result = summarize(traces)["C"]["test"]
        self.assertEqual(result["hit_at_1"], {"numerator": 1, "denominator": 1})
        self.assertEqual(
            {key: result["filter_compliance"][key] for key in ("numerator", "denominator", "filtered_questions")},
            {"numerator": 2, "denominator": 2, "filtered_questions": 1},
        )

    def test_summary_reports_retrieval_latency(self):
        traces = [
            {
                "strategy": "A",
                "split": "dev",
                "filter": {},
                "retrieval_seconds": 0.1,
                "evaluation": {"hit_at_1": False, "evidence_hit_at_3": False},
                "hits": [],
            },
            {
                "strategy": "A",
                "split": "dev",
                "filter": {},
                "retrieval_seconds": 0.2,
                "evaluation": {"hit_at_1": True, "evidence_hit_at_3": True},
                "hits": [],
            },
        ]
        self.assertEqual(
            summarize(traces)["A"]["dev"]["retrieval_latency_seconds"],
            {"count": 2, "mean": 0.15, "median": 0.15, "min": 0.1, "max": 0.2},
        )


if __name__ == "__main__":
    unittest.main()
