"""Run ingestion and retrieval benchmark for the three Day 06 strategies."""
import json
import logging
import os
import statistics
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize, validate_benchmark
from .ingestion import load_manifest
from .store import QdrantVectorStore

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STRATEGIES = ("A", "B", "C")
FIXED_SIZE = 1200
FIXED_OVERLAP = 120
PARAGRAPH_MAX_SIZE = 1200


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _chunk_statistics(chunks: list, embedder: MyEmbedder) -> dict:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "total_characters": sum(lengths),
        "min_characters": min(lengths),
        "median_characters": statistics.median(lengths),
        "max_characters": max(lengths),
        "total_model_tokens": sum(embedder.count_tokens(chunk.text) for chunk in chunks),
    }


def _audit_candidates(pages: list, chunks_by_strategy: dict[str, list], traces: list[dict]) -> dict:
    selected_pages = sorted(pages, key=lambda page: (page.document_id, page.page))[:10]
    coherence = []
    for page in selected_pages:
        offset = min(200, len(page.text) - 1)
        candidates = {"document_id": page.document_id, "page": page.page, "source_offset": offset}
        for strategy in STRATEGIES:
            matches = [
                chunk
                for chunk in chunks_by_strategy[strategy]
                if chunk.document_id == page.document_id
                and chunk.page == page.page
                and chunk.metadata["source_offset_start"] <= offset < chunk.metadata["source_offset_end"]
            ]
            candidates[strategy] = [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "label_0_to_2": None,
                    "reason": "",
                }
                for chunk in matches
            ]
        coherence.append(candidates)

    test_ids = sorted({trace["question_id"] for trace in traces if trace["split"] == "test"})
    test_audits = []
    for strategy in STRATEGIES:
        for question_id in test_ids:
            trace = next(
                item for item in traces
                if item["strategy"] == strategy and item["question_id"] == question_id
            )
            test_audits.append(
                {
                    "strategy": strategy,
                    "question_id": question_id,
                    "filter": trace["filter"],
                    "hits": [
                        {
                            "rank": rank,
                            "chunk_id": hit["chunk_id"],
                            "document_id": hit["document_id"],
                            "page": hit["page"],
                            "source": hit["source"],
                            "metadata": hit["metadata"],
                            "text": hit["text"],
                            "grounding_label_0_to_2": None,
                            "reason": "",
                            "metadata_correct": None,
                        }
                        for rank, hit in enumerate(trace["hits"], start=1)
                    ],
                }
            )
    return {
        "method": (
            "Coherence: first 10 sorted source units, offset 200 (fixed before retrieval); "
            "label the chunk containing that position. Metadata and grounding: all top-3 hits "
            "for the 5 held-out test questions; labels require source comparison."
        ),
        "coherence": coherence,
        "test_metadata_and_grounding": test_audits,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(ROOT / "starter" / ".env")
    data_dir = Path(os.getenv("DAY6_DATA_DIR", str(ROOT / "data"))).resolve()
    results_dir = Path(os.getenv("DAY6_RESULTS_DIR", str(ROOT / "starter" / "results"))).resolve()
    manifest_path = data_dir / "manifest.json"
    benchmark_path = data_dir / "benchmark.json"
    if not manifest_path.is_file() or not benchmark_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path} or {benchmark_path}; prepare the corpus and benchmark first."
        )
    pages, manifest = load_manifest(manifest_path)
    questions = json.loads(benchmark_path.read_text(encoding="utf-8"))["questions"]
    validate_benchmark(questions, pages)

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
    embedder = MyEmbedder(model_name)
    LOGGER.info("Using %s (%d dimensions)", embedder.model_name, embedder.dimensions)

    chunks_by_strategy = {
        "A": [chunk for page in pages for chunk in fixed_size(page, FIXED_SIZE, FIXED_OVERLAP)],
        "B": [chunk for page in pages for chunk in by_paragraph(page, PARAGRAPH_MAX_SIZE)],
    }
    chunks_by_strategy["C"] = chunks_by_strategy["B"]
    traces: list[dict] = []
    resources = {}
    cached_vectors: dict[str, list[list[float]]] = {}
    store = QdrantVectorStore(qdrant_url, embedder.dimensions, qdrant_api_key)
    try:
        for strategy in STRATEGIES:
            chunks = chunks_by_strategy[strategy]
            started = time.perf_counter()
            if strategy == "C":
                vectors = cached_vectors["B"]
                embedding_seconds = 0.0
            else:
                embed_started = time.perf_counter()
                vectors = embedder.embed_documents([chunk.text for chunk in chunks])
                embedding_seconds = time.perf_counter() - embed_started
                if strategy == "B":
                    cached_vectors["B"] = vectors
            upsert_started = time.perf_counter()
            point_count_before = store.count_points(strategy)
            store.upsert(strategy, chunks, vectors)
            upsert_seconds = time.perf_counter() - upsert_started
            resources[strategy] = {
                **_chunk_statistics(chunks, embedder),
                "qdrant_points_before": point_count_before,
                "qdrant_points_after": store.count_points(strategy),
                "embedding_seconds": round(embedding_seconds, 3),
                "vector_store_seconds": round(upsert_seconds, 3),
                "ingestion_seconds": round(time.perf_counter() - started, 3),
                "estimated_cost_usd": 0.0,
            }

        query_vectors = {
            question["id"]: embedder.embed_query(question["question"])
            for question in questions
        }
        for strategy in STRATEGIES:
            for question in questions:
                applied_filter = question.get("filter", {}) if strategy == "C" else {}
                search_started = time.perf_counter()
                hits = store.search(
                    strategy,
                    query_vectors[question["id"]],
                    top_k=3,
                    filters=applied_filter or None,
                )
                retrieval_seconds = time.perf_counter() - search_started
                evaluation = evaluate_query(question, hits)
                traces.append(
                    {
                        "strategy": strategy,
                        "split": question["split"],
                        "question_id": question["id"],
                        "question": question["question"],
                        "filter": applied_filter,
                        "gold": question["gold"],
                        "expected_answer": question["expected_answer"],
                        "retrieval_seconds": round(retrieval_seconds, 6),
                        "evaluation": evaluation,
                        "hits": hits,
                    }
                )
    finally:
        store.close()

    metrics = summarize(traces)
    for strategy in STRATEGIES:
        metrics.setdefault(strategy, {}).update(
            {
                "configuration": {
                    "chunking": (
                        f"fixed_size={FIXED_SIZE}, overlap={FIXED_OVERLAP}"
                        if strategy == "A"
                        else f"paragraph_max_size={PARAGRAPH_MAX_SIZE}"
                    ),
                    "metadata_filter": "question-provided payload filter" if strategy == "C" else "none",
                    "embedding_model": embedder.model_name,
                    "embedding_dimensions": embedder.dimensions,
                    "vector_store": "Qdrant Docker server",
                    "qdrant_server_version": store.server_version,
                    "qdrant_url": qdrant_url,
                    "distance": "cosine",
                    "search": "exact",
                    "resource_metrics": resources[strategy],
                }
            }
        )
    metadata = {
        "run_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_content_units": len(pages),
        "corpus_characters": sum(len(page.text) for page in pages),
        "benchmark_questions": len(questions),
        "split_counts": dict(Counter(question["split"] for question in questions)),
        "model": model_name,
        "embedding_dimensions": embedder.dimensions,
        "versions": {
            "python_packages": {
                "qdrant-client": _package_version("qdrant-client"),
                "sentence-transformers": _package_version("sentence-transformers"),
                "pypdf": _package_version("pypdf"),
            }
        },
        "cost_usd": 0.0,
        "selection_rule": "Select on dev by Evidence Hit@3, then grounding, then resource use; do not tune on test.",
    }
    _write_json(results_dir / "run_metadata.json", metadata)
    _write_json(results_dir / "metrics.json", metrics)
    _write_json(results_dir / "resource_metrics.json", resources)
    (results_dir / "traces.jsonl").write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )
    _write_json(results_dir / "audit_candidates.json", _audit_candidates(pages, chunks_by_strategy, traces))
    LOGGER.info("Wrote traces and results to %s", results_dir)


if __name__ == "__main__":
    main()
