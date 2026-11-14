"""Sentence Transformers adapter; failures are surfaced, never replaced by mock vectors."""
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Real local dense embeddings with the same encoder for queries and documents."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install requirements from starter/requirements.txt to use embeddings") from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = self._model.get_sentence_embedding_dimension()
        if not self.dimensions:
            raise RuntimeError(f"Embedding model did not report dimensions: {model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding provider returned a different number of vectors than inputs")
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("Query must not be empty")
        vector = self._model.encode(
            [question],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vector.tolist()

    def count_tokens(self, text: str) -> int:
        """Count input tokens with the model's tokenizer (including special tokens)."""
        return len(self._model.tokenizer(text, add_special_tokens=True)["input_ids"])
