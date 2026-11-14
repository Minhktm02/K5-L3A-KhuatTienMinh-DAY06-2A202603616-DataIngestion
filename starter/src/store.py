"""Qdrant server vector store with exact cosine search and payload filters."""
import hashlib
import logging
import re
from typing import Any
from urllib.parse import urlparse

from .models import Chunk

LOGGER = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        dimensions: int,
        api_key: str | None = None,
        collection_prefix: str = "day06",
    ) -> None:
        if dimensions < 1:
            raise ValueError("Embedding dimensions must be positive")
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("QDRANT_URL must be an absolute http(s) URL")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Install requirements from starter/requirements.txt to use Qdrant") from exc
        self.dimensions = dimensions
        self.collection_prefix = self._safe_collection_part(collection_prefix)
        self._client = QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=30,
        )
        self.server_version = self._client.info().version

    def create_schema(self) -> None:
        """Collections are created per strategy when vectors are first upserted."""
        return None

    def count_points(self, strategy: str) -> int:
        collection = self._collection_name(strategy, self.collection_prefix)
        if not self._client.collection_exists(collection):
            return 0
        return self._client.count(collection_name=collection, exact=True).count

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert deterministic point IDs and remove obsolete points after a successful write."""
        from qdrant_client import models

        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector")
        if not chunks:
            raise ValueError("Cannot upsert an empty chunk set")
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError(f"All vectors must have {self.dimensions} dimensions")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique within a strategy")

        collection = self._collection_name(strategy, self.collection_prefix)
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=self.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )

        expected = {self._point_id(chunk.chunk_id) for chunk in chunks}
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
                **chunk.metadata,
            }
            points.append(
                models.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=collection, points=points, wait=True)

        stale_ids: list[str] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            stale_ids.extend(str(record.id) for record in records if str(record.id) not in expected)
            if offset is None:
                break
        if stale_ids:
            self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=stale_ids),
                wait=True,
            )
        point_count = self._client.count(collection_name=collection, exact=True).count
        if point_count != len(chunks):
            raise RuntimeError(
                f"Collection {collection} contains {point_count} points after upserting {len(chunks)} chunks"
            )
        LOGGER.info("Stored %d chunks in %s; removed %d stale points", len(points), collection, len(stale_ids))

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return exact top-k cosine results, applying filter fields to Qdrant payloads."""
        from qdrant_client import models

        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector must have {self.dimensions} dimensions")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        collection = self._collection_name(strategy, self.collection_prefix)
        if not self._client.collection_exists(collection):
            raise RuntimeError(f"No indexed collection for strategy {strategy}")
        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )
        result = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            search_params=models.SearchParams(exact=True),
            limit=top_k,
            with_payload=True,
        )
        hits = []
        for point in result.points:
            payload = point.payload or {}
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"chunk_id", "document_id", "source", "page", "text"}
            }
            hits.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "source": payload.get("source"),
                    "page": payload.get("page"),
                    "text": payload.get("text", ""),
                    "score": float(point.score),
                    "metadata": metadata,
                }
            )
        return hits

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _safe_collection_part(value: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
        if not safe:
            raise ValueError("Collection prefix and strategy must contain letters or digits")
        return safe

    @classmethod
    def _collection_name(cls, strategy: str, prefix: str = "day06") -> str:
        return f"{cls._safe_collection_part(prefix)}_{cls._safe_collection_part(strategy)}"

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        digest = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()
        return f"{digest[:8]}-{digest[8:12]}-5{digest[13:16]}-a{digest[17:20]}-{digest[20:32]}"
