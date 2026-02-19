"""Vector indexer for Qdrant storage."""

import logging
from typing import List, Optional

from ..models import Chunk
from ..config import Settings

logger = logging.getLogger(__name__)


class VectorIndexer:
    """Index chunks into Qdrant vector store."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        vector_size: int = None,
    ):
        """
        Initialize Qdrant indexer.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection
            vector_size: Dimension of embedding vectors
        """
        settings = Settings()
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.vector_size = vector_size or settings.EMBEDDING_DIM

        # Lazy load the client
        self._client = None

    @property
    def client(self):
        """Lazy load the Qdrant client."""
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self):
        """Connect to Qdrant server."""
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
        client = QdrantClient(host=self.host, port=self.port)

        # Verify connection
        try:
            client.get_collections()
            logger.info("Connected to Qdrant successfully")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

        return client

    def ensure_collection(self) -> bool:
        """
        Create collection if it doesn't exist.

        Returns:
            True if collection was created, False if it already existed
        """
        from qdrant_client.models import Distance, VectorParams

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            logger.info(f"Collection '{self.collection_name}' already exists")
            return False

        logger.info(f"Creating collection '{self.collection_name}'")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

        # Create payload indexes for filtering
        self._create_payload_indexes()

        logger.info(f"Collection '{self.collection_name}' created successfully")
        return True

    def _create_payload_indexes(self):
        """Create payload field indexes for efficient filtering."""
        from qdrant_client.models import PayloadSchemaType

        index_configs = [
            ("task_id", PayloadSchemaType.KEYWORD),
            ("chapter_number", PayloadSchemaType.INTEGER),
            ("characters", PayloadSchemaType.KEYWORD),
            ("entities_mentioned", PayloadSchemaType.KEYWORD),
            ("event_order", PayloadSchemaType.INTEGER),
            ("series_name", PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in index_configs:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.debug(f"Created index for field: {field_name}")
            except Exception as e:
                logger.warning(f"Failed to create index for {field_name}: {e}")

    def upsert_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> int:
        """
        Upsert chunks with embeddings to Qdrant.

        Args:
            chunks: List of Chunk objects with embeddings
            batch_size: Number of points per upsert batch

        Returns:
            Number of points upserted
        """
        from qdrant_client.models import PointStruct

        # Filter chunks that have embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]

        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"Skipping {len(chunks) - len(valid_chunks)} chunks without embeddings"
            )

        if not valid_chunks:
            logger.warning("No valid chunks to upsert")
            return 0

        # Convert to Qdrant points
        points = [
            PointStruct(
                id=hash(chunk.id) % (2**63),  # Convert string ID to int
                vector=chunk.embedding,
                payload={
                    "text": chunk.text,
                    **chunk.metadata.to_dict(),
                },
            )
            for chunk in valid_chunks
        ]

        # Batch upsert
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total_upserted += len(batch)
            logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")

        logger.info(f"Upserted {total_upserted} points to '{self.collection_name}'")
        return total_upserted

    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }

    def delete_collection(self) -> bool:
        """Delete the collection. Use with caution!"""
        logger.warning(f"Deleting collection '{self.collection_name}'")
        return self.client.delete_collection(self.collection_name)

    def _build_qdrant_filter(self, filter_conditions: Optional[dict]):
        """Build Qdrant Filter from condition dict.

        Args:
            filter_conditions: {"field": "value"} or {"field": ["v1", "v2"]}

        Returns:
            qdrant_client.models.Filter or None
        """
        if not filter_conditions:
            return None

        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        conditions = []
        for field, value in filter_conditions.items():
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(key=field, match=MatchAny(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
        return Filter(must=conditions)

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_conditions: Optional[dict] = None,
        sort_by: str = "relevance",
    ) -> List[dict]:
        """
        Search for similar vectors (legacy dense-only path).

        Args:
            query_vector: Query embedding vector
            limit: Number of results to return
            filter_conditions: Optional filter conditions.
                - 单值: {"field": "value"} → MatchValue
                - 多值: {"field": ["v1", "v2"]} → MatchAny (匹配任意一个)
            sort_by: Sort method ("relevance" or "time")

        Returns:
            List of search results with payload
        """
        qdrant_filter = self._build_qdrant_filter(filter_conditions)

        # For time sorting, we might want to fetch more candidates to ensure
        # we have a good timeline coverage, but for now we stick to limit.
        search_limit = limit if sort_by == "relevance" else limit * 2

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=search_limit,
            query_filter=qdrant_filter,
        )

        results = [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
            for r in response.points
        ]

        if sort_by == "time":
            # Sort by chapter_number first, then event_order
            # Default to 0 if missing
            results.sort(
                key=lambda x: (
                    x["payload"].get("chapter_number", 0),
                    x["payload"].get("event_order", 0),
                )
            )
            results = results[:limit]

        return results

    # ── Hybrid Search Methods ───────────────────────────────────────

    def ensure_hybrid_collection(self) -> bool:
        """Create collection with named dense + sparse vectors for hybrid search.

        Returns:
            True if collection was created, False if it already existed.
        """
        from qdrant_client.models import (
            Distance,
            VectorParams,
            SparseVectorParams,
            SparseIndexParams,
        )

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            logger.info(f"Collection '{self.collection_name}' already exists")
            return False

        logger.info(f"Creating hybrid collection '{self.collection_name}'")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse-bm25": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
                "sparse-lexical": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

        self._create_payload_indexes()
        logger.info(f"Hybrid collection '{self.collection_name}' created successfully")
        return True

    def upsert_hybrid_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> int:
        """Upsert chunks with dense + sparse vectors to a hybrid collection.

        Args:
            chunks: Chunks with embedding, sparse_bm25, sparse_lexical populated.
            batch_size: Batch size for upsert.

        Returns:
            Number of points upserted.
        """
        from qdrant_client.models import PointStruct, SparseVector

        valid_chunks = [c for c in chunks if c.embedding is not None]

        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"Skipping {len(chunks) - len(valid_chunks)} chunks without embeddings"
            )

        if not valid_chunks:
            logger.warning("No valid chunks to upsert")
            return 0

        points = []
        for chunk in valid_chunks:
            vector = {"dense": chunk.embedding}

            if chunk.sparse_bm25 and chunk.sparse_bm25[0]:
                indices, values = chunk.sparse_bm25
                vector["sparse-bm25"] = SparseVector(indices=indices, values=values)

            if chunk.sparse_lexical and chunk.sparse_lexical[0]:
                indices, values = chunk.sparse_lexical
                vector["sparse-lexical"] = SparseVector(indices=indices, values=values)

            points.append(
                PointStruct(
                    id=hash(chunk.id) % (2**63),
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        **chunk.metadata.to_dict(),
                    },
                )
            )

        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total_upserted += len(batch)
            logger.debug(f"Upserted hybrid batch {i // batch_size + 1}: {len(batch)} points")

        logger.info(f"Upserted {total_upserted} hybrid points to '{self.collection_name}'")
        return total_upserted

    def search_by_channel(
        self,
        channel: str,
        query_vector,
        limit: int,
        filter_conditions: Optional[dict] = None,
    ) -> List[dict]:
        """Query a single vector channel.

        Args:
            channel: "dense", "sparse-bm25", or "sparse-lexical"
            query_vector: Dense list[float] or SparseVector
            limit: Max results
            filter_conditions: Optional payload filter dict

        Returns:
            List of result dicts with _channel tag.
        """
        from qdrant_client.models import SparseVector

        qdrant_filter = self._build_qdrant_filter(filter_conditions)

        # For sparse channels, use query with named vector
        if channel.startswith("sparse-"):
            if isinstance(query_vector, (tuple, list)) and len(query_vector) == 2:
                indices, values = query_vector
                if not indices:
                    return []
                query_vector = SparseVector(indices=indices, values=values)
            elif not isinstance(query_vector, SparseVector):
                return []

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=channel,
            limit=limit,
            query_filter=qdrant_filter,
        )

        results = [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
                "_channel": channel,
            }
            for r in response.points
        ]
        return results

    def search_hybrid(
        self,
        query_dense: List[float],
        query_bm25: Optional[tuple] = None,
        query_lexical: Optional[tuple] = None,
        quota_dense: int = 10,
        quota_bm25: int = 10,
        quota_lexical: int = 10,
        filter_conditions: Optional[dict] = None,
    ) -> List[dict]:
        """Three-channel hybrid search: dense + BM25 sparse + M3 lexical sparse.

        Queries each channel independently, unions results, deduplicates by point ID
        (keeping highest score per point).

        Args:
            query_dense: Dense embedding vector (1024-dim).
            query_bm25: (indices, values) BM25 sparse vector. None to skip.
            query_lexical: (indices, values) BGE-M3 lexical sparse vector. None to skip.
            quota_dense: Number of candidates from dense channel (0 to skip).
            quota_bm25: Number of candidates from BM25 channel (0 to skip).
            quota_lexical: Number of candidates from lexical channel (0 to skip).
            filter_conditions: Optional payload filter.

        Returns:
            Deduplicated list of result dicts, each with _channel tag.
        """
        results = []

        if quota_dense > 0 and query_dense:
            dense_results = self.search_by_channel(
                "dense", query_dense, quota_dense, filter_conditions
            )
            results.extend(dense_results)
            logger.debug(f"[hybrid] dense: {len(dense_results)} results")

        if quota_bm25 > 0 and query_bm25 and query_bm25[0]:
            bm25_results = self.search_by_channel(
                "sparse-bm25", query_bm25, quota_bm25, filter_conditions
            )
            results.extend(bm25_results)
            logger.debug(f"[hybrid] bm25: {len(bm25_results)} results")

        if quota_lexical > 0 and query_lexical and query_lexical[0]:
            lexical_results = self.search_by_channel(
                "sparse-lexical", query_lexical, quota_lexical, filter_conditions
            )
            results.extend(lexical_results)
            logger.debug(f"[hybrid] lexical: {len(lexical_results)} results")

        deduped = self._deduplicate_by_id(results)
        logger.debug(f"[hybrid] total={len(results)}, deduped={len(deduped)}")
        return deduped

    @staticmethod
    def _deduplicate_by_id(results: List[dict]) -> List[dict]:
        """Deduplicate results by point ID, keeping the entry with highest score."""
        seen: dict[int, dict] = {}
        for r in results:
            rid = r["id"]
            if rid not in seen or r.get("score", 0) > seen[rid].get("score", 0):
                seen[rid] = r
        # Return sorted by score descending
        return sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
