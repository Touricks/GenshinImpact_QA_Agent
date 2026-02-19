"""Ingestion pipeline orchestrator."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from tqdm import tqdm

from .loader import DocumentLoader
from .chunker import SceneChunker
from .enricher import MetadataEnricher, create_chunks_from_document
from ..common.vector.indexer import VectorIndexer
from ..common.models import Chunk
from ..common.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from pipeline execution."""

    documents_processed: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "documents_processed": self.documents_processed,
            "documents_failed": self.documents_failed,
            "chunks_created": self.chunks_created,
            "chunks_indexed": self.chunks_indexed,
            "error_count": len(self.errors),
        }


class IngestionPipeline:
    """Orchestrate the full ingestion pipeline."""

    def __init__(
        self,
        data_dir: Path,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "genshin_story",
    ):
        """
        Initialize the pipeline.

        Args:
            data_dir: Path to the Data/ directory
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Qdrant collection name
        """
        self.data_dir = Path(data_dir)

        # Initialize components
        self.loader = DocumentLoader(self.data_dir)
        self.chunker = SceneChunker()
        self.enricher = MetadataEnricher()

        # BGE-M3 for dense + lexical sparse
        from ..common.vector.m3_embedder import BGEM3Embedder
        self.m3_embedder = BGEM3Embedder()

        self.indexer = VectorIndexer(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
        )

    def run(self, dry_run: bool = False, skip_embedding: bool = False) -> PipelineStats:
        """
        Execute the full pipeline.

        Args:
            dry_run: If True, parse and chunk but don't index
            skip_embedding: If True, skip embedding generation (for testing)

        Returns:
            PipelineStats with execution statistics
        """
        stats = PipelineStats()
        all_chunks: List[Chunk] = []

        # Reset enricher counter
        self.enricher.reset_counter()

        # Phase 1: Load and chunk documents
        logger.info("Phase 1: Loading and chunking documents...")
        documents = list(self.loader.load_all())
        logger.info(f"Found {len(documents)} documents")

        for doc in tqdm(documents, desc="Processing documents"):
            try:
                chunks = create_chunks_from_document(doc, self.chunker, self.enricher)
                all_chunks.extend(chunks)
                stats.documents_processed += 1
            except Exception as e:
                error_msg = f"Error processing {doc.metadata.file_path}: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
                stats.documents_failed += 1

        stats.chunks_created = len(all_chunks)
        logger.info(
            f"Created {stats.chunks_created} chunks from {stats.documents_processed} documents"
        )

        # Log chunk size statistics
        if all_chunks:
            sizes = [len(c.text) for c in all_chunks]
            logger.info(
                f"Chunk sizes - Min: {min(sizes)}, Max: {max(sizes)}, "
                f"Avg: {sum(sizes) // len(sizes)}"
            )

        if dry_run:
            logger.info("Dry run complete - skipping embedding and indexing")
            return stats

        # Phase 2: Generate embeddings (hybrid: BGE-M3 dense + lexical + BM25)
        if not skip_embedding:
            self._embed_hybrid(all_chunks)

        # Phase 3: Index to Qdrant
        logger.info("Phase 3: Indexing to Qdrant...")
        try:
            self.indexer.ensure_hybrid_collection()
            stats.chunks_indexed = self.indexer.upsert_hybrid_chunks(all_chunks)
            logger.info(f"Indexed {stats.chunks_indexed} chunks to Qdrant")
        except Exception as e:
            error_msg = f"Indexing failed: {e}"
            logger.error(error_msg)
            stats.errors.append(error_msg)

        logger.info(f"Pipeline complete. Stats: {stats.to_dict()}")
        return stats

    def _embed_hybrid(self, chunks: List[Chunk]):
        """Hybrid embedding: BGE-M3 (dense + lexical) + BM25 sparse."""
        texts = [c.text for c in chunks]

        # Phase 2a: BGE-M3 encode → dense (1024) + lexical sparse
        logger.info("Phase 2a: BGE-M3 encoding (dense + lexical sparse)...")
        m3_output = self.m3_embedder.encode(texts)

        for chunk, dense_vec, lexical_weights in zip(
            chunks, m3_output["dense"], m3_output["lexical"]
        ):
            chunk.embedding = self.m3_embedder.dense_to_list(dense_vec)
            chunk.sparse_lexical = self.m3_embedder.lexical_to_sparse(lexical_weights)

        logger.info(f"BGE-M3: {len(chunks)} chunks encoded (dense 1024 + lexical sparse)")

        # Phase 2b: BM25 sparse (jieba + TF-IDF)
        logger.info("Phase 2b: BM25 encoding (jieba + TF-IDF sparse)...")
        from ..common.vector.sparse_embedder import ChineseBM25Embedder, generate_jieba_dict

        # Generate jieba user dict from aliases
        generate_jieba_dict(settings.BM25_USER_DICT_PATH)

        bm25 = ChineseBM25Embedder(user_dict_path=settings.BM25_USER_DICT_PATH)
        bm25.fit(texts)

        for chunk in chunks:
            chunk.sparse_bm25 = bm25.transform(chunk.text)

        # Save BM25 model for runtime use
        bm25.save(settings.BM25_IDF_PATH)
        logger.info(f"BM25: {len(chunks)} chunks encoded, IDF saved to {settings.BM25_IDF_PATH}")

    def validate_chunks(self, chunks: List[Chunk]) -> dict:
        """
        Validate chunk quality.

        Returns:
            Dictionary with validation results
        """
        issues = {
            "empty_chunks": [],
            "too_small": [],
            "too_large": [],
            "no_characters": [],
        }

        for chunk in chunks:
            if not chunk.text.strip():
                issues["empty_chunks"].append(chunk.id)
            elif len(chunk.text) < 100:
                issues["too_small"].append(chunk.id)
            elif len(chunk.text) > 2000:
                issues["too_large"].append(chunk.id)

            if not chunk.metadata.characters:
                issues["no_characters"].append(chunk.id)

        return issues


def run_pipeline(
    data_dir: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    dry_run: bool = False,
) -> PipelineStats:
    """
    Convenience function to run the ingestion pipeline.

    Args:
        data_dir: Path to Data/ directory
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        dry_run: If True, skip indexing

    Returns:
        Pipeline execution statistics
    """
    pipeline = IngestionPipeline(
        data_dir=Path(data_dir),
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
    return pipeline.run(dry_run=dry_run)


@dataclass
class VectorFileTracking:
    """Tracking information for a processed file."""

    file_path: str
    content_hash: str
    last_indexed: str  # ISO timestamp
    chunk_count: int

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "last_indexed": self.last_indexed,
            "chunk_count": self.chunk_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VectorFileTracking":
        return cls(**data)


class IncrementalIngestionPipeline(IngestionPipeline):
    """Pipeline with change detection - skips unchanged files."""

    def __init__(
        self,
        data_dir: Path,
        tracking_file: Optional[Path] = None,
        **kwargs,
    ):
        """
        Initialize incremental pipeline.

        Args:
            data_dir: Path to the Data/ directory
            tracking_file: Path to tracking JSON file
            **kwargs: Additional arguments for IngestionPipeline
        """
        super().__init__(data_dir, **kwargs)

        from ..common.config import Settings

        cfg = Settings()
        self.tracking_file = tracking_file or cfg.VECTOR_TRACKING_FILE
        self.tracking: Dict[str, VectorFileTracking] = self._load_tracking()

    def _load_tracking(self) -> Dict[str, VectorFileTracking]:
        """Load tracking information from disk."""
        if not self.tracking_file.exists():
            return {}

        try:
            data = json.loads(self.tracking_file.read_text(encoding="utf-8"))
            return {
                k: VectorFileTracking.from_dict(v)
                for k, v in data.get("files", {}).items()
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load tracking file: {e}, starting fresh")
            return {}

    def _save_tracking(self) -> None:
        """Save tracking information to disk."""
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "files": {k: v.to_dict() for k, v in self.tracking.items()},
        }

        self.tracking_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug(f"Saved tracking to {self.tracking_file}")

    def _hash_file(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content."""
        content = file_path.read_text(encoding="utf-8")
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _is_changed(self, file_path: Path) -> bool:
        """Check if file is new or modified."""
        key = str(file_path)
        if key not in self.tracking:
            return True  # New file

        current_hash = self._hash_file(file_path)
        return current_hash != self.tracking[key].content_hash

    def _update_tracking(self, file_path: Path, chunk_count: int) -> None:
        """Update tracking for a processed file."""
        key = str(file_path)
        self.tracking[key] = VectorFileTracking(
            file_path=key,
            content_hash=self._hash_file(file_path),
            last_indexed=datetime.now().isoformat(),
            chunk_count=chunk_count,
        )

    def _find_all_files(self) -> List[Path]:
        """Find all dialogue files in data directory (recursive)."""
        return sorted(self.data_dir.rglob("chapter*_dialogue.txt"))

    def get_changed_files(self) -> List[Path]:
        """Get list of new or modified files."""
        all_files = self._find_all_files()
        return [f for f in all_files if self._is_changed(f)]

    def run(self, dry_run: bool = False, skip_embedding: bool = False) -> PipelineStats:
        """
        Execute incremental pipeline - skip unchanged files.

        Args:
            dry_run: If True, parse and chunk but don't index
            skip_embedding: If True, skip embedding generation

        Returns:
            PipelineStats with execution statistics
        """
        stats = PipelineStats()
        all_chunks: List[Chunk] = []

        # Reset enricher counter
        self.enricher.reset_counter()

        # Phase 1: Detect changed files
        logger.info("Phase 1: Detecting changed files...")
        all_files = self._find_all_files()
        changed_files = [f for f in all_files if self._is_changed(f)]

        logger.info(
            f"Found {len(all_files)} files total, {len(changed_files)} changed/new"
        )

        if not changed_files:
            logger.info("No changes detected, skipping processing")
            return stats

        # Track chunk counts per file for tracking update
        file_chunk_counts: Dict[str, int] = {}

        # Phase 2: Load and chunk only changed documents
        logger.info("Phase 2: Loading and chunking changed documents...")
        for file_path in tqdm(changed_files, desc="Processing changed files"):
            try:
                # Use DocumentLoader's private _parse_file method
                task_id = file_path.parent.name
                doc = self.loader._parse_file(file_path, task_id)
                if doc is None:
                    continue

                chunks = create_chunks_from_document(doc, self.chunker, self.enricher)
                all_chunks.extend(chunks)
                file_chunk_counts[str(file_path)] = len(chunks)
                stats.documents_processed += 1
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
                stats.documents_failed += 1

        stats.chunks_created = len(all_chunks)
        logger.info(
            f"Created {stats.chunks_created} chunks from {stats.documents_processed} documents"
        )

        if dry_run:
            logger.info("Dry run complete - skipping embedding and indexing")
            return stats

        # Phase 3: Generate embeddings (hybrid: BGE-M3 dense + lexical + BM25)
        if not skip_embedding and all_chunks:
            self._embed_hybrid(all_chunks)

        # Phase 4: Index to Qdrant
        if all_chunks:
            logger.info("Phase 4: Indexing to Qdrant...")
            try:
                self.indexer.ensure_hybrid_collection()
                stats.chunks_indexed = self.indexer.upsert_hybrid_chunks(all_chunks)
                logger.info(f"Indexed {stats.chunks_indexed} chunks to Qdrant")
            except Exception as e:
                error_msg = f"Indexing failed: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
                return stats

        # Phase 5: Update tracking
        logger.info("Phase 5: Updating tracking...")
        for file_path in changed_files:
            chunk_count = file_chunk_counts.get(str(file_path), 0)
            self._update_tracking(file_path, chunk_count)
        self._save_tracking()

        logger.info(f"Incremental pipeline complete. Stats: {stats.to_dict()}")
        return stats
