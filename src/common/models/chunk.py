"""Chunk models for the ingestion pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk stored in Qdrant payload."""

    # Document-level metadata
    task_id: str
    task_name: str
    chapter_number: int
    chapter_title: str
    series_name: Optional[str] = None

    # Chunk-level metadata
    scene_title: Optional[str] = None
    scene_order: int = 0  # Order within chapter
    chunk_order: int = 0  # Order within scene
    event_order: int = 0  # Global ordering for temporal queries
    characters: List[str] = field(default_factory=list)
    entities_mentioned: List[str] = field(default_factory=list)  # Multi-type entity matches
    has_choice: bool = False  # Whether chunk contains choice options
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Qdrant payload."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "series_name": self.series_name,
            "scene_title": self.scene_title,
            "scene_order": self.scene_order,
            "chunk_order": self.chunk_order,
            "event_order": self.event_order,
            "characters": self.characters,
            "entities_mentioned": self.entities_mentioned,
            "has_choice": self.has_choice,
            "source_file": self.source_file,
        }


@dataclass
class Chunk:
    """A text chunk ready for embedding and indexing."""

    id: str  # Unique identifier
    text: str  # Chunk content
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    # Sparse vectors for hybrid search (indices, values) — Qdrant SparseVector format
    sparse_bm25: Optional[tuple] = None      # (list[int], list[float])
    sparse_lexical: Optional[tuple] = None   # (list[int], list[float]) — BGE-M3 learned sparse

    def __len__(self) -> int:
        """Return text length."""
        return len(self.text)

    def to_dict(self) -> dict:
        """Convert to dictionary including text."""
        result = self.metadata.to_dict()
        result["text"] = self.text
        return result
