"""Metadata enricher for extracting characters and computing event order."""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..common.models import DocumentMetadata, ChunkMetadata, Chunk, SYSTEM_CHARACTERS

logger = logging.getLogger(__name__)

# Default path for alias dictionary — resolved via ACTIVE_REGION
def _get_aliases_path() -> Path:
    from ..common.config.settings import settings
    return Path(__file__).parent.parent / "common" / "config" / "regions" / settings.ACTIVE_REGION / "aliases.json"


def _load_alias_dict(path: Path) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Load alias dictionary and build lookup structures.

    Returns:
        (alias_to_canonical, sorted_search_terms):
        - alias_to_canonical: maps each alias string to its canonical name
        - sorted_search_terms: list of (alias, canonical) sorted longest-first
    """
    if not path.exists():
        logger.warning(f"Alias dictionary not found: {path}")
        return {}, []

    data = json.loads(path.read_text(encoding="utf-8"))
    alias_to_canonical: Dict[str, str] = {}

    for canonical_name, entry in data.items():
        canonical_zh = entry.get("canonical_zh", canonical_name)
        for alias in entry.get("aliases", []):
            alias_to_canonical[alias] = canonical_zh

    # Sort by alias length descending to match longer names first
    sorted_terms = sorted(alias_to_canonical.items(), key=lambda x: len(x[0]), reverse=True)
    return alias_to_canonical, sorted_terms


class MetadataEnricher:
    """Enrich chunks with extracted metadata."""

    # Character extraction pattern
    CHARACTER_PATTERN = re.compile(r"^([^：\n]+)：", re.MULTILINE)

    # Choice pattern
    CHOICE_PATTERN = re.compile(r"^##?\s*选项|^-\s+.+$", re.MULTILINE)

    def __init__(self, aliases_path: Optional[Path] = None):
        """Initialize enricher with optional alias dictionary."""
        self._chunk_counter = 0
        path = aliases_path or _get_aliases_path()
        self._alias_to_canonical, self._sorted_terms = _load_alias_dict(path)

    def reset_counter(self):
        """Reset the chunk counter for a new ingestion run."""
        self._chunk_counter = 0

    def enrich(
        self,
        chunk_text: str,
        scene_title: Optional[str],
        doc_metadata: DocumentMetadata,
        scene_order: int,
        chunk_order: int,
    ) -> Chunk:
        """
        Create an enriched Chunk from text and metadata.

        Args:
            chunk_text: The text content of the chunk
            scene_title: Title of the scene (from ## header)
            doc_metadata: Metadata from the source document
            scene_order: Order of scene within the document
            chunk_order: Order of chunk within the scene

        Returns:
            Enriched Chunk object
        """
        # Extract characters
        characters = self._extract_characters(chunk_text)

        # Extract multi-type entity mentions via alias dictionary
        entities_mentioned = self._extract_entities_mentioned(chunk_text)

        # Compute event order for temporal sorting
        event_order = self._compute_event_order(doc_metadata, scene_order, chunk_order)

        # Check if chunk contains choices
        has_choice = self._has_choice(chunk_text)

        # Generate unique ID
        chunk_id = self._generate_chunk_id(doc_metadata, scene_order, chunk_order)

        # Create metadata
        metadata = ChunkMetadata(
            task_id=doc_metadata.task_id,
            task_name=doc_metadata.task_name,
            chapter_number=doc_metadata.chapter_number,
            chapter_title=doc_metadata.chapter_title,
            series_name=doc_metadata.series_name,
            scene_title=scene_title,
            scene_order=scene_order,
            chunk_order=chunk_order,
            event_order=event_order,
            characters=characters,
            entities_mentioned=entities_mentioned,
            has_choice=has_choice,
            source_file=str(doc_metadata.file_path) if doc_metadata.file_path else None,
        )

        return Chunk(id=chunk_id, text=chunk_text, metadata=metadata)

    def _extract_characters(self, text: str) -> List[str]:
        """Extract unique character names from dialogue text."""
        matches = self.CHARACTER_PATTERN.findall(text)

        # Filter and deduplicate
        characters = set()
        for name in matches:
            name = name.strip()
            # Skip system characters (exact match) and choice markers (prefix match)
            if not name:
                continue
            if name in SYSTEM_CHARACTERS:
                continue
            if name.startswith("选项"):
                continue
            characters.add(name)

        return sorted(list(characters))

    def _extract_entities_mentioned(self, text: str) -> List[str]:
        """Extract entities mentioned in text using the alias dictionary.

        Scans text for all known aliases (longest-first) and returns
        the deduplicated list of canonical entity names.
        """
        if not self._sorted_terms:
            return []

        found: Set[str] = set()
        for alias, canonical in self._sorted_terms:
            if alias in text:
                found.add(canonical)
        return sorted(found)

    def _compute_event_order(
        self, metadata: DocumentMetadata, scene_order: int, chunk_order: int
    ) -> int:
        """
        Compute global event order for temporal queries.

        Format: task_id * 10000 + chapter * 1000 + scene * 10 + chunk
        """
        try:
            task_base = int(metadata.task_id) * 10000
        except ValueError:
            task_base = 0

        chapter_offset = metadata.chapter_number * 1000
        scene_offset = scene_order * 10

        return task_base + chapter_offset + scene_offset + chunk_order

    def _has_choice(self, text: str) -> bool:
        """Check if the chunk contains player choices."""
        return bool(self.CHOICE_PATTERN.search(text))

    def _generate_chunk_id(
        self, metadata: DocumentMetadata, scene_order: int, chunk_order: int
    ) -> str:
        """Generate a unique chunk ID."""
        self._chunk_counter += 1
        return f"{metadata.task_id}_{metadata.chapter_number}_{scene_order}_{chunk_order}_{self._chunk_counter}"


def create_chunks_from_document(
    document,  # RawDocument
    chunker,  # SceneChunker
    enricher: MetadataEnricher,
) -> List[Chunk]:
    """
    Create enriched chunks from a document.

    Args:
        document: RawDocument to process
        chunker: SceneChunker instance
        enricher: MetadataEnricher instance

    Returns:
        List of enriched Chunk objects
    """
    # Get scene chunks
    scene_chunks = chunker.chunk_document(document)

    # Optionally merge small chunks
    scene_chunks = chunker.merge_small_chunks(scene_chunks)

    # Enrich each chunk
    chunks = []
    current_scene_title = None
    scene_order = 0
    chunk_order = 0

    for scene_title, chunk_text in scene_chunks:
        # Track scene changes
        if scene_title != current_scene_title:
            current_scene_title = scene_title
            scene_order += 1
            chunk_order = 0
        else:
            chunk_order += 1

        chunk = enricher.enrich(
            chunk_text=chunk_text,
            scene_title=scene_title,
            doc_metadata=document.metadata,
            scene_order=scene_order,
            chunk_order=chunk_order,
        )
        chunks.append(chunk)

    return chunks
