"""
Data models for the Genshin Story QA System.

This module provides dataclasses for:
- Entities: Character, Faction (Organization), Place (Location), Item, Concept, Event
- Relationships: Relationship, RelationType
- Documents: DocumentMetadata, RawDocument
- Chunks: Chunk, ChunkMetadata
"""

from .entities import (
    Character,
    Faction,
    Organization,  # backward compat alias for Faction
    Place,
    Location,  # backward compat alias for Place
    Item,
    Concept,
    Event,
    KNOWN_ORGANIZATIONS,
    MAIN_CHARACTERS,
    SYSTEM_CHARACTERS,
    ENTITY_ATTRIBUTE_TEMPLATES,
)
from .relationships import (
    Relationship,
    RelationType,
    VALID_RELATION_TYPES,
)
from .document import DocumentMetadata, RawDocument
from .chunk import Chunk, ChunkMetadata

__all__ = [
    # Entities
    "Character",
    "Faction",
    "Organization",
    "Place",
    "Location",
    "Item",
    "Concept",
    "Event",
    "KNOWN_ORGANIZATIONS",
    "MAIN_CHARACTERS",
    "SYSTEM_CHARACTERS",
    "ENTITY_ATTRIBUTE_TEMPLATES",
    # Relationships
    "Relationship",
    "RelationType",
    "VALID_RELATION_TYPES",
    # Documents
    "DocumentMetadata",
    "RawDocument",
    # Chunks
    "Chunk",
    "ChunkMetadata",
]
