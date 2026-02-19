"""
Data ingestion pipeline for processing dialogue files.

Vector Pipeline:
- DocumentLoader: Parse dialogue files from Data/ directory
- SceneChunker: Scene-based semantic chunking
- MetadataEnricher: Extract characters and compute metadata
- VectorIndexer: Index to Qdrant vector database
- JinaReranker: Rerank search results using Jina Reranker v2
- IngestionPipeline: Orchestrate the full pipeline

Graph Pipeline:
- LLMKnowledgeGraphExtractor: LLM-based entity/relationship extraction
- LLMEventExtractor: LLM-based major event extraction
- IncrementalKGExtractor: Incremental KG extraction with caching
- IncrementalEventExtractor: Incremental event extraction with caching
- CharacterValidator: Data quality validation for character names
- EntityNormalizer: Alias to canonical name mapping
"""

from .loader import DocumentLoader
from .chunker import SceneChunker
from .enricher import MetadataEnricher
from ..common.vector.indexer import VectorIndexer
from .reranker import JinaReranker
from .pipeline import IngestionPipeline, run_pipeline

# LLM-based KG extraction
from .llm_kg_extractor import (
    LLMKnowledgeGraphExtractor,
    KnowledgeGraphOutput,
    ExtractedEntity,
    ExtractedRelationship,
    extract_kg_from_text,
    extract_kg_from_file,
)

# Data quality validation
from .character_validator import (
    CharacterValidator,
    InvalidReason,
    ValidationResult,
    validate_character_name,
    filter_character_names,
)

# Entity normalization
from .entity_normalizer import EntityNormalizer

# LLM-based Event extraction
from .event_extractor import (
    LLMEventExtractor,
    EventExtractionOutput,
    ExtractedEvent,
    CharacterRole,
    EventType,
    extract_events_from_text,
    extract_events_from_file,
)
from .incremental_event_extractor import (
    IncrementalEventExtractor,
    EventCache,
    EventFileTrackingInfo,
    write_events_to_graph,
)
from .incremental_kg_extractor import (
    IncrementalKGExtractor,
    KGCache,
    KGFileTrackingInfo,
    write_kg_to_graph,
)

__all__ = [
    # Vector DB pipeline
    "DocumentLoader",
    "SceneChunker",
    "MetadataEnricher",
    "VectorIndexer",
    "JinaReranker",
    "IngestionPipeline",
    "run_pipeline",
    # LLM-based KG extraction
    "LLMKnowledgeGraphExtractor",
    "KnowledgeGraphOutput",
    "ExtractedEntity",
    "ExtractedRelationship",
    "extract_kg_from_text",
    "extract_kg_from_file",
    # Data quality validation
    "CharacterValidator",
    "InvalidReason",
    "ValidationResult",
    "validate_character_name",
    "filter_character_names",
    # Entity normalization
    "EntityNormalizer",
    # LLM-based Event extraction
    "LLMEventExtractor",
    "EventExtractionOutput",
    "ExtractedEvent",
    "CharacterRole",
    "EventType",
    "extract_events_from_text",
    "extract_events_from_file",
    # Incremental Event extraction
    "IncrementalEventExtractor",
    "EventCache",
    "EventFileTrackingInfo",
    "write_events_to_graph",
    # Incremental KG extraction
    "IncrementalKGExtractor",
    "KGCache",
    "KGFileTrackingInfo",
    "write_kg_to_graph",
]
