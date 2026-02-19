# Data Ingestion Architecture Overview

> Complete guide to the dual-database ingestion system for the Genshin story knowledge base.

---

## Architecture Diagram

```
                              ┌──────────────────────────────────────────────────────────────┐
                              │                    Data Source Layer                          │
                              │                    Data/Archon/{task_id}/                     │
                              │                    chapter{N}_dialogue.txt                    │
                              └──────────────────────────┬───────────────────────────────────┘
                                                         │
                             ┌───────────────────────────┴───────────────────────────┐
                             ▼                                                       ▼
              ┌─────────────────────────────────────┐             ┌─────────────────────────────────────┐
              │       Vector Pipeline (Track 1)      │             │       Graph Pipeline (Track 2)       │
              │                                      │             │                                      │
              │  IncrementalIngestionPipeline        │             │  LLMKGExtractor + GraphBuilder       │
              │  ├─ Change Detection (MD5)           │             │  ├─ Change Detection (MD5)          │
              │  ├─ Load & Chunk                     │             │  ├─ LLM Entity/Relation Extraction  │
              │  ├─ Embed (BAAI/bge-base-zh-v1.5)    │             │  ├─ Entity Normalization            │
              │  └─ Index to Qdrant                  │             │  └─ Merge & Deduplicate             │
              └─────────────────┬───────────────────┘             └─────────────────┬───────────────────┘
                                │                                                   │
                                ▼                                                   ▼
              ┌─────────────────────────────────────┐             ┌─────────────────────────────────────┐
              │           Qdrant (Vector DB)         │             │           Neo4j (Graph DB)           │
              │                                      │             │                                      │
              │  Collection: genshin_story           │             │  Nodes: Character, Organization,    │
              │  Payload: text, task_id, chapter,    │             │          MajorEvent                  │
              │           characters, event_order    │             │  Edges: MEMBER_OF, FRIEND_OF,       │
              │                                      │             │          EXPERIENCES, etc.           │
              └─────────────────────────────────────┘             └─────────────────────────────────────┘
```

---

## Module Responsibility Matrix

| Module | Location | Responsibility |
|--------|----------|----------------|
| **DocumentLoader** | `src/ingestion/loader.py` | Parse dialogue files into documents |
| **SceneChunker** | `src/ingestion/chunker.py` | Split documents into semantic chunks |
| **EmbeddingGenerator** | `src/ingestion/embedder.py` | Generate embeddings using BGE model |
| **VectorIndexer** | `src/ingestion/indexer.py` | Qdrant upsert/search operations |
| **IncrementalIngestionPipeline** | `src/ingestion/pipeline.py` | Orchestrate vector incremental updates |
| **LLMKGExtractor** | `src/ingestion/llm_kg_extractor.py` | LLM-based entity/relation extraction |
| **LLMEventExtractor** | `src/ingestion/event_extractor.py` | LLM-based major event extraction |
| **IncrementalEventExtractor** | `src/ingestion/incremental_event_extractor.py` | Incremental event extraction with cache |
| **EntityNormalizer** | `src/ingestion/entity_normalizer.py` | Alias-to-canonical name mapping |
| **CharacterValidator** | `src/ingestion/character_validator.py` | Filter invalid character names |
| **GraphBuilder** | `src/graph/builder.py` | Neo4j node/edge creation |
| **GraphBuilder** | `src/graph/builder.py` | Neo4j node/edge creation with deduplication |

---

## Data Flow Summary

### Track 1: Vector Database (Semantic Search)

```
Dialogue File → Load → Chunk → Embed → Index → Qdrant
```

**Purpose**: Enable semantic search over dialogue content.

**Key Steps**:
1. **Change Detection**: Compare MD5 hash with `.cache/vector/tracking.json`
2. **Load**: Parse `chapter{N}_dialogue.txt` files
3. **Chunk**: Split by scene markers (e.g., `### 场景名称`)
4. **Embed**: Generate 768-dim vectors using BAAI/bge-base-zh-v1.5
5. **Index**: Upsert to Qdrant with metadata payloads

**Reference**: [vector-incremental-pipeline.md](./vector-incremental-pipeline.md)

---

### Track 2: Knowledge Graph (Structured Facts)

```
Dialogue File → LLM Extract → Normalize → Dedupe → Neo4j
```

**Purpose**: Store structured relationships (who knows whom, who did what).

**Key Steps**:
1. **Change Detection**: Compare MD5 hash with `.cache/kg/tracking.json`
2. **Extract**: LLM extracts entities and relationships
3. **Normalize**: Map aliases to canonical names via EntityNormalizer
4. **Dedupe**: GraphBuilder handles duplicate entities/relations via MERGE
5. **Write**: GraphBuilder creates/updates Neo4j nodes and edges

**Reference**: See `src/scripts/build_graph.py` for implementation

---

### Track 2.5: Event Extraction (Plot Events)

```
Dialogue File → LLM Extract Events → Cache → Neo4j MajorEvent
```

**Purpose**: Bridge abstract user queries to concrete narrative moments.

**Key Steps**:
1. **Change Detection**: Compare MD5 hash with `.cache/events/event_tracking.json`
2. **Extract**: LLMEventExtractor identifies major plot events
3. **Cache**: Results stored in `.cache/events/{hash}.json`
4. **Write**: GraphBuilder creates MajorEvent nodes and EXPERIENCES edges

**Reference**: [event-extraction-pipeline.md](./event-extraction-pipeline.md)

---

## Quick Start

### Full Initial Ingestion

```python
from pathlib import Path

# 1. Vector Pipeline (Full)
from src.ingestion.pipeline import IngestionPipeline

vector_pipeline = IngestionPipeline(data_dir=Path("Data/"))
stats = vector_pipeline.run()
print(f"Indexed {stats.chunks_indexed} chunks")

# 2. Graph Pipeline (Full) - Use CLI script
# python -m src.scripts.build_graph --clear

# Or programmatically:
from src.graph.builder import GraphBuilder

with GraphBuilder() as builder:
    builder.setup_schema()
    builder.create_seed_organizations()
    builder.create_seed_characters()
    # See src/scripts/build_graph.py for full pipeline
```

### Incremental Update

```python
from pathlib import Path

# Vector: Only process changed files
from src.ingestion.pipeline import IncrementalIngestionPipeline

pipeline = IncrementalIngestionPipeline(
    data_dir=Path("Data/"),
    tracking_file=Path(".cache/vector/tracking.json"),
)
changed = pipeline.get_changed_files()
print(f"Changed files: {len(changed)}")
stats = pipeline.run()

# Events: Only process changed files
from src.ingestion.incremental_event_extractor import (
    IncrementalEventExtractor,
    write_events_to_graph,
)

event_extractor = IncrementalEventExtractor()
results = event_extractor.extract_incremental(Path("Data/"))
write_events_to_graph(results)
```

---

## Environment Setup

```bash
# 1. Neo4j (Graph Database)
docker-compose up -d neo4j
# Port: 7687, User: neo4j, Password: (check .env)

# 2. Qdrant (Vector Database)
docker run -p 6333:6333 qdrant/qdrant

# 3. Environment Variables
# Copy .env.example to .env and configure:
#   - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
#   - QDRANT_HOST, QDRANT_PORT
#   - GEMINI_API_KEY (for LLM extraction)
```

---

## Cache & Tracking Structure

```
.cache/
├── vector/
│   └── tracking.json       # Vector pipeline file hashes
├── kg/
│   ├── tracking.json       # KG extraction file hashes
│   ├── snapshots/          # KG snapshots for rollback
│   └── {hash}.json         # Cached extraction results
└── events/
    ├── event_tracking.json # Event extraction file hashes
    └── {hash}.json         # Cached event extraction results
```

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [vector-incremental-pipeline.md](./vector-incremental-pipeline.md) | Qdrant incremental ingestion details |
| [event-extraction-pipeline.md](./event-extraction-pipeline.md) | MajorEvent extraction pipeline |
| [entity-normalization.md](./entity-normalization.md) | Alias mapping and character validation |
