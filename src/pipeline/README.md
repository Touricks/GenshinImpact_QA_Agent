# Ingestion Module

Data ingestion pipelines for the dual-database architecture.

## Architecture

```
Dialogue Files (Data/Archon/{task_id}/chapter*_dialogue.txt)
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                      ▼
   Vector Pipeline                        Graph Pipeline
   (Semantic Search)                      (Structured Query)
         │                                      │
         ▼                                      ▼
      Qdrant                                 Neo4j
```

---

## Vector Pipeline

Processes dialogue files into searchable vector embeddings.

| Module | Class | Description |
|--------|-------|-------------|
| `loader.py` | `DocumentLoader` | Parse dialogue files, extract metadata |
| `chunker.py` | `SceneChunker` | Split by scene markers (`## 场景名`) |
| `enricher.py` | `MetadataEnricher` | Add character lists, compute stats |
| `indexer.py` | `VectorIndexer` | Upsert to Qdrant with payloads |
| `reranker.py` | `JinaReranker` | Rerank results (Jina Reranker v2) |
| `pipeline.py` | `IngestionPipeline` | Orchestrate full pipeline |

### Usage

```bash
python -m src.scripts.cli_vector Data/
python -m src.scripts.cli_vector Data/ --incremental
```

### Data Flow

```
Load → Chunk → Enrich → Embed → Index
                                  ↓
                              Qdrant
                          (genshin_story)
```

---

## Graph Pipeline

Extracts structured knowledge using LLM and writes to Neo4j.

| Module | Class | Description |
|--------|-------|-------------|
| `llm_kg_extractor.py` | `LLMKnowledgeGraphExtractor` | Extract entities & relationships |
| `event_extractor.py` | `LLMEventExtractor` | Extract major story events |
| `incremental_kg_extractor.py` | `IncrementalKGExtractor` | Incremental KG extraction with caching |
| `incremental_event_extractor.py` | `IncrementalEventExtractor` | Incremental event extraction with caching |
| `character_validator.py` | `CharacterValidator` | Filter invalid character names |
| `entity_normalizer.py` | `EntityNormalizer` | Map aliases to canonical names |

### Usage

```bash
# Extract KG (entities + relationships)
python -m src.ingestion.incremental_kg_extractor Data/Archon/1608
python -m src.ingestion.incremental_kg_extractor Data/Archon/1608 --write

# Extract Events
python -m src.ingestion.incremental_event_extractor Data/Archon/1608
python -m src.ingestion.incremental_event_extractor Data/Archon/1608 --write

# Status & Maintenance
python -m src.ingestion.incremental_kg_extractor --status
python -m src.ingestion.incremental_event_extractor --status

# Rebuild tracking from cache (fast recovery)
python -m src.ingestion.incremental_kg_extractor --rebuild Data/
python -m src.ingestion.incremental_event_extractor --rebuild Data/

# Cleanup orphan cache files
python -m src.ingestion.incremental_kg_extractor --cleanup --dry-run
python -m src.ingestion.incremental_kg_extractor --cleanup
```

### Data Flow

```
Dialogue → LLM Extract → Validate → Normalize → Neo4j
                                                  ↓
                                    Nodes: Character, Organization, MajorEvent
                                    Edges: FRIEND_OF, MEMBER_OF, EXPERIENCES...
```

---

## Incremental Extractors

Both `IncrementalKGExtractor` and `IncrementalEventExtractor` share the same design:

### Features

| Feature | Description |
|---------|-------------|
| **Content-addressed cache** | Cache keyed by MD5 hash of file content |
| **File tracking** | Track processed files to skip unchanged ones |
| **Incremental updates** | Only process new/modified files |
| **Cache rebuild** | Restore tracking from cache without re-extraction |
| **Orphan cleanup** | Remove cache files not referenced in tracking |

### Cache Structure

```
.cache/
├── kg/
│   ├── kg_tracking.json      # File tracking for KG extraction
│   └── {hash}.json           # Cached KG extraction results
└── events/
    ├── event_tracking.json   # File tracking for event extraction
    └── {hash}.json           # Cached event extraction results
```

### Tracking File Format

```json
{
  "version": "1.0",
  "last_updated": "2026-01-31T00:15:00",
  "files": {
    "Data/Archon/1608/chapter0_dialogue.txt": {
      "file_path": "Data/Archon/1608/chapter0_dialogue.txt",
      "content_hash": "d744c5b319a9a2b1b178f26808651533",
      "last_processed": "2026-01-31T00:10:00",
      "entity_count": 21,
      "relationship_count": 18,
      "task_id": "1608",
      "chapter": 160800
    }
  }
}
```

---

## Key Classes

### IncrementalKGExtractor

```python
from src.ingestion import IncrementalKGExtractor, write_kg_to_graph

extractor = IncrementalKGExtractor()

# Extract from folder
results = extractor.extract_folder(Path("Data/Archon/1608"))

# Write to Neo4j
write_kg_to_graph(results)

# Check status
status = extractor.get_status()
print(f"Tracked: {status['tracked_files']} files")
print(f"Entities: {status['total_entities']}")

# Rebuild tracking from cache
extractor.rebuild_tracking(Path("Data/"))

# Cleanup orphan cache
extractor.cleanup_orphan_cache(dry_run=True)
```

### IncrementalEventExtractor

```python
from src.ingestion import IncrementalEventExtractor, write_events_to_graph

extractor = IncrementalEventExtractor()

# Extract events
results = extractor.extract_folder(Path("Data/Archon/1608"))

# Write to Neo4j
write_events_to_graph(results)

# Rebuild tracking from cache
extractor.rebuild_tracking(Path("Data/"))
```

### CharacterValidator

```python
from src.ingestion import CharacterValidator

validator = CharacterValidator()
result = validator.validate("旅行者")  # ValidationResult(valid=True)
result = validator.validate("选项A")   # ValidationResult(valid=False, reason=SYSTEM_WORD)
```

---

## Environment Variables

```env
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM (for extraction)
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash
EVENT_EXTRACTOR_MODEL=gemini-2.5-flash
```
