# Vector Database Incremental Pipeline

> Internal documentation for the Qdrant vector database incremental ingestion system.

---

## Overview

The incremental pipeline enables efficient updates to the Qdrant vector database by only processing files that have changed since the last ingestion. This avoids re-embedding and re-indexing unchanged content, significantly reducing processing time for partial updates.

### When to Use

| Scenario | Pipeline |
|----------|----------|
| Initial setup / full rebuild | `IngestionPipeline` |
| Adding new dialogue files | `IncrementalIngestionPipeline` |
| Modifying existing files | `IncrementalIngestionPipeline` |
| Clearing and rebuilding | `IngestionPipeline` + `delete_collection()` |

---

## Architecture

```
IncrementalIngestionPipeline
         │
         ├── Phase 1: Change Detection
         │       └── Compare MD5 hashes with tracking.json
         │
         ├── Phase 2: Load & Chunk (changed files only)
         │       ├── DocumentLoader._parse_file()
         │       ├── SceneChunker.chunk()
         │       └── MetadataEnricher.enrich()
         │
         ├── Phase 3: Generate Embeddings
         │       └── EmbeddingGenerator.embed_texts()
         │
         ├── Phase 4: Index to Qdrant
         │       └── VectorIndexer.upsert_chunks()
         │
         └── Phase 5: Update Tracking
                 └── Save file hashes to tracking.json
```

---

## Core Components

### 1. VectorIndexer

**File**: `src/ingestion/indexer.py`

Handles all Qdrant database operations.

```python
from src.ingestion import VectorIndexer

indexer = VectorIndexer(
    host="localhost",
    port=6333,
    collection_name="genshin_story",
    vector_size=768,
)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `ensure_collection()` | Create collection with indexes if not exists |
| `upsert_chunks(chunks, batch_size=100)` | Batch insert/update chunks |
| `search(query_vector, limit, filter_conditions)` | Similarity search |
| `get_collection_info()` | Get collection statistics |
| `delete_collection()` | Remove entire collection |

**Payload Indexes Created:**

```python
# Automatically created for efficient filtering
- task_id: KEYWORD
- chapter_number: INTEGER
- characters: KEYWORD
- event_order: INTEGER
- series_name: KEYWORD
```

---

### 2. IncrementalIngestionPipeline

**File**: `src/ingestion/pipeline.py`

Extends `IngestionPipeline` with change detection.

```python
from pathlib import Path
from src.ingestion.pipeline import IncrementalIngestionPipeline

pipeline = IncrementalIngestionPipeline(
    data_dir=Path("Data/"),
    tracking_file=Path(".cache/vector/tracking.json"),
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="genshin_story",
    batch_size=64,
    device="auto",  # auto, cpu, cuda, mps
)

# Run incremental update
stats = pipeline.run(dry_run=False)
print(f"Processed: {stats.documents_processed}")
print(f"Chunks indexed: {stats.chunks_indexed}")
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `get_changed_files()` | List files that are new or modified |
| `run(dry_run=False)` | Execute incremental pipeline |
| `_is_changed(file_path)` | Check if file hash differs from tracking |
| `_update_tracking(file_path, chunk_count)` | Update tracking after processing |

---

### 3. VectorFileTracking

**File**: `src/ingestion/pipeline.py`

Dataclass for tracking processed files.

```python
@dataclass
class VectorFileTracking:
    file_path: str        # Absolute path to file
    content_hash: str     # MD5 hash of file content
    last_indexed: str     # ISO timestamp of last processing
    chunk_count: int      # Number of chunks created
```

**Tracking File Format** (`.cache/vector/tracking.json`):

```json
{
  "version": "1.0",
  "last_updated": "2026-01-29T22:45:00",
  "files": {
    "/path/to/Data/1600/chapter0_dialogue.txt": {
      "file_path": "/path/to/Data/1600/chapter0_dialogue.txt",
      "content_hash": "a1b2c3d4e5f6...",
      "last_indexed": "2026-01-29T22:45:00",
      "chunk_count": 42
    }
  }
}
```

---

## Usage Examples

### Full Pipeline (Initial Setup)

```python
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline(
    data_dir=Path("Data/"),
    qdrant_host="localhost",
    qdrant_port=6333,
)
stats = pipeline.run()
```

### Incremental Update

```python
from pathlib import Path
from src.ingestion.pipeline import IncrementalIngestionPipeline

pipeline = IncrementalIngestionPipeline(
    data_dir=Path("Data/"),
    tracking_file=Path(".cache/vector/tracking.json"),
)

# Check what files have changed
changed = pipeline.get_changed_files()
print(f"Changed files: {len(changed)}")

# Run incremental update
stats = pipeline.run()
```

### Clear and Rebuild

```python
from src.ingestion import VectorIndexer
from src.ingestion.pipeline import IngestionPipeline

# Delete existing collection
indexer = VectorIndexer()
indexer.delete_collection()

# Full rebuild
pipeline = IngestionPipeline(data_dir=Path("Data/"))
pipeline.run()
```

### Dry Run (Parse Only)

```python
pipeline = IncrementalIngestionPipeline(data_dir=Path("Data/"))
stats = pipeline.run(dry_run=True)  # No embedding or indexing
print(f"Would index {stats.chunks_created} chunks")
```

---

## CLI Usage

### Using ingest.py Script

```bash
# Full ingestion
cd /path/to/AmberProject/src
python -m scripts.ingest Data/

# With custom settings
python -m scripts.ingest Data/ \
    --host localhost \
    --port 6333 \
    --collection genshin_story \
    --batch-size 128 \
    --device cuda

# Dry run (no indexing)
python -m scripts.ingest Data/ --dry-run
```

---

## Pipeline Statistics

The `PipelineStats` dataclass tracks execution metrics:

```python
@dataclass
class PipelineStats:
    documents_processed: int  # Successfully processed files
    documents_failed: int     # Files with errors
    chunks_created: int       # Total chunks generated
    chunks_indexed: int       # Chunks written to Qdrant
    errors: List[str]         # Error messages
```

---

## Change Detection Algorithm

```python
def _is_changed(self, file_path: Path) -> bool:
    """Check if file is new or modified."""
    key = str(file_path)

    # New file (not in tracking)
    if key not in self.tracking:
        return True

    # Compare current hash with stored hash
    current_hash = hashlib.md5(
        file_path.read_text(encoding="utf-8").encode()
    ).hexdigest()

    return current_hash != self.tracking[key].content_hash
```

---

## Chunk ID Generation

Chunks are identified by hashing their unique ID string:

```python
# In VectorIndexer.upsert_chunks()
point_id = hash(chunk.id) % (2**63)  # Convert string to int64
```

The `chunk.id` format: `{task_id}_ch{chapter}_{scene_order}_{chunk_order}`

Example: `1600_ch0_1_0`

---

## Payload Schema

Each indexed point contains:

```python
payload = {
    "text": str,              # Chunk text content
    "task_id": str,           # e.g., "1600"
    "task_name": str,         # e.g., "归途"
    "chapter_number": int,    # e.g., 0
    "chapter_title": str,     # e.g., "墟火"
    "series_name": str,       # e.g., "空月之歌"
    "scene_title": str,       # Scene header
    "scene_order": int,       # Order within chapter
    "chunk_order": int,       # Order within scene
    "event_order": int,       # Global temporal order
    "characters": List[str],  # Characters in chunk
    "has_choice": bool,       # Contains player choice
    "source_file": str,       # Original file path
}
```

---

## Error Handling

The pipeline continues processing on per-file errors:

```python
for file_path in changed_files:
    try:
        # Process file...
        stats.documents_processed += 1
    except Exception as e:
        error_msg = f"Error processing {file_path}: {e}"
        logger.error(error_msg)
        stats.errors.append(error_msg)
        stats.documents_failed += 1
```

Check `stats.errors` after execution for any failures.

---

## Configuration

**Environment Variables** (from `src/.env`):

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

**Settings** (from `src/config/settings.py`):

```python
COLLECTION_NAME = "genshin_story"
EMBEDDING_DIM = 768
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
VECTOR_TRACKING_FILE = Path(".cache/vector/tracking.json")
```

---

## Related Files

| File | Description |
|------|-------------|
| [src/ingestion/indexer.py](../src/ingestion/indexer.py) | VectorIndexer |
| [src/ingestion/pipeline.py](../src/ingestion/pipeline.py) | Pipelines |
| [src/ingestion/embedder.py](../src/ingestion/embedder.py) | EmbeddingGenerator |
| [src/ingestion/chunker.py](../src/ingestion/chunker.py) | SceneChunker |
| [src/ingestion/loader.py](../src/ingestion/loader.py) | DocumentLoader |
| [docs/query/vector-search-api.md](../docs/query/vector-search-api.md) | Query API |
