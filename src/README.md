# Genshin Story QA - Source Code

Core modules for the Genshin Story QA system.

## Directory Structure

```
src/
├── agent/           # ReAct Agent implementation
├── config/          # Configuration and settings
├── graph/           # Neo4j graph database operations
├── ingestion/       # Data ingestion and extraction
├── models/          # Entity and relationship models
├── retrieval/       # Query tools for Agent
└── scripts/         # CLI entry points
    ├── cli_vector.py   # Qdrant vector ingestion
    ├── cli_graph.py    # Neo4j graph building
    └── cli_agent.py    # Agent runner
```

---

## Quick Start

### 1. Environment Setup

Create `.env` in project root:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GEMINI_API_KEY=AIzaSy...
```

### 2. Start Services

```bash
docker-compose up -d neo4j
docker run -p 6333:6333 qdrant/qdrant
```

### 3. CLI Commands

```bash
# Build Neo4j Knowledge Graph
python -m src.scripts.cli_graph --clear      # Full rebuild
python -m src.scripts.cli_graph --skip-seed  # Incremental
python -m src.scripts.cli_graph --stats      # Show stats

# Ingest to Qdrant Vector DB
python -m src.scripts.cli_vector Data/
python -m src.scripts.cli_vector Data/ --incremental

# Run Agent
python -m src.scripts.cli_agent "Who is Mavuika?"
python -m src.scripts.cli_agent --interactive
python -m src.scripts.cli_agent --grading "少女是如何重回世界的？"
```

---

## Module Reference

### scripts/

| CLI | Target | Description |
|-----|--------|-------------|
| `cli_vector.py` | Qdrant | Chunk → Embed → Index |
| `cli_graph.py` | Neo4j | LLM Extract → Build Graph |
| `cli_agent.py` | - | Run ReAct Agent |

### retrieval/

| Tool | Description |
|------|-------------|
| `lookup_knowledge` | Static facts from graph |
| `find_connection` | Path between entities |
| `track_journey` | Temporal evolution |
| `search_memory` | Vector search for content |
| `get_character_events` | Major events query |

### agent/

| Class | Description |
|-------|-------------|
| `GenshinRetrievalAgent` | ReAct Agent with tools |
| `AnswerGrader` | Answer quality evaluation |
| `QueryRefiner` | Query decomposition |

---

## Testing

```bash
python -m tests.integration.verify_graph
python -m tests.integration.test_tools
```
