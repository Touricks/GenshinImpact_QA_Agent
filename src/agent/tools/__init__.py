"""
Retrieval tools for the ReAct Agent.

v2: 6 KG tools + 1 vector tool following the Graph-Vector Complementarity design.

Graph Tools (Neo4j - fast, structured, no long text):
- lookup_knowledge: Entity facts, properties, and direct relationships
- find_connection: Path finding between entities
- get_timeline: Event timeline for any entity type
- trace_causality: Causal/motivation chain traversal
- compare_entities: Side-by-side entity comparison (KG+LLM)
- explore_subgraph: BFS subgraph expansion from a center entity

Vector Tool (Qdrant - slower, returns story text):
- search_memory: Episodic memory, dialogue, plot details

Design Principle:
- Graph = Skeleton & Index (structure, logic, relationships)
- Vector = Flesh & Content (semantics, dialogue, episodes)
- Never store long text in Graph. Never derive relationships from Vector.
"""

from .lookup_knowledge import lookup_knowledge
from .find_connection import find_connection
from .get_timeline import get_timeline
from .trace_causality import trace_causality
from .compare_entities import compare_entities
from .explore_subgraph import explore_subgraph
from .search_memory import search_memory

__all__ = [
    "lookup_knowledge",
    "find_connection",
    "get_timeline",
    "trace_causality",
    "compare_entities",
    "explore_subgraph",
    "search_memory",
]
