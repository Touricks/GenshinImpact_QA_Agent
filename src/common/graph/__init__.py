"""
Graph database module for Neo4j integration (read interface).

This module provides:
- Neo4jConnection: Connection manager for Neo4j database
- GraphSearcher: Query interface for knowledge graph tools
"""

from .connection import Neo4jConnection
from .searcher import GraphSearcher

__all__ = ["Neo4jConnection", "GraphSearcher"]
