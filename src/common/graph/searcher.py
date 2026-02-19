"""
Graph search interface for the graph_search tool.

Provides query methods for retrieving entity relationships from Neo4j.
"""

import logging
from typing import List, Dict, Any, Optional
from .connection import Neo4jConnection

logger = logging.getLogger(__name__)


class GraphSearcher:
    """Query interface for the Neo4j knowledge graph."""

    # Query templates for common operations
    QUERY_TEMPLATES = {
        # Get all relationships for an entity
        "all_relations": """
            MATCH (a {name: $entity})-[r]-(b)
            RETURN
                a.name as source,
                type(r) as relation,
                b.name as target,
                labels(b)[0] as target_type,
                b.description as description,
                properties(r) as rel_properties
            LIMIT $limit
        """,
        # Get specific relationship type
        "specific_relation": """
            MATCH (a {name: $entity})-[r:$rel_type]-(b)
            RETURN
                a.name as source,
                type(r) as relation,
                b.name as target,
                labels(b)[0] as target_type,
                b.description as description,
                properties(r) as rel_properties
            LIMIT $limit
        """,
        # Get faction members
        "faction_members": """
            MATCH (c)-[r:MEMBER_OF]->(f:Faction {name: $entity})
            RETURN
                c.name as name,
                labels(c)[0] as type,
                c.description as description,
                r.description as role
        """,
        # Get entity's faction
        "entity_faction": """
            MATCH (e {name: $entity})-[r:MEMBER_OF]->(f:Faction)
            RETURN
                f.name as faction_name,
                f.description as description,
                r.description as role
        """,
        # Get shortest path between two entities (Excluding generic Region nodes to avoid useless paths)
        "path_between": """
            MATCH path = shortestPath(
                (a {name: $entity1})-[*..4]-(b {name: $entity2})
            )
            WHERE none(n in nodes(path) WHERE n:Region OR n:Nation)
            RETURN
                [n in nodes(path) | n.name] as path_nodes,
                [r in relationships(path) | type(r)] as path_relations,
                length(path) as path_length
        """,
        # Get chunks mentioning a character
        "character_chunks": """
            MATCH (c:Character {name: $entity})-[:MENTIONED_IN]->(ch:Chunk)
            RETURN
                ch.chunk_id as chunk_id,
                ch.task_id as task_id,
                ch.chapter_number as chapter,
                ch.event_order as event_order
            ORDER BY ch.event_order
            LIMIT $limit
        """,
        # Get characters in a chunk
        "chunk_characters": """
            MATCH (c:Character)-[:MENTIONED_IN]->(ch:Chunk {chunk_id: $chunk_id})
            RETURN
                c.name as name,
                c.description as description
        """,
    }

    def __init__(self, connection: Optional[Neo4jConnection] = None):
        """
        Initialize the graph searcher.

        Args:
            connection: Neo4j connection (creates new one if not provided)
        """
        self.conn = connection or Neo4jConnection()

    def close(self):
        """Close the Neo4j connection."""
        self.conn.close()

    def search(
        self,
        entity: str,
        relation: Optional[str] = None,
        depth: int = 1,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Search for entity relationships in the graph.

        This is the main interface for the graph_search tool.

        Args:
            entity: Entity name to search for
            relation: Optional specific relationship type to filter
            depth: Search depth (not fully implemented yet)
            limit: Maximum number of results

        Returns:
            Dictionary with 'entities' list and metadata
        """
        logger.info(f"[Neo4j] search: entity={entity}, relation={relation}, limit={limit}")

        if relation:
            # Search for specific relationship type
            results = self._search_specific_relation(entity, relation, limit)
        else:
            # Search for all relationships
            results = self._search_all_relations(entity, limit)

        logger.debug(f"[Neo4j] search result: {len(results)} relations found")

        return {
            "entity": entity,
            "relation_filter": relation,
            "entities": results,
            "count": len(results),
        }

    def _resolve_canonical_name(self, entity_name: str) -> str:
        """
        Resolve an entity name (or alias) to its canonical name.

        Resolution priority:
        1. aliases.json mapping (fast, covers all known aliases)
        2. Neo4j fulltext index (catches entities not in aliases.json)
        3. Original name as fallback
        """
        from ..alias_resolver import resolve as resolve_alias

        # Priority 1: aliases.json (handles "木偶" → "桑多涅" etc.)
        resolved = resolve_alias(entity_name)
        if resolved != entity_name:
            return resolved

        # Priority 2: Neo4j fulltext index (aliases live in JSON files, not in graph)
        query = """
            CALL db.index.fulltext.queryNodes("entity_fulltext", $name)
            YIELD node, score
            RETURN node.name as name, score
            LIMIT 5
        """
        try:
            results = self.conn.execute(query, {"name": entity_name})

            if not results:
                return entity_name

            # Exact name match (highest priority)
            for res in results:
                if res["name"] == entity_name:
                    return res["name"]

            # High-score match (score > 3.0 indicates strong relevance)
            top_result = results[0]
            if top_result["score"] > 3.0:
                return top_result["name"]

            # Fallback to top score result
            return top_result["name"]

        except Exception:
            # Index might not exist or other error, fallback
            pass

        return entity_name

    def _search_all_relations(self, entity: str, limit: int) -> List[Dict[str, Any]]:
        """Search for all relationships of an entity."""
        canonical_name = self._resolve_canonical_name(entity)
        query = self.QUERY_TEMPLATES["all_relations"]
        return self.conn.execute(query, {"entity": canonical_name, "limit": limit})

    def _search_specific_relation(
        self, entity: str, relation: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Search for specific relationship type."""
        canonical_name = self._resolve_canonical_name(entity)

        # Build query dynamically
        query = f"""
            MATCH (a {{name: $entity}})-[r:{relation}]-(b)
            RETURN
                a.name as source,
                type(r) as relation,
                b.name as target,
                labels(b)[0] as target_type,
                b.description as description,
                properties(r) as rel_properties
            LIMIT $limit
        """
        return self.conn.execute(query, {"entity": canonical_name, "limit": limit})

    def search_history(
        self, entity: str, target: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relationship history (temporal evolution) for an entity.

        Args:
            entity: The main entity to track
            target: Optional target entity to filter history with

        Returns:
            List of relationship events sorted by time.
        """
        logger.info(f"[Neo4j] search_history: entity={entity}, target={target}")

        canonical_source = self._resolve_canonical_name(entity)
        canonical_target = self._resolve_canonical_name(target) if target else None

        filters = "WHERE a.name = $source"
        if canonical_target:
            filters += " AND b.name = $target"

        query = f"""
            MATCH (a)-[r]->(b)
            {filters}
            RETURN
                a.name as source,
                b.name as target,
                type(r) as relation,
                r.chapter as chapter,
                r.task_id as task_id,
                r.evidence as evidence
            ORDER BY r.chapter ASC, r.task_id ASC
        """

        params = {"source": canonical_source}
        if canonical_target:
            params["target"] = canonical_target

        results = self.conn.execute(query, params)
        logger.debug(f"[Neo4j] search_history result: {len(results)} events found")
        return results

    def get_faction_members(self, faction_name: str) -> List[Dict[str, Any]]:
        """Get all members of a faction."""
        canonical_name = self._resolve_canonical_name(faction_name)
        query = self.QUERY_TEMPLATES["faction_members"]
        return self.conn.execute(query, {"entity": canonical_name})

    def get_entity_faction(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get faction(s) an entity belongs to."""
        canonical_name = self._resolve_canonical_name(entity_name)
        query = self.QUERY_TEMPLATES["entity_faction"]
        return self.conn.execute(query, {"entity": canonical_name})

    def get_entity_info(self, entity: str) -> Optional[Dict[str, Any]]:
        """
        Return node properties and labels for an entity.

        Args:
            entity: Entity name (supports alias resolution)

        Returns:
            Dict with node properties and labels, or None if not found
        """
        canonical_name = self._resolve_canonical_name(entity)
        query = """
            MATCH (n {name: $entity})
            RETURN n as node, labels(n) as labels
        """
        results = self.conn.execute(query, {"entity": canonical_name})
        if not results:
            return None
        row = results[0]
        props = dict(row["node"])
        return {"name": canonical_name, "labels": row["labels"], "properties": props}

    def get_path_between(
        self, entity1: str, entity2: str, max_hops: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two entities.

        Args:
            entity1: First entity name
            entity2: Second entity name

        Returns:
            Path information or None if no path exists
        """
        logger.info(f"[Neo4j] get_path_between: {entity1} -> {entity2}")

        canonical_1 = self._resolve_canonical_name(entity1)
        canonical_2 = self._resolve_canonical_name(entity2)
        # Cypher doesn't support parameterized hop bounds, use f-string
        query = f"""
            MATCH path = shortestPath(
                (a {{name: $entity1}})-[*..{max_hops}]-(b {{name: $entity2}})
            )
            WHERE none(n in nodes(path) WHERE n:Region OR n:Nation)
            RETURN
                [n in nodes(path) | n.name] as path_nodes,
                [r in relationships(path) | type(r)] as path_relations,
                length(path) as path_length
        """
        results = self.conn.execute(query, {"entity1": canonical_1, "entity2": canonical_2})
        result = results[0] if results else None

        if result:
            logger.debug(f"[Neo4j] path found: {result.get('path_nodes', [])}")
        else:
            logger.debug(f"[Neo4j] no path found between {entity1} and {entity2}")

        return result

    def get_character_chunks(
        self, char_name: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get chunks that mention a character.

        Args:
            char_name: Character name
            limit: Maximum number of chunks

        Returns:
            List of chunk information
        """
        canonical_name = self._resolve_canonical_name(char_name)
        query = self.QUERY_TEMPLATES["character_chunks"]
        return self.conn.execute(query, {"entity": canonical_name, "limit": limit})

    def get_chunk_characters(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        Get characters mentioned in a chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            List of character information
        """
        query = self.QUERY_TEMPLATES["chunk_characters"]
        return self.conn.execute(query, {"chunk_id": chunk_id})

    def get_major_events(
        self,
        entity: str,
        event_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get major events experienced by a character.

        This method queries MajorEvent nodes connected via EXPERIENCES edges.
        It addresses the "abstract query vs concrete narrative" semantic gap:
        - User asks: "How did the girl return to the world?"
        - This returns: ["献出身体", "化作月光", ...] with summaries

        Args:
            entity: Character name (supports alias resolution)
            event_type: Optional filter by event type
                        (sacrifice/transformation/acquisition/loss/
                         encounter/conflict/revelation/milestone)
            limit: Maximum number of events to return

        Returns:
            List of event dicts sorted by chapter ASC, containing:
            - event_name: Name of the event
            - event_type: Type classification
            - chapter: Chapter number
            - summary: One-sentence summary
            - evidence: Original text evidence
            - role: Character's role (subject/object/witness)
            - outcome: Effect of the event
        """
        logger.info(
            f"[Neo4j] get_major_events: entity={entity}, event_type={event_type}"
        )

        canonical_name = self._resolve_canonical_name(entity)

        if event_type:
            query = """
                MATCH (c:Character {name: $entity})-[r:EXPERIENCES]->(e:MajorEvent)
                WHERE e.event_type = $event_type
                RETURN e.name as event_name,
                       e.event_type as event_type,
                       e.chapter as chapter,
                       e.task_id as task_id,
                       e.summary as summary,
                       e.evidence as evidence,
                       r.role as role,
                       r.outcome as outcome
                ORDER BY e.chapter ASC
                LIMIT $limit
            """
            results = self.conn.execute(
                query,
                {"entity": canonical_name, "event_type": event_type, "limit": limit},
            )
        else:
            query = """
                MATCH (c:Character {name: $entity})-[r:EXPERIENCES]->(e:MajorEvent)
                RETURN e.name as event_name,
                       e.event_type as event_type,
                       e.chapter as chapter,
                       e.task_id as task_id,
                       e.summary as summary,
                       e.evidence as evidence,
                       r.role as role,
                       r.outcome as outcome
                ORDER BY e.chapter ASC
                LIMIT $limit
            """
            results = self.conn.execute(
                query, {"entity": canonical_name, "limit": limit}
            )

        logger.debug(f"[Neo4j] get_major_events result: {len(results)} events found")
        return results

    def get_timeline(self, entity: str) -> List[Dict[str, Any]]:
        """
        Get timeline of events for an entity using three-layer fallback.

        Strategy:
        1. INVOLVED_IN → Event nodes (best for Characters/Items)
        2. TEMPORAL_BEFORE chain (if entity is an Event)
        3. All edges with chapter property (fallback for Places etc.)

        Args:
            entity: Entity name (supports alias resolution)

        Returns:
            List of timeline events sorted by chapter
        """
        logger.info(f"[Neo4j] get_timeline: entity={entity}")
        canonical_name = self._resolve_canonical_name(entity)

        # Layer 1: INVOLVED_IN edges to Event nodes
        query_involved = """
            MATCH (n {name: $entity})-[r:INVOLVED_IN]->(evt:Event)
            RETURN evt.name as event_name, evt.description as description,
                   evt.time_order as time_order, evt.cause as cause,
                   evt.effect as effect,
                   r.chapter as chapter, r.description as role_description
            ORDER BY r.chapter ASC, evt.time_order ASC
        """
        results = self.conn.execute(query_involved, {"entity": canonical_name})
        if results:
            logger.debug(f"[Neo4j] get_timeline: {len(results)} INVOLVED_IN events")
            return [dict(r, source="INVOLVED_IN") for r in results]

        # Layer 2: TEMPORAL_BEFORE chain (if entity is an Event)
        query_temporal = """
            MATCH (e:Event {name: $entity})-[:TEMPORAL_BEFORE*0..5]->(next:Event)
            RETURN next.name as event_name, next.description as description,
                   next.time_order as time_order, next.cause as cause,
                   next.effect as effect
            ORDER BY next.time_order ASC
        """
        results = self.conn.execute(query_temporal, {"entity": canonical_name})
        if results and len(results) > 1:  # >1 because self is always included
            logger.debug(f"[Neo4j] get_timeline: {len(results)} TEMPORAL_BEFORE events")
            return [dict(r, source="TEMPORAL_BEFORE") for r in results]

        # Layer 3: All edges with chapter property (fallback)
        query_fallback = """
            MATCH (n {name: $entity})-[r]->(b)
            WHERE r.chapter IS NOT NULL
            RETURN b.name as event_name, labels(b)[0] as type,
                   type(r) as relation, r.chapter as chapter,
                   r.description as description
            ORDER BY r.chapter ASC
        """
        results = self.conn.execute(query_fallback, {"entity": canonical_name})
        if not results:
            # Try incoming edges too
            query_incoming = """
                MATCH (n {name: $entity})<-[r]-(b)
                WHERE r.chapter IS NOT NULL
                RETURN b.name as event_name, labels(b)[0] as type,
                       type(r) as relation, r.chapter as chapter,
                       r.description as description
                ORDER BY r.chapter ASC
            """
            results = self.conn.execute(query_incoming, {"entity": canonical_name})

        logger.debug(f"[Neo4j] get_timeline: {len(results)} fallback events")
        return [dict(r, source="fallback") for r in results]

    def trace_causality(
        self,
        start: str,
        end: Optional[str] = None,
        max_hops: int = 4,
    ) -> Dict[str, Any]:
        """
        Trace causal/motivation chains via LEADS_TO + MOTIVATED_BY edges.

        Two modes:
        - Dual-param (start, end): Find shortest causal path between two entities
        - Single-param (start only): Expand causal neighborhood

        Args:
            start: Starting entity name
            end: Optional ending entity name
            max_hops: Maximum path length

        Returns:
            Dict with nodes and edges of the causal chain
        """
        logger.info(f"[Neo4j] trace_causality: start={start}, end={end}")
        canonical_start = self._resolve_canonical_name(start)

        if end:
            canonical_end = self._resolve_canonical_name(end)
            query = f"""
                MATCH path = shortestPath(
                    (a {{name: $start}})-[:LEADS_TO|MOTIVATED_BY*..{max_hops}]-(b {{name: $end}})
                )
                RETURN
                    [n in nodes(path) | {{name: n.name, labels: labels(n)}}] as nodes,
                    [r in relationships(path) | {{
                        type: type(r), desc: r.description,
                        src: startNode(r).name, tgt: endNode(r).name
                    }}] as edges,
                    length(path) as path_length
            """
            results = self.conn.execute(
                query, {"start": canonical_start, "end": canonical_end}
            )
            if results:
                return results[0]
            return {"nodes": [], "edges": [], "path_length": 0}
        else:
            # Single-param: expand neighborhood (dedup by entity, keep shortest path)
            query = f"""
                MATCH (a {{name: $start}})-[r:LEADS_TO|MOTIVATED_BY*..{min(max_hops, 3)}]-(b)
                WHERE a <> b
                WITH b, r ORDER BY length(r) ASC
                WITH b, COLLECT(r)[0] AS shortest_path
                RETURN b.name AS name, labels(b)[0] AS type,
                       [rel IN shortest_path | {{type: type(rel), desc: rel.description}}] AS chain
                LIMIT 15
            """
            results = self.conn.execute(query, {"start": canonical_start})
            return {"center": canonical_start, "neighbors": results}

    def explore_subgraph(
        self,
        entity: str,
        depth: int = 1,
        max_nodes: int = 20,
        edge_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        BFS expansion from a center entity.

        Args:
            entity: Center entity name
            depth: Expansion depth (1 or 2)
            max_nodes: Maximum nodes to return
            edge_types: Optional list of edge types to filter (e.g. ["MEMBER_OF", "OPPOSED_TO"])

        Returns:
            Dict with nodes and edges of the subgraph
        """
        logger.info(
            f"[Neo4j] explore_subgraph: entity={entity}, depth={depth}, "
            f"max_nodes={max_nodes}, edge_types={edge_types}"
        )
        canonical_name = self._resolve_canonical_name(entity)
        depth = min(depth, 3)  # Cap at 3 for safety

        if edge_types:
            type_filter = "|".join(edge_types)
            query = f"""
                MATCH path = (center {{name: $entity}})-[:{type_filter}*..{depth}]-(n)
                WITH DISTINCT n, path
                LIMIT {max_nodes}
                RETURN
                    [node in nodes(path) | {{name: node.name, type: labels(node)[0]}}] as nodes,
                    [rel in relationships(path) | {{
                        src: startNode(rel).name, type: type(rel), tgt: endNode(rel).name
                    }}] as edges
            """
        else:
            query = f"""
                MATCH path = (center {{name: $entity}})-[*..{depth}]-(n)
                WITH DISTINCT n, path
                LIMIT {max_nodes}
                RETURN
                    [node in nodes(path) | {{name: node.name, type: labels(node)[0]}}] as nodes,
                    [rel in relationships(path) | {{
                        src: startNode(rel).name, type: type(rel), tgt: endNode(rel).name
                    }}] as edges
            """

        results = self.conn.execute(query, {"entity": canonical_name})

        # Merge all paths into a single subgraph
        all_nodes = {}
        all_edges = set()
        for row in results:
            for node in row.get("nodes", []):
                all_nodes[node["name"]] = node
            for edge in row.get("edges", []):
                all_edges.add((edge["src"], edge["type"], edge["tgt"]))

        return {
            "center": canonical_name,
            "nodes": list(all_nodes.values()),
            "edges": [{"src": s, "type": t, "tgt": g} for s, t, g in all_edges],
            "node_count": len(all_nodes),
            "edge_count": len(all_edges),
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Convenience function for tool integration
def graph_search(
    entity: str,
    relation: Optional[str] = None,
    depth: int = 1,
) -> Dict[str, Any]:
    """
    Graph search function for integration with RAG tools.

    Args:
        entity: Entity name to search for
        relation: Optional relationship type filter
        depth: Search depth

    Returns:
        Search results
    """
    with GraphSearcher() as searcher:
        return searcher.search(entity, relation, depth)


if __name__ == "__main__":
    searcher = GraphSearcher()

    if searcher.conn.verify_connectivity():
        print("Testing GraphSearcher (v2)...")

        print("\n1. All relations for 哥伦比娅:")
        result = searcher.search("哥伦比娅")
        for entity in result["entities"][:5]:
            print(f"  {entity}")

        print("\n2. Faction members of 愚人众:")
        members = searcher.get_faction_members("愚人众")
        for member in members:
            print(f"  {member['name']} ({member.get('role', 'member')})")

        print("\n3. Path between 旅行者 and 哥伦比娅:")
        path = searcher.get_path_between("旅行者", "哥伦比娅")
        if path:
            print(f"  Path: {' -> '.join(path['path_nodes'])}")
            print(f"  Relations: {path['path_relations']}")

    else:
        print("Cannot connect to Neo4j. Make sure it's running.")

    searcher.close()
