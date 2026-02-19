"""
Graph builder for constructing the Neo4j knowledge graph.

v2: Supports 6 entity types (Character, Faction, Place, Item, Concept, Event)
and 9 relationship types with flexible source/target label matching.
"""

from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm

from ..common.graph.connection import Neo4jConnection
from ..common.models.entities import (
    Character,
    Faction,
    Place,
    Item,
    Concept,
    Event,
    Organization,
    Location,
    MAIN_CHARACTERS,
)
from ..common.models.relationships import Relationship, RelationType


# Maps v2 LLM entity_type names to Neo4j node labels
ENTITY_TYPE_TO_LABEL = {
    "Character": "Character",
    "Faction": "Faction",
    "Organization": "Faction",  # legacy alias
    "Place": "Place",
    "Location": "Place",  # legacy alias
    "Item": "Item",
    "Concept": "Concept",
    "Event": "Event",
}


class GraphBuilder:
    """Build and populate the Neo4j knowledge graph."""

    def __init__(self, connection: Optional[Neo4jConnection] = None):
        self.conn = connection or Neo4jConnection()

    def close(self):
        """Close the Neo4j connection."""
        self.conn.close()

    # =========================================================================
    # Schema Setup
    # =========================================================================

    def create_constraints(self):
        """Create unique constraints for all 6 entity types + Chunk."""
        constraints = [
            "CREATE CONSTRAINT character_name IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT faction_name IF NOT EXISTS FOR (f:Faction) REQUIRE f.name IS UNIQUE",
            "CREATE CONSTRAINT place_name IF NOT EXISTS FOR (p:Place) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT item_name IF NOT EXISTS FOR (i:Item) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT event_name IF NOT EXISTS FOR (e:Event) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
        ]
        for constraint in constraints:
            try:
                self.conn.execute(constraint)
            except Exception as e:
                print(f"Constraint warning: {e}")

    def create_indexes(self):
        """Create performance indexes."""
        indexes = [
            "CREATE INDEX character_region IF NOT EXISTS FOR (c:Character) ON (c.region)",
            "CREATE INDEX chunk_task IF NOT EXISTS FOR (ch:Chunk) ON (ch.task_id)",
            # Fulltext indexes covering all named entity types for alias resolution
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Character|Faction|Place|Item|Concept|Event) ON EACH [n.name]",
        ]
        for index in indexes:
            try:
                self.conn.execute(index)
            except Exception as e:
                print(f"Index warning: {e}")

    def setup_schema(self):
        """Set up all constraints and indexes."""
        print("Setting up Neo4j schema...")
        self.create_constraints()
        self.create_indexes()
        print("Schema setup complete.")

    # =========================================================================
    # Node Creation — 6 entity types
    # =========================================================================

    def create_character(self, character: Character) -> None:
        """Create or update a Character node."""
        query = """
        MERGE (c:Character {name: $name})
        SET c.aliases = $aliases,
            c.title = $title,
            c.region = $region,
            c.tribe = $tribe,
            c.description = $description,
            c.first_appearance_task = $first_appearance_task,
            c.first_appearance_chapter = $first_appearance_chapter,
            c.affiliation = $affiliation,
            c.role = $role,
            c.goal = $goal,
            c.fate = $fate
        RETURN c.name as name
        """
        self.conn.execute_write(query, character.to_dict())

    def create_faction(self, faction: Faction) -> None:
        """Create or update a Faction node."""
        query = """
        MERGE (f:Faction {name: $name})
        SET f.type = $type,
            f.region = $region,
            f.description = $description,
            f.leader = $leader,
            f.base = $base,
            f.purpose = $purpose,
            f.status = $status
        RETURN f.name as name
        """
        self.conn.execute_write(query, faction.to_dict())

    # Backward compat alias
    def create_organization(self, organization: Faction) -> None:
        """Create or update a Faction node (backward compat for Organization)."""
        self.create_faction(organization)

    def create_place(self, place: Place) -> None:
        """Create or update a Place node."""
        query = """
        MERGE (p:Place {name: $name})
        SET p.type = $type,
            p.region = $region,
            p.description = $description,
            p.political_status = $political_status,
            p.significance = $significance,
            p.inhabitants = $inhabitants
        RETURN p.name as name
        """
        self.conn.execute_write(query, place.to_dict())

    # Backward compat alias
    def create_location(self, location: Place) -> None:
        """Create or update a Place node (backward compat for Location)."""
        self.create_place(location)

    def create_item(self, item: Item) -> None:
        """Create or update an Item node."""
        query = """
        MERGE (i:Item {name: $name})
        SET i.description = $description,
            i.creator = $creator,
            i.function = $function,
            i.location = $location,
            i.status = $status
        RETURN i.name as name
        """
        self.conn.execute_write(query, item.to_dict())

    def create_concept(self, concept: Concept) -> None:
        """Create or update a Concept node."""
        query = """
        MERGE (c:Concept {name: $name})
        SET c.description = $description,
            c.origin = $origin,
            c.nature = $nature,
            c.related_system = $related_system
        RETURN c.name as name
        """
        self.conn.execute_write(query, concept.to_dict())

    def create_event(self, event: Event) -> None:
        """Create or update an Event node."""
        query = """
        MERGE (e:Event {name: $name})
        SET e.type = $type,
            e.chapter_range = $chapter_range,
            e.description = $description,
            e.agent = $agent,
            e.target = $target,
            e.cause = $cause,
            e.effect = $effect,
            e.legacy = $legacy,
            e.time_order = $time_order
        RETURN e.name as name
        """
        self.conn.execute_write(query, event.to_dict())

    def create_character_simple(
        self,
        name: str,
        task_id: Optional[str] = None,
        chapter: Optional[int] = None,
    ) -> None:
        """Create a simple Character node with just the name."""
        query = """
        MERGE (c:Character {name: $name})
        ON CREATE SET
            c.first_appearance_task = $task_id,
            c.first_appearance_chapter = $chapter
        RETURN c.name as name
        """
        self.conn.execute_write(
            query, {"name": name, "task_id": task_id, "chapter": chapter}
        )

    def create_entity_from_extraction(
        self,
        name: str,
        entity_type: str,
        description: Optional[str] = None,
        attributes: Optional[Dict[str, Optional[str]]] = None,
        text_evidence: Optional[str] = None,
        task_id: Optional[str] = None,
        chapter: Optional[int] = None,
    ) -> None:
        """
        Create or update any entity node from LLM extraction output.

        Maps v2 entity types (Place, Faction) to Neo4j labels.
        Writes attributes as node properties.
        """
        label = ENTITY_TYPE_TO_LABEL.get(entity_type, "Character")
        attrs = attributes or {}

        # Build dynamic SET clause for attributes
        set_parts = ["n.description = $description", "n.text_evidence = $text_evidence"]
        params: Dict[str, Any] = {
            "name": name,
            "description": description,
            "text_evidence": text_evidence,
        }

        if task_id:
            set_parts.append("n.task_id = $task_id")
            params["task_id"] = task_id
        if chapter is not None:
            set_parts.append("n.chapter = $chapter")
            params["chapter"] = chapter

        for attr_key, attr_val in attrs.items():
            if attr_val is not None:
                safe_key = f"attr_{attr_key}"
                set_parts.append(f"n.{attr_key} = ${safe_key}")
                params[safe_key] = attr_val

        set_clause = ", ".join(set_parts)

        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET {set_clause}
        RETURN n.name as name
        """
        try:
            self.conn.execute_write(query, params)
        except Exception as e:
            print(f"Error creating {label} '{name}': {e}")

    # =========================================================================
    # Relationship Creation — v2 flexible label matching
    # =========================================================================

    def create_relationship(self, relationship: Relationship) -> None:
        """
        Create a relationship between two entities.

        v2: Uses flexible label matching — no hardcoded source/target labels.
        Matches any node by name, then creates the typed relationship.
        """
        props = relationship.properties.copy()

        if relationship.chapter is not None:
            props["chapter"] = relationship.chapter
        if relationship.task_id:
            props["task_id"] = relationship.task_id
        if relationship.description:
            props["description"] = relationship.description
        if relationship.confidence:
            props["confidence"] = relationship.confidence
        if relationship.text_evidence:
            props["text_evidence"] = relationship.text_evidence

        prop_sets = ", ".join([f"r.{k} = ${k}" for k in props.keys()])

        # v2: flexible matching — MATCH any node by name, no label constraint
        if "chapter" in props:
            query = f"""
            MATCH (a {{name: $source}})
            MATCH (b {{name: $target}})
            MERGE (a)-[r:{relationship.rel_type.value} {{chapter: $chapter}}]->(b)
            {"SET " + prop_sets if prop_sets else ""}
            RETURN type(r) as rel_type
            """
        else:
            query = f"""
            MATCH (a {{name: $source}})
            MATCH (b {{name: $target}})
            MERGE (a)-[r:{relationship.rel_type.value}]->(b)
            {"SET " + prop_sets if prop_sets else ""}
            RETURN type(r) as rel_type
            """

        params = {"source": relationship.source, "target": relationship.target, **props}

        try:
            self.conn.execute_write(query, params)
        except Exception:
            pass  # Nodes may not exist

    def create_relationship_from_extraction(
        self,
        source: str,
        target: str,
        relation_type: str,
        description: Optional[str] = None,
        confidence: Optional[str] = None,
        text_evidence: Optional[str] = None,
        chapter: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> bool:
        """
        Create a relationship from LLM extraction output.

        Returns True if created successfully.
        """
        props: Dict[str, Any] = {}
        if description:
            props["description"] = description
        if confidence:
            props["confidence"] = confidence
        if text_evidence:
            props["text_evidence"] = text_evidence
        if chapter is not None:
            props["chapter"] = chapter
        if task_id:
            props["task_id"] = task_id

        prop_sets = ", ".join([f"r.{k} = ${k}" for k in props.keys()])
        set_clause = f"SET {prop_sets}" if prop_sets else ""

        # Validate relation type
        try:
            rel_enum = RelationType(relation_type)
        except ValueError:
            print(f"Unknown relation type: {relation_type}")
            return False

        query = f"""
        MATCH (a {{name: $source}})
        MATCH (b {{name: $target}})
        MERGE (a)-[r:{rel_enum.value}]->(b)
        {set_clause}
        RETURN type(r) as rel_type
        """

        params = {"source": source, "target": target, **props}

        try:
            result = self.conn.execute_write(query, params)
            return len(result) > 0
        except Exception as e:
            print(f"Error creating relationship {source}-[{relation_type}]->{target}: {e}")
            return False

    # =========================================================================
    # MajorEvent (legacy — used by incremental_event_extractor)
    # =========================================================================

    def create_major_event(
        self,
        name: str,
        event_type: str,
        chapter: int,
        task_id: str,
        primary_character: str,
        summary: str,
        evidence: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> Optional[str]:
        """Create a MajorEvent node with deduplication."""
        query = """
        MERGE (e:MajorEvent {
            chapter: $chapter,
            event_type: $event_type,
            primary_character: $primary_character
        })
        ON CREATE SET
            e.name = $name,
            e.task_id = $task_id,
            e.summary = $summary,
            e.evidence = $evidence,
            e.outcome = $outcome
        ON MATCH SET
            e.name = $name,
            e.summary = $summary,
            e.evidence = $evidence,
            e.outcome = $outcome
        RETURN e.name as name
        """
        try:
            result = self.conn.execute_write(
                query,
                {
                    "name": name,
                    "event_type": event_type,
                    "chapter": chapter,
                    "task_id": task_id,
                    "primary_character": primary_character,
                    "summary": summary,
                    "evidence": evidence,
                    "outcome": outcome,
                },
            )
            return result[0]["name"] if result else None
        except Exception as e:
            print(f"Error creating MajorEvent: {e}")
            return None

    def create_experiences_edge(
        self,
        character_name: str,
        event_chapter: int,
        event_type: str,
        event_primary_character: str,
        role: str = "subject",
        outcome: Optional[str] = None,
    ) -> bool:
        """Create an EXPERIENCES edge from a Character to a MajorEvent."""
        query = """
        MATCH (c:Character {name: $character_name})
        MATCH (e:MajorEvent {
            chapter: $event_chapter,
            event_type: $event_type,
            primary_character: $event_primary_character
        })
        MERGE (c)-[r:EXPERIENCES]->(e)
        SET r.role = $role,
            r.outcome = $outcome
        RETURN type(r) as rel_type
        """
        try:
            result = self.conn.execute_write(
                query,
                {
                    "character_name": character_name,
                    "event_chapter": event_chapter,
                    "event_type": event_type,
                    "event_primary_character": event_primary_character,
                    "role": role,
                    "outcome": outcome,
                },
            )
            return len(result) > 0
        except Exception as e:
            print(f"Error creating EXPERIENCES edge: {e}")
            return False

    def ingest_extracted_events(
        self,
        events: List[Dict],
        chapter: int,
        task_id: str,
    ) -> int:
        """Ingest a batch of extracted events from LLMEventExtractor."""
        count = 0
        edge_failures = []

        for event in events:
            primary_char = None
            for char in event.get("characters", []):
                if char.get("role") == "subject":
                    primary_char = char.get("name")
                    break
            if not primary_char and event.get("characters"):
                primary_char = event["characters"][0].get("name", "unknown")
            if not primary_char:
                continue

            for char in event.get("characters", []):
                char_name = char.get("name")
                if char_name:
                    self.create_character_simple(char_name, task_id, chapter)

            event_name = self.create_major_event(
                name=event.get("name", ""),
                event_type=event.get("event_type", "milestone"),
                chapter=chapter,
                task_id=task_id,
                primary_character=primary_char,
                summary=event.get("summary", ""),
                evidence=event.get("evidence"),
                outcome=event.get("outcome"),
            )
            if not event_name:
                continue

            for char in event.get("characters", []):
                char_name = char.get("name")
                if char_name:
                    success = self.create_experiences_edge(
                        character_name=char_name,
                        event_chapter=chapter,
                        event_type=event.get("event_type", "milestone"),
                        event_primary_character=primary_char,
                        role=char.get("role", "witness"),
                        outcome=event.get("outcome"),
                    )
                    if not success:
                        edge_failures.append((char_name, event.get("name")))

            count += 1

        if edge_failures:
            print(
                f"Warning: Failed to create {len(edge_failures)} EXPERIENCES edges. "
                f"First 5: {edge_failures[:5]}"
            )
        return count

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def create_characters_batch(
        self,
        characters: Set[str],
        task_id: Optional[str] = None,
        chapter: Optional[int] = None,
    ) -> None:
        """Create multiple character nodes in batch."""
        for name in characters:
            if name in MAIN_CHARACTERS:
                continue
            self.create_character_simple(name, task_id, chapter)

    def create_relationships_batch(self, relationships: List[Relationship]) -> None:
        """Create multiple relationships in batch."""
        for rel in relationships:
            self.create_relationship(rel)

    # =========================================================================
    # Chunk Integration
    # =========================================================================

    def create_chunk(
        self,
        chunk_id: str,
        event_order: int,
        task_id: str,
        chapter_number: int,
        characters: List[str],
    ) -> None:
        """Create a Chunk node and link it to characters."""
        create_query = """
        MERGE (ch:Chunk {chunk_id: $chunk_id})
        SET ch.event_order = $event_order,
            ch.task_id = $task_id,
            ch.chapter_number = $chapter_number
        RETURN ch.chunk_id as id
        """
        self.conn.execute_write(
            create_query,
            {
                "chunk_id": chunk_id,
                "event_order": event_order,
                "task_id": task_id,
                "chapter_number": chapter_number,
            },
        )

        link_query = """
        MATCH (ch:Chunk {chunk_id: $chunk_id})
        MATCH (c:Character {name: $char_name})
        MERGE (c)-[:MENTIONED_IN]->(ch)
        """
        for char_name in characters:
            try:
                self.conn.execute_write(
                    link_query, {"chunk_id": chunk_id, "char_name": char_name}
                )
            except Exception:
                pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear_graph(self) -> None:
        """Delete all nodes and relationships. USE WITH CAUTION."""
        query = "MATCH (n) DETACH DELETE n"
        self.conn.execute_write(query)
        print("Graph cleared.")

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the current graph."""
        queries = {
            "characters": "MATCH (c:Character) RETURN count(c) as count",
            "factions": "MATCH (f:Faction) RETURN count(f) as count",
            "places": "MATCH (p:Place) RETURN count(p) as count",
            "items": "MATCH (i:Item) RETURN count(i) as count",
            "concepts": "MATCH (c:Concept) RETURN count(c) as count",
            "events": "MATCH (e:Event) RETURN count(e) as count",
            "chunks": "MATCH (ch:Chunk) RETURN count(ch) as count",
            "relationships": "MATCH ()-[r]->() RETURN count(r) as count",
        }

        stats = {}
        for name, query in queries.items():
            result = self.conn.execute(query)
            stats[name] = result[0]["count"] if result else 0

        return stats

    def get_relationship_type_stats(self) -> Dict[str, int]:
        """Get counts of each relationship type."""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        """
        result = self.conn.execute(query)
        return {row["rel_type"]: row["count"] for row in result}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
