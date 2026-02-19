"""
Build the Neo4j knowledge graph from dialogue data.

Usage:
    python -m src.scripts.cli_graph [DATA_DIR]
    python -m src.scripts.cli_graph --clear  # Clear and rebuild
    python -m src.scripts.cli_graph --stats  # Show graph statistics

Examples:
    python -m src.scripts.cli_graph Data/staging/Archon/
    python -m src.scripts.cli_graph --clear
    python -m src.scripts.cli_graph --stats
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm

from ..common.graph.connection import Neo4jConnection
from ..pipeline.graph_builder import GraphBuilder
from ..pipeline.llm_kg_extractor import extract_kg_from_file
from ..common.models.relationships import Relationship, RelationType


def parse_file_metadata(file_path: Path) -> dict:
    """
    Parse task_id and chapter from file path.

    Expected: Data/staging/Archon/{task_id}/chapter{N}_dialogue.txt
    """
    task_id = file_path.parent.name
    match = re.search(r"chapter(\d+)", file_path.stem)
    chapter_num = int(match.group(1)) if match else 0

    try:
        global_chapter = int(task_id) * 100 + chapter_num
    except ValueError:
        global_chapter = chapter_num

    return {"task_id": task_id, "chapter": global_chapter}


def build_graph(
    data_dir: str = "Data/staging/",
    clear_existing: bool = False,
) -> None:
    """
    Build the Neo4j knowledge graph from dialogue files.

    v2: No seed data. 6 entity types, 9 relationship types.
    """
    print("=" * 60)
    print("Neo4j Knowledge Graph Builder (v2)")
    print("=" * 60)

    conn = Neo4jConnection()
    if not conn.verify_connectivity():
        print("\nERROR: Cannot connect to Neo4j.")
        print("Make sure Neo4j is running: docker-compose up -d neo4j")
        return

    with GraphBuilder(conn) as builder:
        if clear_existing:
            print("\nClearing existing graph...")
            builder.clear_graph()

        builder.setup_schema()

        # Extract and load from dialogue files
        print(f"\n--- Processing Dialogue Files ---")
        print(f"Data directory: {data_dir}")

        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"ERROR: Data directory not found: {data_dir}")
            return

        dialogue_files = sorted(data_path.rglob("chapter*_dialogue.txt"))
        print(f"Found {len(dialogue_files)} dialogue files")

        all_relationships = []

        for file_path in tqdm(dialogue_files, desc="Processing files"):
            try:
                metadata = parse_file_metadata(file_path)

                try:
                    kg_output = extract_kg_from_file(file_path)
                except Exception as e:
                    print(f"LLM Extraction failed for {file_path.name}: {e}")
                    continue

                # Write all entities using the generic method
                for entity in kg_output.entities:
                    builder.create_entity_from_extraction(
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        attributes=entity.attributes_dict,
                        text_evidence=entity.text_evidence,
                        task_id=metadata["task_id"],
                        chapter=metadata["chapter"],
                    )

                # Collect relationships
                for rel in kg_output.relationships:
                    try:
                        rel_type_enum = RelationType(rel.relation_type)
                        new_rel = Relationship(
                            source=rel.source,
                            target=rel.target,
                            rel_type=rel_type_enum,
                            description=rel.description,
                            confidence=rel.confidence,
                            text_evidence=rel.text_evidence,
                            chapter=metadata["chapter"],
                            task_id=metadata["task_id"],
                        )
                        all_relationships.append(new_rel)
                    except ValueError:
                        print(f"Unknown relation type: {rel.relation_type}")

            except Exception as e:
                print(f"\nError processing {file_path}: {e}")

        # Create relationships (deduplicated)
        print(f"\n--- Creating Relationships ---")
        print(f"Total extracted relationships: {len(all_relationships)}")

        seen = set()
        unique_relationships = []
        for rel in all_relationships:
            key = (rel.source, rel.target, rel.rel_type, rel.chapter)
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        print(f"Unique relationships: {len(unique_relationships)}")

        for rel in tqdm(unique_relationships, desc="Creating relationships"):
            builder.create_relationship(rel)

        # Print statistics
        print("\n--- Graph Statistics ---")
        stats = builder.get_stats()
        for name, count in stats.items():
            print(f"  {name}: {count}")

        print("\n--- Relationship Type Distribution ---")
        rel_stats = builder.get_relationship_type_stats()
        for rel_type, count in rel_stats.items():
            print(f"  {rel_type}: {count}")

        print("\n" + "=" * 60)
        print("Graph build complete!")
        print("=" * 60)


def show_stats() -> None:
    """Show current graph statistics."""
    conn = Neo4jConnection()
    if not conn.verify_connectivity():
        print("Cannot connect to Neo4j.")
        return

    with GraphBuilder(conn) as builder:
        print("\n--- Graph Statistics ---")
        stats = builder.get_stats()
        for name, count in stats.items():
            print(f"  {name}: {count}")

        print("\n--- Relationship Type Distribution ---")
        rel_stats = builder.get_relationship_type_stats()
        for rel_type, count in rel_stats.items():
            print(f"  {rel_type}: {count}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Build Neo4j knowledge graph from dialogue data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="Data/staging/",
        help="Path to data directory (default: Data/staging/)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing graph before building",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show graph statistics only",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        build_graph(
            data_dir=args.data_dir,
            clear_existing=args.clear,
        )


if __name__ == "__main__":
    main()
