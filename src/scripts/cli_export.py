"""
Export and import Neo4j + Qdrant data as portable JSON snapshots.

Usage:
    python -m src.scripts.cli_export export [--output-dir Data/exports/]
    python -m src.scripts.cli_export import Data/exports/2026-02-12_neo4j_qdrant/

Export creates:
    Data/exports/{timestamp}_neo4j_qdrant/
    ├── manifest.json
    ├── neo4j/
    │   ├── nodes.json
    │   └── relationships.json
    └── qdrant/
        └── points.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ..common.graph.connection import Neo4jConnection
from ..pipeline.graph_builder import GraphBuilder
from ..common.config import Settings

# Node labels to export (matches builder.py entity types + Chunk + MajorEvent)
NODE_LABELS = ["Character", "Faction", "Place", "Item", "Concept", "Event", "Chunk", "MajorEvent"]


# ─────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────

def export_neo4j(conn: Neo4jConnection, out_dir: Path) -> dict:
    """Export all Neo4j nodes and relationships to JSON files."""
    neo4j_dir = out_dir / "neo4j"
    neo4j_dir.mkdir(parents=True, exist_ok=True)

    # --- Nodes ---
    all_nodes = []
    for label in NODE_LABELS:
        query = f"MATCH (n:{label}) RETURN n, labels(n) as labels"
        records = conn.execute(query)
        for rec in records:
            node = dict(rec["n"])
            node["_labels"] = rec["labels"]
            # Convert neo4j types to JSON-safe types
            for k, v in node.items():
                if isinstance(v, list):
                    node[k] = [str(x) if not isinstance(x, (str, int, float, bool, type(None))) else x for x in v]
            all_nodes.append(node)

    (neo4j_dir / "nodes.json").write_text(
        json.dumps(all_nodes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Exported {len(all_nodes)} nodes")

    # --- Relationships ---
    query = """
    MATCH (a)-[r]->(b)
    RETURN
        a.name as source, labels(a) as source_labels,
        type(r) as rel_type, properties(r) as props,
        b.name as target, labels(b) as target_labels
    """
    records = conn.execute(query)
    all_rels = []
    for rec in records:
        rel = {
            "source": rec["source"],
            "source_label": rec["source_labels"][0] if rec["source_labels"] else None,
            "rel_type": rec["rel_type"],
            "properties": rec["props"] or {},
            "target": rec["target"],
            "target_label": rec["target_labels"][0] if rec["target_labels"] else None,
        }
        all_rels.append(rel)

    (neo4j_dir / "relationships.json").write_text(
        json.dumps(all_rels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Exported {len(all_rels)} relationships")

    return {"nodes": len(all_nodes), "relationships": len(all_rels)}


def _serialize_vector(vec):
    """Serialize a point's vector field to JSON-safe format.

    Handles both named vectors (dict with dense/sparse) and single vectors (list).
    SparseVector objects are converted to {"indices": [...], "values": [...]}.
    """
    if isinstance(vec, dict):
        out = {}
        for name, v in vec.items():
            if hasattr(v, "indices"):  # SparseVector
                out[name] = {"indices": list(v.indices), "values": list(v.values)}
            else:
                out[name] = v
        return out
    return vec


def export_qdrant(out_dir: Path) -> dict:
    """Export all Qdrant points (with vectors) to JSON.

    Supports both single-vector and named-vector (hybrid) collections.
    """
    from qdrant_client import QdrantClient

    settings = Settings()
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    collection_name = settings.COLLECTION_NAME
    info = client.get_collection(collection_name)

    # Detect vector config type (single vs named)
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict):
        dense_cfg = vectors_config.get("dense")
        vector_dim = dense_cfg.size if dense_cfg else 0
        vector_names = list(vectors_config.keys())
    else:
        vector_dim = vectors_config.size
        vector_names = None

    # Detect sparse vectors
    sparse_config = info.config.params.sparse_vectors or {}
    sparse_names = list(sparse_config.keys()) if sparse_config else []

    qdrant_dir = out_dir / "qdrant"
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    # Scroll through all points
    all_points = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_vectors=True,
            offset=offset,
        )
        for pt in results:
            all_points.append({
                "id": pt.id,
                "vector": _serialize_vector(pt.vector),
                "payload": pt.payload,
            })
        if next_offset is None:
            break
        offset = next_offset

    (qdrant_dir / "points.json").write_text(
        json.dumps(all_points, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mode = "hybrid" if (vector_names or sparse_names) else "single"
    print(f"  Exported {len(all_points)} points (mode={mode}, dense_dim={vector_dim})")
    if vector_names:
        print(f"  Dense vectors: {vector_names}")
    if sparse_names:
        print(f"  Sparse vectors: {sparse_names}")

    return {
        "collection": collection_name,
        "points": len(all_points),
        "vector_dim": vector_dim,
        "vector_names": vector_names,
        "sparse_names": sparse_names,
    }


def run_export(output_dir: str) -> None:
    """Run full export of Neo4j + Qdrant."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    export_name = f"{timestamp}_neo4j_qdrant"
    out_dir = Path(output_dir) / export_name

    if out_dir.exists():
        print(f"Export directory already exists: {out_dir}")
        print("Overwriting...")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Exporting to: {out_dir}")
    print("=" * 60)

    # Neo4j
    print("\n[1/2] Exporting Neo4j...")
    conn = Neo4jConnection()
    if not conn.verify_connectivity():
        print("ERROR: Cannot connect to Neo4j.")
        sys.exit(1)
    try:
        neo4j_stats = export_neo4j(conn, out_dir)
    finally:
        conn.close()

    # Qdrant
    print("\n[2/2] Exporting Qdrant...")
    try:
        qdrant_stats = export_qdrant(out_dir)
    except Exception as e:
        print(f"ERROR: Qdrant export failed: {e}")
        sys.exit(1)

    # Manifest
    settings = Settings()
    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "region": settings.ACTIVE_REGION,
        "neo4j": neo4j_stats,
        "qdrant": qdrant_stats,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nManifest written to {out_dir / 'manifest.json'}")
    print(json.dumps(manifest, indent=2))
    print("\nExport complete!")


# ─────────────────────────────────────────────────────────────
# Import
# ─────────────────────────────────────────────────────────────

def import_neo4j(conn: Neo4jConnection, snapshot_dir: Path) -> None:
    """Import Neo4j nodes and relationships from JSON snapshot."""
    nodes_file = snapshot_dir / "neo4j" / "nodes.json"
    rels_file = snapshot_dir / "neo4j" / "relationships.json"

    if not nodes_file.exists() or not rels_file.exists():
        print("ERROR: Missing neo4j/nodes.json or neo4j/relationships.json")
        sys.exit(1)

    nodes = json.loads(nodes_file.read_text(encoding="utf-8"))
    rels = json.loads(rels_file.read_text(encoding="utf-8"))

    # Clear and set up schema
    builder = GraphBuilder(conn)
    print("  Clearing graph...")
    builder.clear_graph()
    builder.setup_schema()

    # Import nodes
    print(f"  Importing {len(nodes)} nodes...")
    for node in nodes:
        labels = node.pop("_labels", ["Character"])
        label = labels[0]
        name = node.get("name") or node.get("chunk_id")
        if not name:
            continue

        # Build SET clause from all properties
        props = {k: v for k, v in node.items()}
        set_parts = [f"n.{k} = ${k}" for k in props]
        set_clause = ", ".join(set_parts)

        if label == "Chunk":
            # Chunk nodes use chunk_id as unique key
            query = f"MERGE (n:{label} {{chunk_id: $chunk_id}}) SET {set_clause}"
        elif label == "MajorEvent":
            # MajorEvent uses composite key
            query = f"MERGE (n:{label} {{name: $name}}) SET {set_clause}"
        else:
            query = f"MERGE (n:{label} {{name: $name}}) SET {set_clause}"

        try:
            conn.execute_write(query, props)
        except Exception as e:
            print(f"  Warning: Failed to import node {label}:{name}: {e}")

    # Import relationships
    print(f"  Importing {len(rels)} relationships...")
    for rel in rels:
        source = rel["source"]
        target = rel["target"]
        rel_type = rel["rel_type"]
        props = rel.get("properties", {})

        prop_sets = ", ".join([f"r.{k} = $prop_{k}" for k in props])
        set_clause = f"SET {prop_sets}" if prop_sets else ""

        query = f"""
        MATCH (a {{name: $source}})
        MATCH (b {{name: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        {set_clause}
        """

        params = {"source": source, "target": target}
        params.update({f"prop_{k}": v for k, v in props.items()})

        try:
            conn.execute_write(query, params)
        except Exception:
            pass  # Nodes may not exist (e.g. dangling refs)

    # Stats
    stats = builder.get_stats()
    print("  Import stats:")
    for name, count in stats.items():
        print(f"    {name}: {count}")
    builder.close()


def _deserialize_vector(vec):
    """Deserialize a point's vector from JSON back to Qdrant-compatible format.

    Converts sparse vector dicts {"indices": [...], "values": [...]} back to
    SparseVector objects.
    """
    from qdrant_client.models import SparseVector

    if isinstance(vec, dict):
        out = {}
        for name, v in vec.items():
            if isinstance(v, dict) and "indices" in v and "values" in v:
                out[name] = SparseVector(indices=v["indices"], values=v["values"])
            else:
                out[name] = v
        return out
    return vec


def import_qdrant(snapshot_dir: Path, manifest: dict) -> None:
    """Import Qdrant points from JSON snapshot.

    Supports both single-vector and named-vector (hybrid) collections.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct

    points_file = snapshot_dir / "qdrant" / "points.json"
    if not points_file.exists():
        print("ERROR: Missing qdrant/points.json")
        sys.exit(1)

    points_data = json.loads(points_file.read_text(encoding="utf-8"))

    qdrant_meta = manifest.get("qdrant", {})
    collection_name = qdrant_meta.get("collection", "genshin_story")
    vector_dim = qdrant_meta.get("vector_dim", 768)
    vector_names = qdrant_meta.get("vector_names")
    sparse_names = qdrant_meta.get("sparse_names", [])

    settings = Settings()
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    # Delete + recreate collection
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"  Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)

    # Build vectors config based on manifest metadata
    if vector_names:
        # Named vectors (hybrid mode)
        vectors_config = {}
        for vname in vector_names:
            vectors_config[vname] = VectorParams(size=vector_dim, distance=Distance.COSINE)
        sparse_vectors_config = {}
        for sname in sparse_names:
            sparse_vectors_config[sname] = SparseVectorParams()
        mode = "hybrid"
        print(f"  Creating collection '{collection_name}' (mode={mode}, dense_dim={vector_dim})")
        print(f"    Dense vectors: {vector_names}")
        print(f"    Sparse vectors: {sparse_names}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config or None,
        )
    else:
        # Single vector mode
        mode = "single"
        print(f"  Creating collection '{collection_name}' (mode={mode}, vector_dim={vector_dim})")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    # Create payload indexes for common fields
    for field in ["region", "source_file", "entity_type"]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass

    # Batch upsert
    print(f"  Upserting {len(points_data)} points...")
    batch_size = 100
    for i in range(0, len(points_data), batch_size):
        batch = points_data[i : i + batch_size]
        points = [
            PointStruct(
                id=pt["id"],
                vector=_deserialize_vector(pt["vector"]) if vector_names else pt["vector"],
                payload=pt["payload"],
            )
            for pt in batch
        ]
        client.upsert(collection_name=collection_name, points=points)

    info = client.get_collection(collection_name)
    print(f"  Collection '{collection_name}': {info.points_count} points")


def run_import(snapshot_path: str) -> None:
    """Run full import from a JSON snapshot directory."""
    snapshot_dir = Path(snapshot_path)

    manifest_file = snapshot_dir / "manifest.json"
    if not manifest_file.exists():
        print(f"ERROR: No manifest.json found in {snapshot_dir}")
        sys.exit(1)

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    print("=" * 60)
    print(f"Importing from: {snapshot_dir}")
    print(f"Exported at: {manifest.get('exported_at', 'unknown')}")
    print(f"Region: {manifest.get('region', 'unknown')}")
    print("=" * 60)

    # Neo4j
    print("\n[1/2] Importing Neo4j...")
    conn = Neo4jConnection()
    if not conn.verify_connectivity():
        print("ERROR: Cannot connect to Neo4j.")
        sys.exit(1)
    try:
        import_neo4j(conn, snapshot_dir)
    finally:
        conn.close()

    # Qdrant
    print("\n[2/2] Importing Qdrant...")
    try:
        import_qdrant(snapshot_dir, manifest)
    except Exception as e:
        print(f"ERROR: Qdrant import failed: {e}")
        sys.exit(1)

    print("\nImport complete!")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export/import Neo4j + Qdrant snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # export
    export_parser = subparsers.add_parser("export", help="Export Neo4j + Qdrant to JSON")
    export_parser.add_argument(
        "--output-dir",
        default="Data/exports/",
        help="Base output directory (default: Data/exports/)",
    )

    # import
    import_parser = subparsers.add_parser("import", help="Import JSON snapshot into Neo4j + Qdrant")
    import_parser.add_argument(
        "snapshot_dir",
        help="Path to snapshot directory (e.g. Data/exports/2026-02-12_neo4j_qdrant/)",
    )

    args = parser.parse_args()

    if args.command == "export":
        run_export(args.output_dir)
    elif args.command == "import":
        run_import(args.snapshot_dir)


if __name__ == "__main__":
    main()
