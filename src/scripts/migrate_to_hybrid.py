"""Migration script: rebuild Qdrant collection with hybrid search (3 vector types).

Usage:
    .venv/bin/python -m src.scripts.migrate_to_hybrid --rebuild
    .venv/bin/python -m src.scripts.migrate_to_hybrid --swap
    .venv/bin/python -m src.scripts.migrate_to_hybrid --rebuild --swap
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def rebuild(collection_name: str, data_dir: str = None):
    """Create a new hybrid collection and reindex all data."""
    from ..common.config.settings import settings
    from ..pipeline.pipeline import IngestionPipeline

    data_dir = Path(data_dir) if data_dir else settings.DATA_DIR / "processed"
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info(f"=== Rebuilding hybrid collection: {collection_name} ===")
    logger.info(f"Data source: {data_dir}")

    pipeline = IngestionPipeline(
        data_dir=data_dir,
        qdrant_host=settings.QDRANT_HOST,
        qdrant_port=settings.QDRANT_PORT,
        collection_name=collection_name,
        hybrid=True,
    )

    stats = pipeline.run()

    logger.info("=== Rebuild complete ===")
    logger.info(f"Stats: {stats.to_dict()}")

    if stats.errors:
        logger.warning(f"Errors during rebuild: {stats.errors}")

    # Verify collection
    from ..common.vector.indexer import VectorIndexer

    indexer = VectorIndexer(collection_name=collection_name)
    info = indexer.get_collection_info()
    logger.info(f"Collection info: {info}")

    return stats


def swap(new_collection: str, old_collection: str):
    """Swap to the new collection by deleting the old one.

    Note: This does NOT auto-update settings.py COLLECTION_NAME.
    After swap, update .env or settings.py to point to the new collection.
    """
    from ..common.vector.indexer import VectorIndexer

    # Verify new collection exists and has data
    new_indexer = VectorIndexer(collection_name=new_collection)
    try:
        info = new_indexer.get_collection_info()
        points = info.get("points_count", 0)
        if points == 0:
            logger.error(f"New collection '{new_collection}' has 0 points. Aborting swap.")
            sys.exit(1)
        logger.info(f"New collection '{new_collection}': {points} points")
    except Exception as e:
        logger.error(f"Cannot access new collection '{new_collection}': {e}")
        sys.exit(1)

    # Delete old collection if it exists
    old_indexer = VectorIndexer(collection_name=old_collection)
    try:
        old_info = old_indexer.get_collection_info()
        logger.info(f"Old collection '{old_collection}': {old_info.get('points_count', 0)} points")
        old_indexer.delete_collection()
        logger.info(f"Deleted old collection '{old_collection}'")
    except Exception:
        logger.info(f"Old collection '{old_collection}' does not exist, nothing to delete")

    logger.info(f"=== Swap complete ===")
    logger.info(f"Active collection: {new_collection}")
    logger.info(f"Update COLLECTION_NAME in .env or settings.py to '{new_collection}'")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant collection to hybrid search (3 vector types)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Create new hybrid collection and reindex all data",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        help="Delete old collection after verifying new one",
    )
    parser.add_argument(
        "--new-collection",
        default="genshin_story_v3",
        help="Name for the new hybrid collection (default: genshin_story_v3)",
    )
    parser.add_argument(
        "--old-collection",
        default="genshin_story",
        help="Name of the old collection to delete on swap (default: genshin_story)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data source directory (default: Data/processed)",
    )

    args = parser.parse_args()

    if not args.rebuild and not args.swap:
        parser.error("At least one of --rebuild or --swap is required")

    if args.rebuild:
        rebuild(args.new_collection, data_dir=args.data_dir)

    if args.swap:
        swap(args.new_collection, args.old_collection)


if __name__ == "__main__":
    main()
