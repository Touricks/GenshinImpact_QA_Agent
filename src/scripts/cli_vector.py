#!/usr/bin/env python3
"""CLI entry point for ingestion pipeline."""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest Genshin dialogue data into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full ingestion
  python -m src.scripts.ingest Data/

  # Incremental (skip unchanged files)
  python -m src.scripts.ingest Data/ --incremental

  # Dry run (parse and chunk only, no Qdrant)
  python -m src.scripts.ingest Data/ --dry-run

  # Verbose mode
  python -m src.scripts.ingest Data/ -v
        """,
    )

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to Data/ directory containing dialogue files",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant server host (default: localhost)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port (default: 6333)",
    )
    parser.add_argument(
        "--collection",
        default="genshin_story",
        help="Qdrant collection name (default: genshin_story)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without generating embeddings or indexing",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only process new or modified files (skip unchanged)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    if not args.data_dir.is_dir():
        logger.error(f"Path is not a directory: {args.data_dir}")
        sys.exit(1)

    # Import here to avoid slow startup for --help
    from src.pipeline.pipeline import IngestionPipeline, IncrementalIngestionPipeline

    # Create and run pipeline
    if args.incremental:
        logger.info(f"Starting incremental ingestion from: {args.data_dir}")
        pipeline = IncrementalIngestionPipeline(
            data_dir=args.data_dir,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            collection_name=args.collection,
        )
    else:
        logger.info(f"Starting full ingestion from: {args.data_dir}")
        pipeline = IngestionPipeline(
            data_dir=args.data_dir,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            collection_name=args.collection,
        )

    stats = pipeline.run(dry_run=args.dry_run)

    # Print summary
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(f"Documents processed: {stats.documents_processed}")
    print(f"Documents failed:    {stats.documents_failed}")
    print(f"Chunks created:      {stats.chunks_created}")
    print(f"Chunks indexed:      {stats.chunks_indexed}")

    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for error in stats.errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(stats.errors) > 5:
            print(f"  ... and {len(stats.errors) - 5} more")

    # Exit with error code if there were failures
    if stats.documents_failed > 0 or stats.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
