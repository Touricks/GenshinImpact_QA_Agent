"""Validate NK 1601 data compatibility with DocumentLoader and SceneChunker."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "Data" / "staging" / "Archon"
    task_dir = data_dir / "1601"

    if not task_dir.exists():
        logger.error(f"Task directory not found: {task_dir}")
        return

    # --- 1. Load documents ---
    from src.pipeline.loader import DocumentLoader

    loader = DocumentLoader(task_dir)
    docs = list(loader.load_all())
    logger.info(f"Loaded {len(docs)} document(s) from {task_dir}")

    if not docs:
        logger.error("No documents loaded — check file naming / header format")
        return

    for doc in docs:
        m = doc.metadata
        logger.info(
            f"  [{m.task_id}] ch{m.chapter_number}: {m.task_name} / {m.chapter_title}"
        )

    # --- 2. Chunk with SceneChunker ---
    from src.pipeline.chunker import SceneChunker

    chunker = SceneChunker()
    total_chunks = 0

    for doc in docs:
        chunks = chunker.chunk_document(doc)
        total_chunks += len(chunks)
        logger.info(
            f"  ch{doc.metadata.chapter_number}: {len(chunks)} chunks"
        )

        # Print first 3 chunk previews
        for i, (title, text) in enumerate(chunks[:3]):
            preview = text[:80].replace("\n", " ")
            logger.info(f"    [{i}] {title}: {preview}...")

    # --- 3. Summary ---
    logger.info("=" * 60)
    logger.info(f"NK 1601 compatibility validation summary:")
    logger.info(f"  Documents loaded: {len(docs)}")
    logger.info(f"  Total chunks:     {total_chunks}")
    logger.info(f"  Status:           {'PASS' if total_chunks > 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
