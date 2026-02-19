"""
Incremental Event Extractor.

Tracks file changes and extracts major events from dialogue files.
Provides incremental processing with file change tracking and caching.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, asdict

from .event_extractor import (
    LLMEventExtractor,
    EventExtractionOutput,
    ExtractedEvent,
)


@dataclass
class EventFileTrackingInfo:
    """Tracking information for a processed file."""

    file_path: str
    content_hash: str
    last_processed: str
    event_count: int
    task_id: str
    chapter: int


class EventCache:
    """Simple content-addressed cache for event extraction results."""

    def __init__(self, cache_dir: str = ".cache/events"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, content: str) -> str:
        """Generate cache key from content hash."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get(self, content: str) -> Optional[EventExtractionOutput]:
        """Get cached result for content."""
        key = self._hash_key(content)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                return EventExtractionOutput(**data)
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
        return None

    def set(self, content: str, result: EventExtractionOutput):
        """Cache extraction result."""
        key = self._hash_key(content)
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        return {
            "cache_dir": str(self.cache_dir),
            "cached_files": len(cache_files),
        }


class IncrementalEventExtractor:
    """
    Incremental Event Extractor.

    Tracks file changes and only processes modified files.
    Extracts major story events (sacrifices, transformations, etc.)
    for building Event nodes in the knowledge graph.

    Usage:
        extractor = IncrementalEventExtractor()

        # Extract events from a single folder
        events = extractor.extract_folder(Path("Data/Archon/1608"))

        # Extract from all data
        all_events = extractor.extract_all(Path("Data/"))

        # Write to graph
        from src.pipeline.graph_builder import GraphBuilder
        with GraphBuilder() as builder:
            for event_data in all_events:
                builder.ingest_extracted_events(
                    events=[e.model_dump() for e in event_data["events"]],
                    chapter=event_data["chapter"],
                    task_id=event_data["task_id"],
                )
    """

    def __init__(
        self,
        cache_dir: str = ".cache/events",
        tracking_file: Optional[str] = None,
    ):
        """
        Initialize the incremental event extractor.

        Args:
            cache_dir: Directory for cache storage
            tracking_file: Path to tracking JSON file
        """
        self.cache = EventCache(cache_dir)
        self.tracking_file = Path(tracking_file or f"{cache_dir}/event_tracking.json")
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        self._extractor = None
        self.tracking = self._load_tracking()

    @property
    def extractor(self) -> LLMEventExtractor:
        """Lazy load the LLM extractor."""
        if self._extractor is None:
            self._extractor = LLMEventExtractor()
        return self._extractor

    def _load_tracking(self) -> Dict[str, EventFileTrackingInfo]:
        """Load file tracking information from disk."""
        if self.tracking_file.exists():
            try:
                data = json.loads(self.tracking_file.read_text(encoding="utf-8"))
                return {
                    k: EventFileTrackingInfo(**v)
                    for k, v in data.get("files", {}).items()
                }
            except (json.JSONDecodeError, KeyError, TypeError):
                return {}
        return {}

    def _save_tracking(self):
        """Save file tracking information to disk."""
        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "files": {k: asdict(v) for k, v in self.tracking.items()},
        }
        self.tracking_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _hash_content(self, content: str) -> str:
        """Calculate MD5 hash of content."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _hash_file(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content."""
        content = file_path.read_text(encoding="utf-8")
        return self._hash_content(content)

    def _parse_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse task_id and chapter from file path.

        Expected structure: Data/Archon/{task_id}/chapter{N}_dialogue.txt
        Returns: {"task_id": "1608", "chapter": 160800, "chapter_num": 0}
        """
        # Extract task_id from parent folder name
        task_id = file_path.parent.name

        # Extract chapter number from filename
        match = re.search(r"chapter(\d+)", file_path.stem)
        chapter_num = int(match.group(1)) if match else 0

        # Calculate GlobalChapter = TaskID * 100 + ChapterNum
        try:
            global_chapter = int(task_id) * 100 + chapter_num
        except ValueError:
            global_chapter = chapter_num

        return {
            "task_id": task_id,
            "chapter": global_chapter,
            "chapter_num": chapter_num,
        }

    def _extract_characters_from_dialogue(self, text: str) -> List[str]:
        """Extract character names from dialogue text (simple heuristic)."""
        characters: Set[str] = set()
        for line in text.split("\n"):
            if "：" in line and not line.startswith("#"):
                char_name = line.split("：")[0].strip()
                # Filter out non-character lines
                if (
                    char_name
                    and len(char_name) < 20
                    and not char_name.startswith("（")
                    and char_name not in {"选项", "---", "？？？"}
                ):
                    characters.add(char_name)
        return list(characters)

    def get_changed_files(
        self,
        data_dir: Path,
        pattern: str = "**/chapter*_dialogue.txt",
    ) -> List[Path]:
        """
        Get list of files that have changed since last processing.

        Args:
            data_dir: Directory to scan for files
            pattern: Glob pattern for matching files

        Returns:
            List of file paths that are new or modified
        """
        changed = []
        for file_path in data_dir.glob(pattern):
            file_key = str(file_path)
            current_hash = self._hash_file(file_path)

            if file_key not in self.tracking:
                changed.append(file_path)
            elif self.tracking[file_key].content_hash != current_hash:
                changed.append(file_path)

        return changed

    def extract_file(
        self,
        file_path: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract events from a single file with caching.

        Tracking is updated immediately after processing each file to prevent
        data loss if extraction is interrupted.

        Args:
            file_path: Path to the dialogue file
            force: If True, bypass cache and re-extract

        Returns:
            Dict with keys: events, task_id, chapter, file_path
        """
        content = file_path.read_text(encoding="utf-8")
        content_hash = self._hash_content(content)
        metadata = self._parse_file_metadata(file_path)
        file_key = str(file_path)

        # Check cache first
        cached = None
        if not force:
            cached = self.cache.get(content)

        # Fast path: file unchanged and cached - skip processing
        if not force and file_key in self.tracking:
            if self.tracking[file_key].content_hash == content_hash and cached is not None:
                return {
                    "events": cached.events,
                    "task_id": metadata["task_id"],
                    "chapter": metadata["chapter"],
                    "file_path": file_key,
                }

        if cached is not None:
            # Use cached result
            events = cached.events
            event_count = len(events)
        else:
            # Extract characters for context
            characters = self._extract_characters_from_dialogue(content)

            # Extract using LLM
            result = self.extractor.extract(
                dialogue=content,
                characters=characters,
                chapter=metadata["chapter"],
                task_id=metadata["task_id"],
            )

            # Cache result immediately
            self.cache.set(content, result)
            events = result.events
            event_count = len(events)

        # Update tracking immediately after extraction/cache hit
        # This ensures progress is saved even if later files fail
        self.tracking[file_key] = EventFileTrackingInfo(
            file_path=file_key,
            content_hash=content_hash,
            last_processed=datetime.now().isoformat(),
            event_count=event_count,
            task_id=metadata["task_id"],
            chapter=metadata["chapter"],
        )
        self._save_tracking()

        return {
            "events": events,
            "task_id": metadata["task_id"],
            "chapter": metadata["chapter"],
            "file_path": file_key,
        }

    def extract_folder(
        self,
        folder_path: Path,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract events from all dialogue files in a folder.

        Args:
            folder_path: Path to folder (e.g., Data/Archon/1608)
            force: If True, bypass cache

        Returns:
            List of extraction results
        """
        results = []
        dialogue_files = sorted(folder_path.glob("chapter*_dialogue.txt"))

        for file_path in dialogue_files:
            print(f"Processing: {file_path.name}...")
            result = self.extract_file(file_path, force=force)
            results.append(result)
            print(f"  Extracted {len(result['events'])} events")

        return results

    def extract_all(
        self,
        data_dir: Path,
        pattern: str = "**/chapter*_dialogue.txt",
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract events from all dialogue files.

        Args:
            data_dir: Root data directory
            pattern: Glob pattern for files
            force: If True, bypass cache

        Returns:
            List of extraction results for all files
        """
        results = []
        for file_path in sorted(data_dir.glob(pattern)):
            result = self.extract_file(file_path, force=force)
            results.append(result)
        return results

    def extract_incremental(
        self,
        data_dir: Path,
        pattern: str = "**/chapter*_dialogue.txt",
    ) -> List[Dict[str, Any]]:
        """
        Incremental extraction: only process changed files.

        Args:
            data_dir: Root data directory
            pattern: Glob pattern for files

        Returns:
            List of extraction results for changed files only
        """
        changed = self.get_changed_files(data_dir, pattern)
        results = []

        print(f"Found {len(changed)} changed files to process")
        for file_path in changed:
            print(f"Processing: {file_path}...")
            result = self.extract_file(file_path)
            results.append(result)
            print(f"  Extracted {len(result['events'])} events")

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get extraction status."""
        return {
            "tracked_files": len(self.tracking),
            "cache_stats": self.cache.get_stats(),
            "files": [
                {
                    "path": info.file_path,
                    "events": info.event_count,
                    "task_id": info.task_id,
                    "chapter": info.chapter,
                    "last_processed": info.last_processed,
                }
                for info in self.tracking.values()
            ],
        }

    def clear_tracking(self):
        """Clear all tracking information (but keep cache)."""
        self.tracking = {}
        if self.tracking_file.exists():
            self.tracking_file.unlink()

    def cleanup_orphan_cache(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Remove cache files that are not referenced in tracking.

        Cache files are content-addressed by MD5 hash. This method removes
        any cache file whose hash is not in the current tracking info.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Dict with cleanup statistics
        """
        # Get all valid content hashes from tracking
        valid_hashes = {info.content_hash for info in self.tracking.values()}

        # Find all cache files (excluding tracking file)
        cache_files = list(self.cache.cache_dir.glob("*.json"))
        tracking_filename = self.tracking_file.name

        orphans = []
        kept = []

        for cache_file in cache_files:
            if cache_file.name == tracking_filename:
                continue

            # Cache filename is {hash}.json
            file_hash = cache_file.stem

            if file_hash not in valid_hashes:
                orphans.append(cache_file)
                if not dry_run:
                    cache_file.unlink()
            else:
                kept.append(cache_file)

        return {
            "orphans_found": len(orphans),
            "orphans_deleted": 0 if dry_run else len(orphans),
            "files_kept": len(kept),
            "dry_run": dry_run,
            "orphan_files": [str(f) for f in orphans],
        }

    def rebuild_tracking(
        self,
        data_dir: Path,
        pattern: str = "**/chapter*_dialogue.txt",
    ) -> Dict[str, Any]:
        """
        Rebuild tracking from existing cache files.

        This is faster than re-extracting when cache files exist but tracking
        was lost (e.g., due to race conditions or manual deletion).

        Args:
            data_dir: Root data directory to scan
            pattern: Glob pattern for dialogue files

        Returns:
            Dict with rebuild statistics
        """
        stats = {
            "files_scanned": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "already_tracked": 0,
        }

        dialogue_files = sorted(data_dir.glob(pattern))
        print(f"Scanning {len(dialogue_files)} files...")

        for file_path in dialogue_files:
            stats["files_scanned"] += 1
            file_key = str(file_path)

            # Skip if already tracked with same hash
            content = file_path.read_text(encoding="utf-8")
            content_hash = self._hash_content(content)

            if file_key in self.tracking:
                if self.tracking[file_key].content_hash == content_hash:
                    stats["already_tracked"] += 1
                    continue

            # Try to find cache
            cached = self.cache.get(content)
            if cached is None:
                stats["cache_misses"] += 1
                continue

            # Cache hit - rebuild tracking entry
            stats["cache_hits"] += 1
            metadata = self._parse_file_metadata(file_path)

            self.tracking[file_key] = EventFileTrackingInfo(
                file_path=file_key,
                content_hash=content_hash,
                last_processed=datetime.now().isoformat(),
                event_count=len(cached.events),
                task_id=metadata["task_id"],
                chapter=metadata["chapter"],
            )

            print(f"  Restored: {file_path.name} ({len(cached.events)} events)")

        # Save tracking
        self._save_tracking()

        return stats


def write_events_to_graph(
    extraction_results: List[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Write extracted events to the Neo4j graph.

    Args:
        extraction_results: List of extraction results from IncrementalEventExtractor
        dry_run: If True, don't actually write to graph

    Returns:
        Dict with statistics
    """
    from .graph_builder import GraphBuilder

    stats = {
        "total_events": 0,
        "events_written": 0,
        "files_processed": 0,
    }

    if dry_run:
        for result in extraction_results:
            stats["total_events"] += len(result["events"])
            stats["files_processed"] += 1
        return stats

    with GraphBuilder() as builder:
        # Ensure schema is set up
        builder.setup_schema()

        for result in extraction_results:
            events = result["events"]
            chapter = result["chapter"]
            task_id = result["task_id"]

            # Convert Pydantic models to dicts
            event_dicts = [e.model_dump() for e in events]

            count = builder.ingest_extracted_events(
                events=event_dicts,
                chapter=chapter,
                task_id=task_id,
            )

            stats["total_events"] += len(events)
            stats["events_written"] += count
            stats["files_processed"] += 1

    return stats


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Incremental Event Extractor")
    print("=" * 60)

    # Check command line args
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.ingestion.incremental_event_extractor <folder>")
        print("  python -m src.ingestion.incremental_event_extractor Data/Archon/1608")
        print("  python -m src.ingestion.incremental_event_extractor Data/Archon/1608 --write")
        print("  python -m src.ingestion.incremental_event_extractor --cleanup")
        print("  python -m src.ingestion.incremental_event_extractor --cleanup --dry-run")
        print("  python -m src.ingestion.incremental_event_extractor --rebuild Data/")
        print("  python -m src.ingestion.incremental_event_extractor --status")
        sys.exit(1)

    # Initialize extractor
    extractor = IncrementalEventExtractor()

    # Handle special commands
    if "--rebuild" in sys.argv:
        # Find the data directory argument
        data_dir = None
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                data_dir = Path(arg)
                break
        if data_dir is None:
            data_dir = Path("Data/")

        print(f"\nRebuilding tracking from cache for: {data_dir}")
        stats = extractor.rebuild_tracking(data_dir)
        print(f"\n{'='*60}")
        print(f"REBUILD COMPLETE")
        print(f"{'='*60}")
        print(f"Files scanned: {stats['files_scanned']}")
        print(f"Already tracked: {stats['already_tracked']}")
        print(f"Cache hits (restored): {stats['cache_hits']}")
        print(f"Cache misses (need extraction): {stats['cache_misses']}")
        sys.exit(0)

    if "--cleanup" in sys.argv:
        dry_run = "--dry-run" in sys.argv
        print(f"\nCleaning orphan cache files (dry_run={dry_run})...")
        result = extractor.cleanup_orphan_cache(dry_run=dry_run)
        print(f"  Tracked files: {len(extractor.tracking)}")
        print(f"  Cache files kept: {result['files_kept']}")
        print(f"  Orphans found: {result['orphans_found']}")
        if dry_run:
            print(f"  (Dry run - no files deleted)")
            if result['orphan_files']:
                print(f"\n  Would delete:")
                for f in result['orphan_files']:
                    print(f"    {f}")
        else:
            print(f"  Orphans deleted: {result['orphans_deleted']}")
        sys.exit(0)

    if "--status" in sys.argv:
        status = extractor.get_status()
        print(f"\nTracked files: {status['tracked_files']}")
        print(f"Cache stats: {status['cache_stats']}")
        print(f"\nFiles:")
        for f in status['files']:
            print(f"  {f['path']}: {f['events']} events (chapter {f['chapter']})")
        sys.exit(0)

    folder = Path(sys.argv[1])
    write_to_graph = "--write" in sys.argv

    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    # Extract events
    print(f"\nExtracting events from: {folder}")
    results = extractor.extract_folder(folder)

    # Print summary
    total_events = sum(len(r["events"]) for r in results)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {len(results)}")
    print(f"Total events extracted: {total_events}")

    # Print events by type
    type_counts: Dict[str, int] = {}
    for result in results:
        for event in result["events"]:
            t = event.event_type
            type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nEvent type distribution:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # Print sample events
    print(f"\nSample events:")
    for result in results[:2]:
        print(f"\n  File: {result['file_path']}")
        for event in result["events"][:3]:
            print(f"    [{event.event_type}] {event.name}")
            print(f"      Summary: {event.summary[:60]}...")

    # Write to graph if requested
    if write_to_graph:
        print(f"\n{'='*60}")
        print("Writing events to Neo4j graph...")
        stats = write_events_to_graph(results)
        print(f"Events written: {stats['events_written']}")
    else:
        print(f"\nTo write events to graph, run with --write flag")
