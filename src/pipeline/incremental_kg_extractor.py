"""
Incremental Knowledge Graph Extractor.

Tracks file changes and extracts entities/relationships from dialogue files.
Provides incremental processing with file change tracking and caching.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, asdict

from .llm_kg_extractor import (
    LLMKnowledgeGraphExtractor,
    KnowledgeGraphOutput,
    ExtractedEntity,
    ExtractedRelationship,
)


@dataclass
class KGFileTrackingInfo:
    """Tracking information for a processed file."""

    file_path: str
    content_hash: str
    last_processed: str
    entity_count: int
    relationship_count: int
    task_id: str
    chapter: int


class KGCache:
    """Simple content-addressed cache for KG extraction results."""

    def __init__(self, cache_dir: str = ".cache/kg"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, content: str) -> str:
        """Generate cache key from content hash."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get(self, content: str) -> Optional[KnowledgeGraphOutput]:
        """Get cached raw result for content.

        Returns raw (un-normalized) KG output if available.
        Old v1 cache (normalized, no _raw marker) is discarded — returns None
        so that the caller re-extracts and caches the raw version.
        """
        key = self._hash_key(content)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                # Reject old v1 cache that was normalized (no _raw marker)
                if not data.get("_raw", False):
                    return None
                # Strip the _raw marker before parsing
                payload = {k: v for k, v in data.items() if k != "_raw"}
                return KnowledgeGraphOutput(**payload)
            except (json.JSONDecodeError, KeyError, TypeError, Exception):
                return None
        return None

    def set(self, content: str, result: KnowledgeGraphOutput):
        """Cache raw (un-normalized) extraction result with _raw marker."""
        key = self._hash_key(content)
        cache_file = self.cache_dir / f"{key}.json"
        data = json.loads(result.model_dump_json())
        data["_raw"] = True
        cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        return {
            "cache_dir": str(self.cache_dir),
            "cached_files": len(cache_files),
        }


class IncrementalKGExtractor:
    """
    Incremental Knowledge Graph Extractor.

    Tracks file changes and only processes modified files.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/kg",
        tracking_file: Optional[str] = None,
    ):
        self.cache = KGCache(cache_dir)
        self.tracking_file = Path(tracking_file or f"{cache_dir}/kg_tracking.json")
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        self._extractor = None
        self.tracking = self._load_tracking()

    @property
    def extractor(self) -> LLMKnowledgeGraphExtractor:
        """Lazy load the LLM extractor."""
        if self._extractor is None:
            self._extractor = LLMKnowledgeGraphExtractor()
        return self._extractor

    def _load_tracking(self) -> Dict[str, KGFileTrackingInfo]:
        if self.tracking_file.exists():
            try:
                data = json.loads(self.tracking_file.read_text(encoding="utf-8"))
                return {
                    k: KGFileTrackingInfo(**v)
                    for k, v in data.get("files", {}).items()
                }
            except (json.JSONDecodeError, KeyError, TypeError):
                return {}
        return {}

    def _save_tracking(self):
        data = {
            "version": "2.0",
            "last_updated": datetime.now().isoformat(),
            "files": {k: asdict(v) for k, v in self.tracking.items()},
        }
        self.tracking_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _hash_content(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _hash_file(self, file_path: Path) -> str:
        content = file_path.read_text(encoding="utf-8")
        return self._hash_content(content)

    def _parse_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse task_id and chapter from file path.

        Expected structure: Data/staging/Archon/{task_id}/chapter{N}_dialogue.txt
        """
        task_id = file_path.parent.name
        match = re.search(r"chapter(\d+)", file_path.stem)
        chapter_num = int(match.group(1)) if match else 0

        try:
            global_chapter = int(task_id) * 100 + chapter_num
        except ValueError:
            global_chapter = chapter_num

        return {
            "task_id": task_id,
            "chapter": global_chapter,
            "chapter_num": chapter_num,
        }

    def get_changed_files(
        self,
        data_dir: Path,
        pattern: str = "**/chapter*_dialogue.txt",
    ) -> List[Path]:
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
        """Extract KG from a single file with raw caching + lazy normalization.

        Flow:
          1. Check cache for raw (un-normalized) LLM output
          2. If miss → call LLM extract_raw() → cache the raw output
          3. Normalize with latest aliases.json before returning
        """
        content = file_path.read_text(encoding="utf-8")
        content_hash = self._hash_content(content)
        metadata = self._parse_file_metadata(file_path)
        file_key = str(file_path)

        # Try to get raw cached output
        cached_raw = None
        if not force:
            cached_raw = self.cache.get(content)

        # Fast path: file unchanged + raw cache exists → normalize and return
        if not force and file_key in self.tracking:
            if self.tracking[file_key].content_hash == content_hash and cached_raw is not None:
                normalized = self.extractor.normalize_output(cached_raw)
                return {
                    "entities": normalized.entities,
                    "relationships": normalized.relationships,
                    "task_id": metadata["task_id"],
                    "chapter": metadata["chapter"],
                    "file_path": file_key,
                }

        # Get raw output (from cache or fresh LLM call)
        if cached_raw is None:
            cached_raw = self.extractor.extract_raw(content)
            self.cache.set(content, cached_raw)

        # Normalize at read time with latest aliases
        normalized = self.extractor.normalize_output(cached_raw)

        self.tracking[file_key] = KGFileTrackingInfo(
            file_path=file_key,
            content_hash=content_hash,
            last_processed=datetime.now().isoformat(),
            entity_count=len(normalized.entities),
            relationship_count=len(normalized.relationships),
            task_id=metadata["task_id"],
            chapter=metadata["chapter"],
        )
        self._save_tracking()

        return {
            "entities": normalized.entities,
            "relationships": normalized.relationships,
            "task_id": metadata["task_id"],
            "chapter": metadata["chapter"],
            "file_path": file_key,
        }

    def extract_folder(
        self,
        folder_path: Path,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        results = []
        dialogue_files = sorted(folder_path.rglob("chapter*_dialogue.txt"))

        for file_path in dialogue_files:
            print(f"Processing: {file_path.name}...")
            result = self.extract_file(file_path, force=force)
            results.append(result)
            print(f"  Extracted {len(result['entities'])} entities, {len(result['relationships'])} relationships")

        return results

    def extract_all(
        self,
        data_dir: Path,
        pattern: str = "**/chapter*_dialogue.txt",
        force: bool = False,
    ) -> List[Dict[str, Any]]:
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
        changed = self.get_changed_files(data_dir, pattern)
        results = []

        print(f"Found {len(changed)} changed files to process")
        for file_path in changed:
            print(f"Processing: {file_path}...")
            result = self.extract_file(file_path)
            results.append(result)
            print(f"  Extracted {len(result['entities'])} entities, {len(result['relationships'])} relationships")

        return results

    def get_status(self) -> Dict[str, Any]:
        total_entities = sum(info.entity_count for info in self.tracking.values())
        total_rels = sum(info.relationship_count for info in self.tracking.values())
        return {
            "tracked_files": len(self.tracking),
            "total_entities": total_entities,
            "total_relationships": total_rels,
            "cache_stats": self.cache.get_stats(),
            "files": [
                {
                    "path": info.file_path,
                    "entities": info.entity_count,
                    "relationships": info.relationship_count,
                    "task_id": info.task_id,
                    "chapter": info.chapter,
                    "last_processed": info.last_processed,
                }
                for info in self.tracking.values()
            ],
        }

    def generate_review_report(
        self,
        folder_path: Path,
        results: List[Dict[str, Any]],
        output_dir: str = "docs_internal/human_evaluate",
    ) -> Optional[Path]:
        """Generate a human review report for entities not matched by aliases.json.

        Compares raw LLM output against normalized output to identify:
        - Entities that aliases.json did not resolve (need human decision)
        - Entities resolved by fuzzy match (need confirmation)
        - Summary statistics for the extraction run

        Returns the path to the generated report, or None if no review needed.
        """
        normalizer = self.extractor.normalizer
        unresolved: List[Dict[str, Any]] = []
        fuzzy_matched: List[Dict[str, Any]] = []
        resolved_count = 0
        total_count = 0

        for result in results:
            file_path = Path(result["file_path"])
            content = file_path.read_text(encoding="utf-8")
            cached_raw = self.cache.get(content)
            if cached_raw is None:
                continue

            for raw_entity in cached_raw.entities:
                total_count += 1
                raw_name = raw_entity.name
                normalized_name = normalizer.normalize(raw_name, raw_entity.entity_type)

                if normalizer.is_known_entity(raw_name):
                    # Exact match in aliases
                    resolved_count += 1
                elif normalized_name != raw_name:
                    # Fuzzy match — resolved but needs confirmation
                    fuzzy_matched.append({
                        "raw_name": raw_name,
                        "normalized_to": normalized_name,
                        "entity_type": raw_entity.entity_type,
                        "file": file_path.name,
                        "text_evidence": raw_entity.text_evidence,
                    })
                    resolved_count += 1
                else:
                    # Unresolved — needs human review
                    unresolved.append({
                        "name": raw_name,
                        "entity_type": raw_entity.entity_type,
                        "description": raw_entity.description,
                        "file": file_path.name,
                        "text_evidence": raw_entity.text_evidence,
                    })

        if not unresolved and not fuzzy_matched:
            return None

        # Build markdown report
        quest_id = folder_path.name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{timestamp}_{quest_id}.md"

        lines = [
            f"# 人工审核: {quest_id}",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"> 文件数: {len(results)}",
            "",
            "## 统计",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总实体数 | {total_count} |",
            f"| aliases.json 命中 | {resolved_count} ({resolved_count*100//max(total_count,1)}%) |",
            f"| 需人工审核 | {len(unresolved)} |",
            f"| 模糊匹配(需确认) | {len(fuzzy_matched)} |",
            "",
        ]

        if unresolved:
            lines += [
                "## 需人工审核的实体",
                "",
                "以下实体未被 aliases.json 匹配，请判断处置方式：",
                "- **加入 aliases**: 已知角色的遗漏别名 → 更新 aliases.json",
                "- **新实体**: 此 quest 独有的新实体 → 确认后加入 aliases.json",
                "- **动态事件**: LLM 动态提取的事件名 → 不需预定义",
                "- **提取错误**: LLM 幻觉/拼写错误 → 忽略",
                "",
                "| 实体 | 类型 | 来源文件 | 描述 | 处置 |",
                "|------|------|----------|------|------|",
            ]
            for item in unresolved:
                desc = (item["description"] or "")[:40]
                lines.append(
                    f"| `{item['name']}` | {item['entity_type']} "
                    f"| {item['file']} | {desc} | **待审核** |"
                )
            lines.append("")

        if fuzzy_matched:
            lines += [
                "## 模糊匹配(需确认)",
                "",
                "以下实体由模糊匹配归一化，请确认是否正确：",
                "",
                "| 原始名 | 归一化为 | 类型 | 来源文件 | 正确? |",
                "|--------|----------|------|----------|-------|",
            ]
            for item in fuzzy_matched:
                lines.append(
                    f"| `{item['raw_name']}` | `{item['normalized_to']}` "
                    f"| {item['entity_type']} | {item['file']} | **待确认** |"
                )
            lines.append("")

        if unresolved:
            lines += [
                "## 原文证据",
                "",
            ]
            for item in unresolved:
                lines.append(f"### `{item['name']}` ({item['entity_type']})")
                lines.append(f"- 文件: {item['file']}")
                lines.append(f"- 证据: {item['text_evidence']}")
                lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def clear_tracking(self):
        self.tracking = {}
        if self.tracking_file.exists():
            self.tracking_file.unlink()

    def cleanup_orphan_cache(self, dry_run: bool = False) -> Dict[str, Any]:
        valid_hashes = {info.content_hash for info in self.tracking.values()}
        cache_files = list(self.cache.cache_dir.glob("*.json"))
        tracking_filename = self.tracking_file.name

        orphans = []
        kept = []

        for cache_file in cache_files:
            if cache_file.name == tracking_filename:
                continue
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

    def renormalize_all(
        self,
        data_dir: Path,
        pattern: str = "**/chapter*_dialogue.txt",
    ) -> List[Dict[str, Any]]:
        """Re-normalize all cached raw outputs with latest aliases.json.

        No LLM calls. Reads raw cache → applies current aliases → returns results.
        Files without raw cache are skipped (need fresh extraction).
        """
        results = []
        skipped = 0
        dialogue_files = sorted(data_dir.glob(pattern))
        print(f"Re-normalizing {len(dialogue_files)} files (no LLM calls)...")

        for file_path in dialogue_files:
            content = file_path.read_text(encoding="utf-8")
            cached_raw = self.cache.get(content)
            if cached_raw is None:
                skipped += 1
                print(f"  SKIP (no raw cache): {file_path.name}")
                continue

            metadata = self._parse_file_metadata(file_path)
            normalized = self.extractor.normalize_output(cached_raw)

            results.append({
                "entities": normalized.entities,
                "relationships": normalized.relationships,
                "task_id": metadata["task_id"],
                "chapter": metadata["chapter"],
                "file_path": str(file_path),
            })
            print(f"  OK: {file_path.name} ({len(normalized.entities)} entities)")

        print(f"\nRe-normalized: {len(results)} files, skipped: {skipped}")
        return results

    def rebuild_tracking(
        self,
        data_dir: Path,
        pattern: str = "**/chapter*_dialogue.txt",
    ) -> Dict[str, Any]:
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

            content = file_path.read_text(encoding="utf-8")
            content_hash = self._hash_content(content)

            if file_key in self.tracking:
                if self.tracking[file_key].content_hash == content_hash:
                    stats["already_tracked"] += 1
                    continue

            cached = self.cache.get(content)
            if cached is None:
                stats["cache_misses"] += 1
                continue

            stats["cache_hits"] += 1
            metadata = self._parse_file_metadata(file_path)

            self.tracking[file_key] = KGFileTrackingInfo(
                file_path=file_key,
                content_hash=content_hash,
                last_processed=datetime.now().isoformat(),
                entity_count=len(cached.entities),
                relationship_count=len(cached.relationships),
                task_id=metadata["task_id"],
                chapter=metadata["chapter"],
            )

            print(f"  Restored: {file_path.name} ({len(cached.entities)} entities)")

        self._save_tracking()
        return stats


def write_kg_to_graph(
    extraction_results: List[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Write extracted KG to the Neo4j graph.

    v2: Supports all 6 entity types. Maps Place→Place, Faction→Faction labels.
    Writes attributes, description, confidence, text_evidence.
    """
    from .graph_builder import GraphBuilder

    stats = {
        "total_entities": 0,
        "total_relationships": 0,
        "entities_written": 0,
        "relationships_written": 0,
        "files_processed": 0,
    }

    if dry_run:
        for result in extraction_results:
            stats["total_entities"] += len(result["entities"])
            stats["total_relationships"] += len(result["relationships"])
            stats["files_processed"] += 1
        return stats

    with GraphBuilder() as builder:
        builder.setup_schema()

        for result in extraction_results:
            entities = result["entities"]
            relationships = result["relationships"]
            chapter = result["chapter"]
            task_id = result["task_id"]

            # Write entities using the generic method
            for entity in entities:
                if isinstance(entity, ExtractedEntity):
                    e = entity
                else:
                    e = ExtractedEntity(**entity) if isinstance(entity, dict) else entity

                builder.create_entity_from_extraction(
                    name=e.name,
                    entity_type=e.entity_type,
                    description=e.description,
                    attributes=e.attributes_dict,
                    text_evidence=e.text_evidence,
                    task_id=task_id,
                    chapter=chapter,
                )
                stats["entities_written"] += 1

            # Write relationships
            for rel in relationships:
                if isinstance(rel, ExtractedRelationship):
                    r = rel
                else:
                    r = ExtractedRelationship(**rel) if isinstance(rel, dict) else rel

                success = builder.create_relationship_from_extraction(
                    source=r.source,
                    target=r.target,
                    relation_type=r.relation_type,
                    description=r.description,
                    confidence=r.confidence,
                    text_evidence=r.text_evidence,
                    chapter=chapter,
                    task_id=task_id,
                )
                if success:
                    stats["relationships_written"] += 1

            stats["total_entities"] += len(entities)
            stats["total_relationships"] += len(relationships)
            stats["files_processed"] += 1

    return stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Incremental KG Extractor (v2)")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.ingestion.incremental_kg_extractor <folder>")
        print("  python -m src.ingestion.incremental_kg_extractor Data/staging/Archon/1601")
        print("  python -m src.ingestion.incremental_kg_extractor Data/staging/Archon/1601 --write")
        print("  python -m src.ingestion.incremental_kg_extractor --renormalize Data/staging/Archon/1601")
        print("  python -m src.ingestion.incremental_kg_extractor --renormalize Data/staging/ --write")
        print("  python -m src.ingestion.incremental_kg_extractor --cleanup")
        print("  python -m src.ingestion.incremental_kg_extractor --cleanup --dry-run")
        print("  python -m src.ingestion.incremental_kg_extractor --rebuild Data/staging/")
        print("  python -m src.ingestion.incremental_kg_extractor --status")
        print("  python -m src.ingestion.incremental_kg_extractor --stats <folder>")
        sys.exit(1)

    extractor = IncrementalKGExtractor()

    if "--renormalize" in sys.argv:
        data_dir = None
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                data_dir = Path(arg)
                break
        if data_dir is None:
            data_dir = Path("Data/staging/")

        print(f"\nRe-normalizing with latest aliases.json: {data_dir}")
        results = extractor.renormalize_all(data_dir)

        if not results:
            print("No files to re-normalize (no raw cache found).")
            sys.exit(0)

        total_entities = sum(len(r["entities"]) for r in results)
        total_rels = sum(len(r["relationships"]) for r in results)
        print(f"\nTotal: {total_entities} entities, {total_rels} relationships")

        # Generate human review report
        report_path = extractor.generate_review_report(data_dir, results)
        if report_path:
            print(f"\nReview report: {report_path}")

        if "--write" in sys.argv:
            print(f"\nWriting re-normalized KG to Neo4j...")
            write_stats = write_kg_to_graph(results)
            print(f"Entities written: {write_stats['entities_written']}")
            print(f"Relationships written: {write_stats['relationships_written']}")
        sys.exit(0)

    if "--rebuild" in sys.argv:
        data_dir = None
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                data_dir = Path(arg)
                break
        if data_dir is None:
            data_dir = Path("Data/staging/")

        print(f"\nRebuilding tracking from cache for: {data_dir}")
        rebuild_stats = extractor.rebuild_tracking(data_dir)
        print(f"\n{'='*60}")
        print(f"REBUILD COMPLETE")
        print(f"{'='*60}")
        print(f"Files scanned: {rebuild_stats['files_scanned']}")
        print(f"Already tracked: {rebuild_stats['already_tracked']}")
        print(f"Cache hits (restored): {rebuild_stats['cache_hits']}")
        print(f"Cache misses (need extraction): {rebuild_stats['cache_misses']}")
        sys.exit(0)

    if "--cleanup" in sys.argv:
        dry_run = "--dry-run" in sys.argv
        print(f"\nCleaning orphan cache files (dry_run={dry_run})...")
        cleanup_result = extractor.cleanup_orphan_cache(dry_run=dry_run)
        print(f"  Tracked files: {len(extractor.tracking)}")
        print(f"  Cache files kept: {cleanup_result['files_kept']}")
        print(f"  Orphans found: {cleanup_result['orphans_found']}")
        if dry_run:
            print(f"  (Dry run - no files deleted)")
            if cleanup_result['orphan_files']:
                print(f"\n  Would delete:")
                for f in cleanup_result['orphan_files']:
                    print(f"    {f}")
        else:
            print(f"  Orphans deleted: {cleanup_result['orphans_deleted']}")
        sys.exit(0)

    if "--status" in sys.argv:
        status = extractor.get_status()
        print(f"\nTracked files: {status['tracked_files']}")
        print(f"Total entities: {status['total_entities']}")
        print(f"Total relationships: {status['total_relationships']}")
        print(f"Cache stats: {status['cache_stats']}")
        print(f"\nFiles:")
        for f in status['files']:
            print(f"  {f['path']}: {f['entities']} entities, {f['relationships']} rels (chapter {f['chapter']})")
        sys.exit(0)

    folder = Path(sys.argv[1])
    write_to_graph = "--write" in sys.argv
    show_stats = "--stats" in sys.argv

    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    # Extract KG
    print(f"\nExtracting KG from: {folder}")
    results = extractor.extract_folder(folder)

    # Print summary
    total_entities = sum(len(r["entities"]) for r in results)
    total_rels = sum(len(r["relationships"]) for r in results)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {len(results)}")
    print(f"Total entities: {total_entities}")
    print(f"Total relationships: {total_rels}")

    # Count entity types
    entity_types: Dict[str, int] = {}
    for result in results:
        for entity in result["entities"]:
            t = entity.entity_type
            entity_types[t] = entity_types.get(t, 0) + 1

    print(f"\nEntity types:")
    for t, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # Count relationship types
    rel_types: Dict[str, int] = {}
    for result in results:
        for rel in result["relationships"]:
            t = rel.relation_type
            rel_types[t] = rel_types.get(t, 0) + 1

    print(f"\nRelationship types:")
    for t, count in sorted(rel_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # Print sample
    if show_stats:
        print(f"\nSample entities (first file):")
        for result in results[:1]:
            print(f"\n  File: {result['file_path']}")
            for entity in result["entities"][:10]:
                filled = entity.attributes_dict
                attrs_str = f" attrs={filled}" if filled else ""
                print(f"    [{entity.entity_type}] {entity.name}{attrs_str}")

        print(f"\nSample relationships (first file):")
        for result in results[:1]:
            for rel in result["relationships"][:10]:
                print(f"    {rel.source} --[{rel.relation_type}]--> {rel.target}")
                print(f"      desc={rel.description}, conf={rel.confidence}")

    # Generate human review report
    report_path = extractor.generate_review_report(folder, results)
    if report_path:
        print(f"\n--- Human Review ---")
        print(f"Review report: {report_path}")
    else:
        print(f"\nNo entities need human review (all matched by aliases.json)")

    # Write to graph if requested
    if write_to_graph:
        print(f"\n{'='*60}")
        print("Writing KG to Neo4j graph...")
        write_stats = write_kg_to_graph(results)
        print(f"Entities written: {write_stats['entities_written']}")
        print(f"Relationships written: {write_stats['relationships_written']}")
    else:
        print(f"\nTo write KG to graph, run with --write flag")
