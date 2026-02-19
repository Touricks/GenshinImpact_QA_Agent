"""Document loader for parsing dialogue files."""

import re
import logging
from pathlib import Path
from typing import Iterator, Optional

from ..common.models import DocumentMetadata, RawDocument

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load and parse dialogue files from Data/ directory."""

    # Header patterns
    TITLE_PATTERN = re.compile(r"^#\s*(.+?)\s*-\s*第(\d+)章[：:](.+)$")
    SERIES_PATTERN = re.compile(r"^#\s*(.+(?:序幕|第[一二三四五六七八九十\d]+幕|序奏|幕间).*)$")
    SOURCE_PATTERN = re.compile(r"^#\s*来源[：:]\s*(.+)$")
    SUMMARY_MARKER = re.compile(r"^##?\s*剧情简介")
    CHAPTER_FROM_FILENAME = re.compile(r"chapter(\d+)_dialogue\.txt")

    def __init__(self, data_dir: Path):
        """Initialize loader with data directory path."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

    # Character story header pattern: # 角色名 - 角色故事
    CHARACTER_STORY_TITLE = re.compile(r"^#\s*(.+?)\s*-\s*角色故事")

    def load_all(self) -> Iterator[RawDocument]:
        """Load all documents from data directory (recursive)."""
        # Recursively find all dialogue files
        dialogue_files = sorted(self.data_dir.rglob("chapter*_dialogue.txt"))

        # Group by parent directory (task folders)
        task_dirs = sorted(set(f.parent for f in dialogue_files))
        logger.info(f"Found {len(task_dirs)} task directories")

        for task_dir in task_dirs:
            yield from self._load_task(task_dir)

        # Also load character story files
        story_files = sorted(self.data_dir.rglob("story.txt"))
        if story_files:
            logger.info(f"Found {len(story_files)} character story files")
            for story_file in story_files:
                try:
                    doc = self._parse_story_file(story_file)
                    if doc:
                        yield doc
                except Exception as e:
                    logger.error(f"Failed to parse {story_file}: {e}")

    def _load_task(self, task_dir: Path) -> Iterator[RawDocument]:
        """Load all chapters from a task directory."""
        task_id = task_dir.name
        chapter_files = sorted(task_dir.glob("chapter*_dialogue.txt"))

        if not chapter_files:
            logger.warning(f"No chapter files found in {task_dir}")
            return

        for chapter_file in chapter_files:
            try:
                doc = self._parse_file(chapter_file, task_id)
                if doc:
                    yield doc
            except Exception as e:
                logger.error(f"Failed to parse {chapter_file}: {e}")

    def _parse_file(self, file_path: Path, task_id: str) -> Optional[RawDocument]:
        """Parse a single dialogue file."""
        # Read file with UTF-8 encoding
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to UTF-8 with BOM
            content = file_path.read_text(encoding="utf-8-sig")

        if not content.strip():
            logger.warning(f"Empty file: {file_path}")
            return None

        lines = content.split("\n")
        metadata = self._parse_header(lines, task_id, file_path)
        body_content = self._extract_body(lines)

        return RawDocument(metadata=metadata, content=body_content)

    def _parse_header(
        self, lines: list[str], task_id: str, file_path: Path
    ) -> DocumentMetadata:
        """Parse header lines to extract metadata."""
        task_name = "Unknown"
        chapter_number = 0
        chapter_title = "Unknown"
        series_name = None
        source_url = None
        summary = None

        # Try to extract chapter number from filename as fallback
        filename_match = None
        if file_path:
            filename_match = self.CHAPTER_FROM_FILENAME.search(file_path.name)
        if filename_match:
            chapter_number = int(filename_match.group(1))

        # Parse header lines (usually first 10 lines)
        header_lines = lines[:15]
        in_summary = False
        summary_lines = []

        for line in header_lines:
            line = line.strip()

            # Check for title pattern (task_name - 第N章：chapter_title)
            title_match = self.TITLE_PATTERN.match(line)
            if title_match:
                task_name = title_match.group(1).strip()
                chapter_number = int(title_match.group(2))
                chapter_title = title_match.group(3).strip()
                continue

            # Check for series pattern
            series_match = self.SERIES_PATTERN.match(line)
            if series_match and not title_match:
                series_name = series_match.group(1).strip()
                continue

            # Check for source URL
            source_match = self.SOURCE_PATTERN.match(line)
            if source_match:
                source_url = source_match.group(1).strip()
                continue

            # Check for summary section
            if self.SUMMARY_MARKER.match(line):
                in_summary = True
                continue

            # Collect summary content
            if in_summary:
                if line.startswith("---") or line.startswith("##"):
                    in_summary = False
                elif line:
                    summary_lines.append(line)

        if summary_lines:
            summary = "\n".join(summary_lines)

        return DocumentMetadata(
            task_id=task_id,
            task_name=task_name,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            series_name=series_name,
            summary=summary,
            source_url=source_url,
            file_path=file_path,
        )

    def _parse_story_file(self, file_path: Path) -> Optional[RawDocument]:
        """Parse a character story file (story.txt)."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8-sig")

        if not content.strip():
            logger.warning(f"Empty file: {file_path}")
            return None

        lines = content.split("\n")
        task_id = file_path.parent.name  # e.g., "10000116"

        # Parse character name from first line
        character_name = "Unknown"
        source_url = None
        for line in lines[:5]:
            title_match = self.CHARACTER_STORY_TITLE.match(line.strip())
            if title_match:
                character_name = title_match.group(1).strip()
            source_match = self.SOURCE_PATTERN.match(line.strip())
            if source_match:
                source_url = source_match.group(1).strip()

        metadata = DocumentMetadata(
            task_id=task_id,
            task_name=character_name,
            chapter_number=0,
            chapter_title="角色故事",
            series_name=None,
            summary=None,
            source_url=source_url,
            file_path=file_path,
        )

        body_content = self._extract_body(lines)
        return RawDocument(metadata=metadata, content=body_content)

    def _extract_body(self, lines: list[str]) -> str:
        """Extract body content after header section."""
        # Find the first scene marker or dialogue
        body_start = 0
        in_header = True

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Body starts after summary section ends
            if stripped.startswith("---") and in_header:
                body_start = i
                in_header = False
                break
            # Or after ## marker that's not 剧情简介
            if stripped.startswith("## ") and not self.SUMMARY_MARKER.match(stripped):
                body_start = i
                in_header = False
                break

        return "\n".join(lines[body_start:])


def count_files(data_dir: Path) -> dict:
    """Count files in data directory for statistics (recursive)."""
    data_path = Path(data_dir)
    chapter_files = list(data_path.rglob("chapter*_dialogue.txt"))
    story_files = list(data_path.rglob("story.txt"))
    task_dirs = set(f.parent for f in chapter_files)

    return {
        "tasks": len(task_dirs),
        "chapters": len(chapter_files),
        "stories": len(story_files),
    }
