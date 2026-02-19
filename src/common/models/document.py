"""Document models for the ingestion pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DocumentMetadata:
    """Metadata extracted from document header."""

    task_id: str  # e.g., "1600"
    task_name: str  # e.g., "归途"
    chapter_number: int  # e.g., 0
    chapter_title: str  # e.g., "墟火"
    series_name: Optional[str] = None  # e.g., "空月之歌 序奏"
    summary: Optional[str] = None  # 剧情简介 content
    source_url: Optional[str] = None  # Original URL
    file_path: Optional[Path] = None  # Absolute path to source file

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "series_name": self.series_name,
            "summary": self.summary,
            "source_url": self.source_url,
            "file_path": str(self.file_path) if self.file_path else None,
        }


@dataclass
class RawDocument:
    """Raw document with metadata and content."""

    metadata: DocumentMetadata
    content: str  # Full text after header
    scenes: List[str] = field(default_factory=list)  # Scene-split content

    def __len__(self) -> int:
        """Return content length."""
        return len(self.content)
