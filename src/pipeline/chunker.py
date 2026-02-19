"""Scene-based chunker for dialogue content."""

import re
import logging
from typing import List, Tuple

from ..common.models import RawDocument
from ..common.config import Settings

logger = logging.getLogger(__name__)


class SceneChunker:
    """Scene-based semantic chunker for dialogue content."""

    # Scene delimiters
    SCENE_HEADER = re.compile(r"^##\s+(.+)$")
    HORIZONTAL_RULE = re.compile(r"^---\s*$")
    CHOICE_MARKER = re.compile(r"^##\s*选项")

    # Dialogue pattern
    DIALOGUE_PATTERN = re.compile(r"^([^：\n]+)：(.+)$")

    def __init__(
        self,
        max_chunk_size: int = None,
        min_chunk_size: int = None,
        overlap: int = None,
    ):
        """Initialize chunker with size constraints."""
        settings = Settings()
        self.max_chunk_size = max_chunk_size or settings.MAX_CHUNK_SIZE
        self.min_chunk_size = min_chunk_size or settings.MIN_CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP

    def chunk_document(self, document: RawDocument) -> List[Tuple[str, str]]:
        """
        Split document into semantic chunks.

        Returns:
            List of (scene_title, chunk_text) tuples
        """
        content = document.content
        if not content.strip():
            return []

        # Step 1: Split by scenes
        scenes = self._split_by_scenes(content)
        logger.debug(f"Split into {len(scenes)} scenes")

        # Step 2: Process each scene
        chunks = []
        for scene_title, scene_content in scenes:
            scene_chunks = self._process_scene(scene_title, scene_content)
            chunks.extend(scene_chunks)

        logger.debug(f"Created {len(chunks)} chunks from document")
        return chunks

    def _split_by_scenes(self, content: str) -> List[Tuple[str, str]]:
        """Split content by scene markers (## headers and ---)."""
        scenes = []
        lines = content.split("\n")

        current_title = None
        current_lines = []

        for line in lines:
            # Check for scene header
            header_match = self.SCENE_HEADER.match(line.strip())
            if header_match:
                # Save previous scene if exists
                if current_lines:
                    scene_text = "\n".join(current_lines).strip()
                    if scene_text:
                        scenes.append((current_title, scene_text))
                # Start new scene
                current_title = header_match.group(1).strip()
                current_lines = []
                continue

            # Check for horizontal rule (also a scene break)
            if self.HORIZONTAL_RULE.match(line.strip()):
                if current_lines:
                    scene_text = "\n".join(current_lines).strip()
                    if scene_text:
                        scenes.append((current_title, scene_text))
                current_title = None
                current_lines = []
                continue

            current_lines.append(line)

        # Don't forget the last scene
        if current_lines:
            scene_text = "\n".join(current_lines).strip()
            if scene_text:
                scenes.append((current_title, scene_text))

        return scenes

    def _process_scene(self, title: str, content: str) -> List[Tuple[str, str]]:
        """Process a single scene, splitting if too large."""
        content = content.strip()
        if not content:
            return []

        # If scene is within size limits, return as-is
        if len(content) <= self.max_chunk_size:
            if len(content) >= self.min_chunk_size:
                return [(title, content)]
            # Scene too small, will be merged later
            return [(title, content)]

        # Scene too large, split by dialogue turns
        return self._split_large_scene(title, content)

    def _split_large_scene(self, title: str, content: str) -> List[Tuple[str, str]]:
        """Split a large scene into smaller chunks at dialogue boundaries."""
        chunks = []
        lines = content.split("\n")

        current_chunk_lines = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # Check if adding this line would exceed max size
            if current_size + line_size > self.max_chunk_size and current_chunk_lines:
                # Check if this is a dialogue boundary
                is_dialogue_start = self.DIALOGUE_PATTERN.match(line.strip())

                if is_dialogue_start or current_size > self.max_chunk_size * 0.8:
                    # Save current chunk
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        chunks.append((title, chunk_text))

                    # Start new chunk with overlap
                    overlap_lines = self._get_overlap_lines(current_chunk_lines)
                    current_chunk_lines = overlap_lines
                    current_size = sum(len(l) + 1 for l in current_chunk_lines)

            current_chunk_lines.append(line)
            current_size += line_size

        # Don't forget the last chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append((title, chunk_text))

        return chunks

    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get overlap lines from the end of a chunk."""
        if not lines:
            return []

        # Calculate how many lines to include for overlap
        total_size = 0
        overlap_lines = []

        for line in reversed(lines):
            line_size = len(line) + 1
            if total_size + line_size > self.overlap:
                break
            overlap_lines.insert(0, line)
            total_size += line_size

        return overlap_lines

    def merge_small_chunks(
        self, chunks: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Merge consecutive small chunks that share the same scene title."""
        if not chunks:
            return []

        merged = []
        current_title = chunks[0][0]
        current_text = chunks[0][1]

        for title, text in chunks[1:]:
            # If same title and combined size is acceptable, merge
            combined_size = len(current_text) + len(text) + 1
            if title == current_title and combined_size <= self.max_chunk_size:
                current_text = current_text + "\n\n" + text
            else:
                # Save current and start new
                if current_text.strip():
                    merged.append((current_title, current_text.strip()))
                current_title = title
                current_text = text

        # Don't forget the last one
        if current_text.strip():
            merged.append((current_title, current_text.strip()))

        return merged
