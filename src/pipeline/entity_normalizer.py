"""
Entity Normalizer Module.
Dynamically loads alias data from regions/{ACTIVE_REGION}/aliases.json.
Supports all 6 entity types: Character, Place, Faction, Item, Concept, Event.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import difflib

logger = logging.getLogger(__name__)


class EntityNormalizer:
    """
    Normalizes entity names against aliases loaded from the active region config.
    """

    def __init__(self, aliases_path: Optional[Path] = None):
        """
        Initialize the normalizer by loading aliases from config.

        Args:
            aliases_path: Path to aliases.json. If None, auto-resolves from ACTIVE_REGION.
        """
        if aliases_path is None:
            aliases_path = self._resolve_aliases_path()

        # alias_text -> canonical_name
        self._alias_map: Dict[str, str] = {}
        # canonical_name -> entity_type
        self._type_map: Dict[str, str] = {}
        # List of all canonical names for fuzzy matching
        self._canonical_names: List[str] = []

        self._load_aliases(aliases_path)

    @staticmethod
    def _resolve_aliases_path() -> Path:
        """Resolve aliases.json path from active region config."""
        try:
            from src.common.config.settings import settings
            region = settings.ACTIVE_REGION
        except Exception:
            region = "nodkrai"
        return Path(__file__).resolve().parent.parent / "common" / "config" / "regions" / region / "aliases.json"

    def _load_aliases(self, path: Path) -> None:
        """Load alias mappings from JSON file."""
        if not path.exists():
            logger.warning(f"Aliases file not found: {path}")
            return

        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load aliases: {e}")
            return

        for key, entry in raw.items():
            canonical = entry.get("canonical_zh", key)
            entity_type = entry.get("type", "Character")
            aliases = entry.get("aliases", [])

            self._canonical_names.append(canonical)
            self._type_map[canonical] = entity_type

            # Map every alias to the canonical name
            for alias in aliases:
                self._alias_map[alias] = canonical
            # Also map the canonical name to itself
            self._alias_map[canonical] = canonical

        logger.info(
            f"Loaded {len(self._canonical_names)} entities with "
            f"{len(self._alias_map)} alias mappings"
        )

    def normalize(self, name: str, entity_type: Optional[str] = None) -> str:
        """
        Normalize an entity name to its canonical form.

        Args:
            name: The raw entity name extracted from text.
            entity_type: Optional entity type hint.

        Returns:
            The normalized canonical name, or the original name if no match.
        """
        if not name:
            return name

        # 1. Exact alias lookup
        if name in self._alias_map:
            return self._alias_map[name]

        # 2. Case-insensitive alias lookup
        name_lower = name.lower()
        for alias, canonical in self._alias_map.items():
            if alias.lower() == name_lower:
                return canonical

        # 3. Fuzzy match — type-aware to prevent cross-type mismatches
        #    e.g. Event "薇尔米娜的背叛" must NOT match Character "薇尔米娜"
        candidates = self._canonical_names
        if entity_type:
            same_type = [n for n in self._canonical_names
                         if self._type_map.get(n) == entity_type]
            if same_type:
                candidates = same_type

        matches = difflib.get_close_matches(
            name, candidates, n=1, cutoff=0.7
        )
        if matches:
            logger.debug(f"Fuzzy normalized '{name}' -> '{matches[0]}'")
            return matches[0]

        return name

    def get_entity_type(self, canonical_name: str) -> Optional[str]:
        """Get the entity type for a canonical name."""
        return self._type_map.get(canonical_name)

    def is_known_entity(self, name: str) -> bool:
        """Check if a name matches any known entity."""
        return name in self._alias_map
