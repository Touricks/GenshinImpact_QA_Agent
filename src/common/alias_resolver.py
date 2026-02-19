"""
Shared alias resolution for all tools (KG + Vector).

Loads alias→canonical mappings from the active region's aliases.json.
Provides a single resolve() function used by GraphSearcher and search_memory.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_aliases: Optional[Dict[str, str]] = None


def _get_aliases_path() -> Path:
    """Get path to active region's aliases.json."""
    from .config.settings import settings

    return (
        Path(__file__).parent / "config" / "regions" / settings.ACTIVE_REGION / "aliases.json"
    )


def load_aliases() -> Dict[str, str]:
    """Load all alias→canonical mappings from active region's aliases.json.

    Returns a flat dict: {"木偶": "桑多涅", "Sandrone": "桑多涅", ...}
    """
    global _aliases
    if _aliases is not None:
        return _aliases

    path = _get_aliases_path()
    if not path.exists():
        logger.warning(f"[Alias] aliases.json not found: {path}")
        _aliases = {}
        return _aliases

    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    for canonical, info in data.items():
        if not isinstance(info, dict):
            continue
        canonical_zh = info.get("canonical_zh", canonical)
        for alias in info.get("aliases", []):
            mapping[alias] = canonical_zh
        mapping[canonical_zh] = canonical_zh
    _aliases = mapping
    return _aliases


def resolve(name: str) -> str:
    """Resolve an alias to its canonical name.

    Returns the canonical name if found, otherwise the original name.
    """
    aliases = load_aliases()
    canonical = aliases.get(name)
    if canonical and canonical != name:
        logger.info(f"[Alias] '{name}' -> '{canonical}'")
        return canonical
    return name


def get_all_names(name: str) -> List[str]:
    """Get all known names for an entity (canonical + all aliases).

    Useful for Qdrant MatchAny filtering.
    """
    aliases = load_aliases()
    names = {name}

    canonical = aliases.get(name, name)
    names.add(canonical)
    for alias, canon in aliases.items():
        if canon == canonical:
            names.add(alias)

    return list(names)
