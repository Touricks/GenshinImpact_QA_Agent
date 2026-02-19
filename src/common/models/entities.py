"""
Entity models for the knowledge graph.

Defines dataclasses for Character, Faction, Place, Item, Concept, and Event nodes.
v2 schema: 6 entity types with structured attribute templates.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Character:
    """Character entity for the knowledge graph."""

    name: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    source_refs: List[str] = field(default_factory=list)
    # Legacy fields (kept for backward compat)
    title: Optional[str] = None
    region: Optional[str] = None
    tribe: Optional[str] = None
    first_appearance_task: Optional[str] = None
    first_appearance_chapter: Optional[int] = None
    # v2 attribute template
    affiliation: Optional[str] = None
    role: Optional[str] = None
    goal: Optional[str] = None
    fate: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j."""
        d = {
            "name": self.name,
            "aliases": self.aliases,
            "description": self.description,
            "title": self.title,
            "region": self.region,
            "tribe": self.tribe,
            "first_appearance_task": self.first_appearance_task,
            "first_appearance_chapter": self.first_appearance_chapter,
            "affiliation": self.affiliation,
            "role": self.role,
            "goal": self.goal,
            "fate": self.fate,
        }
        if self.source_refs:
            d["source_refs"] = self.source_refs
        return d


@dataclass
class Faction:
    """Faction entity (tribe, guild, nation, faction). v2 name for Organization."""

    name: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    source_refs: List[str] = field(default_factory=list)
    # Legacy
    org_type: Optional[str] = None
    region: Optional[str] = None
    # v2 attribute template
    leader: Optional[str] = None
    base: Optional[str] = None
    purpose: Optional[str] = None
    status: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j."""
        d = {
            "name": self.name,
            "type": self.org_type,
            "region": self.region,
            "description": self.description,
            "leader": self.leader,
            "base": self.base,
            "purpose": self.purpose,
            "status": self.status,
        }
        if self.aliases:
            d["aliases"] = self.aliases
        if self.source_refs:
            d["source_refs"] = self.source_refs
        return d


# Backward compat alias
Organization = Faction


@dataclass
class Place:
    """Place entity. v2 name for Location."""

    name: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    source_refs: List[str] = field(default_factory=list)
    # Legacy
    location_type: Optional[str] = None
    region: Optional[str] = None
    # v2 attribute template
    political_status: Optional[str] = None
    significance: Optional[str] = None
    inhabitants: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j."""
        d = {
            "name": self.name,
            "type": self.location_type,
            "region": self.region,
            "description": self.description,
            "political_status": self.political_status,
            "significance": self.significance,
            "inhabitants": self.inhabitants,
        }
        if self.aliases:
            d["aliases"] = self.aliases
        if self.source_refs:
            d["source_refs"] = self.source_refs
        return d


# Backward compat alias
Location = Place


@dataclass
class Item:
    """Item entity (artifact, weapon, tool)."""

    name: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    source_refs: List[str] = field(default_factory=list)
    # v2 attribute template
    creator: Optional[str] = None
    function: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j."""
        d = {
            "name": self.name,
            "description": self.description,
            "creator": self.creator,
            "function": self.function,
            "location": self.location,
            "status": self.status,
        }
        if self.aliases:
            d["aliases"] = self.aliases
        if self.source_refs:
            d["source_refs"] = self.source_refs
        return d


@dataclass
class Concept:
    """Concept entity (lore system, power, phenomenon)."""

    name: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    source_refs: List[str] = field(default_factory=list)
    # v2 attribute template
    origin: Optional[str] = None
    nature: Optional[str] = None
    related_system: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j."""
        d = {
            "name": self.name,
            "description": self.description,
            "origin": self.origin,
            "nature": self.nature,
            "related_system": self.related_system,
        }
        if self.aliases:
            d["aliases"] = self.aliases
        if self.source_refs:
            d["source_refs"] = self.source_refs
        return d


@dataclass
class Event:
    """Event entity (quest, battle, ceremony)."""

    name: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    source_refs: List[str] = field(default_factory=list)
    # Legacy
    event_type: Optional[str] = None
    chapter_range: List[int] = field(default_factory=list)
    # v2 attribute template
    agent: Optional[str] = None
    target: Optional[str] = None
    cause: Optional[str] = None
    effect: Optional[str] = None
    legacy: Optional[str] = None
    time_order: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j."""
        d = {
            "name": self.name,
            "type": self.event_type,
            "chapter_range": self.chapter_range,
            "description": self.description,
            "agent": self.agent,
            "target": self.target,
            "cause": self.cause,
            "effect": self.effect,
            "legacy": self.legacy,
            "time_order": self.time_order,
        }
        if self.aliases:
            d["aliases"] = self.aliases
        if self.source_refs:
            d["source_refs"] = self.source_refs
        return d


# =============================================================================
# v2 attribute template definitions (for prompt generation & validation)
# =============================================================================

ENTITY_ATTRIBUTE_TEMPLATES: Dict[str, List[str]] = {
    "Character": ["affiliation", "role", "goal", "fate"],
    "Place": ["political_status", "significance", "inhabitants"],
    "Faction": ["leader", "base", "purpose", "status"],
    "Item": ["creator", "function", "location", "status"],
    "Concept": ["origin", "nature", "related_system"],
    "Event": ["agent", "target", "cause", "effect", "legacy", "time_order"],
}


# =============================================================================
# Seed Data: Load from external JSON config files with inline fallback
# =============================================================================

def _get_region_config_dir() -> Path:
    """Get config directory for the active region."""
    try:
        from src.common.config.settings import settings
        return Path(__file__).resolve().parent.parent / "config" / "regions" / settings.ACTIVE_REGION
    except Exception:
        return Path(__file__).resolve().parent.parent / "config" / "regions" / "nodkrai"


def _load_seed_characters(path: Path) -> dict[str, "Character"]:
    """Load MAIN_CHARACTERS from seed_characters.json."""
    try:
        with open(path / "seed_characters.json", encoding="utf-8") as f:
            raw = json.load(f)
        result: dict[str, Character] = {}
        for key, val in raw.items():
            result[key] = Character(
                name=val["name"],
                aliases=val.get("aliases", []),
                title=val.get("title"),
                region=val.get("region"),
                tribe=val.get("tribe"),
                description=val.get("description"),
            )
        return result
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {
            "旅行者": Character(name="旅行者", aliases=["玩家", "Traveler"]),
            "派蒙": Character(name="派蒙", description="旅行者的向导"),
        }


def _load_seed_organizations(path: Path) -> dict[str, "Faction"]:
    """Load KNOWN_ORGANIZATIONS from seed_organizations.json."""
    try:
        with open(path / "seed_organizations.json", encoding="utf-8") as f:
            raw = json.load(f)
        result: dict[str, Faction] = {}
        for key, val in raw.items():
            result[key] = Faction(
                name=val["name"],
                org_type=val["org_type"],
                region=val.get("region"),
                description=val.get("description"),
            )
        return result
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {
            "愚人众": Faction(name="愚人众", org_type="faction", region="至冬", description="至冬的执行者组织"),
        }


MAIN_CHARACTERS = _load_seed_characters(_get_region_config_dir())
KNOWN_ORGANIZATIONS = _load_seed_organizations(_get_region_config_dir())

# System characters to filter out during extraction
SYSTEM_CHARACTERS = {
    "？？？",
    "选项",
    "---",
    "黑雾诅咒",
    "小机器人",
    "受伤的绒翼龙",
}
