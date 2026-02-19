"""
Relationship models for the knowledge graph.

v2 schema: 9 relationship types strictly aligned with design doc.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
from enum import Enum


class RelationType(str, Enum):
    """v2 relationship types for the knowledge graph (9 types)."""

    MEMBER_OF = "MEMBER_OF"            # Character/Faction → Faction
    LOCATED_AT = "LOCATED_AT"          # Any → Place
    CREATED_BY = "CREATED_BY"          # Item/Concept → Character/Faction
    LEADS_TO = "LEADS_TO"              # Any → Any (因果/使能/依赖, description 标注子类型)
    MOTIVATED_BY = "MOTIVATED_BY"      # Character/Faction → Concept/Event
    INVOLVED_IN = "INVOLVED_IN"        # Any 6 类 → Event
    TEMPORAL_BEFORE = "TEMPORAL_BEFORE" # Event → Event (严格时间顺序)
    OPPOSED_TO = "OPPOSED_TO"          # Character/Faction → Character/Faction
    ORIGINATES_FROM = "ORIGINATES_FROM" # Item/Concept → Place/Concept


# All valid relation type values for validation
VALID_RELATION_TYPES = {rt.value for rt in RelationType}


@dataclass
class Relationship:
    """A relationship between two entities."""

    source: str
    target: str
    rel_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    chapter: Optional[int] = None
    task_id: Optional[str] = None
    # v2 fields
    description: Optional[str] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None
    text_evidence: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j."""
        d = {
            "source": self.source,
            "target": self.target,
            "type": self.rel_type.value,
            "properties": self.properties,
            "chapter": self.chapter,
            "task_id": self.task_id,
        }
        if self.description:
            d["description"] = self.description
        if self.confidence:
            d["confidence"] = self.confidence
        if self.text_evidence:
            d["text_evidence"] = self.text_evidence
        return d
