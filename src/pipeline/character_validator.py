"""
Character name validation for data quality control.

Filters out invalid character names during extraction:
- System text (conditions, UI elements)
- Combined speakers (multiple names with &)
- Narrative text (sentences)
- Generic references (众人, 呼救的人)
- Objects and non-character entities
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Tuple


class InvalidReason(Enum):
    """Reasons why a character name is invalid."""

    SYSTEM_TEXT = "system_text"  # [条件, 开始条件
    COMBINED_SPEAKER = "combined_speaker"  # 派蒙&卡齐娜
    NARRATIVE_TEXT = "narrative_text"  # Contains sentence punctuation
    UI_ELEMENT = "ui_element"  # 选项1, 玩家选项
    GENERIC_REFERENCE = "generic_reference"  # 众人, 呼救的人
    OBJECT_OR_THING = "object_or_thing"  # 嘟嘟通讯仪, 比赛公告板
    TOO_LONG = "too_long"  # Name > 15 characters
    EMPTY = "empty"  # Empty or whitespace only
    BLACKLISTED = "blacklisted"  # In explicit blacklist


@dataclass
class ValidationResult:
    """Result of character name validation."""

    is_valid: bool
    name: str
    reason: Optional[InvalidReason] = None
    normalized_name: Optional[str] = None


class CharacterValidator:
    """
    Validates and filters character names extracted from dialogue.

    This class implements multiple validation rules to ensure only
    valid character names are stored in the knowledge graph.
    """

    # System text patterns (starts with brackets or system keywords)
    SYSTEM_TEXT_PATTERNS = [
        re.compile(r"^\["),  # [条件
        re.compile(r"^【"),  # 【系统】
        re.compile(r"^（"),  # （旁白）
        re.compile(r"^\("),  # (voice) - ASCII parenthesis
        re.compile(r"^开始条件"),
        re.compile(r"^结束条件"),
        re.compile(r"^查看$"),
        re.compile(r"^调查$"),
        re.compile(r"^触发$"),
        re.compile(r"^任务"),
    ]

    # Combined speaker pattern (multiple names with &)
    COMBINED_SPEAKER_PATTERN = re.compile(r"&")

    # Narrative text patterns (contains sentence punctuation)
    NARRATIVE_PATTERNS = [
        re.compile(r"。"),  # Period
        re.compile(r"，.*，"),  # Multiple commas
        re.compile(r"、.*、"),  # Multiple enumeration marks
        re.compile(r"！.*。"),  # Exclamation then period
        re.compile(r"？.*。"),  # Question then period
        re.compile(r"……"),  # Ellipsis
        re.compile(r".{20,}"),  # Very long (>20 chars) likely narrative
        re.compile(r"^你[^，。]{5,}"),  # Starts with 你 followed by narrative
        re.compile(r"回想起.+的话$"),  # "回想起XX的话" pattern
    ]

    # UI element patterns
    UI_PATTERNS = [
        re.compile(r"^选项\d*$"),  # 选项, 选项1
        re.compile(r"^玩家选项$"),
        re.compile(r"^选择\d*$"),
        re.compile(r"^分支\d*$"),
        re.compile(r"^对话选项$"),
    ]

    # Generic reference patterns (non-specific speakers)
    GENERIC_PATTERNS = [
        re.compile(r"^众人"),  # 众人, 众人齐声
        re.compile(r"的人$"),  # 呼救的人, 受伤的人
        re.compile(r"的声音$"),  # XX的声音
        re.compile(r"^嘈杂"),  # 嘈杂的怒吼
        re.compile(r"^远处"),  # 远处的XX
        re.compile(r"^某"),  # 某人, 某个, 某处的声音
        re.compile(r"^有人$"),
        re.compile(r"^旁人$"),
        re.compile(r"^路人"),
        re.compile(r"^神秘"),  # 神秘的声音
        re.compile(r"^来自.+的"),  # 来自过去的声音, 来自深处的
    ]

    # Object/thing patterns (not characters)
    OBJECT_PATTERNS = [
        re.compile(r"通讯仪$"),  # 嘟嘟通讯仪
        re.compile(r"公告板$"),  # 比赛公告板
        re.compile(r"告示$"),
        re.compile(r"信件$"),
        re.compile(r"留言$"),
        re.compile(r"^长翎"),  # 长翎鹮 (bird species name)
        re.compile(r"回忆）$"),  # XX（回忆）treated as separate
        re.compile(r"机仆$"),  # 门禁机仆, 通行机仆
        re.compile(r"情报$"),  # 战场情报
    ]

    # Explicit blacklist
    BLACKLIST: Set[str] = {
        "？？？",
        "???",
        "---",
        "选项",
        "黑雾诅咒",
        "小机器人",
        "受伤的绒翼龙",
        "「木偶」",
        "Saurian companion",
        "奇怪的鸟",
        "某个声音",
        "系统",
        "旁白",
        "画外音",
        "致敬",  # Game mechanic
        "深渊低语",  # Narrative description
    }

    # Characters with parenthetical notes to normalize
    # e.g., "丽莎（回忆）" -> "丽莎"
    PARENTHETICAL_PATTERN = re.compile(r"^(.+?)（[^）]+）$")

    def __init__(self, additional_blacklist: Optional[Set[str]] = None):
        """
        Initialize the validator.

        Args:
            additional_blacklist: Extra names to blacklist
        """
        self.blacklist = self.BLACKLIST.copy()
        if additional_blacklist:
            self.blacklist.update(additional_blacklist)

    def validate(self, name: str) -> ValidationResult:
        """
        Validate a character name.

        Args:
            name: Character name to validate

        Returns:
            ValidationResult with validity status and reason if invalid
        """
        # Strip whitespace
        name = name.strip()

        # Check empty
        if not name:
            return ValidationResult(False, name, InvalidReason.EMPTY)

        # Check blacklist
        if name in self.blacklist:
            return ValidationResult(False, name, InvalidReason.BLACKLISTED)

        # Check system text
        for pattern in self.SYSTEM_TEXT_PATTERNS:
            if pattern.search(name):
                return ValidationResult(False, name, InvalidReason.SYSTEM_TEXT)

        # Check combined speaker
        if self.COMBINED_SPEAKER_PATTERN.search(name):
            return ValidationResult(False, name, InvalidReason.COMBINED_SPEAKER)

        # Check narrative text
        for pattern in self.NARRATIVE_PATTERNS:
            if pattern.search(name):
                return ValidationResult(False, name, InvalidReason.NARRATIVE_TEXT)

        # Check UI elements
        for pattern in self.UI_PATTERNS:
            if pattern.match(name):
                return ValidationResult(False, name, InvalidReason.UI_ELEMENT)

        # Check generic references
        for pattern in self.GENERIC_PATTERNS:
            if pattern.search(name):
                return ValidationResult(False, name, InvalidReason.GENERIC_REFERENCE)

        # Check objects
        for pattern in self.OBJECT_PATTERNS:
            if pattern.search(name):
                return ValidationResult(False, name, InvalidReason.OBJECT_OR_THING)

        # Check length (after all pattern checks)
        if len(name) > 15:
            return ValidationResult(False, name, InvalidReason.TOO_LONG)

        # Normalize parenthetical notes
        normalized = self._normalize_name(name)

        return ValidationResult(True, name, normalized_name=normalized)

    def _normalize_name(self, name: str) -> str:
        """
        Normalize a character name.

        Handles:
        - Parenthetical notes: 丽莎（回忆） -> 丽莎
        - Whitespace

        Args:
            name: Character name

        Returns:
            Normalized name
        """
        # Remove parenthetical notes
        match = self.PARENTHETICAL_PATTERN.match(name)
        if match:
            return match.group(1).strip()

        return name.strip()

    def filter_names(self, names: List[str]) -> Tuple[List[str], List[Tuple[str, InvalidReason]]]:
        """
        Filter a list of character names.

        Args:
            names: List of character names

        Returns:
            Tuple of (valid_names, invalid_names_with_reasons)
        """
        valid = []
        invalid = []

        for name in names:
            result = self.validate(name)
            if result.is_valid:
                # Use normalized name if available
                valid.append(result.normalized_name or result.name)
            else:
                invalid.append((result.name, result.reason))

        return valid, invalid

    def split_combined_speaker(self, name: str) -> List[str]:
        """
        Split a combined speaker into individual names.

        Args:
            name: Potentially combined speaker name (e.g., "派蒙&卡齐娜")

        Returns:
            List of individual names
        """
        if "&" not in name:
            return [name]

        parts = name.split("&")
        return [p.strip() for p in parts if p.strip()]

    def extract_valid_from_combined(self, name: str) -> List[str]:
        """
        Extract valid character names from a potentially combined speaker.

        First splits by &, then validates each part.

        Args:
            name: Character name (possibly combined)

        Returns:
            List of valid character names
        """
        parts = self.split_combined_speaker(name)
        valid_names = []

        for part in parts:
            result = self.validate(part)
            if result.is_valid:
                valid_names.append(result.normalized_name or result.name)

        return valid_names


# Convenience function
def validate_character_name(name: str) -> ValidationResult:
    """
    Validate a single character name.

    Args:
        name: Character name to validate

    Returns:
        ValidationResult
    """
    validator = CharacterValidator()
    return validator.validate(name)


def filter_character_names(names: List[str]) -> List[str]:
    """
    Filter a list of character names, returning only valid ones.

    Args:
        names: List of character names

    Returns:
        List of valid character names
    """
    validator = CharacterValidator()
    valid, _ = validator.filter_names(names)
    return valid
