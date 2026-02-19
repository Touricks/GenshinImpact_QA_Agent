"""
LLM-based Major Event Extractor.

Extracts significant plot events (sacrifices, transformations, revelations, etc.)
from dialogue text to build Event nodes in the knowledge graph.

This addresses the "abstract query vs concrete narrative" semantic gap problem:
- User asks: "How did the girl return to the world?"
- DB contains: "献出身体", "权能转交", "化作月光"
- Solution: Event nodes bridge abstract concepts to concrete story moments
"""

import os
from pathlib import Path
from typing import List, Optional, Literal, Set
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
# This file is at: src/ingestion/event_extractor.py
# Project root is at: ../../
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)


# =============================================================================
# Event Type Taxonomy
# =============================================================================

class EventType(str, Enum):
    """Classification of major story events."""

    SACRIFICE = "sacrifice"  # 牺牲 - Character pays a significant price
    TRANSFORMATION = "transformation"  # 转变 - Character state/form changes
    ACQUISITION = "acquisition"  # 获得 - Character gains power/item
    LOSS = "loss"  # 失去 - Character loses something important
    ENCOUNTER = "encounter"  # 相遇 - Important character meeting
    CONFLICT = "conflict"  # 冲突 - Battle/confrontation
    REVELATION = "revelation"  # 揭示 - Truth/secret revealed
    MILESTONE = "milestone"  # 里程碑 - Important turning point


# Trigger keywords for each event type (used in prompt guidance)
EVENT_TRIGGER_KEYWORDS = {
    EventType.SACRIFICE: ["献出", "放弃", "牺牲", "舍弃", "付出"],
    EventType.TRANSFORMATION: ["变成", "化作", "转变", "蜕变", "觉醒"],
    EventType.ACQUISITION: ["获得", "得到", "继承", "接受", "掌握"],
    EventType.LOSS: ["失去", "丢失", "消失", "离开", "死亡"],
    EventType.ENCOUNTER: ["遇见", "相遇", "邂逅", "初见", "重逢"],
    EventType.CONFLICT: ["战斗", "对抗", "冲突", "决战", "击败"],
    EventType.REVELATION: ["揭示", "发现", "真相", "揭露", "原来"],
    EventType.MILESTONE: ["进入", "离开", "开始", "结束", "抵达"],
}


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class CharacterRole(BaseModel):
    """A character's role in an event."""

    name: str = Field(description="角色名称")
    role: Literal["subject", "object", "witness"] = Field(
        description="角色在事件中的角色：subject(主动方), object(被动方), witness(见证者)"
    )


class ExtractedEvent(BaseModel):
    """A single major event extracted from text."""

    name: str = Field(description="事件简称（5-15字）")
    event_type: Literal[
        "sacrifice",
        "transformation",
        "acquisition",
        "loss",
        "encounter",
        "conflict",
        "revelation",
        "milestone",
    ] = Field(description="事件类型")
    characters: List[CharacterRole] = Field(
        description="涉及的角色及其角色"
    )
    summary: str = Field(
        description="事件摘要（一句话，30-50字）"
    )
    outcome: Optional[str] = Field(
        default=None,
        description="事件结果/影响"
    )
    evidence: str = Field(
        description="原文引用（支持该事件的对话片段，50字以内）"
    )


class EventExtractionOutput(BaseModel):
    """Complete event extraction output for a dialogue chunk."""

    events: List[ExtractedEvent] = Field(
        default_factory=list,
        description="提取的重大事件列表"
    )

    def get_primary_characters(self) -> Set[str]:
        """Get all primary characters (subjects) from events."""
        chars = set()
        for event in self.events:
            for char in event.characters:
                if char.role == "subject":
                    chars.add(char.name)
        return chars

    def filter_by_type(self, event_type: EventType) -> List[ExtractedEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type.value]


# =============================================================================
# Extraction Prompt
# =============================================================================

EVENT_EXTRACTION_PROMPT = """你是一个原神（Genshin Impact）剧情分析专家。请从以下对话中提取**重大事件**（转折点、牺牲、获得、失去等）。

## 对话内容
{dialogue}

## 角色列表（参考）
{characters}

## 章节信息
章节: {chapter}
任务ID: {task_id}

## 事件类型说明
- **sacrifice** (牺牲): 角色付出重大代价。关键词：献出、放弃、牺牲
- **transformation** (转变): 角色状态/形态改变。关键词：变成、化作、蜕变、觉醒
- **acquisition** (获得): 角色获得力量/物品。关键词：获得、得到、继承
- **loss** (失去): 角色失去某物/某人。关键词：失去、丢失、死亡、离开
- **encounter** (相遇): 重要人物相遇。关键词：遇见、相遇、重逢
- **conflict** (冲突): 战斗/对抗。关键词：战斗、对抗、击败
- **revelation** (揭示): 真相/秘密揭露。关键词：揭示、发现、原来
- **milestone** (里程碑): 重要转折点。关键词：进入、离开、开始、结束

## 提取规则
1. **只提取重大事件**，忽略日常对话和闲聊
2. 每个事件必须至少涉及一个角色
3. evidence 必须是原文引用
4. 事件名称要简洁准确（5-15字）
5. 如果对话中没有重大事件，返回空列表

## 角色定义
- **subject**: 事件的主动发起者/主体
- **object**: 事件的承受者/客体
- **witness**: 事件的见证者

## 输出格式
请直接输出包含events数组的JSON对象。如果没有重大事件，返回空events数组。
"""


# =============================================================================
# LLM Event Extractor
# =============================================================================

class LLMEventExtractor:
    """
    Extract major story events using LLM.

    This class produces structured Event data that can be inserted
    into the Neo4j knowledge graph as Event nodes with EXPERIENCES edges.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the LLM event extractor.

        Args:
            model: Model name (defaults to DATA_MODEL from settings)
            api_key: API key (defaults to GEMINI_API_KEY env var)
        """
        from llama_index.llms.google_genai import GoogleGenAI
        from .entity_normalizer import EntityNormalizer
        from ..common.config import settings

        self.model = model or settings.DATA_MODEL
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")

        # Use native Google GenAI LLM for schema-enforced structured output
        self.llm = GoogleGenAI(
            model=self.model,
            api_key=api_key,
        )
        self.structured_llm = self.llm.as_structured_llm(EventExtractionOutput)
        self.normalizer = EntityNormalizer()

    def _build_prompt(
        self,
        dialogue: str,
        characters: List[str],
        chapter: int,
        task_id: str,
    ) -> str:
        """Build the extraction prompt with context."""
        return EVENT_EXTRACTION_PROMPT.format(
            dialogue=dialogue,
            characters=", ".join(characters) if characters else "未知",
            chapter=chapter,
            task_id=task_id,
        )

    def extract(
        self,
        dialogue: str,
        characters: Optional[List[str]] = None,
        chapter: int = 0,
        task_id: str = "",
    ) -> EventExtractionOutput:
        """
        Extract major events from dialogue text.

        Args:
            dialogue: Dialogue text to analyze
            characters: Optional list of known characters in the dialogue
            chapter: Chapter number for context
            task_id: Task ID for context

        Returns:
            EventExtractionOutput containing extracted events
        """
        prompt = self._build_prompt(
            dialogue=dialogue,
            characters=characters or [],
            chapter=chapter,
            task_id=task_id,
        )
        response = self.structured_llm.complete(prompt)
        output = response.raw

        # Normalize character names in events
        for event in output.events:
            for char_role in event.characters:
                normalized = self.normalizer.normalize(char_role.name)
                if normalized != char_role.name:
                    char_role.name = normalized

        return output

    def extract_from_chunk(
        self,
        chunk_text: str,
        chunk_metadata: dict,
    ) -> EventExtractionOutput:
        """
        Extract events from a chunk with metadata.

        Args:
            chunk_text: The chunk text content
            chunk_metadata: Metadata dict with 'characters', 'chapter', 'task_id'

        Returns:
            EventExtractionOutput
        """
        return self.extract(
            dialogue=chunk_text,
            characters=chunk_metadata.get("characters", []),
            chapter=chunk_metadata.get("chapter", 0),
            task_id=chunk_metadata.get("task_id", ""),
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_events_from_text(
    text: str,
    chapter: int = 0,
    task_id: str = "",
) -> EventExtractionOutput:
    """
    Convenience function to extract events from text.

    Args:
        text: Dialogue text
        chapter: Chapter number
        task_id: Task ID

    Returns:
        EventExtractionOutput
    """
    extractor = LLMEventExtractor()
    return extractor.extract(text, chapter=chapter, task_id=task_id)


def extract_events_from_file(
    file_path: Path,
    chapter: int = 0,
    task_id: str = "",
) -> EventExtractionOutput:
    """
    Extract events from a dialogue file.

    Args:
        file_path: Path to dialogue file
        chapter: Chapter number
        task_id: Task ID

    Returns:
        EventExtractionOutput
    """
    text = file_path.read_text(encoding="utf-8")
    return extract_events_from_text(text, chapter=chapter, task_id=task_id)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    # Test with sample dialogue from chapter 1608
    test_dialogue = """
少女：我想…可以的话，请…记得我。

旅行者：当然。我会记得你。

少女：谢谢…那么…永别了。

（少女将身体化作月光，洒向挪德卡莱的大地）

派蒙：她…她消失了…

旅行者：不，她没有消失。她只是…以另一种方式存在了。
"""

    print("Testing LLM Event Extractor...")
    print("=" * 60)
    print("Input dialogue:")
    print(test_dialogue)
    print("=" * 60)

    try:
        extractor = LLMEventExtractor()
        result = extractor.extract(
            dialogue=test_dialogue,
            characters=["少女", "旅行者", "派蒙"],
            chapter=1608,
            task_id="1608",
        )

        print("\nExtracted Events:")
        for event in result.events:
            print(f"\n  Event: {event.name}")
            print(f"  Type: {event.event_type}")
            print(f"  Characters:")
            for char in event.characters:
                print(f"    - {char.name} ({char.role})")
            print(f"  Summary: {event.summary}")
            if event.outcome:
                print(f"  Outcome: {event.outcome}")
            print(f"  Evidence: {event.evidence}")

        print("\n" + "=" * 60)
        print("JSON Output:")
        print(result.model_dump_json(indent=2))

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
