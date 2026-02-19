"""
LLM-based Knowledge Graph Extractor.

Uses LLM (via OpenAI-compatible API) to extract entities and relationships
from dialogue text for building knowledge graphs.

v2: 6 entity types + 9 relationship types + structured attribute templates.
"""

import os
from pathlib import Path
from typing import List, Optional, Literal, Set, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)


# =============================================================================
# Pydantic Models for Structured Output (v2)
# =============================================================================

class EntityAttributes(BaseModel):
    """Structured attribute template — union of all entity type attributes.
    LLM fills relevant fields based on entity_type; others stay None.
    Gemini API requires fixed schema (no additionalProperties/Dict).
    """
    # Character
    affiliation: Optional[str] = Field(default=None, description="所属组织/势力")
    role: Optional[str] = Field(default=None, description="职务/头衔")
    goal: Optional[str] = Field(default=None, description="目标")
    fate: Optional[str] = Field(default=None, description="命运/结局")
    # Place
    political_status: Optional[str] = Field(default=None, description="政治状态")
    significance: Optional[str] = Field(default=None, description="重要性")
    inhabitants: Optional[str] = Field(default=None, description="居民")
    # Faction
    leader: Optional[str] = Field(default=None, description="领袖")
    base: Optional[str] = Field(default=None, description="据点")
    purpose: Optional[str] = Field(default=None, description="目的")
    status: Optional[str] = Field(default=None, description="现状")
    # Item
    creator: Optional[str] = Field(default=None, description="制造者")
    function: Optional[str] = Field(default=None, description="功能")
    location: Optional[str] = Field(default=None, description="所在地")
    # Concept
    origin: Optional[str] = Field(default=None, description="起源")
    nature: Optional[str] = Field(default=None, description="本质")
    related_system: Optional[str] = Field(default=None, description="关联体系")
    # Event
    agent: Optional[str] = Field(default=None, description="主动方")
    target: Optional[str] = Field(default=None, description="受影响方")
    cause: Optional[str] = Field(default=None, description="原因")
    effect: Optional[str] = Field(default=None, description="结果")
    legacy: Optional[str] = Field(default=None, description="遗留影响")
    time_order: Optional[str] = Field(default=None, description="时间线位置")

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Return only non-None attributes as a dict."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ExtractedEntity(BaseModel):
    """A single extracted entity from text."""
    name: str = Field(description="实体名称（中文）")
    entity_type: Literal["Character", "Place", "Faction", "Item", "Concept", "Event"] = Field(
        description="实体类型"
    )
    description: Optional[str] = Field(
        default=None,
        description="实体简述"
    )
    text_evidence: str = Field(
        description="支持该实体存在的原文引用"
    )
    attributes: EntityAttributes = Field(
        default_factory=EntityAttributes,
        description="类型特定的结构化属性（按实体类型填写相关字段）"
    )

    @property
    def attributes_dict(self) -> Dict[str, Optional[str]]:
        """Get attributes as a plain dict (non-None values only)."""
        return self.attributes.to_dict()


class ExtractedRelationship(BaseModel):
    """A single extracted relationship between entities."""
    source: str = Field(description="关系源实体名称")
    target: str = Field(description="关系目标实体名称")
    relation_type: Literal[
        "MEMBER_OF",
        "LOCATED_AT",
        "CREATED_BY",
        "LEADS_TO",
        "MOTIVATED_BY",
        "INVOLVED_IN",
        "TEMPORAL_BEFORE",
        "OPPOSED_TO",
        "ORIGINATES_FROM",
    ] = Field(description="关系类型（9类之一）")
    description: str = Field(
        description="关系描述。leads_to 需标注子类型（因果/使能/依赖）"
    )
    text_evidence: str = Field(
        description="支持该关系的原文引用"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="置信度"
    )


class KnowledgeGraphOutput(BaseModel):
    """Complete knowledge graph extraction output."""
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="所有提取的实体"
    )
    relationships: List[ExtractedRelationship] = Field(
        default_factory=list,
        description="实体间的关系"
    )

    def get_entity_names(self) -> Set[str]:
        """Get all entity names as a set."""
        return {e.name for e in self.entities}

    def get_characters(self) -> List[ExtractedEntity]:
        return [e for e in self.entities if e.entity_type == "Character"]

    def get_entities_by_type(self, entity_type: str) -> List[ExtractedEntity]:
        return [e for e in self.entities if e.entity_type == entity_type]


# =============================================================================
# Extraction Prompt (v2 — NK context)
# =============================================================================

EXTRACTION_PROMPT = """你是一个原神（Genshin Impact）剧情文本分析专家。请从以下对话文本中提取知识图谱（实体+关系）。

## 一、实体类型定义（6类）

### 1. Character（角色）
有名字的人物、神灵、龙等有意识的存在。
- 判断标准：对话参与者、被提及的人物、神话/历史人物
- 属性模板: affiliation（所属组织）, role（职务/头衔）, goal（目标）, fate（命运/结局）
- 旅行者别名：杜麦尼、玩家、Traveler
- 派蒙别名：应急食品、飞行的小精灵
- 反例："选项"、"？？？"、"小机器人" 不是角色

### 2. Place（地点）
有名字的地理位置、建筑、区域。
- 属性模板: political_status（政治状态）, significance（重要性）, inhabitants（居民）
- 例：挪德卡莱、那夏镇、希汐岛、帕哈岛、皮拉米达城、苦壑崖

### 3. Faction（组织/势力）
有组织结构的团体、部族、军团、商会。
- 属性模板: leader（领袖）, base（据点）, purpose（目的）, status（现状）
- 例：执灯人军团、霜月之子、伏尼契商会、愚人众、魇夜之莺

### 4. Item（物品）
有名字的重要物品、武器、工具、遗物。
- 属性模板: creator（制造者）, function（功能）, location（所在地）, status（现状）
- 例：月髓、银灯芯、螺旋剑、德肋庇革劳诺之箭、三宝磨

### 5. Concept（概念）
抽象的力量体系、信仰、现象、法则。
- 属性模板: origin（起源）, nature（本质）, related_system（关联体系）
- 例：月矩力、狂猎、虚假之天、三月女神、圣嗣之血

### 6. Event（事件）
有名称的具体历史事件、战役、灾难。
- 属性模板: agent（主动方）, target（受影响方）, cause（原因）, effect（结果）, legacy（遗留影响）, time_order（时间线位置描述）
- 例：葬火之战、苦壑崖事件、寒天之钉

## 二、关系类型定义（9类，仅使用以下类型）

### 1. MEMBER_OF
某实体隶属于某组织/势力。
- 适用: Character→Faction, Faction→Faction
- 例: 叶洛亚 → 执灯人军团

### 2. LOCATED_AT
实体位于某地点。
- 适用: 任意→Place
- 例: 银月之庭 → 希汐岛

### 3. CREATED_BY
物品/概念由某人或组织创造。
- 适用: Item/Concept→Character/Faction
- 例: 伊涅芙 → 爱诺

### 4. LEADS_TO（语义拓宽：因果/使能/依赖）
因果链、使能条件、依赖关系。description 字段必须标注子类型。
- 子类型：
  - 因果: A直接导致B (例: 葬火之战 →[因果] 月亮碎裂)
  - 使能: A为B提供可能性 (例: 月髓 →[使能] 压制狂猎)
  - 依赖: A依赖B才能运作 (例: 三宝磨 →[依赖] 圣嗣之血)
- ⚠️ 注意区分 leads_to（因果/逻辑联系）vs temporal_before（纯时间顺序）

### 5. MOTIVATED_BY
行为动机、信仰驱动。
- 适用: Character/Faction→Concept/Event
- 例: 多托雷 → 夺取三月权能

### 6. INVOLVED_IN（全类型适用）
实体参与某事件。**重点：不限于角色！当 Item/Concept/Place/Faction 在事件中起关键作用时也要提取。**
- 适用: 任意6类 → Event
- 例: 月髓 → 最终决战（Item→Event）
- 例: 狂猎 → 苦壑崖事件（Concept→Event）
- 例: 执灯人军团 → 最终决战（Faction→Event）
- 例: 苦壑崖 → 苦壑崖决胜之战（Place→Event，地点是事件发生地）
- 例: 那夏镇 → 镇民混乱事件（Place→Event，地点受事件影响）

### 7. TEMPORAL_BEFORE
严格时间顺序。仅 Event→Event。
- ⚠️ 仅用于"A发生在B之前"的纯时间排列，不含因果关系

### 8. OPPOSED_TO
对立、敌对。
- 适用: Character/Faction → Character/Faction
- 例: 执灯人军团 → 拉乌斯万格

### 9. ORIGINATES_FROM
起源、来源。
- 适用: Item/Concept → Place/Concept
- 例: 月矩力 → 古月遗骸

## 三、提取要求

1. **text_evidence 必填**：每个实体和关系都必须引用原文片段作为证据。
2. **confidence 必填**：high/medium/low。仅当原文明确表述时标 high。
3. **attributes 尽量填充**：按照实体类型的属性模板，从文本中尽可能填写。未提及的留空（null），**不要编造**。
4. **leads_to 必须标注子类型**：description 开头写明"因果:"、"使能:"或"依赖:"。
5. **involved_in 覆盖非角色**：特别注意 Item、Concept、Faction、Place 参与事件的情况。Place 作为事件发生地或受影响地点时必须提取。
6. **不要**提取"？？？"、"选项"、"小机器人"等系统文本为实体。
7. 优先使用中文名。

## 对话文本
{text}

请输出严格的JSON格式。
"""


# =============================================================================
# LLM Knowledge Graph Extractor
# =============================================================================

class LLMKnowledgeGraphExtractor:
    """
    Extract complete knowledge graph using LLM.

    This class is independent of Neo4j and produces Pydantic objects
    that can be serialized to JSON for caching or testing.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from llama_index.llms.google_genai import GoogleGenAI
        from .entity_normalizer import EntityNormalizer
        from ..common.config import settings

        self.model = model or settings.DATA_MODEL

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")

        self.llm = GoogleGenAI(
            model=self.model,
            api_key=api_key,
        )
        self.structured_llm = self.llm.as_structured_llm(KnowledgeGraphOutput)
        self.normalizer = EntityNormalizer()

    def _build_prompt(self, text: str) -> str:
        """Build the extraction prompt with the input text."""
        return EXTRACTION_PROMPT.format(text=text)

    def extract_raw(self, text: str, max_retries: int = 2) -> KnowledgeGraphOutput:
        """
        Extract entities and relationships WITHOUT normalization.

        This is the cacheable step — raw LLM output is preserved so that
        normalization can be re-applied when aliases.json changes.
        Retries once on truncated JSON (transient Gemini issue).
        If retry also fails, exits immediately to avoid silent data loss.
        """
        import sys
        import time
        prompt = self._build_prompt(text)
        for attempt in range(max_retries):
            try:
                response = self.structured_llm.complete(prompt)
                return response.raw
            except Exception as e:
                is_json_error = "json" in str(e).lower()
                if attempt < max_retries - 1 and is_json_error:
                    print(f"  Retry {attempt + 1}/{max_retries} (truncated JSON)...")
                    time.sleep(2)
                    continue
                if is_json_error:
                    print(
                        f"\n[FATAL] JSON truncation after {max_retries} retries.\n"
                        f"  Likely cause: output too large for Gemini max_output_tokens.\n"
                        f"  Suggestion: split the input text or reduce entity count.\n"
                        f"  Error: {e}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                raise

    def normalize_output(self, kg: KnowledgeGraphOutput) -> KnowledgeGraphOutput:
        """
        Apply current aliases.json to a KG output. Can be re-run any time.

        Returns a deep copy — the cached raw data is never mutated.
        """
        import copy
        result = copy.deepcopy(kg)
        for entity in result.entities:
            normalized = self.normalizer.normalize(entity.name, entity.entity_type)
            if normalized != entity.name:
                entity.name = normalized
        for rel in result.relationships:
            rel.source = self.normalizer.normalize(rel.source)
            rel.target = self.normalizer.normalize(rel.target)
        return result

    def extract(self, text: str) -> KnowledgeGraphOutput:
        """
        Extract and normalize in one step (convenience wrapper).

        For pipeline use, prefer extract_raw() + normalize_output() separately
        so that raw output can be cached independently of normalization.
        """
        raw = self.extract_raw(text)
        return self.normalize_output(raw)

    def extract_entities_only(self, text: str) -> List[ExtractedEntity]:
        result = self.extract(text)
        return result.entities

    def extract_relationships_only(self, text: str) -> List[ExtractedRelationship]:
        result = self.extract(text)
        return result.relationships

    def extract_character_names(self, text: str) -> Set[str]:
        result = self.extract(text)
        return {e.name for e in result.entities if e.entity_type == "Character"}


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_kg_from_text(text: str) -> KnowledgeGraphOutput:
    extractor = LLMKnowledgeGraphExtractor()
    return extractor.extract(text)


def extract_kg_from_file(file_path: Path) -> KnowledgeGraphOutput:
    text = file_path.read_text(encoding="utf-8")
    return extract_kg_from_text(text)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    test_text = """
奈芙尔：那个地方叫「苦壑崖」，三年前调查分队在那里遭遇了狂猎。

叶洛亚：我已经带队去侦察过了，狂猎的活动范围在扩大。

尼基塔：执灯人军团必须阻止狂猎蔓延。银灯芯的结界是我们唯一的防线。

派蒙：银灯芯？那是什么？

叶洛亚：铸灯者索洛维留下的三件圣物之一，可以展开结界抵御狂猎。
"""

    print("Testing LLM Knowledge Graph Extractor (v2)...")
    print("=" * 60)
    print("Input text:")
    print(test_text)
    print("=" * 60)

    try:
        extractor = LLMKnowledgeGraphExtractor()
        result = extractor.extract(test_text)

        print("\nExtracted Entities:")
        for entity in result.entities:
            attrs_str = f" attrs={entity.attributes_dict}" if entity.attributes_dict else ""
            print(f"  - {entity.name} [{entity.entity_type}]{attrs_str}")

        print("\nExtracted Relationships:")
        for rel in result.relationships:
            print(f"  - {rel.source} --[{rel.relation_type}]--> {rel.target}")
            print(f"    desc: {rel.description}")
            print(f"    confidence: {rel.confidence}")

        print("\nJSON Output:")
        print(result.model_dump_json(indent=2))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
