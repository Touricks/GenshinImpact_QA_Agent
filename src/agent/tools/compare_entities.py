"""
知识图谱工具：compare_entities - 对比两个实体的属性和关系。

KG+LLM 混合工具：
1. KG 部分：获取两个实体的属性和关系
2. LLM 部分：生成属性对齐矩阵
"""

import json
import logging
import os

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ...common.graph.searcher import GraphSearcher

logger = logging.getLogger(__name__)


def _format_entity_profile(info: dict, relations: list) -> str:
    """Format entity info + relations into a profile string for LLM."""
    if not info:
        return "(未找到)"

    lines = [f"名称: {info['name']}"]
    lines.append(f"类型: {', '.join(info['labels'])}")

    # Properties
    props = info.get("properties", {})
    for key, val in props.items():
        if key != "name" and val:
            lines.append(f"{key}: {str(val)[:200]}")

    # Relations
    if relations:
        lines.append(f"\n关系 ({len(relations)} 条):")
        for rel in relations[:25]:  # Limit to 25 relations
            target = rel.get("target", "?")
            rtype = rel.get("relation", "?")
            target_type = rel.get("target_type", "?")
            lines.append(f"  - [{rtype}] → {target} ({target_type})")

    return "\n".join(lines)


class CompareEntitiesInput(BaseModel):
    """compare_entities 工具输入参数。"""
    entity_a: str = Field(description="第一个实体名称")
    entity_b: str = Field(description="第二个实体名称")


@tool(args_schema=CompareEntitiesInput)
def compare_entities(entity_a: str, entity_b: str) -> str:
    """对比两个实体的属性和关系，生成属性对齐矩阵。

    适用场景：回答"X和Y有何不同"、"X和Y的对比"等对比类问题。"""
    with GraphSearcher() as searcher:
        # Get entity info
        info_a = searcher.get_entity_info(entity_a)
        info_b = searcher.get_entity_info(entity_b)

        # Get relations
        rels_a = searcher.search(entity_a, limit=25)
        rels_b = searcher.search(entity_b, limit=25)

    profile_a = _format_entity_profile(info_a, rels_a.get("entities", []))
    profile_b = _format_entity_profile(info_b, rels_b.get("entities", []))

    if not info_a and not info_b:
        return (
            f"在知识图谱中未找到 '{entity_a}' 和 '{entity_b}'。\n"
            f"建议使用 lookup_knowledge 分别查询。"
        )

    # Try LLM alignment
    try:
        comparison = _llm_compare(entity_a, entity_b, profile_a, profile_b)
        if comparison:
            return comparison
    except Exception as e:
        logger.warning(f"LLM comparison failed: {e}, falling back to simple diff")

    # Fallback: simple side-by-side comparison
    return _simple_compare(entity_a, entity_b, info_a, info_b, rels_a, rels_b)


def _llm_compare(
    entity_a: str, entity_b: str, profile_a: str, profile_b: str
) -> str:
    """Use LLM to generate alignment matrix."""
    from ..llm_factory import create_chat_model
    from ...common.config import Settings
    settings = Settings()

    llm = create_chat_model(settings.GRADER_MODEL)

    prompt = f"""请对比以下两个实体，生成一个属性对齐矩阵。

## 实体 A: {entity_a}
{profile_a}

## 实体 B: {entity_b}
{profile_b}

## 输出格式
用 Markdown 表格输出对比结果，包含以下列：
| 维度 | {entity_a} | {entity_b} |

维度应包括但不限于：类型、所属阵营、角色定位、目标、命运/结局、关键关系。
只对比双方都有数据的维度。简洁回答，每格不超过 30 字。"""

    import asyncio

    async def _run():
        response = await llm.ainvoke(prompt)
        return response.content

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, _run()).result(timeout=30)
        else:
            result = asyncio.run(_run())
    except RuntimeError:
        result = asyncio.run(_run())

    comparison_text = str(result).strip()
    return f"## 实体对比：{entity_a} vs {entity_b}\n\n{comparison_text}"


def _simple_compare(
    entity_a: str, entity_b: str,
    info_a: dict, info_b: dict,
    rels_a: dict, rels_b: dict,
) -> str:
    """Fallback: simple side-by-side comparison without LLM."""
    lines = [f"## 实体对比：{entity_a} vs {entity_b}"]
    lines.append("")

    # Compare properties
    props_a = info_a.get("properties", {}) if info_a else {}
    props_b = info_b.get("properties", {}) if info_b else {}

    all_keys = set(props_a.keys()) | set(props_b.keys())
    skip_keys = {"name", "embedding", "aliases"}
    compare_keys = sorted(all_keys - skip_keys)

    if compare_keys:
        lines.append("| 属性 | " + entity_a + " | " + entity_b + " |")
        lines.append("|------|------|------|")
        for key in compare_keys:
            val_a = str(props_a.get(key, "-"))[:50]
            val_b = str(props_b.get(key, "-"))[:50]
            lines.append(f"| {key} | {val_a} | {val_b} |")
        lines.append("")

    # Compare relation types
    rel_types_a = set()
    for rel in rels_a.get("entities", []):
        rel_types_a.add(rel.get("relation", ""))
    rel_types_b = set()
    for rel in rels_b.get("entities", []):
        rel_types_b.add(rel.get("relation", ""))

    shared = rel_types_a & rel_types_b
    only_a = rel_types_a - rel_types_b
    only_b = rel_types_b - rel_types_a

    lines.append("**关系类型对比：**")
    if shared:
        lines.append(f"- 共有：{', '.join(shared)}")
    if only_a:
        lines.append(f"- 仅 {entity_a}：{', '.join(only_a)}")
    if only_b:
        lines.append(f"- 仅 {entity_b}：{', '.join(only_b)}")

    return "\n".join(lines)
