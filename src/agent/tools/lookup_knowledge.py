"""
知识图谱工具：lookup_knowledge - 查询实体的静态信息和直接关系。

此工具查询 Neo4j 知识图谱获取实体的基本信息。
返回结构化数据（非长文本）。
"""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ...common.graph.searcher import GraphSearcher

# 关系类型 → 中文语义标注，帮助模型理解 KG 输出
RELATION_LABELS = {
    "MOTIVATED_BY": "信仰/动机",
    "MEMBER_OF": "成员",
    "OPPOSED_TO": "对立",
    "ORIGINATES_FROM": "起源/来源",
    "LOCATED_AT": "所在地",
    "INVOLVED_IN": "参与",
    "LEADS_TO": "导致/引发",
    "CREATED_BY": "创造者",
    "TEMPORAL_BEFORE": "时间先于",
}


class LookupKnowledgeInput(BaseModel):
    """lookup_knowledge 工具输入参数。"""
    entity: str = Field(description="实体名称（角色、组织或地点），支持别名自动解析")
    relation: Optional[str] = Field(default=None, description="可选的关系类型过滤，如 FRIEND_OF, MEMBER_OF, MOTIVATED_BY")


@tool(args_schema=LookupKnowledgeInput)
def lookup_knowledge(entity: str, relation: Optional[str] = None) -> str:
    """查询知识图谱获取实体（角色、组织、地点）的基本信息、属性和直接关系。

    适用场景：回答"X是谁"、"X的属性"等事实类问题。几乎每个问题都应先用此工具确认实体存在。
    返回的关系类型含义：MOTIVATED_BY=信仰/动机, MEMBER_OF=所属组织, LOCATED_AT=所在地, OPPOSED_TO=对立, INVOLVED_IN=参与事件, LEADS_TO=导致/引发, CREATED_BY=创造者。"""
    with GraphSearcher() as searcher:
        entity_info = searcher.get_entity_info(entity)
        result = searcher.search(entity, relation=relation, limit=25)

    if not result["entities"] and not entity_info:
        return f"在知识图谱中未找到 '{entity}' 的信息。建议使用 search_memory 搜索包含此实体的故事内容。"

    # 格式化输出为结构化文本
    lines = [f"## 实体信息：{result['entity']}"]

    # Node self-properties
    if entity_info:
        labels = entity_info.get("labels", [])
        lines.append(f"**类型**: {', '.join(labels)}")
        props = entity_info.get("properties", {})
        skip_keys = {"name", "embedding", "aliases"}
        for key, val in props.items():
            if key not in skip_keys and val:
                lines.append(f"**{key}**: {str(val)[:200]}")

    if result["relation_filter"]:
        lines.append(f"(已过滤关系类型：{result['relation_filter']})")

    lines.append("")

    for item in result["entities"]:
        target = item.get("target", "未知")
        relation_type = item.get("relation", "RELATED_TO")
        target_type = item.get("target_type", "实体")
        description = item.get("description", "")
        rel_props = item.get("rel_properties", {})
        chapter = rel_props.get("chapter", "")
        task_id = rel_props.get("task_id", "")

        label = RELATION_LABELS.get(relation_type, "")
        rel_display = f"{label} {relation_type}" if label else relation_type
        line = f"- [{rel_display}] → {target} ({target_type})"
        if chapter:
            line += f" [第{str(chapter)[:3]}章"
            if task_id:
                line += f", 任务{task_id}"
            line += "]"
        if description:
            line += f": {description[:100]}..."
        lines.append(line)

    lines.append("")
    lines.append(f"共找到 {result['count']} 条关系。")

    return "\n".join(lines)
