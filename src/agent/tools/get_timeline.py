"""
知识图谱工具：get_timeline - 获取实体的事件时间线。

此工具合并了原 track_journey 和 get_character_events 的功能。
输入任意实体 → 返回参与的事件序列。

三层退化策略：
1. INVOLVED_IN → Event 节点（Character/Item 最佳）
2. TEMPORAL_BEFORE 链（如果实体本身是 Event）
3. 全部带 chapter 属性的边（Place 等的兜底策略）
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ...common.graph.searcher import GraphSearcher


class GetTimelineInput(BaseModel):
    """get_timeline 工具输入参数。"""
    entity: str = Field(description="实体名称（角色、物品、事件、地点等），支持别名自动解析")


@tool(args_schema=GetTimelineInput)
def get_timeline(entity: str) -> str:
    """获取实体的事件时间线，按时间排序。支持 Character/Item/Event/Place 等所有实体类型。

    适用场景：回答"X的经历"、"X的行动路线"、"X发生了什么"等时间线类问题。可与explore_subgraph组合使用，先获取时间线骨架再展开关联实体。"""
    with GraphSearcher() as searcher:
        events = searcher.get_timeline(entity)

    if not events:
        return (
            f"在知识图谱中未找到 '{entity}' 的时间线。\n\n"
            f"建议：\n"
            f"- 使用 lookup_knowledge(entity=\"{entity}\") 确认实体存在。\n"
            f"- 使用 search_memory(query=\"{entity}\", sort_by=\"time\") "
            f"按时间顺序搜索故事内容。"
        )

    source = events[0].get("source", "unknown")
    lines = [f"## 时间线：{entity}"]
    lines.append(f"（数据来源：{source}，共 {len(events)} 条事件）")
    lines.append("")

    current_chapter = None
    for evt in events:
        chapter = evt.get("chapter", "")
        event_name = evt.get("event_name", "未知事件")

        # Group by chapter
        if chapter and chapter != current_chapter:
            if current_chapter is not None:
                lines.append("")
            lines.append(f"### 第 {chapter} 章")
            current_chapter = chapter

        # Event entry
        description = evt.get("description", "")
        cause = evt.get("cause", "")
        effect = evt.get("effect", "")
        role_desc = evt.get("role_description", "")
        relation = evt.get("relation", "")

        line = f"- **{event_name}**"
        if relation:
            line += f" [{relation}]"
        lines.append(line)

        if role_desc:
            lines.append(f"  - 角色：{role_desc[:150]}")
        if description:
            lines.append(f"  - 描述：{description[:200]}")
        if cause:
            lines.append(f"  - 起因：{cause[:150]}")
        if effect:
            lines.append(f"  - 影响：{effect[:150]}")

    lines.append("")
    lines.append(
        "**提示**: 如需详细剧情内容，请使用 search_memory 搜索时间线中的特定事件。"
    )

    return "\n".join(lines)
