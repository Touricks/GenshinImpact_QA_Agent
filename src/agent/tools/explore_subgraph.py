"""
知识图谱工具：explore_subgraph - 以实体为中心展开关系子图。

BFS 展开关系子图，支持深度控制、节点数限制和边类型过滤。
"""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ...common.graph.searcher import GraphSearcher


class ExploreSubgraphInput(BaseModel):
    """explore_subgraph 工具输入参数。"""
    entity: str = Field(description="中心实体名称")
    depth: int = Field(default=1, ge=1, le=3, description="展开深度")
    max_nodes: int = Field(default=20, ge=1, description="最大返回节点数")
    edge_types: Optional[str] = Field(
        default=None,
        description="边类型过滤，逗号分隔。可用: INVOLVED_IN, LEADS_TO, LOCATED_AT, MEMBER_OF, OPPOSED_TO, MOTIVATED_BY, CREATED_BY, ORIGINATES_FROM, TEMPORAL_BEFORE",
    )


@tool(args_schema=ExploreSubgraphInput)
def explore_subgraph(
    entity: str,
    depth: int = 1,
    max_nodes: int = 20,
    edge_types: Optional[str] = None,
) -> str:
    """以实体为中心展开关系子图（BFS），支持深度控制和边类型过滤。

    适用场景：分析阵营/势力/关系网络的最高效工具。回答"X周围有哪些势力"、"X的社会关系网"等网络类问题。也可用于复杂分析：对核心实体展开子图后综合分析。可用边类型：INVOLVED_IN, LEADS_TO, LOCATED_AT, MEMBER_OF, OPPOSED_TO, MOTIVATED_BY, CREATED_BY, ORIGINATES_FROM。"""
    # Parse edge_types string to list
    type_list = None
    if edge_types:
        type_list = [t.strip() for t in edge_types.split(",") if t.strip()]

    with GraphSearcher() as searcher:
        result = searcher.explore_subgraph(
            entity, depth=depth, max_nodes=max_nodes, edge_types=type_list
        )

    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    if not nodes:
        return (
            f"在知识图谱中未找到 '{entity}' 的子图。\n\n"
            f"建议：\n"
            f"- 使用 lookup_knowledge(\"{entity}\") 确认实体名称。\n"
            f"- 尝试不使用 edge_types 过滤。"
        )

    lines = [f"## 子图：{entity}"]
    lines.append(
        f"（深度={depth}, 节点数={result['node_count']}, "
        f"边数={result['edge_count']}）"
    )
    if edge_types:
        lines.append(f"（边类型过滤：{edge_types}）")
    lines.append("")

    # Group nodes by type
    nodes_by_type = {}
    for node in nodes:
        ntype = node.get("type", "未知")
        nodes_by_type.setdefault(ntype, []).append(node["name"])

    lines.append("**节点：**")
    for ntype, names in sorted(nodes_by_type.items()):
        lines.append(f"- {ntype} ({len(names)}): {', '.join(names)}")

    lines.append("")
    lines.append("**关系：**")
    for edge in edges:
        lines.append(
            f"- {edge['src']} -[{edge['type']}]-> {edge['tgt']}"
        )

    return "\n".join(lines)
