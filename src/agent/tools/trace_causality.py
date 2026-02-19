"""
知识图谱工具：trace_causality - 追踪因果/动机链。

仅遍历 LEADS_TO + MOTIVATED_BY 边，追踪事件之间的因果和动机关系。
支持双参数（路径查找）和单参数（邻域展开）两种模式。
"""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ...common.graph.searcher import GraphSearcher


class TraceCausalityInput(BaseModel):
    """trace_causality 工具输入参数。"""
    start: str = Field(description="起始实体名称")
    end: Optional[str] = Field(default=None, description="可选的终点实体名称，提供时查找因果路径，不提供时展开因果邻域")


@tool(args_schema=TraceCausalityInput)
def trace_causality(start: str, end: Optional[str] = None) -> str:
    """追踪因果关系链或动机链（LEADS_TO/MOTIVATED_BY）。

    适用场景：回答"为什么X会Y"、"X的动机是什么"、"X导致了什么"等因果/动机类问题。双参数模式(start+end)查因果路径，单参数模式(start)展开因果邻域。"""
    with GraphSearcher() as searcher:
        result = searcher.trace_causality(start, end=end)

    if end:
        # Dual-param mode: path
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])

        if not nodes:
            return (
                f"在因果图中未找到 '{start}' 到 '{end}' 的路径"
                f"（LEADS_TO/MOTIVATED_BY 边，4步以内）。\n\n"
                f"建议：\n"
                f"- 使用 find_connection(\"{start}\", \"{end}\") 查找全类型路径。\n"
                f"- 使用 search_memory 搜索两者的关联。"
            )

        lines = [f"## 因果路径：{start} → {end}"]
        lines.append(f"（路径长度：{result.get('path_length', 0)} 步）")
        lines.append("")

        # Build chain representation
        chain_parts = []
        for i, node in enumerate(nodes):
            name = node.get("name", "?")
            labels = node.get("labels", [])
            label = labels[0] if labels else "?"
            chain_parts.append(f"{name}({label})")
            if i < len(edges):
                edge = edges[i]
                edge_type = edge.get("type", "?")
                chain_parts.append(f" -[{edge_type}]-> ")

        lines.append("".join(chain_parts))
        lines.append("")

        # Edge details
        lines.append("**因果链详情：**")
        for edge in edges:
            desc = edge.get("desc", "")
            lines.append(
                f"- {edge.get('src', '?')} → {edge.get('tgt', '?')} "
                f"[{edge.get('type', '?')}]"
            )
            if desc:
                lines.append(f"  说明：{desc[:200]}")

        return "\n".join(lines)
    else:
        # Single-param mode: neighborhood
        neighbors = result.get("neighbors", [])

        if not neighbors:
            return (
                f"在因果图中未找到 '{start}' 的因果邻域。\n\n"
                f"建议：\n"
                f"- 使用 lookup_knowledge(\"{start}\") 查看实体所有关系。\n"
                f"- 使用 explore_subgraph(\"{start}\") 查看完整邻域。"
            )

        lines = [f"## 因果邻域：{start}"]
        lines.append(f"（共 {len(neighbors)} 个因果关联实体）")
        lines.append("")

        for neighbor in neighbors:
            name = neighbor.get("name", "?")
            ntype = neighbor.get("type", "?")
            chain = neighbor.get("chain", [])

            lines.append(f"- **{name}** ({ntype})")
            for link in chain:
                link_type = link.get("type", "?")
                link_desc = link.get("desc", "")
                lines.append(f"  - [{link_type}] {link_desc[:80]}")

        return "\n".join(lines)
