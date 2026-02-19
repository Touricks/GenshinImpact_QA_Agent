"""
知识图谱工具：find_connection - 查找两个实体之间的关系路径。

此工具在 Neo4j 知识图谱中查找最短连接路径。
返回逻辑链（非长文本）。
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ...common.graph.searcher import GraphSearcher


class FindConnectionInput(BaseModel):
    """find_connection 工具输入参数。"""
    entity1: str = Field(description="第一个实体名称")
    entity2: str = Field(description="第二个实体名称")
    max_hops: int = Field(default=4, ge=1, le=6, description="最大跳数，增大可找到更远的路径")


@tool(args_schema=FindConnectionInput)
def find_connection(entity1: str, entity2: str, max_hops: int = 4) -> str:
    """查找知识图谱中两个实体之间的最短连接路径，返回逻辑链。

    适用场景：回答"A和B什么关系"、"A怎么认识B"等关系类问题。也可用于复杂分析中查实体间路径。"""
    with GraphSearcher() as searcher:
        path = searcher.get_path_between(entity1, entity2, max_hops=max_hops)

    if not path:
        return (
            f"在知识图谱中未找到 '{entity1}' 和 '{entity2}' 之间的直接连接（{max_hops}步以内）。\n\n"
            f"建议：\n"
            f"- 使用 lookup_knowledge 分别查看每个实体的关系。\n"
            f"- 使用 search_memory 搜索两者同时出现的故事内容。"
        )

    # 格式化路径为可读的链
    nodes = path["path_nodes"]
    relations = path["path_relations"]

    # 构建链：A -[关系1]-> B -[关系2]-> C
    chain_parts = [nodes[0]]
    for i, rel in enumerate(relations):
        chain_parts.append(f" -[{rel}]-> ")
        chain_parts.append(nodes[i + 1])

    chain = "".join(chain_parts)

    lines = [
        f"## 关系路径：{entity1} ↔ {entity2}",
        "",
        f"**路径**（{path['path_length']} 步）：",
        chain,
        "",
        "**路径中的节点：**",
    ]

    for node in nodes:
        lines.append(f"- {node}")

    return "\n".join(lines)
