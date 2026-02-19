"""Genshin Retrieval Agent v4 module.

Usage:
    from src.agent import GenshinRetrievalAgent, create_agent

    agent = create_agent()
    response = await agent.run("少女是谁？")
    answer, response = await agent.run_with_grading("木偶为何对月髓感兴趣？")
"""

from .agent import GenshinRetrievalAgent, create_agent

__all__ = [
    "GenshinRetrievalAgent",
    "create_agent",
]
