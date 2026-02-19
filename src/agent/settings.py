"""Agent 配置参数 — 代理层.

实际配置位于 src/common/config/settings.py (AGENT_* 字段)。
本模块保持 `from .settings import agent_settings` 兼容，不需要修改引用方。
"""

from src.common.config.settings import settings


class AgentSettings:
    """代理对象，从全局 settings 读取 AGENT_* 字段。"""

    @property
    def max_iterations(self) -> int:
        return settings.AGENT_MAX_ITERATIONS

    @property
    def max_tool_calls(self) -> int:
        return settings.AGENT_MAX_TOOL_CALLS

    @property
    def max_evidence(self) -> int:
        return settings.AGENT_MAX_EVIDENCE

    @property
    def faithfulness_threshold(self) -> float:
        return settings.AGENT_FAITHFULNESS_THRESHOLD

    @property
    def completeness_threshold(self) -> float:
        return settings.AGENT_COMPLETENESS_THRESHOLD

    @property
    def relevance_threshold(self) -> float:
        return settings.AGENT_RELEVANCE_THRESHOLD

    @property
    def tool_output_truncate(self) -> int:
        return settings.AGENT_TOOL_OUTPUT_TRUNCATE

    @property
    def grader_tool_output_limit(self) -> int:
        return settings.AGENT_GRADER_TOOL_OUTPUT_LIMIT

    # ── v4 参数 ────────────────────────────────────────────────
    @property
    def v4_max_iterations(self) -> int:
        return settings.AGENT_V4_MAX_ITERATIONS

    @property
    def v4_max_tool_calls(self) -> int:
        return settings.AGENT_V4_MAX_TOOL_CALLS

    @property
    def v4_max_tools_per_iter(self) -> int:
        return settings.AGENT_V4_MAX_TOOLS_PER_ITER

    @property
    def v4_wall_timeout(self) -> int:
        return settings.AGENT_V4_WALL_TIMEOUT


agent_settings = AgentSettings()
