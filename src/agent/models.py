"""Agent Data Models.

Shared Pydantic models used across Workflow steps, Grader, Tracer, and API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── ToolCallRecord ──────────────────────────────────────────────────

class ToolCallRecord(BaseModel):
    """统一的工具调用记录 — 贯穿 Agent、Grader、Tracer、API 四层."""

    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    output_display: str | None = None
    duration_ms: int = 0
    query_event_id: str | None = None
    global_index: int | None = None

    def truncated(self, max_len: int) -> str:
        if len(self.output) > max_len:
            return self.output[:max_len] + "..."
        return self.output


# ── Faithfulness ────────────────────────────────────────────────────

class FaithfulnessResult(BaseModel):
    """Faithfulness 评估结果 — 单分数模式."""
    score: float = 0.0
    passed: bool = False


# ── Final Grading ──────────────────────────────────────────────────

class ProductionGrade(BaseModel):
    faithfulness: float = 0.0
    completeness: float = 0.0
    answer_relevance: float = 0.0
    passed: bool = False


# ── AgentResponse (API 输出) ────────────────────────────────────────

class GradingResult(BaseModel):
    faithfulness: float = 0.0
    completeness: float = 0.0
    answer_relevance: float = 0.0
    passed: bool = False


class AgentResponse(BaseModel):
    """API 响应结构."""
    question: str = ""
    answer_text: str = ""
    raw_answer: str = ""  # humanize 前的原始回答
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    grading: GradingResult = Field(default_factory=GradingResult)
    query_count: int = 0
    total_duration_ms: int = 0
