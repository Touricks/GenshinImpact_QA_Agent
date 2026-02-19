"""Agent Grader — Faithfulness + Completeness + Relevance.

三维度单分数评估，每个维度一次 LLM 调用，三维度并行执行。
使用 with_structured_output 确保输出格式正确。

Uses GRADER_MODEL (fast model).
"""

import asyncio
import logging

from pydantic import BaseModel, Field

from .models import (
    FaithfulnessResult,
    ProductionGrade,
    ToolCallRecord,
)
from .settings import agent_settings

logger = logging.getLogger(__name__)


# ── Structured Output Models ─────────────────────────────────────

class FaithfulnessOutput(BaseModel):
    """Faithfulness 评估结果。"""
    faithfulness: float = Field(ge=0.0, le=1.0, description="忠实度分数: 1.0=全部有支持, 0.0=完全编造")
    reasoning: str = Field(description="评估理由")


class CompletenessOutput(BaseModel):
    """Completeness 评估结果。"""
    completeness: float = Field(ge=0.0, le=1.0, description="完整度分数: 1.0=充分利用, 0.0=完全忽略")
    reasoning: str = Field(description="评估理由")


class RelevanceOutput(BaseModel):
    """Relevance 评估结果。"""
    relevance: float = Field(ge=0.0, le=1.0, description="相关度分数: 1.0=完全回答, 0.0=无关")
    reasoning: str = Field(description="评估理由")


# ── Faithfulness Grader Prompt ──────────────────────────────────────

FAITHFULNESS_PROMPT = """你是一个 Faithfulness 评估器。判断答案整体是否有工具调用结果的支持。

## 用户问题
{question}

## Agent 答案
{answer}

## 工具调用记录 (带索引)
{tool_calls}

## 评估标准
- 1.0: 答案中所有事实性陈述都能在工具输出中找到直接或间接支持
- 0.8: 绝大部分陈述有支持，少量细节为合理推断
- 0.5: 部分陈述有支持，但存在明显无依据的内容
- 0.2: 答案大部分内容无法从工具输出中验证
- 0.0: 答案完全是编造的，或与工具输出矛盾

请根据以上标准评估，给出 faithfulness 分数和理由。
"""

# ── Answer Relevance Prompt ─────────────────────────────────────────

COMPLETENESS_PROMPT = """你是一个 Completeness 评估器。判断答案是否充分利用了工具返回的信息。

## 用户问题
{question}

## Agent 答案
{answer}

## 工具调用记录 (带索引)
{tool_calls}

## 评估标准
- 1.0: 工具返回的所有关键信息都被纳入答案
- 0.8: 答案纳入了大部分工具信息，遗漏少量非核心细节
- 0.5: 答案遗漏了工具返回中的重要信息
- 0.2: 答案说"无法查到"但工具实际返回了相关信息
- 0.0: 答案完全忽略了工具返回的内容，或工具未返回任何有用信息且答案无内容

## 判定要点
- 重点关注工具返回中**与问题直接相关**的信息是否被纳入答案
- 如果工具返回信息但答案声称"无法查到"或"没有相关信息"，completeness 应 ≤ 0.2
- 如果工具未返回任何有用信息，答案如实说明，completeness 可为 0.5

请根据以上标准评估，给出 completeness 分数和理由。
"""

RELEVANCE_PROMPT = """你是一个 Answer Relevance 评估器。判断答案是否回答了用户问题。

## 用户问题
{question}

## Agent 答案
{answer}

## 评估标准
- 1.0: 完全回答了问题的所有方面
- 0.8: 回答了主要方面，遗漏部分细节
- 0.5: 只部分回答了问题
- 0.2: 答案与问题相关但未真正回答
- 0.0: 答案与问题无关或拒绝回答

请根据以上标准评估，给出 relevance 分数和理由。
"""


class FaithfulnessGrader:
    """Faithfulness 评估 — 单分数模式。"""

    def __init__(self, llm):
        self.structured_llm = llm.with_structured_output(FaithfulnessOutput)

    async def grade(
        self,
        answer: str,
        tool_calls: list[ToolCallRecord],
        question: str = "",
    ) -> FaithfulnessResult:
        limit = agent_settings.grader_tool_output_limit
        if tool_calls:
            tc_str = "\n".join(
                f"[{i}] {tc.tool_name}({tc.input}) → {tc.truncated(limit)}"
                for i, tc in enumerate(tool_calls)
            )
        else:
            tc_str = "(没有调用任何工具)"

        prompt = FAITHFULNESS_PROMPT.format(
            question=question,
            answer=answer,
            tool_calls=tc_str,
        )

        try:
            result = await self.structured_llm.ainvoke(prompt)
            score = result.faithfulness
            if score < 0.3:
                logger.warning(f"Faithfulness low ({score}), reasoning: {result.reasoning[:200]}")
            threshold = agent_settings.faithfulness_threshold
            return FaithfulnessResult(score=score, passed=score >= threshold)
        except Exception as e:
            logger.error(f"Faithfulness grading failed: {e}")
        return FaithfulnessResult(score=0.0, passed=False)


class AnswerRelevanceGrader:
    """Answer Relevance 评估。"""

    def __init__(self, llm):
        self.structured_llm = llm.with_structured_output(RelevanceOutput)

    async def grade(self, question: str, answer: str) -> float:
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        try:
            result = await self.structured_llm.ainvoke(prompt)
            score = result.relevance
            if score < 0.3:
                logger.warning(f"Relevance low ({score}), reasoning: {result.reasoning[:200]}")
            return score
        except Exception as e:
            logger.error(f"Relevance grading failed: {e}")
        return 0.0


class CompletenessGrader:
    """Completeness 评估: 答案是否充分利用了 tool output 中的可用信息。"""

    def __init__(self, llm):
        self.structured_llm = llm.with_structured_output(CompletenessOutput)

    async def grade(
        self,
        question: str,
        answer: str,
        tool_calls: list[ToolCallRecord],
    ) -> float:
        limit = agent_settings.grader_tool_output_limit
        if tool_calls:
            tc_str = "\n".join(
                f"[{i}] {tc.tool_name}({tc.input}) → {tc.truncated(limit)}"
                for i, tc in enumerate(tool_calls)
            )
        else:
            tc_str = "(没有调用任何工具)"

        prompt = COMPLETENESS_PROMPT.format(
            question=question,
            answer=answer,
            tool_calls=tc_str,
        )
        try:
            result = await self.structured_llm.ainvoke(prompt)
            return result.completeness
        except Exception as e:
            logger.error(f"Completeness grading failed: {e}")
        return 0.0


class ProductionGraderService:
    """生产环境 Grader: Faithfulness + Completeness + Answer Relevance."""

    def __init__(self, llm):
        self.faithfulness_grader = FaithfulnessGrader(llm)
        self.completeness_grader = CompletenessGrader(llm)
        self.relevance_grader = AnswerRelevanceGrader(llm)

    async def grade(
        self,
        question: str,
        answer: str,
        tool_calls: list[ToolCallRecord],
    ) -> ProductionGrade:
        faith_result, completeness, relevance = await asyncio.gather(
            self.faithfulness_grader.grade(answer, tool_calls, question),
            self.completeness_grader.grade(question, answer, tool_calls),
            self.relevance_grader.grade(question, answer),
        )

        return ProductionGrade(
            faithfulness=faith_result.score,
            completeness=completeness,
            answer_relevance=relevance,
            passed=(
                faith_result.passed
                and completeness >= agent_settings.completeness_threshold
                and relevance >= agent_settings.relevance_threshold
            ),
        )
