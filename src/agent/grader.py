"""Agent Grader — Faithfulness + Completeness (golden_answer 模式).

评估环境（有 golden_answer）：两维度基于 golden_answer 评估。
生产环境（无 golden_answer）：直接返回默认值，零 LLM 调用。

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
    faithfulness: float = Field(ge=0.0, le=1.0, description="忠实度分数: 1.0=核心事实完全一致, 0.0=完全错误")
    reasoning: str = Field(description="评估理由")


class CompletenessOutput(BaseModel):
    """Completeness 评估结果。"""
    completeness: float = Field(ge=0.0, le=1.0, description="完整度分数: 1.0=覆盖所有要点, 0.0=未覆盖任何要点")
    reasoning: str = Field(description="评估理由")


# ── Prompts (golden_answer 模式) ──────────────────────────────────

FAITHFULNESS_PROMPT = """你是一个 Faithfulness 评估器。对比 Agent 回答与标准答案，评估事实正确性。

## 用户问题
{question}

## 标准答案 (Golden Answer)
{golden_answer}

## Agent 回答
{answer}

## 重要原则
- 标准答案可能不完整，不能覆盖所有正确事实
- **Agent 回答中包含标准答案未提及的信息，不应扣分**（标准答案不是唯一正确叙事）
- 只有 Agent 回答与标准答案**明确矛盾**的内容才应扣分
- 逐条对比事实点，不要因叙事结构或表述顺序不同而降分

## 评估标准
- 1.0: Agent 回答与标准答案无矛盾，核心事实一致
- 0.8: 核心正确，少量细节与标准答案矛盾
- 0.5: 部分事实与标准答案一致，部分关键信息矛盾
- 0.2: 大部分关键信息与标准答案矛盾
- 0.0: 完全错误或答非所问

请逐条对比事实点，区分"矛盾"与"额外信息"，给出 faithfulness 分数和理由。
"""

COMPLETENESS_PROMPT = """你是一个 Completeness 评估器。评估 Agent 回答是否全面回答了用户问题的所有方面。

## 用户问题
{question}

## 标准答案 (Golden Answer, 作为要点参考)
{golden_answer}

## Agent 回答
{answer}

## 评估标准
- 1.0: 覆盖了问题的所有方面和标准答案的所有关键要点
- 0.8: 覆盖大部分方面，遗漏少量细节
- 0.5: 只覆盖约一半要点
- 0.2: 只覆盖少量要点
- 0.0: 未覆盖任何要点

请根据以上标准评估，给出 completeness 分数和理由。
"""


class FaithfulnessGrader:
    """Faithfulness 评估 — 基于 golden_answer 的事实正确性。"""

    def __init__(self, llm):
        self.structured_llm = llm.with_structured_output(FaithfulnessOutput)

    async def grade(
        self,
        question: str,
        answer: str,
        golden_answer: str,
    ) -> FaithfulnessResult:
        prompt = FAITHFULNESS_PROMPT.format(
            question=question,
            golden_answer=golden_answer,
            answer=answer,
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


class CompletenessGrader:
    """Completeness 评估 — 基于 golden_answer 的覆盖率。"""

    def __init__(self, llm):
        self.structured_llm = llm.with_structured_output(CompletenessOutput)

    async def grade(
        self,
        question: str,
        answer: str,
        golden_answer: str,
    ) -> float:
        prompt = COMPLETENESS_PROMPT.format(
            question=question,
            golden_answer=golden_answer,
            answer=answer,
        )
        try:
            result = await self.structured_llm.ainvoke(prompt)
            return result.completeness
        except Exception as e:
            logger.error(f"Completeness grading failed: {e}")
        return 0.0


class ProductionGraderService:
    """Grader: 有 golden_answer 时两维度 LLM 评估，无则跳过。"""

    def __init__(self, llm):
        self.faithfulness_grader = FaithfulnessGrader(llm)
        self.completeness_grader = CompletenessGrader(llm)

    async def grade(
        self,
        question: str,
        answer: str,
        tool_calls: list[ToolCallRecord],
        golden_answer: str | None = None,
    ) -> ProductionGrade:
        if not golden_answer:
            return ProductionGrade(faithfulness=1.0, completeness=1.0, total=1.0)

        faith_result, completeness = await asyncio.gather(
            self.faithfulness_grader.grade(question, answer, golden_answer),
            self.completeness_grader.grade(question, answer, golden_answer),
        )

        faith = faith_result.score
        total = round((faith + completeness) / 2, 4)

        return ProductionGrade(
            faithfulness=faith,
            completeness=completeness,
            total=total,
        )
