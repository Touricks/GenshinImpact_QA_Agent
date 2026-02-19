"""Genshin Retrieval Agent — v4 Single ReAct Workflow.

入口文件: 创建 LLM、加载工具、构建 Workflow 并运行。
v4: 单 ReAct 循环 (solve→humanize→grade)
LLM 策略: solve → REASONING_MODEL, 其他步骤 → GRADER_MODEL.
"""

import logging
import os
import time
from typing import Tuple

from ..common.config import Settings
from .models import AgentResponse
from .tracer import AgentTracer

logger = logging.getLogger(__name__)


def _ensure_google_api_key():
    """Ensure GOOGLE_API_KEY is set in environment."""
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key
    # Avoid "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" warning
    if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        del os.environ["GEMINI_API_KEY"]


class GenshinRetrievalAgent:
    """ReAct Agent with v4 Single ReAct Workflow.

    单 ReAct 循环 + post-hoc grade (更快更简洁)
    """

    def __init__(
        self,
        session_id: str = "default",
        model: str = None,
        verbose: bool = False,
        enable_grader: bool = True,
    ):
        self.session_id = session_id
        self.verbose = verbose
        self.enable_grader = enable_grader
        self.settings = Settings()
        self.model = model or self.settings.REASONING_MODEL
        self._workflow = None
        self._tracer = AgentTracer()

    def _ensure_initialized(self):
        """Lazy initialization."""
        if self._workflow is not None:
            return

        _ensure_google_api_key()
        logger.info(f"Initializing v4 Agent: reasoning={self.model}, grader={self.settings.GRADER_MODEL}")

        from .llm_factory import create_chat_model

        # REASONING_MODEL — for solve_question (需要强推理 + tool calling)
        reasoning_llm = create_chat_model(self.model, thinking_level=self.settings.AGENT_THINKING_LEVEL)

        # GRADER_MODEL — for grader, humanizer (快速模型, BaseChatModel 直接传递)
        grader_llm = create_chat_model(self.settings.GRADER_MODEL)

        # Load tools (7 tools)
        from .tools import (
            lookup_knowledge,
            find_connection,
            get_timeline,
            trace_causality,
            compare_entities,
            explore_subgraph,
        )
        from .tools.search_memory import search_memory

        tools = [
            lookup_knowledge,
            find_connection,
            get_timeline,
            trace_causality,
            compare_entities,
            explore_subgraph,
            search_memory,
        ]

        # Ablation: filter tools based on mode
        if self.settings.ABLATION_MODE == "vector_only":
            tools = [t for t in tools if t.name == "search_memory"]
            logger.info(f"[Ablation] vector_only mode: {len(tools)} tool(s)")

        from .workflow_v4 import AgentV4Workflow
        self._workflow = AgentV4Workflow(
            reasoning_llm=reasoning_llm,
            grader_llm=grader_llm,
            tools=tools,
            tracer=self._tracer,
        )

        logger.info("v4 Agent initialized successfully")

    async def run(self, query: str) -> str:
        """Run a single query (stateless). Returns answer text."""
        self._ensure_initialized()
        result = await self._workflow.run(question=query)
        return result.answer_text

    async def run_with_grading(self, query: str, golden_answer: str = "") -> Tuple[str, AgentResponse]:
        """Run with full grading. Returns (answer_text, full_response)."""
        self._ensure_initialized()

        start_time = time.time()
        result = await self._workflow.run(question=query, golden_answer=golden_answer)

        total_duration = int((time.time() - start_time) * 1000)
        result.total_duration_ms = total_duration

        # Save trace (raw_answer = pre-humanize, answer_text = post-humanize)
        trace_path = self._tracer.end_trace(
            final_response=result.raw_answer or result.answer_text,
            passed=result.grading.total >= 0.7,
            total_duration_ms=total_duration,
            humanized_response=result.answer_text,
        )
        if trace_path:
            logger.info(f"Trace saved: {trace_path}")

        return result.answer_text, result


def create_agent(
    session_id: str = "default",
    model: str = None,
    verbose: bool = False,
    enable_grader: bool = True,
    **kwargs,
) -> GenshinRetrievalAgent:
    """Factory function to create a GenshinRetrievalAgent."""
    return GenshinRetrievalAgent(
        session_id=session_id,
        model=model,
        verbose=verbose,
        enable_grader=enable_grader,
    )
