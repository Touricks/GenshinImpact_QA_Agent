"""Agent v4 Workflow — LangGraph StateGraph ReAct loop.

替代手动 while 循环，使用 LangGraph StateGraph 实现 ReAct 工作流。
Graph: START → solve_llm ↔ solve_tools → humanize → grade → END

LLM 策略: ReAct solve → REASONING_MODEL, humanize + grade → GRADER_MODEL.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from operator import add
from typing import Annotated, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .models import (
    ToolCallRecord,
    AgentResponse,
    GradingResult,
)
from .settings import agent_settings
from .grader import ProductionGraderService
from .combiner import ResponseHumanizer

logger = logging.getLogger(__name__)


def _extract_text_content(content) -> str:
    """Extract text from AIMessage.content (handles str or list[dict] with thinking blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # Skip "thinking" blocks — they are internal reasoning
        return "".join(parts)
    return str(content) if content else ""


# ── State ──────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """LangGraph state for v4 ReAct workflow."""

    # Core
    messages: Annotated[list, add_messages]
    question: str
    qe_id: str

    # Solve tracking
    tool_calls: Annotated[list[ToolCallRecord], add]  # append via reducer
    total_tool_call_count: int       # overwrite (last-write-wins)
    solve_iterations: int            # overwrite
    solve_start_ts: float            # timestamp of first solve_llm call

    # Eval
    golden_answer: str  # empty string = production mode

    # Output
    raw_answer: str
    humanized_answer: str
    grading: Optional[GradingResult]


# ── Workflow ───────────────────────────────────────────────────────────


class AgentV4Workflow:
    """v4 LangGraph StateGraph ReAct workflow.

    Graph: START → solve_llm ↔ solve_tools → humanize → grade → END
    """

    def __init__(
        self,
        reasoning_llm,
        grader_llm,
        tools: list,
        tracer=None,
        **kwargs,
    ):
        self.reasoning_llm = reasoning_llm
        self.grader_llm = grader_llm
        self.tracer = tracer

        # Tools are pre-decorated with @tool — use directly
        self.tool_map = {t.name: t for t in tools}
        self.llm_with_tools = reasoning_llm.bind_tools(tools)

        self.humanizer = ResponseHumanizer(grader_llm)
        self.final_grader = ProductionGraderService(grader_llm)

        self.graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("solve_llm", self._solve_llm_node)
        graph.add_node("solve_tools", self._solve_tools_node)
        graph.add_node("force_answer", self._force_answer_node)
        graph.add_node("humanize", self._humanize_node)
        graph.add_node("grade", self._grade_node)

        graph.add_edge(START, "solve_llm")
        graph.add_conditional_edges("solve_llm", self._should_continue, {
            "solve_tools": "solve_tools",
            "force_answer": "force_answer",
            "humanize": "humanize",
        })
        graph.add_edge("solve_tools", "solve_llm")
        graph.add_edge("force_answer", "humanize")
        graph.add_edge("humanize", "grade")
        graph.add_edge("grade", END)

        return graph.compile(checkpointer=MemorySaver())

    # ── Nodes ─────────────────────────────────────────────────────

    async def _solve_llm_node(self, state: AgentState) -> dict:
        solve_start = state.get("solve_start_ts") or time.time()
        iteration = state.get("solve_iterations", 0) + 1
        tool_count = state.get("total_tool_call_count", 0)

        llm_start = time.time()
        response: AIMessage = await self.llm_with_tools.ainvoke(state["messages"])
        llm_ms = int((time.time() - llm_start) * 1000)
        wall_ms = int((time.time() - solve_start) * 1000)

        tc_names = [tc["name"] for tc in (response.tool_calls or [])]
        tc_summary = ", ".join(tc_names) if tc_names else "(final answer)"
        logger.info(
            f"[v4:LLM] iter={iteration} llm={llm_ms}ms wall={wall_ms}ms "
            f"tools_so_far={tool_count} → {tc_summary}"
        )

        if self.tracer:
            self.tracer.log_iteration(
                state["qe_id"], iteration, llm_ms, wall_ms,
                tc_names, tool_count,
            )

        return {
            "messages": [response],
            "solve_iterations": iteration,
            "solve_start_ts": solve_start,
        }

    def _should_continue(self, state: AgentState) -> str:
        last_msg = state["messages"][-1]

        if not last_msg.tool_calls:
            return "humanize"

        max_tool_calls = agent_settings.v4_max_tool_calls
        max_iterations = agent_settings.v4_max_iterations

        if state.get("total_tool_call_count", 0) >= max_tool_calls:
            return "force_answer"
        if state.get("solve_iterations", 0) >= max_iterations:
            return "force_answer"

        return "solve_tools"

    async def _solve_tools_node(self, state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        qe_id = state["qe_id"]
        iteration = state.get("solve_iterations", 0)
        max_tool_calls = agent_settings.v4_max_tool_calls
        per_iter_limit = agent_settings.v4_max_tools_per_iter
        total_count = state.get("total_tool_call_count", 0)

        new_records: list[ToolCallRecord] = []
        new_messages: list[ToolMessage] = []
        limit_reached = False
        iter_count = 0

        for tc in last_msg.tool_calls:
            if total_count >= max_tool_calls:
                logger.warning(f"[v4:{qe_id}] tool call limit reached ({max_tool_calls})")
                limit_reached = True
                break
            if iter_count >= per_iter_limit:
                logger.info(f"[v4:{qe_id}] per-iter limit reached ({per_iter_limit}), deferring remaining")
                limit_reached = True
                break

            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            lc_tool = self.tool_map.get(tool_name)
            if not lc_tool:
                tool_output = f"Error: unknown tool '{tool_name}'"
                tc_ms = 0
            else:
                tc_start = time.time()
                try:
                    tool_output = str(lc_tool.invoke(tool_args))
                except Exception as e:
                    tool_output = f"Error: {e}"
                tc_ms = int((time.time() - tc_start) * 1000)

            logger.info(
                f"[v4:Tool] {tool_name}({tool_args}) → {tc_ms}ms, "
                f"output={len(tool_output)}ch"
            )

            total_count += 1
            iter_count += 1

            record = ToolCallRecord(
                tool_name=tool_name,
                input=tool_args,
                output=tool_output,
                duration_ms=tc_ms,
                query_event_id=qe_id,
            )
            record.output_display = record.truncated(agent_settings.tool_output_truncate)
            new_records.append(record)

            if self.tracer:
                self.tracer.log_query_event_tool_call(
                    qe_id, record.tool_name, record.input,
                    record.output_display or record.output, record.duration_ms,
                    iteration=iteration,
                )

            new_messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call_id,
            ))

        # Add placeholder ToolMessages for skipped calls (every tool_call_id needs a ToolMessage)
        if limit_reached:
            at_global_limit = total_count >= max_tool_calls
            executed_ids = {m.tool_call_id for m in new_messages}
            for tc in last_msg.tool_calls:
                if tc["id"] not in executed_ids:
                    if at_global_limit:
                        msg = f"工具调用已跳过：已达到全局上限 ({max_tool_calls})。请根据已有信息直接给出最终回答。"
                    else:
                        msg = f"本轮已执行 {per_iter_limit} 个工具，剩余调用已跳过。如仍需要此信息，请在下一轮重新调用。"
                    new_messages.append(ToolMessage(
                        content=msg,
                        tool_call_id=tc["id"],
                    ))

        if self.tracer:
            self.tracer.flush_partial()

        return {
            "messages": new_messages,
            "tool_calls": new_records,
            "total_tool_call_count": total_count,
        }

    async def _force_answer_node(self, state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        qe_id = state["qe_id"]

        # Add placeholder ToolMessages for all un-responded tool calls
        existing_ids = {
            m.tool_call_id for m in state["messages"]
            if isinstance(m, ToolMessage)
        }
        placeholders = []
        for tc in last_msg.tool_calls:
            if tc["id"] not in existing_ids:
                placeholders.append(ToolMessage(
                    content="工具调用已跳过：已达到上限。请根据已有信息直接给出最终回答。",
                    tool_call_id=tc["id"],
                ))

        logger.warning(
            f"[v4:{qe_id}] forcing answer "
            f"(iterations={state.get('solve_iterations', 0)}, "
            f"tool_calls={state.get('total_tool_call_count', 0)})"
        )

        # Call LLM without tools to force a text answer
        messages = list(state["messages"]) + placeholders
        solve_start = state.get("solve_start_ts") or time.time()
        llm_start = time.time()
        response: AIMessage = await self.reasoning_llm.ainvoke(messages)
        llm_ms = int((time.time() - llm_start) * 1000)
        wall_ms = int((time.time() - solve_start) * 1000)

        iteration = state.get("solve_iterations", 0) + 1
        logger.info(
            f"[v4:LLM] iter={iteration}(force) llm={llm_ms}ms wall={wall_ms}ms"
        )

        if self.tracer:
            self.tracer.log_iteration(
                qe_id, iteration, llm_ms, wall_ms,
                [], state.get("total_tool_call_count", 0),
                is_force_answer=True,
            )

        return {"messages": placeholders + [response]}

    async def _humanize_node(self, state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        raw_answer = _extract_text_content(last_msg.content)

        humanize_start = time.time()
        humanized = await self.humanizer.humanize(state["question"], raw_answer)
        humanize_ms = int((time.time() - humanize_start) * 1000)
        logger.info(f"[v4:Humanize] {len(raw_answer)}ch → {len(humanized)}ch ({humanize_ms}ms)")

        return {
            "raw_answer": raw_answer,
            "humanized_answer": humanized,
        }

    async def _grade_node(self, state: AgentState) -> dict:
        grade_start = time.time()
        final_grade = await self.final_grader.grade(
            question=state["question"],
            answer=state["humanized_answer"],
            tool_calls=state.get("tool_calls", []),
            golden_answer=state.get("golden_answer") or None,
        )
        grade_ms = int((time.time() - grade_start) * 1000)
        logger.info(
            f"[v4:Grade] faith={final_grade.faithfulness:.2f} "
            f"comp={final_grade.completeness:.2f} "
            f"total={final_grade.total:.2f} ({grade_ms}ms)"
        )

        grading = GradingResult(
            faithfulness=final_grade.faithfulness,
            completeness=final_grade.completeness,
            total=final_grade.total,
        )

        if self.tracer:
            self.tracer.log_final_grading({
                "faithfulness": grading.faithfulness,
                "completeness": grading.completeness,
                "total": grading.total,
                "duration_ms": grade_ms,
            })

        return {"grading": grading}

    # ── Orchestrator ──────────────────────────────────────────────

    async def run(self, question: str, golden_answer: str = "") -> AgentResponse:
        """Execute StateGraph: solve → humanize → grade → AgentResponse."""
        from .prompts import AGENT_V4_SYSTEM_PROMPT

        start_time = time.time()
        qe_id = str(uuid.uuid4())[:8]

        max_iterations = agent_settings.v4_max_iterations
        max_tool_calls = agent_settings.v4_max_tool_calls

        if self.tracer:
            self.tracer.start_trace(question, {
                "version": "v4-langgraph",
                "max_iterations": max_iterations,
                "max_tool_calls": max_tool_calls,
            }, golden_answer=golden_answer)
            self.tracer.log_routing("single", "v4 LangGraph StateGraph ReAct", 0)
            self.tracer.start_query_event(qe_id, question, "", False)

        system_prompt = AGENT_V4_SYSTEM_PROMPT.format(
            max_tool_calls=max_tool_calls,
        )

        from ..common.config.settings import settings as global_settings
        if global_settings.ABLATION_MODE == "vector_only":
            system_prompt += (
                "\n\n## [消融实验] 仅 search_memory 可用\n"
                "上述 KG-First 规则不适用。当前只有 search_memory（向量搜索）工具。"
                "请用不同关键词多次搜索以全面覆盖信息。"
            )

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ],
            "question": question,
            "qe_id": qe_id,
            "tool_calls": [],
            "total_tool_call_count": 0,
            "solve_iterations": 0,
            "solve_start_ts": 0.0,
            "golden_answer": golden_answer,
            "raw_answer": "",
            "humanized_answer": "",
            "grading": None,
        }

        # ── Invoke graph with retry ──
        MAX_RETRIES = 2
        wall_timeout = agent_settings.v4_wall_timeout
        final_state = None

        for retry in range(MAX_RETRIES + 1):
            try:
                config = {"configurable": {"thread_id": f"{qe_id}-{retry}"}}
                final_state = await asyncio.wait_for(
                    self.graph.ainvoke(initial_state, config),
                    timeout=wall_timeout,
                )

                answer = final_state.get("humanized_answer", "")
                tool_calls = final_state.get("tool_calls", [])

                if not answer and not tool_calls and retry < MAX_RETRIES:
                    wait = 2 ** retry
                    logger.warning(f"[v4:{qe_id}] empty response, retry {retry+1}/{MAX_RETRIES}")
                    await asyncio.sleep(wait)
                    continue
                break
            except asyncio.TimeoutError:
                logger.error(
                    f"[v4:{qe_id}] wall-clock timeout ({wall_timeout}s), "
                    f"retry={retry+1}/{MAX_RETRIES+1}"
                )
                final_state = {
                    **initial_state,
                    "humanized_answer": f"求解超时：已超过最大等待时间 ({wall_timeout}s)。",
                    "raw_answer": "求解超时",
                }
                break
            except Exception as e:
                err_str = str(e).lower()
                retryable = any(k in err_str for k in ["503", "500", "unavailable", "empty"])
                if retryable and retry < MAX_RETRIES:
                    wait = 2 ** retry
                    logger.warning(f"[v4:{qe_id}] {e}, retry {retry+1}/{MAX_RETRIES} in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"[v4:{qe_id}] failed after {retry+1} attempts: {e}")
                final_state = {
                    **initial_state,
                    "humanized_answer": f"求解失败: {e}",
                    "raw_answer": f"求解失败: {e}",
                }
                break

        # ── Extract results ──
        raw_answer = final_state.get("raw_answer", "")
        answer = final_state.get("humanized_answer", "") or raw_answer
        tool_calls = final_state.get("tool_calls", [])
        grading = final_state.get("grading") or GradingResult()

        total_duration = int((time.time() - start_time) * 1000)
        tool_output_chars = sum(len(tc.output) for tc in tool_calls)
        logger.info(
            f"[v4:Solve] done in {total_duration}ms, {len(tool_calls)} tool calls, "
            f"prompt={len(system_prompt)}ch, tool_output={tool_output_chars}ch, "
            f"answer={len(answer)}ch"
        )

        if self.tracer:
            self.tracer.end_query_event(qe_id, answer, {
                "solve_duration_ms": total_duration,
                "tool_call_count": len(tool_calls),
            })

        return AgentResponse(
            question=question,
            answer_text=answer,
            raw_answer=raw_answer,
            tool_calls=tool_calls,
            grading=grading,
            query_count=1,
            total_duration_ms=total_duration,
        )
