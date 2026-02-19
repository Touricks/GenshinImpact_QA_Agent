"""Agent Tracer — query_events[] based trace logging.

记录完整的 Workflow 执行链:
routing → query_events[] → final_grading
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AgentTracer:
    """Trace 记录器."""

    def __init__(self, log_dir: str = "logger/traces"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace: Optional[Dict[str, Any]] = None

    def start_trace(self, query: str, config: Dict[str, Any]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        query_hash = hashlib.md5(query.encode()).hexdigest()[:6]
        trace_id = f"{timestamp}-{query_hash}"

        self.current_trace = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "config": config,
            "routing": None,
            "query_events": [],
            "final_grading": None,
            "final_response": None,
            "query_count": 0,
            "total_duration_ms": 0,
            "passed": False,
        }
        logger.info(f"[Tracer] Started trace: {trace_id}")
        return trace_id

    def log_routing(self, topology: str, reasoning: str, duration_ms: int = 0):
        if self.current_trace is None:
            return
        self.current_trace["routing"] = {
            "topology": topology,
            "reasoning": reasoning,
            "duration_ms": duration_ms,
        }

    def start_query_event(
        self,
        qe_id: str,
        question: str,
        context: str = "",
        is_refinement: bool = False,
    ):
        if self.current_trace is None:
            return
        qe = {
            "id": qe_id,
            "question": question,
            "context": context[:2000] if context else "",
            "is_refinement": is_refinement,
            "iterations": [],
            "tool_calls": [],
            "answer": None,
            "grading": None,
            "start_time": datetime.now().isoformat(),
            "total_duration_ms": 0,
        }
        self.current_trace["query_events"].append(qe)

    def log_iteration(
        self,
        qe_id: str,
        iteration: int,
        llm_ms: int,
        wall_ms: int,
        tools_requested: list[str],
        tools_so_far: int,
        is_force_answer: bool = False,
    ):
        """记录一轮 ReAct iteration 的 LLM 调用数据。"""
        if self.current_trace is None:
            return
        for qe in self.current_trace["query_events"]:
            if qe["id"] == qe_id:
                qe["iterations"].append({
                    "iteration": iteration,
                    "llm_duration_ms": llm_ms,
                    "wall_duration_ms": wall_ms,
                    "tools_requested": tools_requested,
                    "tools_so_far": tools_so_far,
                    "is_force_answer": is_force_answer,
                })
                break

    def log_query_event_tool_call(
        self,
        qe_id: str,
        tool: str,
        input_data: Dict[str, Any],
        output: str,
        duration_ms: int = 0,
        iteration: int = 0,
    ):
        if self.current_trace is None:
            return
        for qe in self.current_trace["query_events"]:
            if qe["id"] == qe_id:
                entry = {
                    "tool": tool,
                    "input": input_data,
                    "output": output[:6000],
                    "duration_ms": duration_ms,
                }
                if iteration:
                    entry["iteration"] = iteration
                qe["tool_calls"].append(entry)
                break

    def end_query_event(self, qe_id: str, answer: str, grading: Dict[str, Any] = None):
        if self.current_trace is None:
            return
        for qe in self.current_trace["query_events"]:
            if qe["id"] == qe_id:
                qe["answer"] = answer[:2000]
                qe["grading"] = grading
                qe["end_time"] = datetime.now().isoformat()
                break

    def flush_partial(self):
        """写入当前 trace 的快照（不清空 current_trace），用于超时/崩溃后事后分析。"""
        if self.current_trace is None:
            return
        trace_id = self.current_trace["trace_id"]
        filepath = self.log_dir / f"{trace_id}.partial.json"
        snapshot = {**self.current_trace, "status": "in_progress"}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def _cleanup_partial(self):
        """删除当前 trace 的 partial 文件（正常完成后调用）。"""
        if self.current_trace is None:
            return
        trace_id = self.current_trace["trace_id"]
        partial = self.log_dir / f"{trace_id}.partial.json"
        if partial.exists():
            partial.unlink()

    def log_final_grading(self, grading: Dict[str, Any]):
        if self.current_trace is None:
            return
        self.current_trace["final_grading"] = grading

    def end_trace(
        self,
        final_response: str,
        passed: bool,
        total_duration_ms: int,
        humanized_response: str = None,
    ) -> str:
        if self.current_trace is None:
            return ""

        self.current_trace["final_response"] = final_response
        self.current_trace["humanized_response"] = humanized_response
        self.current_trace["passed"] = passed
        self.current_trace["total_duration_ms"] = total_duration_ms
        self.current_trace["query_count"] = len(self.current_trace["query_events"])
        self.current_trace["end_timestamp"] = datetime.now().isoformat()

        self._cleanup_partial()

        trace_id = self.current_trace["trace_id"]
        filepath = self.log_dir / f"{trace_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.current_trace, f, ensure_ascii=False, indent=2)

        logger.info(f"[Tracer] Trace saved: {filepath} (passed={passed}, duration={total_duration_ms}ms)")
        self.current_trace = None
        return str(filepath)
