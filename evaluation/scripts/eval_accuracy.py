"""Golden Answer Accuracy Evaluation.

独立 CLI 脚本：读取 agent eval 结果或 trace 文件，与 golden answer 对比，
用 LLM-as-judge 打 correctness + coverage 分数，输出报告到 evaluation/results/.

Usage:
    .venv/bin/python evaluation/scripts/eval_accuracy.py \
        --from-eval logger/eval/tier1_*.json logger/eval/tier2_*.json
    .venv/bin/python evaluation/scripts/eval_accuracy.py \
        --from-traces logger/traces/
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for name in ("httpx", "httpcore", "google", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "reference" / "dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"

# ── LLM Prompt ────────────────────────────────────────────────────

ACCURACY_PROMPT = """你是一个答案准确度评估器。对比 Agent 的回答与标准答案，评估准确度。

## 用户问题
{question}

## 标准答案 (Golden Answer)
{golden_answer}

## Agent 回答
{agent_answer}

## 评估维度

### Correctness (事实正确性)
Agent 回答中的核心事实是否与标准答案一致？
- 1.0: 核心事实完全正确，无矛盾
- 0.8: 核心事实正确，有少量非关键细节偏差
- 0.5: 部分正确，部分关键信息错误或矛盾
- 0.2: 大部分信息错误或与标准答案矛盾
- 0.0: 完全错误或答非所问

### Coverage (要点覆盖率)
Agent 回答覆盖了标准答案中多少关键要点？
- 1.0: 覆盖所有关键要点
- 0.8: 覆盖大部分要点，遗漏少量细节
- 0.5: 覆盖约一半要点
- 0.2: 只覆盖少量要点
- 0.0: 未覆盖任何要点

## 输出格式

严格返回 JSON (不要其他文字):
```json
{{
  "correctness": 0.0,
  "coverage": 0.0,
  "reasoning": "评分理由"
}}
```"""


# ── Data Loading ──────────────────────────────────────────────────

def load_golden_dataset(path: Path) -> dict[str, dict]:
    """Load golden dataset, return {id: record} map."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q for q in data["dataset"]}


def load_eval_files(paths: list[Path]) -> list[dict]:
    """Load agent results from eval JSON files."""
    results = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        results.extend(data.get("results", []))
    return results


def load_traces(trace_dir: Path) -> list[dict]:
    """Load agent results from trace JSON files.

    Traces don't have a question ID, so we match by question text later.
    """
    results = []
    for p in sorted(trace_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            trace = json.load(f)
        results.append({
            "id": None,  # no ID in traces; matched by question text
            "question": trace.get("query", ""),
            "complexity": None,
            "answer": trace.get("humanized_response") or trace.get("final_response", ""),
            "faithfulness": (trace.get("final_grading") or {}).get("faithfulness"),
            "answer_relevance": (trace.get("final_grading") or {}).get("answer_relevance"),
            "passed": trace.get("passed"),
        })
    return results


def match_results_to_golden(
    agent_results: list[dict],
    golden: dict[str, dict],
) -> list[dict]:
    """Match agent results to golden answers by ID.

    For trace results without ID, fallback to question text matching.
    Returns list of matched pairs.
    """
    # Build question text → id map for fallback
    question_to_id = {q["question"]: qid for qid, q in golden.items()}

    matched = []
    for r in agent_results:
        qid = r.get("id")
        if not qid and r.get("question"):
            qid = question_to_id.get(r["question"])
        if not qid or qid not in golden:
            logger.warning(f"Unmatched result: id={r.get('id')}, q={r.get('question', '')[:40]}...")
            continue
        g = golden[qid]
        matched.append({
            "id": qid,
            "complexity": r.get("complexity") or g.get("complexity"),
            "question": g["question"],
            "golden_answer": g["golden_answer"],
            "agent_answer": r["answer"],
            "agent_self_grading": {
                "faithfulness": r.get("faithfulness"),
                "answer_relevance": r.get("answer_relevance"),
                "passed": r.get("passed"),
            },
        })
    return matched


# ── LLM Grading ───────────────────────────────────────────────────

def _init_llm():
    """Initialize grader LLM (same pattern as src/agent/agent.py)."""
    from dotenv import load_dotenv
    load_dotenv()

    if not os.environ.get("GOOGLE_API_KEY"):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key

    from src.common.config import Settings
    settings = Settings()

    from llama_index.llms.google_genai import GoogleGenAI
    llm = GoogleGenAI(
        model=settings.GRADER_MODEL,
        is_function_calling_model=False,
    )
    return llm, settings.GRADER_MODEL


def _parse_score_json(text: str) -> dict:
    """Extract JSON from LLM response text."""
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start < 0 or json_end <= json_start:
        return {}
    try:
        return json.loads(text[json_start:json_end])
    except json.JSONDecodeError:
        return {}


async def grade_one(llm, item: dict) -> dict:
    """Grade a single (agent_answer, golden_answer) pair."""
    prompt = ACCURACY_PROMPT.format(
        question=item["question"],
        golden_answer=item["golden_answer"],
        agent_answer=item["agent_answer"],
    )
    try:
        response = await llm.acomplete(prompt)
        data = _parse_score_json(str(response))
        item["correctness"] = float(data.get("correctness", 0.0))
        item["coverage"] = float(data.get("coverage", 0.0))
        item["reasoning"] = data.get("reasoning", "")
    except Exception as e:
        logger.error(f"Grading failed for {item['id']}: {e}")
        item["correctness"] = 0.0
        item["coverage"] = 0.0
        item["reasoning"] = f"ERROR: {e}"
    return item


async def grade_all(items: list[dict]) -> list[dict]:
    """Grade all items concurrently."""
    llm, model_name = _init_llm()
    logger.info(f"Grading {len(items)} questions with {model_name}")
    tasks = [grade_one(llm, item) for item in items]
    return await asyncio.gather(*tasks)


# ── Report Generation ─────────────────────────────────────────────

def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate statistics."""
    if not results:
        return {"avg_correctness": 0.0, "avg_coverage": 0.0, "by_complexity": {}}

    avg_c = sum(r["correctness"] for r in results) / len(results)
    avg_v = sum(r["coverage"] for r in results) / len(results)

    by_complexity = defaultdict(list)
    for r in results:
        by_complexity[r["complexity"]].append(r)

    complexity_stats = {}
    for c in sorted(by_complexity):
        group = by_complexity[c]
        complexity_stats[str(c)] = {
            "count": len(group),
            "avg_correctness": round(sum(r["correctness"] for r in group) / len(group), 4),
            "avg_coverage": round(sum(r["coverage"] for r in group) / len(group), 4),
        }

    return {
        "avg_correctness": round(avg_c, 4),
        "avg_coverage": round(avg_v, 4),
        "by_complexity": complexity_stats,
    }


def save_report(results: list[dict], source_names: list[str], dataset_path: str, model: str) -> Path:
    """Save JSON report to evaluation/results/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"accuracy_{ts}.json"

    # Sort results by ID for readability
    results.sort(key=lambda r: r["id"])

    report = {
        "timestamp": ts,
        "dataset": str(dataset_path),
        "source": source_names,
        "model": model,
        "total": len(results),
        "summary": compute_summary(results),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def print_report(results: list[dict], source_names: list[str], golden_total: int):
    """Print summary table to terminal."""
    results.sort(key=lambda r: r["id"])

    print(f"\n{'='*80}")
    print("Agent Accuracy Report")
    print(f"{'='*80}")
    print(f"Source: {', '.join(source_names)}")
    print(f"Matched: {len(results)}/{golden_total} questions")
    print()

    # Header
    header = f"{'ID':<8}{'Cmplx':>5}{'Correct':>9}{'Cover':>8}{'Faith':>8}{'Relev':>8}  Reasoning"
    print(header)
    print("-" * len(header) + "-" * 30)

    for r in results:
        sg = r.get("agent_self_grading", {})
        faith = sg.get("faithfulness")
        relev = sg.get("answer_relevance")
        faith_s = f"{faith:.2f}" if faith is not None else "  -"
        relev_s = f"{relev:.2f}" if relev is not None else "  -"
        reasoning = (r.get("reasoning") or "")[:50]
        print(
            f"{r['id']:<8}{r['complexity']:>5}"
            f"{r['correctness']:>9.2f}{r['coverage']:>8.2f}"
            f"{faith_s:>8}{relev_s:>8}"
            f"  {reasoning}"
        )

    # Summary by complexity
    summary = compute_summary(results)
    print(f"\nBy Complexity:")
    for c, stats in summary["by_complexity"].items():
        print(f"  Complexity {c} ({stats['count']} questions): "
              f"correctness={stats['avg_correctness']:.2f}, coverage={stats['avg_coverage']:.2f}")

    print(f"\nOverall:")
    print(f"  Correctness: {summary['avg_correctness']:.2f}")
    print(f"  Coverage:    {summary['avg_coverage']:.2f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate agent accuracy against golden answers"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-eval", nargs="+", type=Path, metavar="FILE",
        help="Eval result JSON files (logger/eval/tier*.json)",
    )
    source.add_argument(
        "--from-traces", type=Path, metavar="DIR",
        help="Trace directory (logger/traces/)",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help=f"Golden dataset path (default: {DEFAULT_DATASET.relative_to(PROJECT_ROOT)})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load golden dataset
    golden = load_golden_dataset(args.dataset)
    logger.info(f"Loaded {len(golden)} golden answers from {args.dataset}")

    # Load agent results
    if args.from_eval:
        agent_results = load_eval_files(args.from_eval)
        source_names = [p.name for p in args.from_eval]
    else:
        agent_results = load_traces(args.from_traces)
        source_names = [args.from_traces.name]
    logger.info(f"Loaded {len(agent_results)} agent results")

    # Match
    matched = match_results_to_golden(agent_results, golden)
    logger.info(f"Matched {len(matched)}/{len(agent_results)} results to golden answers")

    if not matched:
        logger.error("No matched results. Check IDs in eval files vs golden dataset.")
        sys.exit(1)

    # Grade
    graded = asyncio.run(grade_all(matched))

    # Report
    _, model_name = _init_llm()
    report_path = save_report(graded, source_names, str(args.dataset), model_name)
    logger.info(f"Report saved to {report_path}")

    print_report(graded, source_names, len(golden))


if __name__ == "__main__":
    main()
