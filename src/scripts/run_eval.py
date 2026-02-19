"""Parallel eval runner — run testsets with ProcessPoolExecutor.

Usage:
    .venv/bin/python -m src.scripts.run_eval --testset smoke
    .venv/bin/python -m src.scripts.run_eval --testset basic --workers 3
    .venv/bin/python -m src.scripts.run_eval --ids NK-01 NK-07 NK-15
    .venv/bin/python -m src.scripts.run_eval --testset full --workers 1
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "TestData" / "cleandata" / "Data_only_nodkrai"
QUESTIONS_DIR = DATASET_DIR / "questions"
TESTSETS_DIR = DATASET_DIR / "testsets"
LOGGER_DIR = PROJECT_ROOT / "logger"
EVAL_OUTPUT_DIR = LOGGER_DIR / "eval"
TRACES_DIR = LOGGER_DIR / "traces"
BACKUP_DIR = LOGGER_DIR / "backup"


def _setup_logging():
    """Configure logging for the main process."""
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    for name in ("httpx", "httpcore", "google", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ── Worker function (runs in subprocess) ─────────────────────────

def _run_one_question(question: dict) -> dict:
    """Run a single question in an isolated subprocess.

    Creates its own agent instance, runs the question with grading,
    and returns a result dict.
    """
    import asyncio
    import logging
    import os
    import time
    import warnings

    # Suppress third-party UserWarnings (jieba/pkg_resources, transformers tokenizer)
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message="You're using a.*tokenizer.*fast tokenizer")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Suppress noisy loggers in worker
    for name in ("httpx", "httpcore", "google", "urllib3", "jieba", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)

    qid = question["id"]
    start = time.time()

    try:
        from src.agent import create_agent

        agent = create_agent(enable_grader=True)
        answer, response = asyncio.run(
            agent.run_with_grading(question["question"])
        )
        duration_ms = int((time.time() - start) * 1000)

        return {
            "id": qid,
            "complexity": question.get("complexity", 0),
            "answer": answer[:2000],
            "faithfulness": response.grading.faithfulness,
            "completeness": response.grading.completeness,
            "answer_relevance": response.grading.answer_relevance,
            "passed": response.grading.passed,
            "query_count": response.query_count,
            "tool_calls": len(response.tool_calls),
            "duration_ms": duration_ms,
            "error": None,
        }
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        return {
            "id": qid,
            "complexity": question.get("complexity", 0),
            "answer": "",
            "faithfulness": 0.0,
            "completeness": 0.0,
            "answer_relevance": 0.0,
            "passed": False,
            "query_count": 0,
            "tool_calls": 0,
            "duration_ms": duration_ms,
            "error": str(e),
        }


# ── Data loading ─────────────────────────────────────────────────

def load_testset(name: str) -> list[str]:
    """Load question IDs from a testset file."""
    path = TESTSETS_DIR / f"{name}.json"
    if not path.exists():
        logger.error(f"Testset not found: {path}")
        logger.info(f"Available: {[f.stem for f in TESTSETS_DIR.glob('*.json')]}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["ids"]


def load_questions(ids: list[str]) -> list[dict]:
    """Load individual question files by ID."""
    questions = []
    for qid in ids:
        path = QUESTIONS_DIR / f"{qid}.json"
        if not path.exists():
            logger.warning(f"Question file not found: {path}, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            questions.append(json.load(f))
    return questions


# ── Result output ────────────────────────────────────────────────

def write_results(label: str, results: list[dict]):
    """Write results JSON compatible with eval pipeline."""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = EVAL_OUTPUT_DIR / f"{label}_{ts}.json"
    payload = {
        "label": label,
        "timestamp": ts,
        "count": len(results),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Results written: {path}")
    return path


def print_summary(results: list[dict], total_time: float):
    """Print a summary table to stdout."""
    print(f"\n{'='*80}")
    print(f"{'ID':<8} {'Cmplx':>5} {'Faith':>6} {'Comp':>6} {'Relev':>6} {'Pass':>5} {'Tools':>5} {'Time':>8} {'Err':>4}")
    print(f"{'-'*80}")

    passed = 0
    for r in sorted(results, key=lambda x: x["id"]):
        err = "ERR" if r.get("error") else ""
        p = "Y" if r["passed"] else "N"
        if r["passed"]:
            passed += 1
        dur = f"{r['duration_ms']/1000:.0f}s"
        print(
            f"{r['id']:<8} {r['complexity']:>5} "
            f"{r['faithfulness']:>6.2f} {r['completeness']:>6.2f} {r['answer_relevance']:>6.2f} "
            f"{p:>5} {r['tool_calls']:>5} {dur:>8} {err:>4}"
        )

    print(f"{'='*80}")
    print(f"Total: {len(results)} questions, {passed} passed, {total_time:.0f}s elapsed")
    print(f"{'='*80}")


# ── Archive previous logs ─────────────────────────────────────────

def _archive_previous_logs():
    """Move existing traces/ and eval/ contents to backup/{timestamp}/."""
    has_traces = TRACES_DIR.exists() and any(TRACES_DIR.iterdir())
    has_eval = EVAL_OUTPUT_DIR.exists() and any(EVAL_OUTPUT_DIR.iterdir())
    if not has_traces and not has_eval:
        return

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = BACKUP_DIR / ts
    dest.mkdir(parents=True, exist_ok=True)

    moved = 0
    for src_dir in (TRACES_DIR, EVAL_OUTPUT_DIR):
        if not src_dir.exists():
            continue
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.move(str(f), str(dest / f.name))
                moved += 1

    if moved:
        logger.info(f"Archived {moved} previous log files → {dest}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel eval runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--testset", help="Testset name (e.g. smoke, basic, full)")
    group.add_argument("--ids", nargs="+", help="Question IDs (e.g. NK-01 NK-07)")
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of parallel workers (default: 2)"
    )
    args = parser.parse_args()

    _setup_logging()
    _archive_previous_logs()

    # Load questions
    if args.testset:
        ids = load_testset(args.testset)
        label = args.testset
    else:
        ids = args.ids
        label = "custom"

    questions = load_questions(ids)
    if not questions:
        logger.error("No questions loaded")
        sys.exit(1)

    workers = min(args.workers, len(questions))
    logger.info(f"Running {len(questions)} questions with {workers} workers")

    # Execute
    results = []
    start_time = time.time()

    if workers <= 1:
        # Serial execution
        for i, q in enumerate(questions, 1):
            logger.info(f"[{i}/{len(questions)}] Running {q['id']}...")
            result = _run_one_question(q)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            logger.info(f"[{i}/{len(questions)}] {q['id']} → {status} ({result['duration_ms']/1000:.0f}s)")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_id = {}
            for q in questions:
                future = pool.submit(_run_one_question, q)
                future_to_id[future] = q["id"]

            completed = 0
            for future in as_completed(future_to_id):
                completed += 1
                qid = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "PASS" if result["passed"] else "FAIL"
                    err = f" [ERROR: {result['error'][:50]}]" if result.get("error") else ""
                    logger.info(
                        f"[{completed}/{len(questions)}] {qid} → {status} "
                        f"({result['duration_ms']/1000:.0f}s){err}"
                    )
                except Exception as e:
                    logger.error(f"[{completed}/{len(questions)}] {qid} → CRASH: {e}")
                    results.append({
                        "id": qid, "complexity": 0, "answer": "",
                        "faithfulness": 0, "completeness": 0, "answer_relevance": 0,
                        "passed": False, "query_count": 0, "tool_calls": 0,
                        "duration_ms": 0, "error": str(e),
                    })

    total_time = time.time() - start_time

    # Output
    print_summary(results, total_time)
    write_results(label, results)


if __name__ == "__main__":
    main()
