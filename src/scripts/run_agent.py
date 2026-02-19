"""Agent Runner — clean CLI for testing the agent.

Usage:
    python -m src.scripts.run_agent "少女是谁？"              # simple
    python -m src.scripts.run_agent -g "少女是谁？"           # with grading
    python -m src.scripts.run_agent -i                        # interactive
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from src.agent import create_agent


def _setup_logging():
    """Configure logging with timestamp-based file + console."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"logger/run_agent_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    # Suppress noisy loggers
    for name in ("httpx", "httpcore", "google", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_file


async def _run_simple(query: str):
    """Run a single query, print the answer."""
    agent = create_agent(enable_grader=False)
    answer = await agent.run(query)
    print(answer)


async def _run_grading(query: str):
    """Run with grading, print answer + metrics."""
    agent = create_agent(enable_grader=True)
    answer, response = await agent.run_with_grading(query)

    print(f"\n{'='*60}")
    print(f"Question : {query}")
    print(f"{'='*60}")
    print(f"Answer   : {answer}")
    print(f"{'─'*60}")
    print(f"Faithfulness     : {response.grading.faithfulness:.2f}")
    print(f"Completeness     : {response.grading.completeness:.2f}")
    print(f"Total            : {response.grading.total:.2f}")
    print(f"Query Count      : {response.query_count}")
    print(f"Tool Calls       : {len(response.tool_calls)}")
    print(f"Duration         : {response.total_duration_ms}ms")
    print(f"{'='*60}")


async def _run_interactive():
    """Interactive REPL loop."""
    agent = create_agent(enable_grader=False)
    print("v4 Agent Interactive Mode (type /quit to exit)")
    print("=" * 60)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query == "/quit":
            break
        answer = await agent.run(query)
        print(f"\n{answer}")


def main():
    parser = argparse.ArgumentParser(description="Agent Runner (v4)")
    parser.add_argument("query", nargs="?", help="Question to ask the agent")
    parser.add_argument("-g", "--grading", action="store_true", help="Enable grading mode")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive REPL")
    args = parser.parse_args()

    log_file = _setup_logging()
    logging.getLogger(__name__).info(f"Log file: {log_file}")

    if args.interactive:
        asyncio.run(_run_interactive())
    elif args.query:
        if args.grading:
            asyncio.run(_run_grading(args.query))
        else:
            asyncio.run(_run_simple(args.query))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
