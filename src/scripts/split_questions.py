"""Split dataset.json into per-question files and generate testset definitions.

Usage:
    .venv/bin/python -m src.scripts.split_questions
    .venv/bin/python -m src.scripts.split_questions --dataset path/to/dataset.json
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    PROJECT_ROOT / "TestData" / "cleandata" / "Data_only_nodkrai" / "dataset.json"
)

# Default testset definitions: name -> {description, ids}
# "ids" can be:
#   - list of IDs: explicit list
#   - "all": all questions
#   - {"complexity": N}: filter by complexity
DEFAULT_TESTSETS = {
    "smoke": {
        "description": "Quick 3-question sanity check (complexity 1/2/5)",
        "ids": ["NK-01", "NK-07", "NK-15"],
    },
    "basic": {
        "description": "Tier-1 basic recall questions",
        "ids": {"complexity": 1},
    },
    "full": {
        "description": "All questions",
        "ids": "all",
    },
}


def split_questions(dataset_path: Path, outdir: Path) -> int:
    """Split dataset into individual question files."""
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    questions_dir = outdir / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for q in data["dataset"]:
        qid = q["id"]
        out_path = questions_dir / f"{qid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(q, f, ensure_ascii=False, indent=2)
        count += 1

    return count


def generate_testsets(dataset_path: Path, outdir: Path) -> int:
    """Generate testset definition files."""
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    all_questions = data["dataset"]
    all_ids = [q["id"] for q in all_questions]

    testsets_dir = outdir / "testsets"
    testsets_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for name, spec in DEFAULT_TESTSETS.items():
        # Resolve IDs
        ids_spec = spec["ids"]
        if ids_spec == "all":
            ids = all_ids
        elif isinstance(ids_spec, dict) and "complexity" in ids_spec:
            target = ids_spec["complexity"]
            ids = [q["id"] for q in all_questions if q["complexity"] == target]
        else:
            ids = ids_spec

        testset = {
            "name": name,
            "description": spec["description"],
            "count": len(ids),
            "ids": ids,
        }
        out_path = testsets_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(testset, f, ensure_ascii=False, indent=2)
        count += 1
        print(f"  testsets/{name}.json  ({len(ids)} questions)")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset.json into per-question files + testsets"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (defaults to same dir as dataset)",
    )
    args = parser.parse_args()

    outdir = args.outdir or args.dataset.parent

    print(f"Source: {args.dataset}")
    print(f"Output: {outdir}/")
    print()

    # Step 1: Split into individual question files
    print("Splitting questions...")
    q_count = split_questions(args.dataset, outdir)
    print(f"  questions/  ({q_count} files)")
    print()

    # Step 2: Generate testset definitions
    print("Generating testsets...")
    t_count = generate_testsets(args.dataset, outdir)
    print(f"\nDone: {q_count} question files, {t_count} testsets")


if __name__ == "__main__":
    main()
