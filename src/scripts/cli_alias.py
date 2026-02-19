"""CLI for managing alias entries.

Usage:
    python -m src.scripts.cli_alias add --canonical-zh "秘所" --type Place --region nodkrai
    python -m src.scripts.cli_alias from-review docs_internal/human_evaluate/xxx.md --region nodkrai
    python -m src.scripts.cli_alias merge --region nodkrai
    python -m src.scripts.cli_alias validate --region nodkrai
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from src.common.config.merge_aliases import (
    TYPE_FILES,
    TYPE_TO_FILE,
    GLOBAL_KEYS,
    merge_aliases,
    merge_and_write,
    _aliases_dir,
    _load_type_file,
)
from src.common.config.settings import settings

VALID_TYPES = list(TYPE_TO_FILE.keys())


def _write_type_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def add_alias(
    canonical_zh: str,
    entity_type: str,
    region: str,
    canonical_en: str = "",
    aliases: Optional[List[str]] = None,
    note: str = "",
    is_global: bool = False,
) -> Path:
    """Add a single alias entry to the appropriate split file, then merge.

    Returns the path of the modified split file.
    """
    if entity_type not in TYPE_TO_FILE:
        raise ValueError(f"Invalid type '{entity_type}'. Must be one of: {VALID_TYPES}")

    file_stem = TYPE_TO_FILE[entity_type]
    scope = "_global" if is_global else region
    file_path = _aliases_dir() / scope / f"{file_stem}.json"

    data = _load_type_file(file_path)

    if canonical_zh in data:
        raise ValueError(f"Key '{canonical_zh}' already exists in {file_path}")

    if aliases is None:
        aliases = [canonical_zh]

    entry = {
        "canonical_zh": canonical_zh,
        "canonical_en": canonical_en,
        "aliases": aliases,
        "type": entity_type,
        "note": note,
    }

    data[canonical_zh] = entry
    _write_type_file(file_path, data)
    print(f"Added '{canonical_zh}' -> {file_path.relative_to(_aliases_dir())}")

    # Auto-merge
    merge_and_write(region)
    return file_path


def parse_review_table(md_path: Path) -> list[dict]:
    """Parse the human_evaluate markdown table, extracting '待审核' entries.

    Returns list of dicts with keys: entity, type, source, description.
    """
    text = md_path.read_text(encoding="utf-8")
    entries = []

    # Match table rows: | `entity` | Type | source | description | **待审核** |
    pattern = re.compile(
        r"^\|\s*`([^`]+)`\s*\|"   # entity (backtick-wrapped)
        r"\s*(\w+)\s*\|"           # type
        r"\s*([^|]*?)\s*\|"        # source file
        r"\s*([^|]*?)\s*\|"        # description
        r"\s*\*\*待审核\*\*\s*\|", # disposition
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        entries.append({
            "entity": m.group(1).strip(),
            "type": m.group(2).strip(),
            "source": m.group(3).strip(),
            "description": m.group(4).strip(),
        })
    return entries


def cmd_from_review(md_path: Path, region: str) -> None:
    """Interactive import from a human review report."""
    entries = parse_review_table(md_path)
    if not entries:
        print("No '待审核' entries found in the review file.")
        return

    added = 0
    for i, e in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}] {e['entity']} ({e['type']}): {e['description']}")
        choice = input("  [a]dd / [s]kip / [e]dit: ").strip().lower()

        if choice == "s":
            print("  -> Skipped")
            continue

        canonical_zh = e["entity"]
        entity_type = e["type"]
        note = e["description"]
        aliases_list = [canonical_zh]

        if choice == "e":
            new_name = input(f"  canonical_zh [{canonical_zh}]: ").strip()
            if new_name:
                canonical_zh = new_name
            new_type = input(f"  type [{entity_type}]: ").strip()
            if new_type:
                entity_type = new_type
            new_note = input(f"  note [{note}]: ").strip()
            if new_note:
                note = new_note
            extra = input(f"  extra aliases (comma-sep) []: ").strip()
            if extra:
                aliases_list.extend(a.strip() for a in extra.split(",") if a.strip())

        canonical_en = input(f"  canonical_en []: ").strip()

        try:
            add_alias(
                canonical_zh=canonical_zh,
                entity_type=entity_type,
                region=region,
                canonical_en=canonical_en,
                aliases=aliases_list,
                note=note,
                is_global=False,
            )
            added += 1
        except ValueError as err:
            print(f"  ERROR: {err}")

    if added:
        merge_and_write(region)
    print(f"\nDone: {added} added, {len(entries) - added} skipped.")


def cmd_validate(region: str) -> None:
    """Validate alias files for the given region."""
    aliases_dir = _aliases_dir()
    errors: list[str] = []
    all_keys: dict[str, str] = {}  # key -> source

    for scope in ("_global", region):
        scope_dir = aliases_dir / scope
        for type_name in TYPE_FILES:
            path = scope_dir / f"{type_name}.json"
            data = _load_type_file(path)
            source = f"{scope}/{type_name}.json"

            for key, entry in data.items():
                # Check required fields
                if not entry.get("canonical_zh"):
                    errors.append(f"{source}: key '{key}' missing canonical_zh")
                if not entry.get("type"):
                    errors.append(f"{source}: key '{key}' missing type")

                # Check type matches file
                expected_file = TYPE_TO_FILE.get(entry.get("type", ""))
                if expected_file and expected_file != type_name:
                    errors.append(
                        f"{source}: key '{key}' has type '{entry['type']}' "
                        f"but is in {type_name}.json (expected {expected_file}.json)"
                    )

                # Check cross-file duplicate keys
                if key in all_keys:
                    errors.append(
                        f"Duplicate key '{key}': in {all_keys[key]} and {source}"
                    )
                all_keys[key] = source

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"Validation OK: {len(all_keys)} entries across _global + {region}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cli_alias",
        description="Manage alias entries for entity normalization",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- add ---
    p_add = sub.add_parser("add", help="Add a single alias entry")
    p_add.add_argument("--canonical-zh", required=True)
    p_add.add_argument("--canonical-en", default="")
    p_add.add_argument("--type", required=True, choices=VALID_TYPES, dest="entity_type")
    p_add.add_argument("--aliases", nargs="*", default=None,
                        help="Alias strings (default: [canonical_zh])")
    p_add.add_argument("--note", default="")
    p_add.add_argument("--region", default=settings.ACTIVE_REGION)
    p_add.add_argument("--global", action="store_true", dest="is_global",
                        help="Write to _global/ instead of region/")

    # --- from-review ---
    p_review = sub.add_parser("from-review", help="Import from human review report")
    p_review.add_argument("md_path", type=Path, help="Path to review .md file")
    p_review.add_argument("--region", default=settings.ACTIVE_REGION)

    # --- merge ---
    p_merge = sub.add_parser("merge", help="Manually trigger merge")
    p_merge.add_argument("--region", default=settings.ACTIVE_REGION)

    # --- validate ---
    p_validate = sub.add_parser("validate", help="Validate alias files")
    p_validate.add_argument("--region", default=settings.ACTIVE_REGION)

    args = parser.parse_args()

    if args.command == "add":
        add_alias(
            canonical_zh=args.canonical_zh,
            entity_type=args.entity_type,
            region=args.region,
            canonical_en=args.canonical_en,
            aliases=args.aliases,
            note=args.note,
            is_global=args.is_global,
        )
    elif args.command == "from-review":
        cmd_from_review(args.md_path, args.region)
    elif args.command == "merge":
        merge_and_write(args.region)
    elif args.command == "validate":
        cmd_validate(args.region)


if __name__ == "__main__":
    main()
