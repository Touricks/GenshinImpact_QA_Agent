"""Merge split alias files into a single regions/{region}/aliases.json.

Source of truth: src/config/aliases/{_global,region}/*.json
Generated artifact: src/config/regions/{region}/aliases.json

Usage:
    python -m src.config.merge_aliases [--region nodkrai]
    python -m src.config.merge_aliases --split --region nodkrai   # one-time migration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

# Six canonical type file names
TYPE_FILES = ("characters", "places", "factions", "items", "concepts", "events")

# Keys that belong in _global/ (cross-region entities)
GLOBAL_KEYS = frozenset([
    "旅行者", "荧", "派蒙", "戴因斯雷布", "尼伯龙根",
    "愚人众", "虚假之天", "葬火之战",
])

# Mapping from type field value -> file stem
TYPE_TO_FILE = {
    "Character": "characters",
    "Place": "places",
    "Faction": "factions",
    "Item": "items",
    "Concept": "concepts",
    "Event": "events",
}

_CONFIG_DIR = Path(__file__).resolve().parent  # src/config/


def _aliases_dir() -> Path:
    return _CONFIG_DIR / "aliases"


def _regions_dir() -> Path:
    return _CONFIG_DIR / "regions"


def _load_type_file(path: Path) -> dict:
    """Load a single type JSON file, returning {} if missing."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_aliases(region: str) -> dict:
    """Merge _global/* and {region}/* into a single dict.

    Raises ValueError on key conflicts (same key in global and region)
    or duplicate keys across different type files.
    """
    aliases_dir = _aliases_dir()
    merged: dict = {}
    seen_keys: Dict[str, str] = {}  # key -> "scope/type" for error msgs

    for scope in ("_global", region):
        scope_dir = aliases_dir / scope
        for type_name in TYPE_FILES:
            path = scope_dir / f"{type_name}.json"
            data = _load_type_file(path)
            for key, entry in data.items():
                source = f"{scope}/{type_name}.json"
                if key in merged:
                    prev = seen_keys[key]
                    raise ValueError(
                        f"Duplicate key '{key}': found in {prev} and {source}"
                    )
                merged[key] = entry
                seen_keys[key] = source

    return merged


def merge_and_write(region: str) -> Path:
    """Merge and write to regions/{region}/aliases.json. Returns output path."""
    merged = merge_aliases(region)
    out_dir = _regions_dir() / region
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aliases.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {len(merged)} entries -> {out_path}")
    return out_path


def split_aliases(region: str) -> None:
    """One-time migration: split existing aliases.json into type+scope files.

    Reads regions/{region}/aliases.json, classifies each entry,
    writes split files, then verifies roundtrip.
    """
    src_path = _regions_dir() / region / "aliases.json"
    with open(src_path, encoding="utf-8") as f:
        original = json.load(f)

    # Classify into buckets
    buckets: dict = {}  # "scope/type_name" -> dict
    for key, entry in original.items():
        scope = "_global" if key in GLOBAL_KEYS else region
        type_field = entry.get("type", "")
        file_stem = TYPE_TO_FILE.get(type_field)
        if not file_stem:
            raise ValueError(
                f"Unknown type '{type_field}' for key '{key}'"
            )
        bucket_key = f"{scope}/{file_stem}"
        buckets.setdefault(bucket_key, {})[key] = entry

    # Write each bucket
    aliases_dir = _aliases_dir()
    for bucket_key, data in buckets.items():
        scope, type_name = bucket_key.split("/")
        out_dir = aliases_dir / scope
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{type_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"  {out_path.relative_to(_CONFIG_DIR)}: {len(data)} entries")

    # Verify roundtrip
    merged = merge_aliases(region)
    assert merged == original, "Roundtrip verification failed!"
    print(f"Roundtrip OK: {len(original)} entries match.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge alias split files")
    parser.add_argument(
        "--region", default="nodkrai", help="Region name (default: nodkrai)"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="One-time: split existing aliases.json into type files"
    )
    args = parser.parse_args()

    if args.split:
        split_aliases(args.region)
    else:
        merge_and_write(args.region)


if __name__ == "__main__":
    main()
