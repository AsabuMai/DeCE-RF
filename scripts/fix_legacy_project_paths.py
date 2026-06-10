from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PREFIXES = [
    "/workspace/rf_h_edit",
]
TARGET_PREFIX = str(ROOT)
SEARCH_ROOTS = [
    ROOT / "outputs" / "pretty_matrix",
]
SUFFIXES = {".json", ".txt"}


def replace_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        new_value = value
        count = 0
        for prefix in LEGACY_PREFIXES:
            if prefix in new_value:
                new_value = new_value.replace(prefix, TARGET_PREFIX)
                count += 1
        return new_value, count
    if isinstance(value, list):
        out = []
        total = 0
        for item in value:
            new_item, count = replace_value(item)
            out.append(new_item)
            total += count
        return out, total
    if isinstance(value, dict):
        out = {}
        total = 0
        for key, item in value.items():
            new_item, count = replace_value(item)
            out[key] = new_item
            total += count
        return out, total
    return value, 0


def update_json(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    new_data, count = replace_value(data)
    if count:
        path.write_text(json.dumps(new_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return count


def update_text(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    new_text = text
    count = 0
    for prefix in LEGACY_PREFIXES:
        count += new_text.count(prefix)
        new_text = new_text.replace(prefix, TARGET_PREFIX)
    if count:
        path.write_text(new_text, encoding="utf-8")
    return count


def main() -> int:
    files_seen = 0
    replacements = 0
    for root in SEARCH_ROOTS:
        for path in root.rglob("*"):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            files_seen += 1
            if path.suffix == ".json":
                replacements += update_json(path)
            else:
                replacements += update_text(path)
    print(f"files_seen={files_seen}")
    print(f"replacements={replacements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
