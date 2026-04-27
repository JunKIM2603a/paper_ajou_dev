from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def expand_search_space(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not search_space:
        return [{}]

    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo, strict=True)))
    return combinations


def sanitize_run_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")

