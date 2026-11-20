"""Persist TensorBoard upload cursors; advance only after the API accepts a batch."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for file_key, tags in parsed.items():
        if isinstance(file_key, str) and isinstance(tags, dict):
            out[file_key] = {str(k): int(v) for k, v in tags.items() if isinstance(v, (int, float))}
    return out


def save(path: Path, cursor: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(cursor, separators=(",", ":")).encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
        handle.write(data)
        temp_name = handle.name
    path.replace(temp_name)


def apply_ack(cursor: dict[str, dict[str, int]], points: list[dict[str, Any]]) -> None:
    for point in points:
        file_key = point.get("_event_file")
        tag = point.get("tag")
        step = point.get("step")
        if not isinstance(file_key, str) or not isinstance(tag, str):
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        tag_state = cursor.setdefault(file_key, {})
        tag_state[tag] = max(tag_state.get(tag, -1), step_int)
