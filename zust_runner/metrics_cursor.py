"""Persist TensorBoard file offsets. The offset moves only after the API accepts the bytes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

CURSOR_VERSION = 2


def empty() -> dict[str, Any]:
    return {"version": CURSOR_VERSION, "files": {}}


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty()
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return empty()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return empty()
    if not isinstance(parsed, dict) or parsed.get("version") != CURSOR_VERSION:
        return empty()
    files = parsed.get("files")
    if not isinstance(files, dict):
        return empty()
    cleaned: dict[str, dict[str, int]] = {}
    for key, entry in files.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        try:
            cleaned[key] = {
                "offset": int(entry.get("offset") or 0),
                "dev": int(entry.get("dev") or 0),
                "ino": int(entry.get("ino") or 0),
            }
        except (TypeError, ValueError):
            continue
    return {"version": CURSOR_VERSION, "files": cleaned}


def save(path: Path, cursor: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": CURSOR_VERSION, "files": cursor.get("files") or {}}
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
        handle.write(data)
        temp_name = handle.name
    Path(temp_name).replace(path)


def apply_file_ack(cursor: dict[str, Any], items: list[dict[str, Any]]) -> None:
    """Advance each file to the furthest acked record. Callers pass a queue prefix."""
    files: dict[str, dict[str, int]] = cursor.setdefault("files", {})
    for item in items:
        rel = item.get("_file")
        if not isinstance(rel, str):
            continue
        try:
            end = int(item.get("_offset"))
            dev = int(item.get("_dev") or 0)
            ino = int(item.get("_ino") or 0)
        except (TypeError, ValueError):
            continue
        entry = files.setdefault(rel, {"offset": 0, "dev": dev, "ino": ino})
        if int(entry.get("dev") or 0) != dev or int(entry.get("ino") or 0) != ino:
            entry["offset"] = 0
            entry["dev"] = dev
            entry["ino"] = ino
        entry["offset"] = max(int(entry.get("offset") or 0), end)


def reset_files(cursor: dict[str, Any], relative_paths: list[str]) -> None:
    files: dict[str, dict[str, int]] = cursor.setdefault("files", {})
    for rel in relative_paths:
        entry = files.get(rel)
        if entry is not None:
            entry["offset"] = 0
