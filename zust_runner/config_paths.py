"""Resolve training config paths inside a clone or repo subfolder."""

from __future__ import annotations

from pathlib import Path


def normalize_config_rel(config_rel: str, root_path: str = ".") -> str:
    """Return config path relative to the training root folder (ZUST_ROOT_PATH)."""
    normalized = config_rel.replace("\\", "/").lstrip("/")
    root = root_path.strip().replace("\\", "/").strip("/")
    if root in ("", "."):
        return normalized
    prefix = f"{root}/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def _config_rel_candidates(config_rel: str, project_root: Path) -> list[str]:
    normalized = normalize_config_rel(config_rel, ".")
    if ".." in normalized.split("/"):
        return []
    candidates = [normalized]
    prefix = f"{project_root.name}/"
    if normalized.startswith(prefix):
        stripped = normalized[len(prefix) :]
        if stripped and stripped not in candidates:
            candidates.append(stripped)
    return candidates


def find_config_file(
    project_root: Path,
    config_rel: str,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    """Locate the config YAML under the training root or repository clone."""
    project_resolved = project_root.resolve()
    for rel in _config_rel_candidates(config_rel, project_root):
        destination = (project_resolved / rel).resolve()
        if destination != project_resolved and project_resolved not in destination.parents:
            continue
        if destination.is_file():
            return destination
    if repo_root is None:
        return None
    repo_resolved = repo_root.resolve()
    normalized = config_rel.replace("\\", "/").lstrip("/")
    if ".." in normalized.split("/"):
        return None
    destination = (repo_resolved / normalized).resolve()
    if destination != repo_resolved and repo_resolved not in destination.parents:
        return None
    if destination.is_file():
        return destination
    return None


def _path_within_any_anchor(path: Path, anchors: list[Path]) -> bool:
    resolved = path.resolve()
    for anchor in anchors:
        anchor_resolved = anchor.resolve()
        if resolved == anchor_resolved or anchor_resolved in resolved.parents:
            return True
    return False