"""Upload a folder from the training checkout to the Hugging Face Hub."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo

_TOKEN_KEYS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _token_from_mapping(values: dict[str, str]) -> str:
    for key in _TOKEN_KEYS:
        value = values.get(key, "").strip()
        if value:
            return value
    return ""


def _env_file_candidates(checkout: Path) -> list[Path]:
    from zust_runner.runner_cache import EFFECTIVE_ENV_FILE, outbound_dir

    candidates = [outbound_dir() / EFFECTIVE_ENV_FILE, checkout / ".env"]
    root = os.environ.get("ZUST_ROOT_PATH", "").strip().replace("\\", "/").strip("/")
    if root and root not in {".", "./"}:
        candidates.append(checkout / root / ".env")
    return candidates


def _token(checkout: Path) -> str:
    found = _token_from_mapping({key: os.environ.get(key, "") for key in _TOKEN_KEYS})
    if found:
        return found
    for path in _env_file_candidates(checkout):
        if not path.is_file():
            continue
        found = _token_from_mapping(_parse_env_text(path.read_text(encoding="utf-8", errors="replace")))
        if found:
            return found
    raise RuntimeError(
        "Hugging Face token missing. Add HF_TOKEN to run env_content or the pod environment."
    )


def run_huggingface_push(
    action: dict[str, Any],
    checkout: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> None:
    folder_path = str(action.get("folder_path") or "").strip().replace("\\", "/").lstrip("/")
    repo_id = str(action.get("repo_id") or "").strip()
    if not folder_path or not repo_id:
        raise ValueError("huggingface_push action requires folder_path and repo_id")
    if ".." in folder_path.split("/"):
        raise ValueError("folder_path must stay inside the repository")

    source = (checkout / folder_path).resolve()
    checkout_resolved = checkout.resolve()
    if source != checkout_resolved and checkout_resolved not in source.parents:
        raise ValueError("folder_path must stay inside the repository")
    if not source.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    token = _token(checkout)
    api = HfApi(token=token)

    def emit(message: str) -> None:
        if log is not None:
            log(message)
        else:
            print(message, flush=True)

    if action.get("create_repo"):
        create_repo(
            repo_id,
            token=token,
            private=bool(action.get("private")),
            exist_ok=True,
        )

    replace_existing = bool(action.get("replace_existing"))
    if replace_existing:
        emit(f"Replacing huggingface.co/{repo_id} with {folder_path} ...")
    else:
        emit(f"Uploading {folder_path} to huggingface.co/{repo_id} ...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(source),
        repo_type="model",
        commit_message=(
            "Replace repository contents from Zust training run"
            if replace_existing
            else "Upload from Zust training run"
        ),
        **({"delete_patterns": "*"} if replace_existing else {}),
    )
    emit(f"Uploaded {folder_path} to https://huggingface.co/{repo_id}")


def main() -> int:
    import json

    action = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    checkout = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    try:
        run_huggingface_push(action, checkout)
    except Exception as exc:
        print(f"Hugging Face upload failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
