"""Upload a folder from the training checkout to the Hugging Face Hub."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo


def _token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    raise RuntimeError(
        "Hugging Face token missing. Add HF_TOKEN to run env_content or the pod environment."
    )


def run_huggingface_push(action: dict[str, Any], checkout: Path) -> None:
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

    token = _token()
    api = HfApi(token=token)
    if action.get("create_repo"):
        create_repo(
            repo_id,
            token=token,
            private=bool(action.get("private")),
            exist_ok=True,
        )

    print(f"Uploading {source} to huggingface.co/{repo_id} ...", flush=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(source),
        repo_type="model",
        commit_message="Upload from Zust training run",
    )
    print(f"Uploaded to https://huggingface.co/{repo_id}", flush=True)


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
