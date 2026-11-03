"""Runner path checks."""

import tempfile
from pathlib import Path

import pytest

from zust_runner.main import _validate_repo_relative_path


def test_runner_rejects_traversal_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with pytest.raises(ValueError):
            _validate_repo_relative_path(root, "../outside.yaml")
