"""TensorBoard log dir resolution for the runner."""

from pathlib import Path

import yaml

from zust_runner.metrics import resolve_training_log_dir


def test_resolve_training_log_dir_relative(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    config = checkout / "config.yaml"
    config.write_text(yaml.dump({"train": {"log_dir": "training_session/tb"}}), encoding="utf-8")
    resolved = resolve_training_log_dir(checkout, "config.yaml")
    assert resolved == (checkout / "training_session/tb").resolve()


def test_resolve_training_log_dir_disabled(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    config = checkout / "config.yaml"
    config.write_text(yaml.dump({"train": {}}), encoding="utf-8")
    assert resolve_training_log_dir(checkout, "config.yaml") is None
