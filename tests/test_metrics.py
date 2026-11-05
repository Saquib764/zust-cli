"""TensorBoard log dir resolution and metric tail cursors for the runner."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from zust_runner.metrics import _collect_points, resolve_training_log_dir


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


def _scalar_event(step: int, value: float) -> SimpleNamespace:
    return SimpleNamespace(step=step, value=value, wall_time=1_700_000_000.0)


def test_collect_points_tracks_cursors_per_event_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    log_dir.mkdir()
    old_file = log_dir / "events.out.tfevents.old"
    new_file = log_dir / "events.out.tfevents.new"
    old_file.write_text("stub", encoding="utf-8")
    new_file.write_text("stub", encoding="utf-8")

    def fake_accumulator(path: str) -> SimpleNamespace:
        if path.endswith("old"):
            return SimpleNamespace(
                Reload=lambda: None,
                Tags=lambda: {"scalars": ["train/loss"]},
                Scalars=lambda tag: [_scalar_event(187, 0.5)],
            )
        return SimpleNamespace(
            Reload=lambda: None,
            Tags=lambda: {"scalars": ["train/loss"]},
            Scalars=lambda tag: [_scalar_event(1, 0.1), _scalar_event(2, 0.2)],
        )

    sent: dict[str, dict[str, int]] = {}
    with patch("zust_runner.metrics.EventAccumulator", side_effect=fake_accumulator):
        first = _collect_points(log_dir, sent)
    assert sorted(p["step"] for p in first) == [1, 2, 187]

    old_file.unlink()
    with patch("zust_runner.metrics.EventAccumulator", side_effect=fake_accumulator):
        second = _collect_points(log_dir, sent)
    assert second == []


def test_collect_points_after_tensorboard_wipe_resends_low_steps(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    log_dir.mkdir()
    generation = {"file": "events.out.tfevents.gen1"}

    def write_generation(name: str, steps: list[int]) -> None:
        path = log_dir / name
        path.write_text("stub", encoding="utf-8")
        generation["file"] = name

    write_generation("events.out.tfevents.gen1", [187])

    def fake_accumulator(path: str) -> SimpleNamespace:
        if path.endswith("gen1"):
            return SimpleNamespace(
                Reload=lambda: None,
                Tags=lambda: {"scalars": ["train/loss"]},
                Scalars=lambda tag: [_scalar_event(187, 0.5)],
            )
        return SimpleNamespace(
            Reload=lambda: None,
            Tags=lambda: {"scalars": ["train/loss"]},
            Scalars=lambda tag: [_scalar_event(1, 0.1), _scalar_event(50, 0.2)],
        )

    sent: dict[str, dict[str, int]] = {}
    with patch("zust_runner.metrics.EventAccumulator", side_effect=fake_accumulator):
        _collect_points(log_dir, sent)

    for path in log_dir.glob("events.out.tfevents.*"):
        path.unlink()
    write_generation("events.out.tfevents.gen2", [1, 50])

    with patch("zust_runner.metrics.EventAccumulator", side_effect=fake_accumulator):
        after_wipe = _collect_points(log_dir, sent)

    assert [p["step"] for p in after_wipe] == [1, 50]
