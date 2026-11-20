"""TensorBoard log dir resolution and metric tail cursors for the runner."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from zust_runner.metrics import collect_new_scalars, resolve_training_log_dir
from zust_runner.metrics_cursor import apply_ack


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


def test_collect_new_scalars_tracks_cursors_per_event_file(tmp_path: Path) -> None:
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

    cursor: dict[str, dict[str, int]] = {}
    with patch("zust_runner.metrics.EventAccumulator", side_effect=fake_accumulator):
        first = collect_new_scalars(log_dir, cursor)
    assert sorted(p["step"] for p in first) == [1, 2, 187]
    apply_ack(cursor, first)

    old_file.unlink()
    with patch("zust_runner.metrics.EventAccumulator", side_effect=fake_accumulator):
        second = collect_new_scalars(log_dir, cursor)
    assert second == []


def test_collect_new_scalars_after_tensorboard_wipe_resends_low_steps(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    log_dir.mkdir()

    def write_generation(name: str) -> None:
        path = log_dir / name
        path.write_text("stub", encoding="utf-8")

    write_generation("events.out.tfevents.gen1")

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

    cursor: dict[str, dict[str, int]] = {}
    with patch("zust_runner.metrics.EventAccumulator", side_effect=fake_accumulator):
        first = collect_new_scalars(log_dir, cursor)
    apply_ack(cursor, first)

    for path in log_dir.glob("events.out.tfevents.*"):
        path.unlink()
    write_generation("events.out.tfevents.gen2")

    with patch("zust_runner.metrics.EventAccumulator", side_effect=fake_accumulator):
        after_wipe = collect_new_scalars(log_dir, cursor)

    assert [p["step"] for p in after_wipe] == [1, 50]
