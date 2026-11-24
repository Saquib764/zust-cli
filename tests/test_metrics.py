"""TensorBoard log dir resolution and metric tail cursors for the runner."""

from pathlib import Path

import yaml

from tensorboard.compat.proto.event_pb2 import Event

from zust_runner.config_paths import find_config_file, normalize_config_rel
from zust_runner.event_tail import crc32c, encode_record, read_records
from zust_runner.metrics import read_new_scalars, resolve_training_log_dir
from zust_runner.metrics_cursor import apply_file_ack, load, save


def test_normalize_config_rel_strips_root_prefix() -> None:
    assert normalize_config_rel("probe-pulse/config.yaml", "probe-pulse") == "config.yaml"
    assert normalize_config_rel("config.yaml", "probe-pulse") == "config.yaml"


def test_resolve_training_log_dir_repo_relative_config_in_subfolder(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    workdir = checkout / "probe-pulse"
    workdir.mkdir(parents=True)
    config = workdir / "config.yaml"
    config.write_text(yaml.dump({"train": {"log_dir": "training_session/tb"}}), encoding="utf-8")
    resolved = resolve_training_log_dir(
        workdir,
        "probe-pulse/config.yaml",
        repo_root=checkout,
    )
    assert resolved == (workdir / "training_session/tb").resolve()


def test_find_config_file_uses_repo_root_fallback(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    workdir = checkout / "probe-pulse"
    workdir.mkdir(parents=True)
    config = workdir / "config.yaml"
    config.write_text("train: {}\n", encoding="utf-8")
    assert find_config_file(workdir, "probe-pulse/config.yaml", repo_root=checkout) == config


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


def _event_bytes(step: int, value: float, tag: str = "train/loss") -> bytes:
    event = Event(wall_time=1_700_000_000.0, step=step)
    summary = event.summary.value.add()
    summary.tag = tag
    summary.simple_value = value
    return encode_record(event.SerializeToString())


def test_crc32c_known_vector() -> None:
    assert crc32c(b"123456789") == 0xE3069283


def test_read_records_stops_before_a_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents.test"
    payload = _event_bytes(1, 0.5)
    path.write_bytes(payload + b"\x01\x02")
    records, offset = read_records(path, 0)
    assert len(records) == 1
    assert offset == len(payload)
    again, offset_again = read_records(path, offset)
    assert again == []
    assert offset_again == offset


def test_save_creates_cursor_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics-cursor.json"
    cursor = {"version": 2, "files": {"events.out.tfevents.test": {"offset": 4, "dev": 1, "ino": 2}}}
    save(path, cursor)
    assert load(path) == cursor


def test_load_discards_step_cursor(tmp_path: Path) -> None:
    path = tmp_path / "metrics-cursor.json"
    path.write_text('{"events.out.tfevents.test": {"train/loss": 4}}', encoding="utf-8")
    assert load(path) == {"version": 2, "files": {}}


def test_read_new_scalars_tails_and_rereads_a_replaced_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    nested = log_dir / "run"
    nested.mkdir(parents=True)
    event_file = nested / "events.out.tfevents.child"
    event_file.write_bytes(_event_bytes(1, 0.1) + _event_bytes(2, 0.2))
    state: dict[str, dict[str, int]] = {}
    first, reset = read_new_scalars(log_dir, state, attempt=3)
    assert reset == []
    assert [point["step"] for point in first] == [1, 2]
    assert first[0]["attempt"] == 3
    assert first[0]["_file"] == "run/events.out.tfevents.child"
    cursor = {"version": 2, "files": {}}
    apply_file_ack(cursor, first)
    second, _reset = read_new_scalars(log_dir, state, attempt=3)
    assert second == []

    event_file.unlink()
    event_file.write_bytes(_event_bytes(7, 0.7))
    replaced, reset = read_new_scalars(log_dir, state, attempt=4)
    assert reset == ["run/events.out.tfevents.child"]
    assert [point["step"] for point in replaced] == [7]


def test_read_new_scalars_keeps_a_repeated_step(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    log_dir.mkdir()
    event_file = log_dir / "events.out.tfevents.test"
    event_file.write_bytes(_event_bytes(4, 1.0) + _event_bytes(4, 0.25))
    points, _reset = read_new_scalars(log_dir, {}, attempt=1)
    assert [point["value"] for point in points] == [1.0, 0.25]
