"""Read new TensorBoard event records from a byte offset.

Event files are TFRecords. A short trailing record is a torn write: leave the
offset where it was and try again. A complete record with a bad checksum is
skipped so one corrupt event cannot pin the file.
"""

from __future__ import annotations

import struct
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event

_MASK = 0xFFFFFFFF
_CRC_POLY = 0x82F63B78
_MAX_RECORD = 32 * 1024 * 1024


def _crc_table() -> list[int]:
    table: list[int] = []
    for index in range(256):
        crc = index
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC_POLY if crc & 1 else crc >> 1
        table.append(crc)
    return table


_TABLE = _crc_table()


def crc32c(data: bytes) -> int:
    crc = _MASK
    for byte in data:
        crc = _TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ _MASK


def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & _MASK


def encode_record(payload: bytes) -> bytes:
    length = struct.pack("<Q", len(payload))
    return length + struct.pack("<I", masked_crc32c(length)) + payload + struct.pack("<I", masked_crc32c(payload))


def read_records(path: Path, offset: int) -> tuple[list[tuple[bytes, int]], int]:
    """Return complete records and the offset to resume from."""
    size = path.stat().st_size
    if offset < 0 or offset > size:
        offset = 0
    records: list[tuple[bytes, int]] = []
    with path.open("rb") as handle:
        while offset + 12 <= size:
            handle.seek(offset)
            header = handle.read(12)
            if len(header) < 12:
                break
            length, length_crc = struct.unpack("<QI", header)
            if length_crc != masked_crc32c(header[:8]) or length > _MAX_RECORD:
                break
            if offset + 12 + length + 4 > size:
                break
            payload = handle.read(length)
            crc_raw = handle.read(4)
            if len(payload) < length or len(crc_raw) < 4:
                break
            end = offset + 12 + length + 4
            data_crc = struct.unpack("<I", crc_raw)[0]
            if data_crc != masked_crc32c(payload):
                offset = end
                continue
            records.append((payload, end))
            offset = end
    return records, offset


def scalars_from_record(payload: bytes) -> list[dict[str, object]]:
    """Scalar summaries in one event. Other summary kinds are ignored."""
    event = Event()
    try:
        event.ParseFromString(payload)
    except Exception:
        return []
    if not event.HasField("summary"):
        return []
    points: list[dict[str, object]] = []
    for value in event.summary.value:
        tag = value.tag
        if not tag:
            continue
        parsed = _scalar_value(value)
        if parsed is None:
            continue
        number, is_nan = parsed
        point: dict[str, object] = {
            "tag": tag,
            "step": int(event.step),
            "value": number,
            "is_nan": is_nan,
        }
        if event.wall_time:
            point["wall_time"] = float(event.wall_time)
        points.append(point)
    return points


def _scalar_value(value: object) -> tuple[float, bool] | None:
    metadata = getattr(value, "metadata", None)
    plugin = ""
    if metadata is not None and metadata.HasField("plugin_data"):
        plugin = metadata.plugin_data.plugin_name
    if plugin and plugin != "scalars":
        return None
    if value.HasField("tensor"):  # type: ignore[attr-defined]
        number = _tensor_scalar(value.tensor)  # type: ignore[attr-defined]
        if number is None:
            return None
        return _finite(number)
    if value.HasField("histo") or value.HasField("image") or value.HasField("audio"):  # type: ignore[attr-defined]
        return None
    return _finite(float(value.simple_value))  # type: ignore[attr-defined]


def _tensor_scalar(tensor: object) -> float | None:
    float_val = getattr(tensor, "float_val", None)
    if float_val:
        return float(float_val[0])
    double_val = getattr(tensor, "double_val", None)
    if double_val:
        return float(double_val[0])
    content = getattr(tensor, "tensor_content", b"") or b""
    if len(content) >= 4:
        return struct.unpack("<f", content[:4])[0]
    return None


def _finite(number: float) -> tuple[float, bool]:
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0, True
    return number, False
