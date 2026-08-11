"""Canonical serialization used by frozen research contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml


def plain(value: Any) -> Any:
    """Return JSON-shaped values with deterministic mapping key order."""
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value is not canonical JSON data: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Serialize one value as canonical UTF-8 JSON."""
    return json.dumps(
        plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def checksum(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON."""
    return sha256(canonical_bytes(value)).hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without evaluating executable constructors."""
    with Path(path).open(encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, Mapping):
        raise ValueError("Research YAML must contain one top-level mapping")
    return plain(value)


def dump_yaml(value: Any) -> str:
    """Render canonical data as stable, human-readable YAML."""
    return yaml.safe_dump(plain(value), sort_keys=True, allow_unicode=True)
