"""
TRACE-ID — Canonical JSON Serializer
Produces deterministic, byte-reproducible JSON from a Python dict.

Requirements:
  - Deterministic key ordering (sorted alphabetically, recursive)
  - No random fields
  - UTF-8 encoding
  - Consistent float formatting
  - No trailing whitespace or newlines that vary by platform

The SAME manifest dict must ALWAYS produce the SAME bytes.
"""

from __future__ import annotations

import json
from typing import Any


def _sort_recursive(obj: Any) -> Any:
    """
    Recursively sort dict keys and convert non-serializable types.

    - dicts → sorted by key
    - lists → preserved order (list order is meaningful)
    - floats → kept as floats (JSON standard representation)
    - everything else → passed through
    """
    if isinstance(obj, dict):
        return {k: _sort_recursive(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_sort_recursive(item) for item in obj]
    if isinstance(obj, bool):
        return obj  # must check before int since bool is subclass of int
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        # Ensure consistent float representation — avoid locale issues
        return obj
    if obj is None:
        return None
    return obj


def canonicalize(manifest: dict) -> bytes:
    """
    Produce a deterministic canonical UTF-8 JSON byte string from a manifest dict.

    Properties guaranteed:
      - Keys sorted recursively (ASCII order)
      - No extra whitespace
      - Compact separators: ',' and ':'
      - UTF-8 encoded (not ASCII-escaped)
      - No trailing newline

    Args:
        manifest: Evidence manifest dict. Must be JSON-serializable.

    Returns:
        UTF-8 encoded bytes of the canonical JSON representation.

    Example:
        >>> canonicalize({"b": 2, "a": 1})
        b'{"a":1,"b":2}'
    """
    sorted_manifest = _sort_recursive(manifest)
    json_str = json.dumps(
        sorted_manifest,
        sort_keys=True,          # Belt-and-suspenders: also sort at dumps level
        separators=(",", ":"),   # Compact — no spaces
        ensure_ascii=False,      # Preserve Unicode characters (UTF-8 output)
        allow_nan=False,         # Reject NaN/Inf — not valid JSON
    )
    return json_str.encode("utf-8")


def canonicalize_str(manifest: dict) -> str:
    """Return the canonical JSON as a string (UTF-8 decoded)."""
    return canonicalize(manifest).decode("utf-8")
