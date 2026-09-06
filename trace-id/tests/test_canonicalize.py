"""
Tests for canonical JSON serialization.

Verifies:
  - Keys are sorted alphabetically
  - Compact separators (no spaces)
  - UTF-8 encoding
  - Recursive sorting of nested structures
  - No random or non-deterministic fields
"""

from __future__ import annotations

import json

import pytest

from app.evidence.canonicalize import canonicalize, canonicalize_str


class TestCanonicalize:
    def test_keys_sorted(self):
        """Keys are sorted alphabetically."""
        result = canonicalize({"z": 3, "a": 1, "m": 2})
        parsed = json.loads(result)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_compact_format(self):
        """Output uses compact separators — no spaces."""
        result = canonicalize({"a": 1, "b": 2})
        assert result == b'{"a":1,"b":2}'
        assert b" " not in result

    def test_utf8_encoded(self):
        """Output is UTF-8 bytes, not ASCII-escaped."""
        result = canonicalize({"name": "José"})
        assert isinstance(result, bytes)
        decoded = result.decode("utf-8")
        assert "José" in decoded
        # Should NOT use ASCII escape \u00e9 for é
        assert "\\u" not in decoded

    def test_nested_keys_sorted(self):
        """Nested dict keys are also sorted."""
        result = canonicalize({"outer": {"z": 1, "a": 2}})
        parsed = json.loads(result)
        inner_keys = list(parsed["outer"].keys())
        assert inner_keys == sorted(inner_keys)

    def test_list_order_preserved(self):
        """List element order is preserved (not sorted)."""
        result = canonicalize({"items": [3, 1, 2]})
        parsed = json.loads(result)
        assert parsed["items"] == [3, 1, 2]

    def test_bool_values(self):
        """Boolean values serialize correctly."""
        result = canonicalize({"flag": True, "other": False})
        assert b'"flag":true' in result
        assert b'"other":false' in result

    def test_none_value(self):
        """None serializes to null."""
        result = canonicalize({"key": None})
        assert b'"key":null' in result

    def test_empty_dict(self):
        result = canonicalize({})
        assert result == b"{}"

    def test_empty_list(self):
        result = canonicalize({"items": []})
        assert b'"items":[]' in result

    def test_deeply_nested(self):
        """Three levels of nesting are sorted."""
        data = {
            "c": {"b": {"z": 1, "a": 2}, "a": 3},
            "a": 0,
        }
        result = canonicalize(data)
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["a", "c"]
        assert list(parsed["c"].keys()) == ["a", "b"]
        assert list(parsed["c"]["b"].keys()) == ["a", "z"]

    def test_float_representation(self):
        """Floats are not mangled."""
        result = canonicalize({"x": 0.823456})
        parsed = json.loads(result)
        assert abs(parsed["x"] - 0.823456) < 1e-9

    def test_integer_not_float(self):
        """Integers stay integers."""
        result = canonicalize({"n": 42})
        assert b'"n":42' in result
        assert b'"n":42.0' not in result

    def test_canonicalize_str_returns_string(self):
        result = canonicalize_str({"a": 1})
        assert isinstance(result, str)
        assert result == '{"a":1}'

    def test_same_input_same_output(self, sample_manifest):
        """Canonicalize is idempotent."""
        b1 = canonicalize(sample_manifest)
        b2 = canonicalize(sample_manifest)
        assert b1 == b2

    def test_key_order_invariance(self, sample_manifest):
        """Reversed key order produces same canonical form."""
        reversed_manifest = dict(reversed(list(sample_manifest.items())))
        assert canonicalize(sample_manifest) == canonicalize(reversed_manifest)
