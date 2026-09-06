"""
Tests for SHA-256 deterministic hashing.

Verifies:
  - Same input always produces same hash
  - Different input produces different hash
  - Hash format is correct (64-char hex)
  - Empty manifest and edge cases
"""

from __future__ import annotations

import pytest

from app.evidence.hasher import hash_manifest, hash_bytes
from app.evidence.canonicalize import canonicalize


class TestHashManifest:
    def test_deterministic_same_dict(self, sample_manifest):
        """Same dict always produces same hash."""
        h1 = hash_manifest(sample_manifest)
        h2 = hash_manifest(sample_manifest)
        assert h1 == h2

    def test_hash_is_64_hex_chars(self, sample_manifest):
        """SHA-256 output is a 64-character hex string."""
        h = hash_manifest(sample_manifest)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_dict_different_hash(self, sample_manifest):
        """Modified manifest produces different hash."""
        h1 = hash_manifest(sample_manifest)
        modified = dict(sample_manifest)
        modified["face_similarity"] = 0.999999  # changed
        h2 = hash_manifest(modified)
        assert h1 != h2

    def test_key_order_does_not_matter(self, sample_manifest):
        """Hash is identical regardless of key insertion order."""
        import copy
        # Reverse the dict key order
        reversed_manifest = dict(reversed(list(sample_manifest.items())))
        h1 = hash_manifest(sample_manifest)
        h2 = hash_manifest(reversed_manifest)
        assert h1 == h2

    def test_known_hash_value(self):
        """Verify a known canonical hash for a simple dict."""
        simple = {"a": 1, "b": 2}
        canonical = canonicalize(simple)
        assert canonical == b'{"a":1,"b":2}'

        import hashlib
        expected = hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
        assert hash_manifest(simple) == expected

    def test_whitespace_in_values_preserved(self):
        """Whitespace within string values must be preserved."""
        m1 = {"key": "hello world"}
        m2 = {"key": "hello  world"}  # double space
        assert hash_manifest(m1) != hash_manifest(m2)

    def test_empty_manifest(self):
        """Empty dict produces a valid hash."""
        h = hash_manifest({})
        assert len(h) == 64

    def test_unicode_values(self):
        """Unicode characters in values produce correct hash."""
        m = {"name": "José García", "city": "München"}
        h = hash_manifest(m)
        assert len(h) == 64
        # Same unicode → same hash
        assert hash_manifest(m) == hash_manifest(m)

    def test_nested_dict_deterministic(self):
        """Nested dicts are canonicalized recursively."""
        m1 = {"outer": {"b": 2, "a": 1}}
        m2 = {"outer": {"a": 1, "b": 2}}
        assert hash_manifest(m1) == hash_manifest(m2)

    def test_float_precision(self):
        """Different float values produce different hashes."""
        m1 = {"similarity": 0.823456}
        m2 = {"similarity": 0.823457}
        assert hash_manifest(m1) != hash_manifest(m2)


class TestHashBytes:
    def test_known_sha256(self):
        """SHA-256 of empty bytes is known."""
        import hashlib
        result = hash_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_deterministic(self):
        data = b"TRACE-ID test data"
        assert hash_bytes(data) == hash_bytes(data)
