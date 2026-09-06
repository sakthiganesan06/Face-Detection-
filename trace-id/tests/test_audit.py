"""
Tests for audit/replay tamper detection.

Verifies:
  - VERIFIED result when manifest is unmodified
  - TAMPERED result when any field is modified
  - demonstrate_tamper creates different hash
  - Tamper detection for different field types
  - No modification of original manifest file
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evidence.canonicalize import canonicalize
from app.evidence.hasher import hash_manifest, save_manifest
from app.audit.verifier import demonstrate_tamper, audit_manifest


class TestTamperDetection:
    def test_unmodified_hash_is_stable(self, sample_manifest):
        """The same manifest always hashes identically."""
        h1 = hash_manifest(sample_manifest)
        h2 = hash_manifest(dict(sample_manifest))  # copy
        assert h1 == h2

    def test_modified_float_field_changes_hash(self, sample_manifest):
        """Modifying a float field changes the hash."""
        original_hash = hash_manifest(sample_manifest)
        tampered = dict(sample_manifest)
        tampered["face_similarity"] = tampered["face_similarity"] + 0.001
        tampered_hash = hash_manifest(tampered)
        assert original_hash != tampered_hash

    def test_modified_string_field_changes_hash(self, sample_manifest):
        """Modifying a string field changes the hash."""
        original_hash = hash_manifest(sample_manifest)
        tampered = dict(sample_manifest)
        tampered["verification_result"] = "TAMPERED_RESULT"
        assert hash_manifest(tampered) != original_hash

    def test_added_field_changes_hash(self, sample_manifest):
        """Adding a new field changes the hash."""
        original_hash = hash_manifest(sample_manifest)
        tampered = dict(sample_manifest)
        tampered["extra_field"] = "injected"
        assert hash_manifest(tampered) != original_hash

    def test_deleted_field_changes_hash(self, sample_manifest):
        """Removing a field changes the hash."""
        original_hash = hash_manifest(sample_manifest)
        tampered = {k: v for k, v in sample_manifest.items() if k != "face_similarity"}
        assert hash_manifest(tampered) != original_hash

    def test_demonstrate_tamper_changes_hash(self, sample_manifest, tmp_dir):
        """demonstrate_tamper creates a different hash."""
        # Save the manifest to a temp file
        manifest_path = tmp_dir / "manifest.json"
        save_manifest(sample_manifest, manifest_path=manifest_path, hash_path=tmp_dir / "manifest.sha256")

        result = demonstrate_tamper(manifest_path, field="visual_similarity")

        assert result["hashes_differ"]
        assert result["original_hash"] != result["tampered_hash"]

    def test_demonstrate_tamper_does_not_modify_original(self, sample_manifest, tmp_dir):
        """demonstrate_tamper saves a SEPARATE tampered file, not modifying the original."""
        manifest_path = tmp_dir / "manifest.json"
        save_manifest(sample_manifest, manifest_path=manifest_path, hash_path=tmp_dir / "manifest.sha256")

        original_content = manifest_path.read_text(encoding="utf-8")
        demonstrate_tamper(manifest_path, field="visual_similarity")
        after_content = manifest_path.read_text(encoding="utf-8")

        # Original must be unchanged
        assert original_content == after_content

    def test_tampered_file_exists_separately(self, sample_manifest, tmp_dir):
        """Tampered file is saved with _TAMPERED_DEMO suffix."""
        manifest_path = tmp_dir / "manifest.json"
        save_manifest(sample_manifest, manifest_path=manifest_path, hash_path=tmp_dir / "manifest.sha256")

        result = demonstrate_tamper(manifest_path)
        tampered_path = Path(result["tampered_manifest"])

        assert tampered_path.exists()
        assert "TAMPERED" in tampered_path.name

    def test_offline_audit_verified_on_unmodified(self, sample_manifest, tmp_dir):
        """Offline audit returns VERIFIED when manifest is unmodified."""
        manifest_path = tmp_dir / "manifest.json"
        hash_path = tmp_dir / "manifest.sha256"
        save_manifest(sample_manifest, manifest_path=manifest_path, hash_path=hash_path)

        # The save_manifest stores the original hash in the .sha256 file
        result = audit_manifest(manifest_path, use_mock=True)
        # In mock mode, compares local hash against saved .sha256
        assert result.local_hash == result.chain_hash
        assert result.verified

    def test_offline_audit_tampered_on_modified(self, sample_manifest, tmp_dir):
        """Offline audit returns TAMPERED when manifest is modified after saving hash."""
        manifest_path = tmp_dir / "manifest.json"
        hash_path = tmp_dir / "manifest.sha256"

        # Save original + hash
        original_hash, _ = save_manifest(
            sample_manifest, manifest_path=manifest_path, hash_path=hash_path
        )

        # Now modify the manifest on disk (simulating tampering)
        tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
        tampered["face_similarity"] = 0.999
        manifest_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
        # Leave the .sha256 file unchanged (it has the original hash)

        result = audit_manifest(manifest_path, use_mock=True)
        # The recomputed hash of the modified manifest differs from the saved hash
        assert not result.verified
        assert result.local_hash != result.chain_hash

    def test_missing_manifest_returns_error(self, tmp_dir):
        """Missing manifest file returns ERROR status."""
        result = audit_manifest(tmp_dir / "nonexistent.json", use_mock=True)
        assert result.status == "ERROR"
        assert not result.verified


class TestFaceVerification:
    """Test face similarity threshold logic."""

    def test_similar_embeddings_pass(self, random_embedding, similar_embedding):
        """Embeddings from the same person (similar) should pass."""
        from app.face.verifier import verify_embeddings
        result = verify_embeddings(random_embedding, similar_embedding, threshold=0.45)
        # similar_embedding is similar (same person, slight noise) — should pass threshold
        # Note: cosine similarity with 5% noise on 512-dim vector is typically ~0.6-0.85
        assert result.similarity > 0.45  # Must exceed FACE_SIMILARITY_THRESHOLD
        assert result.passed

    def test_different_embeddings_fail(self, random_embedding, different_embedding):
        """Completely different embeddings should fail."""
        from app.face.verifier import verify_embeddings
        result = verify_embeddings(random_embedding, different_embedding, threshold=0.45)
        # Cosine similarity of random vectors is near 0
        assert result.similarity < 0.45
        assert not result.passed

    def test_identical_embeddings_similarity_is_one(self, random_embedding):
        """Identical embedding vectors have similarity = 1.0."""
        from app.face.verifier import verify_embeddings
        result = verify_embeddings(random_embedding, random_embedding.copy())
        assert abs(result.similarity - 1.0) < 1e-5
