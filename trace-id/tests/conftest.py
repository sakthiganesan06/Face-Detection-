"""
TRACE-ID — pytest fixtures and shared test utilities.

All external APIs (SerpApi, Google Cloud Vision, blockchain, DINOv2, InsightFace)
are mocked in unit tests. No live API calls are made during testing.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Ensure project root is in path ────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── Force MOCK_MODE for all tests ─────────────────────────────────────────────
os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("FACE_SIMILARITY_THRESHOLD", "0.45")
os.environ.setdefault("VISUAL_SIMILARITY_THRESHOLD", "0.60")
os.environ.setdefault("FACE_WEIGHT", "0.60")
os.environ.setdefault("VISUAL_WEIGHT", "0.30")
os.environ.setdefault("DISCOVERY_WEIGHT", "0.10")


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def sample_manifest() -> dict:
    """A minimal valid evidence manifest dict."""
    return {
        "version": "1.0",
        "record_id": "abcdef1234567890abcdef1234567890",
        "platform": "example.com",
        "post_url": "https://example.com/post/123",
        "image_url": "https://example.com/images/photo.jpg",
        "domain": "example.com",
        "candidate_type": "web",
        "content_image_sha256": "e3b0c44298fc1c149afb4c8996fb92427ae41e4649b934ca495991b7852b855",
        "face_model": "InsightFace buffalo_l (ArcFace R100)",
        "face_similarity": 0.823456,
        "face_verification": "PASS",
        "face_threshold_used": 0.45,
        "visual_model": "DINOv2 ViT-S/14",
        "visual_similarity": 0.712345,
        "visual_verification": "PASS",
        "visual_threshold_used": 0.60,
        "final_score": 0.706780,
        "score_weights": {
            "face_weight": 0.60,
            "visual_weight": 0.30,
            "discovery_weight": 0.10,
        },
        "discovery_sources": ["google_cloud_vision", "google_lens"],
        "discovery_confidence": 1.0,
        "verification_result": "MATCH",
        "verification_statement": "Candidate image passed face similarity verification.",
        "discovered_at": "2026-09-06T15:00:00+00:00",
        "pipeline": "TRACE-ID v1.0",
        "mock_mode": True,
    }


@pytest.fixture
def random_embedding() -> np.ndarray:
    """Random L2-normalized embedding vector (512-dim)."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def similar_embedding(random_embedding: np.ndarray) -> np.ndarray:
    """An embedding very similar to random_embedding."""
    noise = np.random.default_rng(7).standard_normal(512).astype(np.float32) * 0.05
    noisy = random_embedding + noise
    return noisy / np.linalg.norm(noisy)


@pytest.fixture
def different_embedding() -> np.ndarray:
    """An embedding very different from random_embedding (random seed 99)."""
    rng = np.random.default_rng(99)
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)
