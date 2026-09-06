"""
TRACE-ID — Evidence Manifest Builder
Constructs the deterministic evidence manifest dict.

The manifest captures:
  - Pipeline version and record ID
  - Verified candidate details (URL, domain, platform)
  - Similarity scores (face and visual)
  - Discovery sources
  - Content image SHA-256 (of the downloaded candidate image)
  - Models used
  - Timestamps (ISO 8601 UTC)
  - Verification result wording (never "identity confirmed")

The manifest DOES NOT include:
  - Raw face embeddings (biometric data)
  - Private personal information
  - Unverified candidates
  - Claims of legal identity
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.candidates.collector import ScoredCandidate
from app import config

# ── Manifest version ──────────────────────────────────────────────────────────
MANIFEST_VERSION = "1.0"


def build_manifest(
    best: ScoredCandidate,
    input_image_path: str,
    discovery_sources: list[str] | None = None,
    recognized_person: str | None = None,
    verified_social_posts: list[dict] | None = None,
) -> dict:
    """
    Build the deterministic evidence manifest for the best verified candidate.

    Args:
        best:                  The highest-scoring accepted ScoredCandidate.
        input_image_path:      Path to the original input image (for SHA-256).
        discovery_sources:     Override discovery sources list (defaults to candidate's).
        recognized_person:     Name of recognized public entity/person if identified.
        verified_social_posts: List of verified social media posts found.

    Returns:
        Evidence manifest dict. Safe to serialize to JSON.
        Does NOT contain raw embeddings or PII.
    """
    candidate = best.candidate

    # Deterministic record ID: SHA-256 of (page_url + image_url + timestamp_utc)
    now_utc = datetime.now(timezone.utc)
    record_seed = (
        candidate.page_url
        + candidate.image_url
        + now_utc.isoformat()
    )
    record_id = hashlib.sha256(record_seed.encode()).hexdigest()[:32]

    # SHA-256 of the downloaded candidate image (not the input image)
    content_image_sha256 = candidate.image_sha256 or ""

    # Discovery sources
    sources = discovery_sources or candidate.discovery_sources or [candidate.source_engine]

    # Determine platform
    platform = candidate.domain or "unknown"

    manifest = {
        "version": MANIFEST_VERSION,
        "record_id": record_id,
        "recognized_person": recognized_person or "Unknown / Unspecified",
        "platform": platform,
        "post_url": candidate.page_url,
        "image_url": candidate.image_url,
        "domain": candidate.domain,
        "candidate_type": candidate.candidate_type,
        "content_image_sha256": content_image_sha256,
        "verified_social_posts": verified_social_posts or [],
        # ── Models ────────────────────────────────────────────────────────────
        "face_model": "InsightFace buffalo_l (ArcFace R100)",
        "face_similarity": round(best.face_result.similarity, 6),
        "face_verification": best.face_result.verdict,
        "face_threshold_used": best.face_result.threshold_used,
        # ── Visual ────────────────────────────────────────────────────────────
        "visual_model": "DINOv2 ViT-S/14",
        "visual_similarity": round(best.visual_result.similarity, 6),
        "visual_verification": best.visual_result.verdict,
        "visual_threshold_used": best.visual_result.threshold_used,
        # ── Scoring ───────────────────────────────────────────────────────────
        "final_score": round(best.final_score, 6),
        "score_weights": {
            "face_weight": config.FACE_WEIGHT,
            "visual_weight": config.VISUAL_WEIGHT,
            "discovery_weight": config.DISCOVERY_WEIGHT,
        },
        # ── Discovery ─────────────────────────────────────────────────────────
        "discovery_sources": sorted(set(sources)),
        "discovery_confidence": candidate.discovery_confidence,
        # ── Verification result ───────────────────────────────────────────────
        # IMPORTANT: wording must never claim "identity confirmed" or similar.
        "verification_result": "MATCH",
        "verification_statement": (
            "Candidate image passed face similarity verification and "
            "independent visual verification. "
            "This verifies image similarity, NOT legal identity or account ownership."
        ),
        # ── Timestamps ────────────────────────────────────────────────────────
        "discovered_at": now_utc.isoformat(),
        "pipeline": "TRACE-ID v1.0",
        "mock_mode": config.MOCK_MODE,
    }

    return manifest


def build_no_match_manifest(
    input_image_path: str,
    candidates_evaluated: int,
    engines_used: list[str],
) -> dict:
    """
    Build a manifest for the NO_VERIFIED_MATCH case.

    This manifest is NOT uploaded to blockchain since there is no verified match.
    It serves as an audit trail of the search attempt.
    """
    now_utc = datetime.now(timezone.utc)
    return {
        "version": MANIFEST_VERSION,
        "verification_result": "NO_VERIFIED_MATCH",
        "verification_statement": (
            "No candidate passed both face verification and visual verification. "
            "No blockchain anchoring performed."
        ),
        "candidates_evaluated": candidates_evaluated,
        "engines_used": engines_used,
        "discovered_at": now_utc.isoformat(),
        "pipeline": "TRACE-ID v1.0",
        "mock_mode": config.MOCK_MODE,
    }
