"""
TRACE-ID — Candidate Collector + Scorer
Orchestrates candidate collection, verification, and scoring.

Scoring formula (documented in README and config):
  final_score = FACE_WEIGHT * face_similarity
              + VISUAL_WEIGHT * visual_similarity
              + DISCOVERY_WEIGHT * discovery_confidence

Face verification PASS is mandatory — a candidate with a high visual score
but failed face verification CANNOT be accepted as a match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app import config
from app.face.detector import DetectedFace, FaceDetector
from app.face.verifier import FaceVerificationResult, verify_face_pair
from app.search.base import Candidate
from app.visual.dinov2 import VisualVerificationResult, get_verifier as get_visual_verifier
from app.candidates.downloader import DownloadResult, download_image

logger = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    """A candidate that has been verified and scored."""

    candidate: Candidate
    face_result: FaceVerificationResult
    visual_result: VisualVerificationResult
    final_score: float
    accepted: bool  # True if both face and visual passed AND final_score is valid

    @property
    def verdict(self) -> str:
        return "PASS" if self.accepted else "FAIL"


def compute_final_score(
    face_similarity: float,
    visual_similarity: float,
    discovery_confidence: float,
    face_weight: float = config.FACE_WEIGHT,
    visual_weight: float = config.VISUAL_WEIGHT,
    discovery_weight: float = config.DISCOVERY_WEIGHT,
) -> float:
    """
    Compute the transparent weighted final score.

    Formula:
        final_score = face_weight * face_similarity
                    + visual_weight * visual_similarity
                    + discovery_weight * discovery_confidence

    All inputs should be in [0, 1].
    """
    score = (
        face_weight * max(0.0, face_similarity)
        + visual_weight * max(0.0, visual_similarity)
        + discovery_weight * max(0.0, min(1.0, discovery_confidence))
    )
    return round(score, 6)


def verify_and_score_candidate(
    candidate: Candidate,
    input_embedding: np.ndarray,
    input_image_path: str,
    input_visual_emb: Optional[np.ndarray] = None,
) -> ScoredCandidate:
    """
    Download, verify (face + visual), and score a single candidate.
    """
    face_detector = FaceDetector()
    visual_verifier = get_visual_verifier()

    # ── Step 1: Download ──────────────────────────────────────────────────────
    download_url = candidate.image_url or candidate.thumbnail_url or candidate.page_url
    if not download_url:
        logger.warning("Candidate has no downloadable URL — marking unavailable")
        return _unavailable_result(candidate)

    dl_result = download_image(download_url)

    if not dl_result.success:
        logger.info(
            "  Image unavailable: %s — %s", download_url[:60], dl_result.reason
        )
        candidate.image_available = False
        return _unavailable_result(candidate)

    candidate.image_available = True
    candidate.local_image_path = dl_result.local_path
    candidate.image_sha256 = dl_result.sha256

    # ── Step 2: Load downloaded image ─────────────────────────────────────────
    try:
        candidate_image = cv2.imread(dl_result.local_path)
        if candidate_image is None:
            raise ValueError("cv2.imread returned None")
    except Exception as exc:
        logger.warning("Could not read downloaded image %s: %s", dl_result.local_path, exc)
        return _unavailable_result(candidate)

    # ── Step 3: Face verification ──────────────────────────────────────────────
    try:
        candidate_faces = face_detector.detect(candidate_image)
        candidate.face_count = len(candidate_faces)
        face_result = verify_face_pair(input_embedding, candidate_faces)
    except Exception as exc:
        logger.warning("Face verification error for %s: %s", download_url[:60], exc)
        candidate.face_count = 0
        face_result = FaceVerificationResult(
            similarity=0.0,
            passed=False,
            threshold_used=config.FACE_SIMILARITY_THRESHOLD,
            candidate_face_index=-1,
        )


    # ── Step 4: Visual verification (DINOv2) ──────────────────────────────────
    try:
        query_ref = input_visual_emb if input_visual_emb is not None else input_image_path
        visual_result = visual_verifier.compare(query_ref, candidate_image)
    except Exception as exc:
        logger.warning("Visual verification error for %s: %s", download_url[:60], exc)
        visual_result = VisualVerificationResult(
            similarity=0.0,
            passed=False,
            threshold_used=config.VISUAL_SIMILARITY_THRESHOLD,
        )

    # ── Step 5: Final score ────────────────────────────────────────────────────
    final_score = compute_final_score(
        face_similarity=face_result.similarity,
        visual_similarity=visual_result.similarity,
        discovery_confidence=candidate.discovery_confidence,
    )

    # ── Step 6: Acceptance decision ────────────────────────────────────────────
    # Face verification is mandatory — even a perfect visual score cannot
    # compensate for a failed face verification.
    accepted = face_result.passed  # face_passed is the mandatory gate

    # Update candidate fields
    candidate.face_similarity = face_result.similarity
    candidate.face_passed = face_result.passed
    candidate.visual_similarity = visual_result.similarity
    candidate.visual_passed = visual_result.passed
    candidate.final_score = final_score

    logger.info(
        "  [%s] face=%.3f(%s) visual=%.3f(%s) score=%.3f | %s",
        "PASS" if accepted else "FAIL",
        face_result.similarity,
        face_result.verdict,
        visual_result.similarity,
        visual_result.verdict,
        final_score,
        (download_url[:50] + "...") if len(download_url) > 50 else download_url,
    )

    return ScoredCandidate(
        candidate=candidate,
        face_result=face_result,
        visual_result=visual_result,
        final_score=final_score,
        accepted=accepted,
    )


def _unavailable_result(candidate: Candidate) -> ScoredCandidate:
    """Create a failed ScoredCandidate for an unavailable image."""
    face_result = FaceVerificationResult(
        similarity=0.0,
        passed=False,
        threshold_used=config.FACE_SIMILARITY_THRESHOLD,
        candidate_face_index=-1,
    )
    visual_result = VisualVerificationResult(
        similarity=0.0,
        passed=False,
        threshold_used=config.VISUAL_SIMILARITY_THRESHOLD,
    )
    candidate.face_similarity = 0.0
    candidate.face_passed = False
    candidate.visual_similarity = 0.0
    candidate.visual_passed = False
    candidate.final_score = 0.0
    return ScoredCandidate(
        candidate=candidate,
        face_result=face_result,
        visual_result=visual_result,
        final_score=0.0,
        accepted=False,
    )


def select_best_candidate(
    scored: list[ScoredCandidate],
) -> Optional[ScoredCandidate]:
    """
    Select the best accepted candidate by final_score.

    Returns None if no candidate passed face verification.
    """
    accepted = [s for s in scored if s.accepted]
    if not accepted:
        return None
    # Sort by final_score descending
    accepted.sort(key=lambda s: s.final_score, reverse=True)
    return accepted[0]


def select_verified_candidates(
    scored: list[ScoredCandidate],
) -> list[ScoredCandidate]:
    """
    Select and rank all verified candidates that passed the face similarity gate.
    Sorted by face_similarity descending.
    """
    verified = [s for s in scored if s.accepted]
    verified.sort(key=lambda s: s.face_result.similarity, reverse=True)
    return verified


def determine_pipeline_state(
    faces_detected_in_input: bool,
    discovery_count: int,
    scored_candidates: list[ScoredCandidate],
) -> str:
    """
    Determine the canonical pipeline result state.

    Possible states:
      - FACE_NOT_DETECTED
      - NO_DISCOVERY_RESULTS
      - NO_FACE_IN_CANDIDATES
      - VERIFIED_MATCH
      - POTENTIAL_MATCH
      - NO_VERIFIED_MATCH
    """
    if not faces_detected_in_input:
        return "FACE_NOT_DETECTED"
    if discovery_count == 0 or not scored_candidates:
        return "NO_DISCOVERY_RESULTS"

    # Check if any candidate has detected faces
    has_candidate_face = any(
        s.candidate.image_available and s.face_result.candidate_face_index >= 0
        for s in scored_candidates
    )
    if not has_candidate_face:
        return "NO_FACE_IN_CANDIDATES"

    # Check for verified matches
    verified = [s for s in scored_candidates if s.accepted]
    if verified:
        return "VERIFIED_MATCH"

    # Check for potential matches (borderline band)
    potential = [s for s in scored_candidates if s.face_result.potential_match]
    if potential:
        return "POTENTIAL_MATCH"

    return "NO_VERIFIED_MATCH"
