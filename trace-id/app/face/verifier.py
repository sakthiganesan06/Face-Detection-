"""
TRACE-ID — Face Verifier
Computes cosine similarity between two face embeddings and applies a
configurable threshold to produce a PASS/FAIL verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app import config
from app.face.detector import DetectedFace

logger = logging.getLogger(__name__)


@dataclass
class FaceVerificationResult:
    """Result of comparing an input face embedding against a candidate face."""

    similarity: float
    """Cosine similarity in [−1, 1]. For ArcFace normalized embeddings: [0, 1]."""

    passed: bool
    """True if similarity >= FACE_SIMILARITY_THRESHOLD."""

    threshold_used: float
    """The threshold value applied during this comparison."""

    candidate_face_index: int
    """Index of the candidate face that gave this (best) result."""

    potential_match: bool = False
    """True if similarity is close to threshold (within 0.08 margin)."""

    @property
    def verdict(self) -> str:
        if self.candidate_face_index < 0:
            return "REJECT_NO_FACE"
        if self.passed:
            return "MATCH"
        if self.potential_match:
            return "POTENTIAL_MATCH"
        return "NO_MATCH"



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Both vectors should be L2-normalized before calling this.
    For already-normalized unit vectors: similarity = dot(a, b).

    Returns a float in [−1, 1]. For normalized ArcFace embeddings the
    practical range is [0, 1] since the model is trained with ArcFace loss.
    """
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        logger.warning("Near-zero embedding norm detected — returning 0.0 similarity")
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def verify_face_pair(
    input_embedding: np.ndarray,
    candidate_faces: list[DetectedFace],
    threshold: float | None = None,
) -> FaceVerificationResult:
    """
    Compare input face embedding against all faces detected in a candidate image.

    When multiple candidate faces exist, selects the one that produces
    the highest similarity score and reports that as the result.

    Args:
        input_embedding:  L2-normalized ArcFace embedding from the input image.
        candidate_faces:  List of DetectedFace objects from the candidate image.
        threshold:        Override for FACE_SIMILARITY_THRESHOLD (uses config default if None).

    Returns:
        FaceVerificationResult with the best similarity found.
    """
    if threshold is None:
        threshold = config.FACE_SIMILARITY_THRESHOLD

    if not candidate_faces:
        logger.debug("No faces in candidate image — verification FAIL")
        return FaceVerificationResult(
            similarity=0.0,
            passed=False,
            threshold_used=threshold,
            candidate_face_index=-1,
            potential_match=False,
        )

    best_sim = -1.0
    best_idx = 0

    for face in candidate_faces:
        sim = cosine_similarity(input_embedding, face.embedding)
        if sim > best_sim:
            best_sim = sim
            best_idx = face.index

    passed = best_sim >= threshold
    potential = (best_sim >= max(0.28, threshold - 0.08)) and not passed

    logger.debug(
        "Face verification: best_similarity=%.4f threshold=%.4f verdict=%s",
        best_sim,
        threshold,
        "MATCH" if passed else ("POTENTIAL_MATCH" if potential else "FAIL"),
    )

    return FaceVerificationResult(
        similarity=round(best_sim, 6),
        passed=passed,
        threshold_used=threshold,
        candidate_face_index=best_idx,
        potential_match=potential,
    )


def verify_embeddings(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    threshold: float | None = None,
) -> FaceVerificationResult:
    """
    Convenience function to compare two embeddings directly (without DetectedFace wrappers).
    Used in unit tests and audit tools.
    """
    if threshold is None:
        threshold = config.FACE_SIMILARITY_THRESHOLD

    sim = cosine_similarity(embedding_a, embedding_b)
    return FaceVerificationResult(
        similarity=round(sim, 6),
        passed=sim >= threshold,
        threshold_used=threshold,
        candidate_face_index=0,
    )
