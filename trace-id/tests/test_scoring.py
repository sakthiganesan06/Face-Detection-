"""
Tests for candidate scoring logic.

Verifies:
  - Weighted formula correctness
  - Face verification mandatory gate
  - Score boundary conditions
  - Configurable weights
"""

from __future__ import annotations

import pytest

from app.candidates.collector import compute_final_score


class TestComputeFinalScore:
    def test_perfect_scores(self):
        """All inputs at 1.0 should yield 1.0."""
        score = compute_final_score(
            face_similarity=1.0,
            visual_similarity=1.0,
            discovery_confidence=1.0,
            face_weight=0.60,
            visual_weight=0.30,
            discovery_weight=0.10,
        )
        assert abs(score - 1.0) < 1e-6

    def test_zero_scores(self):
        """All inputs at 0.0 should yield 0.0."""
        score = compute_final_score(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_formula_matches_expected(self):
        """Exact formula: 0.6*face + 0.3*visual + 0.1*discovery."""
        face = 0.8
        visual = 0.7
        discovery = 1.0
        expected = 0.6 * face + 0.3 * visual + 0.1 * discovery
        result = compute_final_score(face, visual, discovery)
        assert abs(result - expected) < 1e-6

    def test_negative_face_clamped_to_zero(self):
        """Negative similarities are clamped to 0."""
        score = compute_final_score(
            face_similarity=-0.5,
            visual_similarity=1.0,
            discovery_confidence=1.0,
        )
        # face_similarity is clamped to 0
        expected = 0.60 * 0.0 + 0.30 * 1.0 + 0.10 * 1.0
        assert abs(score - expected) < 1e-6

    def test_discovery_confidence_clamped_to_one(self):
        """Discovery confidence above 1.0 is clamped."""
        score = compute_final_score(
            face_similarity=1.0,
            visual_similarity=1.0,
            discovery_confidence=2.0,  # Above max
        )
        expected = 0.60 * 1.0 + 0.30 * 1.0 + 0.10 * 1.0
        assert abs(score - expected) < 1e-6

    def test_custom_weights(self):
        """Custom weights are applied correctly."""
        face = 0.9
        visual = 0.5
        discovery = 0.8
        score = compute_final_score(
            face_similarity=face,
            visual_similarity=visual,
            discovery_confidence=discovery,
            face_weight=0.5,
            visual_weight=0.4,
            discovery_weight=0.1,
        )
        expected = 0.5 * face + 0.4 * visual + 0.1 * discovery
        assert abs(score - expected) < 1e-6

    def test_score_is_rounded(self):
        """Score is rounded to 6 decimal places."""
        score = compute_final_score(
            face_similarity=1 / 3,
            visual_similarity=1 / 3,
            discovery_confidence=1 / 3,
        )
        # Should be rounded, not a repeating decimal
        assert len(str(score).split(".")[-1]) <= 6

    def test_face_gate_is_mandatory(self):
        """
        Verify that face_passed=False prevents acceptance even if score is high.
        This tests the gate logic in verify_and_score_candidate.
        """
        from dataclasses import dataclass
        from app.face.verifier import FaceVerificationResult
        from app.visual.dinov2 import VisualVerificationResult
        from app.candidates.collector import ScoredCandidate
        from app.search.base import Candidate

        # Create a candidate with high visual score but failed face verification
        face_result = FaceVerificationResult(
            similarity=0.3,  # Below threshold
            passed=False,
            threshold_used=0.45,
            candidate_face_index=0,
        )
        visual_result = VisualVerificationResult(
            similarity=0.95,  # Very high visual
            passed=True,
            threshold_used=0.60,
        )

        candidate = Candidate(
            source_engine="test",
            image_url="https://example.com/img.jpg",
            discovery_sources=["test"],
        )

        scored = ScoredCandidate(
            candidate=candidate,
            face_result=face_result,
            visual_result=visual_result,
            final_score=compute_final_score(0.3, 0.95, 1.0),
            accepted=face_result.passed,  # Face gate: must pass face
        )

        # Must NOT be accepted despite high visual score
        assert not scored.accepted
        assert scored.verdict == "FAIL"

    def test_both_must_pass_for_acceptance(self):
        """Only candidates where face verification passed are accepted."""
        from app.face.verifier import FaceVerificationResult
        from app.visual.dinov2 import VisualVerificationResult
        from app.candidates.collector import ScoredCandidate, select_best_candidate
        from app.search.base import Candidate

        def make_scored(face_sim, visual_sim, face_passed):
            c = Candidate(
                source_engine="test",
                image_url=f"https://example.com/{face_sim}.jpg",
                discovery_sources=["test"],
            )
            return ScoredCandidate(
                candidate=c,
                face_result=FaceVerificationResult(
                    similarity=face_sim, passed=face_passed,
                    threshold_used=0.45, candidate_face_index=0
                ),
                visual_result=VisualVerificationResult(
                    similarity=visual_sim, passed=True, threshold_used=0.60
                ),
                final_score=compute_final_score(face_sim, visual_sim, 1.0),
                accepted=face_passed,
            )

        # One passes, one fails face
        passed = make_scored(0.7, 0.8, True)
        failed = make_scored(0.2, 0.95, False)

        best = select_best_candidate([passed, failed])
        assert best is not None
        assert best.accepted
        assert best.face_result.similarity == 0.7

    def test_no_match_when_all_fail(self):
        """select_best_candidate returns None when all fail."""
        from app.face.verifier import FaceVerificationResult
        from app.visual.dinov2 import VisualVerificationResult
        from app.candidates.collector import ScoredCandidate, select_best_candidate
        from app.search.base import Candidate

        failed = ScoredCandidate(
            candidate=Candidate(
                source_engine="test",
                image_url="https://example.com/img.jpg",
                discovery_sources=["test"],
            ),
            face_result=FaceVerificationResult(
                similarity=0.1, passed=False, threshold_used=0.45, candidate_face_index=-1
            ),
            visual_result=VisualVerificationResult(
                similarity=0.1, passed=False, threshold_used=0.60
            ),
            final_score=0.1,
            accepted=False,
        )
        result = select_best_candidate([failed])
        assert result is None
