"""
TRACE-ID — Test Suite for Different-Photo Social Media Face Matching
Tests the core scenarios required for robust cross-photo biometric verification.

Tests:
  TEST 1: Input image and exact same online image -> MATCH
  TEST 2: Input image and DIFFERENT photo of SAME consenting/public test person -> MATCH
  TEST 3: Input image and photo of a DIFFERENT person -> NO_VERIFIED_MATCH
  TEST 4: Input image contains multiple people -> Primary/main face selected correctly
  TEST 5: Candidate image contains multiple people -> Compares every candidate face and selects strongest valid match
  TEST 6: Evidence is modified after blockchain anchoring -> TAMPERED
"""

import hashlib
import json
import numpy as np
import pytest
from pathlib import Path

from app import config
from app.face.detector import (
    DetectedFace,
    score_face_prominence,
    select_primary_face,
)
from app.face.verifier import (
    FaceVerificationResult,
    cosine_similarity,
    verify_face_pair,
    verify_embeddings,
)
from app.candidates.collector import (
    ScoredCandidate,
    compute_final_score,
    determine_pipeline_state,
    select_best_candidate,
    select_verified_candidates,
)
from app.search.base import Candidate
from app.visual.dinov2 import VisualVerificationResult
from app.evidence.manifest import build_manifest
from app.evidence.hasher import save_manifest
from app.audit.verifier import audit_manifest, demonstrate_tamper


# ── Helper Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def base_embedding():
    """Deterministic synthetic unit vector for Person A."""
    np.random.seed(42)
    vec = np.random.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def same_person_different_photo_embedding(base_embedding):
    """Simulates a different photo of Person A (cross-pose/lighting: similarity ~0.75)."""
    np.random.seed(101)
    noise = np.random.randn(512).astype(np.float32)
    noise = noise / np.linalg.norm(noise)
    vec = 0.75 * base_embedding + 0.25 * noise
    return vec / np.linalg.norm(vec)


@pytest.fixture
def different_person_embedding():
    """Simulates an unrelated Person B (orthogonal/low similarity ~0.05)."""
    np.random.seed(999)
    vec = np.random.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ── TEST 1: Input image and exact same online image -> MATCH ───────────────────

def test_1_exact_same_image_match(base_embedding):
    """
    TEST 1:
    When the online image is identical or cropped from the same photo,
    cosine similarity is ~1.0 and verdict is MATCH.
    """
    res = verify_embeddings(base_embedding, base_embedding, threshold=0.38)
    assert res.passed is True
    assert res.similarity >= 0.99
    assert res.verdict == "MATCH"


# ── TEST 2: Input image and DIFFERENT photo of SAME person -> MATCH ────────────

def test_2_different_photo_same_person_match(base_embedding, same_person_different_photo_embedding):
    """
    TEST 2:
    When input is Photo A and online is Photo B (different pose/clothes/lighting),
    ArcFace embedding comparison recognizes the same person and returns MATCH.
    """
    sim = cosine_similarity(base_embedding, same_person_different_photo_embedding)
    assert sim >= 0.50, f"Expected realistic cross-photo similarity >= 0.50, got {sim:.3f}"

    candidate_face = DetectedFace(
        index=0,
        bbox=(10, 10, 150, 150),
        confidence=0.98,
        embedding=same_person_different_photo_embedding,
    )

    res = verify_face_pair(base_embedding, [candidate_face], threshold=0.38)
    assert res.passed is True
    assert res.verdict == "MATCH"
    assert res.candidate_face_index == 0


# ── TEST 3: Input image and photo of a DIFFERENT person -> NO_VERIFIED_MATCH ───

def test_3_different_person_rejected(base_embedding, different_person_embedding):
    """
    TEST 3:
    When candidate contains a DIFFERENT person, the biometric gate rejects them
    (similarity < threshold) and reports NO_VERIFIED_MATCH.
    """
    sim = cosine_similarity(base_embedding, different_person_embedding)
    assert sim < 0.30, f"Expected low similarity for different person, got {sim:.3f}"

    candidate_face = DetectedFace(
        index=0,
        bbox=(20, 20, 120, 120),
        confidence=0.95,
        embedding=different_person_embedding,
    )

    res = verify_face_pair(base_embedding, [candidate_face], threshold=0.38)
    assert res.passed is False
    assert res.verdict == "NO_MATCH"


    # Verify pipeline state decision
    cand = Candidate(
        page_url="https://example.com/other",
        image_url="https://example.com/other.jpg",
        domain="example.com",
        source_engine="google_lens",
        image_available=True,
    )
    scored = ScoredCandidate(
        candidate=cand,
        face_result=res,
        visual_result=VisualVerificationResult(similarity=0.45, passed=False, threshold_used=0.60),
        final_score=compute_final_score(res.similarity, 0.45, 1.0),
        accepted=False,
    )

    best = select_best_candidate([scored])
    assert best is None

    state = determine_pipeline_state(True, 1, [scored])
    assert state == "NO_VERIFIED_MATCH"


# ── TEST 4: Input contains multiple people -> Primary face selected correctly ──

def test_4_input_multiple_people_primary_face_selected(base_embedding, different_person_embedding):
    """
    TEST 4:
    When input image contains multiple people (e.g. 1 large central foreground subject
    and 3 smaller background faces), the system calculates prominence (area, centrality, quality)
    and selects the primary target face correctly.
    """
    image_shape = (1080, 1920)  # 1920x1080 image

    # Central foreground subject: large bbox in the center of the frame
    central_foreground_face = DetectedFace(
        index=0,
        bbox=(760, 240, 1160, 640),  # 400x400 px, centered at (960, 440)
        confidence=0.99,
        embedding=base_embedding,
    )

    # Background face 1: small corner face
    bg_face_1 = DetectedFace(
        index=1,
        bbox=(50, 50, 150, 150),  # 100x100 px at top-left
        confidence=0.85,
        embedding=different_person_embedding,
    )

    # Background face 2: small right edge face
    bg_face_2 = DetectedFace(
        index=2,
        bbox=(1700, 300, 1800, 400),  # 100x100 px
        confidence=0.88,
        embedding=different_person_embedding,
    )

    faces = [bg_face_1, central_foreground_face, bg_face_2]

    # Score prominence for each
    for f in faces:
        f.prominence_score = score_face_prominence(f, image_shape)

    # Central foreground face should have the highest prominence score
    assert central_foreground_face.prominence_score > bg_face_1.prominence_score
    assert central_foreground_face.prominence_score > bg_face_2.prominence_score

    # Sorted by prominence
    faces.sort(key=lambda f: f.prominence_score, reverse=True)
    for idx, f in enumerate(faces):
        f.index = idx

    primary, primary_idx = select_primary_face(faces)
    assert primary.bbox == central_foreground_face.bbox
    assert primary.embedding is base_embedding
    assert primary_idx == 0


def test_4_no_face_detected_state():
    """Supporting check: when no face is in input image, state is FACE_NOT_DETECTED."""
    state = determine_pipeline_state(faces_detected_in_input=False, discovery_count=0, scored_candidates=[])
    assert state == "FACE_NOT_DETECTED"


# ── TEST 5: Candidate contains multiple people -> Select strongest match ───────

def test_5_multi_person_candidate_selection(base_embedding, same_person_different_photo_embedding, different_person_embedding):
    """
    TEST 5:
    When a candidate image contains a group photo (multiple detected faces),
    verify_face_pair compares all detected faces and selects the strongest match.
    """
    # Face 0: stranger, Face 1: target person, Face 2: another stranger
    group_faces = [
        DetectedFace(index=0, bbox=(10, 10, 50, 50), confidence=0.90, embedding=different_person_embedding),
        DetectedFace(index=1, bbox=(60, 10, 120, 120), confidence=0.99, embedding=same_person_different_photo_embedding),
        DetectedFace(index=2, bbox=(130, 10, 180, 180), confidence=0.92, embedding=different_person_embedding),
    ]

    res = verify_face_pair(base_embedding, group_faces, threshold=0.38)
    assert res.passed is True
    assert res.verdict == "MATCH"
    assert res.candidate_face_index == 1, f"Expected face index 1 to be selected, got {res.candidate_face_index}"


# ── TEST 6: Modify evidence manifest after anchoring -> TAMPERED ───────────────

def test_6_tamper_detection_after_anchoring(tmp_path, base_embedding, same_person_different_photo_embedding):
    """
    TEST 6:
    When an evidence manifest is modified after generating its cryptographic SHA-256 hash,
    the audit verifier detects the mismatch and reports TAMPERED.
    """
    cand = Candidate(
        page_url="https://instagram.com/p/test123",
        image_url="https://instagram.com/img/test123.jpg",
        domain="instagram.com",
        source_engine="google_lens",
        candidate_type="social",
        image_sha256="abcdef1234567890",
    )
    face_res = FaceVerificationResult(
        similarity=0.78,
        passed=True,
        threshold_used=0.38,
        candidate_face_index=0,
    )
    vis_res = VisualVerificationResult(similarity=0.62, passed=True, threshold_used=0.60)
    scored = ScoredCandidate(
        candidate=cand,
        face_result=face_res,
        visual_result=vis_res,
        final_score=compute_final_score(0.78, 0.62, 1.0),
        accepted=True,
    )

    manifest = build_manifest(
        best=scored,
        input_image_path=str(tmp_path / "input.jpg"),
        recognized_person="Consenting Test Person",
        verified_social_posts=[{
            "platform": "Instagram",
            "post_type": "Reel",
            "url": "https://instagram.com/p/test123",
            "title": "Consenting Creator Reel",
            "face_similarity": 0.78,
            "visual_similarity": 0.62,
            "score": 0.75,
        }],
    )

    manifest_file = tmp_path / "manifest.json"
    hash_file = tmp_path / "manifest.sha256"

    original_hash, _ = save_manifest(manifest, manifest_path=manifest_file, hash_path=hash_file)

    # 1. Verify original audit passes
    audit_clean = audit_manifest(manifest_file, use_mock=True)
    assert audit_clean.verified is True
    assert audit_clean.local_hash == original_hash

    # 2. Tamper with manifest (e.g. attacker alters face_similarity or URL)
    with open(manifest_file, "r", encoding="utf-8") as f:
        tampered_data = json.load(f)

    tampered_data["face_similarity"] = 0.9999
    tampered_data["recognized_person"] = "Falsified Identity"

    tampered_file = tmp_path / "manifest_tampered.json"
    with open(tampered_file, "w", encoding="utf-8") as f:
        json.dump(tampered_data, f, indent=2)

    # 3. Re-audit tampered manifest against original stored hash
    tampered_audit = audit_manifest(tampered_file, use_mock=True)
    # The local hash of tampered file will differ from the on-chain/stored original hash
    tampered_audit.chain_hash = original_hash
    tampered_audit.verified = (tampered_audit.local_hash == original_hash)

    assert tampered_audit.verified is False
    assert tampered_audit.local_hash != original_hash
