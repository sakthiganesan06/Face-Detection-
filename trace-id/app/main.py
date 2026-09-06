"""
TRACE-ID — Main CLI Pipeline
Traceable Identity & Digital Evidence
"Discover. Verify. Anchor."

Usage:
    python -m app.main --image data/input/test.jpg
    python -m app.main --image data/input/test.jpg --face-index 0
    python -m app.main --image data/input/test.jpg --max-candidates 12
    python -m app.main --audit data/evidence/manifest.json
    python -m app.main --audit data/evidence/manifest.json --mock-blockchain
    python -m app.main --tamper-demo data/evidence/manifest.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click

# Reconfigure stdout/stderr for Unicode support on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("trace_id")
for mod in ["app.face", "app.search", "app.visual", "app.candidates",
            "app.evidence", "app.blockchain", "app.audit", "app.ocr"]:
    logging.getLogger(mod).setLevel(logging.WARNING)


def _hr(char="=", width=60):
    print(char * width)


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    image_path: str,
    max_candidates: int | None = None,
    selected_face_index: int | None = None,
) -> int:
    """
    Execute the full TRACE-ID pipeline satisfying Hacker House Goa 2026 Task #3.

    Returns:
        0 on success (VERIFIED_MATCH or POTENTIAL_MATCH),
        1 on NO_VERIFIED_MATCH / NO_DISCOVERY_RESULTS / NO_FACE_IN_CANDIDATES,
        2 on fatal error / FACE_NOT_DETECTED.
    """
    from app import config

    if max_candidates is not None:
        import app.config as _cfg
        _cfg.MAX_CANDIDATES = max_candidates

    start_time = time.time()

    print("========================================")
    print("TRACE-ID PIPELINE")
    print("========================================")
    if config.MOCK_MODE:
        print("\n[MOCK MODE] Search results are SIMULATED.")

    # ── [1] INPUT ─────────────────────────────────────────────────────────────
    from app.face.detector import (
        validate_image_path,
        load_image,
        FaceDetector,
        select_primary_face,
    )
    import cv2

    print("\n[1] INPUT")
    print(f"Image: {image_path}")

    try:
        resolved_path = validate_image_path(image_path)
        image_bgr = load_image(resolved_path)
        h, w = image_bgr.shape[:2]
        print(f"Resolution: {w}x{h}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        print("\n[11] FINAL RESULT\nERROR")
        return 2

    # ── [2] FACE DETECTION ────────────────────────────────────────────────────
    print("\n[2] FACE DETECTION")
    try:
        detector = FaceDetector()
        faces = detector.detect(image_bgr)
    except RuntimeError as exc:
        print(f"Error initializing face detector: {exc}")
        print("\n[11] FINAL RESULT\nERROR")
        return 2

    if not faces:
        print("Faces detected: 0")
        print("\n[11] FINAL RESULT\nNO_FACE / FACE_NOT_DETECTED\nReason: No face detected in the input image.")
        return 2

    print(f"Faces detected: {len(faces)}")
    for f in faces:
        print(f"\nFace {f.index}:")
        print(f"  bbox = {f.bbox}")
        print(f"  area = {f.area}")
        print(f"  confidence = {f.confidence:.2f}")

    selected_face, primary_idx = select_primary_face(faces, selected_index=selected_face_index)
    print(f"\nPrimary face:")
    print(f"  Face {primary_idx}")
    print(f"  Area: {selected_face.area}")
    print(f"  Confidence: {selected_face.confidence:.2f}")

    # ── [3] FACE EMBEDDING ────────────────────────────────────────────────────
    print("\n[3] FACE EMBEDDING")
    input_embedding = selected_face.embedding
    print(f"Model: InsightFace / ArcFace ({config.INSIGHTFACE_MODEL_NAME})")
    print(f"Embedding dimension: {len(input_embedding)}")

    # ── [4] OCR / WATERMARK DETECTION ─────────────────────────────────────────
    print("\n[4] OCR")
    detected_texts: list[str] = []
    try:
        from app.ocr.extractor import OCRExtractor
        ocr = OCRExtractor()
        detected_texts = ocr.extract_text(image_bgr)
    except Exception as exc:
        logger.debug("OCR execution note: %s", exc)

    if detected_texts:
        print("Detected text:")
        for t in detected_texts:
            print(f"- {t}")
    else:
        print("Detected text: None")

    # ── [5] REVERSE IMAGE SEARCH ──────────────────────────────────────────────
    print("\n[5] REVERSE IMAGE SEARCH")
    from app.search.serpapi_lens import SerpApiLensEngine
    from app.search.serpapi_reverse import SerpApiReverseImageEngine
    from app.search.cloud_vision import CloudVisionEngine
    from app.search.context_search import ContextSearchEngine
    from app.search.page_scraper import expand_candidates_from_pages
    from app.search.aggregator import aggregate_candidates
    from app.visual.dinov2 import get_verifier as get_visual_verifier

    candidates_per_engine: dict = {}
    raw_candidate_count = 0

    # Face crop for face-focused discovery (ideal for finding different photos / reels)
    pad_x = int((selected_face.bbox[2] - selected_face.bbox[0]) * 0.25)
    pad_y = int((selected_face.bbox[3] - selected_face.bbox[1]) * 0.25)
    c_x1 = max(0, selected_face.bbox[0] - pad_x)
    c_y1 = max(0, selected_face.bbox[1] - pad_y)
    c_x2 = min(w, selected_face.bbox[2] + pad_x)
    c_y2 = min(h, selected_face.bbox[3] + pad_y)
    crop_img = image_bgr[c_y1:c_y2, c_x1:c_x2]
    crop_path = resolved_path.parent / f".crop_{resolved_path.name}"
    cv2.imwrite(str(crop_path), crop_img)

    # A. Google Lens
    lens_engine = SerpApiLensEngine()
    lens_exact = 0
    lens_visual = 0
    lens_pages = 0
    try:
        lens_candidates = lens_engine.search(str(resolved_path))
        candidates_per_engine["google_lens"] = lens_candidates
        raw_candidate_count += len(lens_candidates)
        lens_exact = lens_engine.last_results_breakdown.get("exact_matches", 0)
        lens_visual = lens_engine.last_results_breakdown.get("visual_matches", len(lens_candidates))
        lens_pages = lens_engine.last_results_breakdown.get("pages", sum(1 for c in lens_candidates if c.page_url))

        if crop_img.size > 0:
            try:
                crop_candidates = lens_engine.search(str(crop_path))
                candidates_per_engine["google_lens_face"] = crop_candidates
                raw_candidate_count += len(crop_candidates)
                lens_visual += len(crop_candidates)
                lens_pages += sum(1 for c in crop_candidates if c.page_url)
            except Exception:
                pass
    except Exception as exc:
        logger.debug("Google Lens error: %s", exc)
        candidates_per_engine["google_lens"] = []

    print("Google Lens:")
    print(f"Exact matches: {lens_exact}")
    print(f"Visual matches: {lens_visual}")
    print(f"Matching pages: {lens_pages}")

    # B. Google Reverse Image
    rev_engine = SerpApiReverseImageEngine()
    rev_img_results = 0
    rev_inline_images = 0
    try:
        rev_candidates = rev_engine.search(str(resolved_path))
        candidates_per_engine["google_reverse_image"] = rev_candidates
        raw_candidate_count += len(rev_candidates)
        rev_img_results = rev_engine.last_results_breakdown.get("image_results", len(rev_candidates))
        rev_inline_images = rev_engine.last_results_breakdown.get("inline_images", 0)
    except Exception as exc:
        logger.debug("Google Reverse Image error: %s", exc)
        candidates_per_engine["google_reverse_image"] = []

    print("\nGoogle Reverse Image:")
    print(f"Image results: {rev_img_results}")
    print(f"Inline images: {rev_inline_images}")

    # C. Google Cloud Vision
    cv_engine = CloudVisionEngine()
    cv_pages = 0
    cv_full = 0
    cv_partial = 0
    cv_visual = 0
    try:
        cv_candidates = cv_engine.search(str(resolved_path))
        candidates_per_engine["google_cloud_vision"] = cv_candidates
        raw_candidate_count += len(cv_candidates)
        cv_pages = cv_engine.last_results_breakdown.get("matching_pages", sum(1 for c in cv_candidates if c.page_url))
        cv_visual = cv_engine.last_results_breakdown.get("matching_images", len(cv_candidates))
    except Exception as exc:
        logger.debug("Cloud Vision error: %s", exc)
        candidates_per_engine["google_cloud_vision"] = []

    print("\nGoogle Cloud Vision:")
    print(f"Matching pages: {cv_pages}")
    print(f"Full matches: {cv_full}")
    print(f"Partial matches: {cv_partial}")
    print(f"Visual matches: {cv_visual}")

    # D. Optional Contextual / OCR Search (Layer 5)
    if detected_texts:
        context_engine = ContextSearchEngine()
        context_queries = ocr.build_context_queries(detected_texts)
        if context_queries:
            try:
                context_candidates = context_engine.search_queries(context_queries)
                if context_candidates:
                    candidates_per_engine["ocr_context"] = context_candidates
                    raw_candidate_count += len(context_candidates)
            except Exception:
                pass

    # Clean up crop temp file
    try:
        if crop_path.exists():
            crop_path.unlink()
    except Exception:
        pass

    # ── [6] CANDIDATE COLLECTION (Layers 1-6) ─────────────────────────────────
    print("\n[6] CANDIDATE COLLECTION")

    # Layer 6: Extract images from discovered public web pages
    all_initial = []
    for c_list in candidates_per_engine.values():
        all_initial.extend(c_list)

    try:
        page_extracted_candidates = expand_candidates_from_pages(all_initial, max_pages_to_scrape=4)
        if page_extracted_candidates:
            candidates_per_engine["page_scraper"] = page_extracted_candidates
            raw_candidate_count += len(page_extracted_candidates)
    except Exception as exc:
        logger.debug("Page scraping notice: %s", exc)

    all_candidates = aggregate_candidates(
        candidates_per_engine, max_candidates=config.MAX_CANDIDATES
    )

    print(f"Total raw candidates: {raw_candidate_count}")
    print(f"After deduplication: {len(all_candidates)}")

    if not all_candidates:
        print("\n[11] FINAL RESULT\nNO_DISCOVERY_RESULTS\nReason: No candidates found online.")
        return 1

    # ── [7] FACE VERIFICATION ─────────────────────────────────────────────────
    print("\n[7] FACE VERIFICATION")
    from app.candidates.collector import (
        verify_and_score_candidate,
        select_best_candidate,
        select_verified_candidates,
        determine_pipeline_state,
    )

    visual_verifier = get_visual_verifier()
    input_visual_emb = visual_verifier.embed(image_bgr)

    from concurrent.futures import ThreadPoolExecutor

    def _eval_cand(c):
        return verify_and_score_candidate(
            candidate=c,
            input_embedding=input_embedding,
            input_image_path=str(resolved_path),
            input_visual_emb=input_visual_emb,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        scored_candidates = list(executor.map(_eval_cand, all_candidates))

    for idx, sc in enumerate(scored_candidates, 1):
        cand_url = sc.candidate.page_url or sc.candidate.image_url
        has_face = sc.candidate.image_available and sc.face_result.candidate_face_index >= 0
        face_count = sc.candidate.face_count if hasattr(sc.candidate, "face_count") else (1 if has_face else 0)
        source_label = sc.candidate.source_engine or "web"

        result_label = sc.face_result.verdict
        if sc.accepted:
            result_label = "MATCH"
        elif not has_face:
            result_label = "REJECT_NO_FACE"
        else:
            result_label = "NO_MATCH"

        print(f"\nCandidate {idx}")
        print(f"Source: {source_label}")
        print(f"URL: {cand_url}")
        print(f"Faces detected: {face_count}")
        if has_face:
            print(f"Best face similarity: {sc.face_result.similarity:.4f}")
            print(f"DINOv2 similarity: {sc.visual_result.similarity:.4f}")
        print(f"Result: {result_label}")

    candidates_with_faces = sum(1 for sc in scored_candidates if sc.candidate.image_available and sc.face_result.candidate_face_index >= 0)
    highest_face_sim = max((sc.face_result.similarity for sc in scored_candidates), default=0.0)
    print("\n--- Discovery & Verification Summary ---")
    print(f"Total candidates discovered: {raw_candidate_count}")
    print(f"Candidates after deduplication: {len(all_candidates)}")
    print(f"Candidates containing faces: {candidates_with_faces}")
    print(f"Highest ArcFace similarity: {highest_face_sim:.4f}")

    best = select_best_candidate(scored_candidates)
    verified_candidates = select_verified_candidates(scored_candidates)
    pipeline_state = determine_pipeline_state(True, len(all_candidates), scored_candidates)

    # ── [8] VERIFIED MATCH ────────────────────────────────────────────────────
    print("\n[8] VERIFIED MATCH")
    if best:
        platform_str = best.candidate.domain or "Web"
        if "instagram" in platform_str.lower():
            platform_str = "Instagram"
        elif "x.com" in platform_str.lower() or "twitter" in platform_str.lower():
            platform_str = "X (Twitter)"
        elif "linkedin" in platform_str.lower():
            platform_str = "LinkedIn"
        elif "facebook" in platform_str.lower():
            platform_str = "Facebook"
        elif "youtube" in platform_str.lower():
            platform_str = "YouTube"

        print(f"Platform: {platform_str}")
        print(f"Post URL: {best.candidate.page_url or best.candidate.image_url}")
        print(f"Face similarity: {best.face_result.similarity:.2f}")
        print(f"Visual similarity: {best.visual_result.similarity:.2f}")

        if len(verified_candidates) > 1:
            print(f"\nAll Verified Matches ({len(verified_candidates)} photos):")
            for v_idx, vc in enumerate(verified_candidates, 1):
                v_url = vc.candidate.page_url or vc.candidate.image_url
                print(f"  [{v_idx}] {vc.candidate.domain}: {v_url} (face_sim: {vc.face_result.similarity:.2f})")
    else:
        print("None")

    # ── [9] EVIDENCE MANIFEST & SHA-256 ───────────────────────────────────────
    print("\n[9] EVIDENCE")
    from app.search.entity import extract_person_name, categorize_social_post
    from app.evidence.manifest import build_manifest, build_no_match_manifest
    from app.evidence.hasher import save_manifest

    all_titles = [sc.candidate.title for sc in scored_candidates if sc.candidate.title]
    recognized_person = extract_person_name(all_titles)

    verified_social = []
    for sc in verified_candidates:
        if sc.candidate.candidate_type == "social" or any(s in sc.candidate.domain for s in ["instagram", "x.com", "twitter", "linkedin", "facebook", "tiktok", "threads", "youtube"]):
            social_info = categorize_social_post(sc.candidate.page_url, sc.candidate.domain)
            verified_social.append({
                "platform": social_info["platform"],
                "post_type": social_info["post_type"],
                "url": sc.candidate.page_url,
                "title": sc.candidate.title,
                "face_similarity": round(sc.face_result.similarity, 4),
                "visual_similarity": round(sc.visual_result.similarity, 4),
                "score": round(sc.final_score, 4),
            })

    if best:
        manifest = build_manifest(
            best=best,
            input_image_path=str(resolved_path),
            recognized_person=recognized_person,
            verified_social_posts=verified_social,
        )
    else:
        manifest = build_no_match_manifest(
            input_image_path=str(resolved_path),
            reason=pipeline_state,
        )

    evidence_hash, manifest_file = save_manifest(manifest)
    print(f"Content SHA-256: {evidence_hash}")

    # ── [10] BLOCKCHAIN ANCHORING ─────────────────────────────────────────────
    print("\n[10] BLOCKCHAIN")
    network_name = "Ethereum Sepolia" if "sepolia" in config.POLYGON_RPC_URL.lower() else "Polygon Amoy"
    print(f"Network: {network_name}")
    print(f"Contract: {config.CONTRACT_ADDRESS or 'Not deployed'}")

    receipt = None
    if best and not config.MOCK_MODE and config.CONTRACT_ADDRESS and config.PRIVATE_KEY:
        try:
            from app.blockchain.client import BlockchainClient
            client = BlockchainClient()
            receipt = client.anchor_evidence(
                record_id_hex=manifest.get("record_id", ""),
                evidence_hash_hex=evidence_hash,
            )
            if receipt and receipt.get("transaction_hash"):
                print(f"Transaction hash: {receipt['transaction_hash']}")
                print(f"Block number: {receipt.get('block_number', 'Confirmed')}")
        except Exception as exc:
            print(f"Blockchain notice: {exc}")
    else:
        if not best:
            print("[Info] Blockchain anchoring skipped (no verified match).")
        else:
            print("[Info] Blockchain anchoring skipped (Mock/Configuration).")

    # ── [11] FINAL RESULT ─────────────────────────────────────────────────────
    print("\n[11] FINAL RESULT\n")
    if best:
        print("MATCH FOUND")
        print(f"Evidence anchored on {network_name}")
        print("========================================")
        return 0
    else:
        print("NO_VERIFIED_MATCH")
        print("Reason: Web discovery returned candidate images, but none passed face verification.")
        print("========================================")
        return 1


# ═════════════════════════════════════════════════════════════════════════════
#  AUDIT MODE
# ═════════════════════════════════════════════════════════════════════════════

def run_audit(manifest_path: str, mock_blockchain: bool) -> int:
    """Run audit mode — verify manifest against blockchain."""
    from app.audit.verifier import audit_manifest

    _hr()
    print("========================================")
    print("BLOCKCHAIN AUDIT")
    print("========================================")
    print(f"Manifest: {manifest_path}")
    if mock_blockchain:
        print("Mode: offline (comparing against local .sha256 file)")
    else:
        print("Mode: live blockchain query")
    print()

    result = audit_manifest(manifest_path, use_mock=mock_blockchain)

    print(f"Local evidence hash:\n{result.local_hash}")
    print(f"\nOn-chain evidence hash:\n{result.chain_hash}")

    if result.verified:
        print("\nStatus:\nVERIFIED — EVIDENCE UNCHANGED")
        print("========================================")
        return 0
    else:
        print("\nStatus:\nTAMPERED — EVIDENCE HASH DOES NOT MATCH")
        print("========================================")
        return 1


# ═════════════════════════════════════════════════════════════════════════════
#  TAMPER DEMO MODE
# ═════════════════════════════════════════════════════════════════════════════

def run_tamper_demo(manifest_path: str) -> int:
    """Demonstrate tamper detection."""
    from app.audit.verifier import demonstrate_tamper, audit_manifest

    _hr()
    print("========================================")
    print("TRACE-ID — TAMPER DETECTION DEMO")
    print("========================================")
    print(f"Original manifest: {manifest_path}\n")

    result = demonstrate_tamper(manifest_path)
    if not result["hashes_differ"]:
        print("ERROR: Tampered manifest produced same hash (unexpected)")
        return 2

    print("1. Original Evidence Verification:")
    print(f"  Field: {result['field_modified']} = {result['original_value']}")
    print(f"  Local SHA-256:    {result['original_hash']}")
    print(f"  On-chain SHA-256: {result['original_hash']}")
    print("  Status: VERIFIED — EVIDENCE UNCHANGED\n")

    print("2. Simulating Malicious Manifest Modification:")
    print(f"  Altered: {result['field_modified']} = {result['tampered_value']}")
    print(f"  Tampered SHA-256: {result['tampered_hash']}\n")

    print("3. Re-running Blockchain Audit on Tampered Evidence:")
    print(f"  Local evidence hash:\n  {result['tampered_hash']}")
    print(f"  On-chain evidence hash:\n  {result['original_hash']}")
    print("  Status: TAMPERED — EVIDENCE HASH DOES NOT MATCH\n")

    print("Result: VERIFIED -> TAMPERED successfully demonstrated.")
    print("========================================")
    return 0


# ═════════════════════════════════════════════════════════════════════════════
#  CLI ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

@click.command()
@click.option(
    "--image", "--input", "-i",
    type=click.Path(),
    default=None,
    help="Path to input image for full pipeline run.",
)
@click.option(
    "--max-candidates", "-n",
    type=int,
    default=None,
    help="Maximum number of candidates to evaluate (overrides .env).",
)
@click.option(
    "--face-index", "-f",
    type=int,
    default=None,
    help="Index of detected face to select (0 = primary most prominent face).",
)
@click.option(
    "--audit", "-a",
    type=click.Path(),
    default=None,
    help="Path to manifest.json for audit/replay verification.",
)
@click.option(
    "--mock-blockchain",
    is_flag=True,
    default=False,
    help="In audit mode: compare against local .sha256 file instead of querying blockchain.",
)
@click.option(
    "--tamper-demo", "-t",
    type=click.Path(),
    default=None,
    help="Demonstrate tamper detection on a manifest file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging (DEBUG level).",
)
def main(
    image: str | None,
    max_candidates: int | None,
    face_index: int | None,
    audit: str | None,
    mock_blockchain: bool,
    tamper_demo: str | None,
    verbose: bool,
) -> None:
    """
    TRACE-ID — Face ID + Multi-Source Reverse Image Search + Blockchain Verification

    \b
    Commands:
      Full pipeline:
        python -m app.main --image data/input/test.jpg
        python -m app.main --image data/input/test.jpg --face-index 0

      Audit manifest:
        python -m app.main --audit data/evidence/manifest.json

      Tamper demo:
        python -m app.main --tamper-demo data/evidence/manifest.json
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for mod in ["app.face", "app.search", "app.visual", "app.candidates",
                    "app.evidence", "app.blockchain", "app.audit", "app.ocr"]:
            logging.getLogger(mod).setLevel(logging.DEBUG)

    if tamper_demo:
        sys.exit(run_tamper_demo(tamper_demo))

    if audit:
        sys.exit(run_audit(audit, mock_blockchain=mock_blockchain))

    if image:
        sys.exit(run_pipeline(image, max_candidates=max_candidates, selected_face_index=face_index))

    # No mode specified
    click.echo("TRACE-ID — No mode specified. Use --help for usage.")
    click.echo("  Full pipeline: python -m app.main --image <path>")
    click.echo("  Audit:         python -m app.main --audit <manifest.json>")
    click.echo("  Tamper demo:   python -m app.main --tamper-demo <manifest.json>")
    sys.exit(1)


if __name__ == "__main__":
    main()
