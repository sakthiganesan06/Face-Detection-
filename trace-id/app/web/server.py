"""
TRACE-ID — Web Application Server
FastAPI backend providing REST endpoints and serving the modern Web UI.
"""

from __future__ import annotations

import cv2
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.audit.verifier import audit_manifest, demonstrate_tamper
from app.blockchain.client import BlockchainClient
from app.candidates.collector import select_best_candidate, verify_and_score_candidate
from app.evidence.hasher import save_manifest
from app.evidence.manifest import build_manifest, build_no_match_manifest
from app.face.detector import FaceDetector, load_image, validate_image_path, select_primary_face
from app.search.aggregator import aggregate_candidates
from app.search.cloud_vision import CloudVisionEngine
from app.search.entity import categorize_social_post, extract_person_name
from app.search.serpapi_lens import SerpApiLensEngine
from app.search.serpapi_reverse import SerpApiReverseImageEngine
from app.visual.dinov2 import get_verifier as get_visual_verifier

logger = logging.getLogger("trace_id_web")

app = FastAPI(
    title="TRACE-ID Web API",
    description="Face ID + Multi-Source Reverse Image Search + Blockchain Verification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/status")
async def get_system_status():
    """Return live system configuration and blockchain network info."""
    return {
        "status": "ready",
        "mock_mode": config.MOCK_MODE,
        "contract_address": config.CONTRACT_ADDRESS or "Not deployed",
        "rpc_url": config.POLYGON_RPC_URL,
        "network": "Ethereum Sepolia Testnet" if "sepolia" in config.POLYGON_RPC_URL.lower() else "Polygon Amoy Testnet",
        "serpapi_configured": bool(config.SERPAPI_API_KEY),
        "vision_configured": bool(config.GOOGLE_APPLICATION_CREDENTIALS and Path(config.GOOGLE_APPLICATION_CREDENTIALS).exists()),
        "thresholds": {
            "face_similarity": config.FACE_SIMILARITY_THRESHOLD,
            "visual_similarity": config.VISUAL_SIMILARITY_THRESHOLD,
        },
    }


@app.on_event("startup")
async def startup_event():
    """Warm up ML models on startup so requests are fast."""
    try:
        FaceDetector()._ensure_loaded()
        get_visual_verifier()._ensure_loaded()
        logger.info("ML Models (ArcFace & DINOv2) preloaded and warm.")
    except Exception as exc:
        logger.warning("Model warm-up note: %s", exc)


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    max_candidates: int = Form(12),
    face_index: Optional[int] = Form(None),
):
    """
    Execute the full end-to-end TRACE-ID pipeline on an uploaded image.
    """
    from concurrent.futures import ThreadPoolExecutor

    start_time = time.time()
    temp_dir = Path(tempfile.mkdtemp(prefix="traceid_upload_"))
    temp_file_path = temp_dir / file.filename

    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Load image & detect faces
        detector = FaceDetector()
        img_bgr = load_image(temp_file_path)
        faces = detector.detect(img_bgr)

        if not faces:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the uploaded image. Please upload a clear photo containing a face.",
            )

        primary_face, chosen_idx = select_primary_face(faces, selected_index=face_index)
        input_embedding = primary_face.embedding

        # Pre-compute query DINOv2 visual embedding once
        visual_verifier = get_visual_verifier()
        input_visual_emb = visual_verifier.embed(img_bgr)

        # Save cropped face with 25% padding for face-focused reverse search (ideal for finding reels / other posts)
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = primary_face.bbox
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)
        face_crop_img = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_file_path = temp_dir / f"crop_{file.filename}"
        cv2.imwrite(str(crop_file_path), face_crop_img)

        # 2. Reverse image search (Full image + Face crop)
        candidates_by_engine = {}
        engines_used = []

        # A. Google Lens (SerpApi)
        if config.SERPAPI_API_KEY or config.MOCK_MODE:
            try:
                lens_engine = SerpApiLensEngine()
                lens_candidates = lens_engine.search(str(temp_file_path))
                candidates_by_engine["google_lens"] = lens_candidates
                engines_used.append("google_lens")

                if face_crop_img.size > 0:
                    try:
                        crop_candidates = lens_engine.search(str(crop_file_path))
                        candidates_by_engine["google_lens_face"] = crop_candidates
                    except Exception as ce:
                        logger.warning("Face crop search note: %s", ce)
            except Exception as e:
                logger.error("Google Lens search failed: %s", e)

        # B. Google Reverse Image (SerpApi)
        if config.SERPAPI_API_KEY or config.MOCK_MODE:
            try:
                rev_engine = SerpApiReverseImageEngine()
                rev_candidates = rev_engine.search(str(temp_file_path))
                candidates_by_engine["google_reverse_image"] = rev_candidates
                engines_used.append("google_reverse_image")
            except Exception as e:
                logger.warning("Google Reverse Image search notice: %s", e)

        # C. Google Cloud Vision
        if config.GOOGLE_APPLICATION_CREDENTIALS and not config.MOCK_MODE:
            try:
                vision_engine = CloudVisionEngine()
                vision_candidates = vision_engine.search(str(temp_file_path))
                candidates_by_engine["google_cloud_vision"] = vision_candidates
                engines_used.append("google_cloud_vision")
            except Exception as e:
                logger.warning("Cloud Vision search notice: %s", e)

        # Aggregate candidates
        aggregated = aggregate_candidates(
            candidates_by_engine, max_candidates=max_candidates
        )


        if not aggregated:
            return JSONResponse(
                content={
                    "success": False,
                    "status": "NO_SEARCH_RESULTS",
                    "message": "No matching candidates found online for this image.",
                    "face_detected": True,
                    "face_confidence": round(float(primary_face.confidence), 4),
                }
            )

        # 3. Verify & score candidates in parallel
        def _score_candidate(cand):
            return verify_and_score_candidate(
                candidate=cand,
                input_embedding=input_embedding,
                input_image_path=str(temp_file_path),
                input_visual_emb=input_visual_emb,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            scored_candidates = list(executor.map(_score_candidate, aggregated))

        best = select_best_candidate(scored_candidates)

        # 4. Extract Recognized Person & Social Media Matches
        all_titles = [sc.candidate.title for sc in scored_candidates if sc.candidate.title]
        recognized_person = extract_person_name(all_titles)

        verified_social_posts = []
        all_candidate_cards = []

        for sc in scored_candidates:
            card_info = {
                "domain": sc.candidate.domain,
                "title": sc.candidate.title,
                "page_url": sc.candidate.page_url,
                "image_url": sc.candidate.image_url or sc.candidate.thumbnail_url,
                "face_similarity": round(float(sc.face_result.similarity), 4),
                "visual_similarity": round(float(sc.visual_result.similarity), 4),
                "final_score": round(float(sc.final_score), 4),
                "face_passed": bool(sc.face_result.passed),
                "accepted": bool(sc.accepted),
                "candidate_type": sc.candidate.candidate_type,
            }
            all_candidate_cards.append(card_info)

            if sc.accepted and (
                sc.candidate.candidate_type == "social"
                or any(s in sc.candidate.domain for s in ["instagram", "x.com", "twitter", "linkedin", "facebook", "tiktok", "threads", "youtube"])
            ):
                social_meta = categorize_social_post(sc.candidate.page_url, sc.candidate.domain)
                verified_social_posts.append({
                    "platform": social_meta["platform"],
                    "post_type": social_meta["post_type"],
                    "url": sc.candidate.page_url,
                    "title": sc.candidate.title,
                    "thumbnail": sc.candidate.thumbnail_url or sc.candidate.image_url,
                    "face_similarity": round(float(sc.face_result.similarity), 4),
                    "visual_similarity": round(float(sc.visual_result.similarity), 4),
                    "score": round(float(sc.final_score), 4),
                })

        if not best:
            scored_candidates.sort(key=lambda s: s.face_result.similarity, reverse=True)
            top_sim = round(scored_candidates[0].face_result.similarity * 100, 1) if scored_candidates else 0
            return JSONResponse(
                content={
                    "success": False,
                    "status": "NO_VERIFIED_MATCH",
                    "message": f"Scanned {len(scored_candidates)} web & social sources. Highest facial similarity was {top_sim}% (below {round(config.FACE_SIMILARITY_THRESHOLD * 100)}% verified biometric threshold).",
                    "face_detected": True,
                    "face_confidence": round(float(primary_face.confidence), 4),
                    "candidates_evaluated": len(scored_candidates),
                    "all_candidates": all_candidate_cards,
                }
            )

        # 5. Build Evidence Manifest & Hash
        manifest = build_manifest(
            best=best,
            input_image_path=str(temp_file_path),
            recognized_person=recognized_person,
            verified_social_posts=verified_social_posts,
        )
        evidence_hash, _ = save_manifest(manifest)

        # 6. Anchor to Blockchain
        blockchain_receipt = None
        if not config.MOCK_MODE and config.CONTRACT_ADDRESS and config.PRIVATE_KEY:
            try:
                bc_client = BlockchainClient()
                blockchain_receipt = bc_client.anchor_evidence(
                    record_id_hex=manifest["record_id"],
                    evidence_hash_hex=evidence_hash,
                )
            except Exception as exc:
                logger.warning("Blockchain anchoring note: %s", exc)
                # If already anchored or network issue, generate informational receipt
                blockchain_receipt = {
                    "contract_address": config.CONTRACT_ADDRESS,
                    "record_id": manifest["record_id"],
                    "evidence_hash": evidence_hash,
                    "status_note": str(exc),
                }

        elapsed = time.time() - start_time

        return JSONResponse(
            content={
                "success": True,
                "status": "VERIFIED_MATCH",
                "elapsed_seconds": round(elapsed, 2),
                "recognized_person": recognized_person or "Public Subject Identified",
                "best_match": {
                    "domain": best.candidate.domain,
                    "title": best.candidate.title,
                    "page_url": best.candidate.page_url,
                    "image_url": best.candidate.image_url or best.candidate.thumbnail_url,
                    "face_similarity": round(float(best.face_result.similarity), 4),
                    "visual_similarity": round(float(best.visual_result.similarity), 4),
                    "final_score": round(float(best.final_score), 4),
                    "candidate_type": best.candidate.candidate_type,
                },
                "verified_social_posts": verified_social_posts,
                "all_candidates": all_candidate_cards,
                "evidence": {
                    "record_id": manifest["record_id"],
                    "evidence_hash": evidence_hash,
                    "manifest": manifest,
                },
                "blockchain_receipt": blockchain_receipt,
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/audit")
async def audit_record():
    """Audit the current manifest against on-chain Ethereum smart contract."""
    manifest_path = config.MANIFEST_PATH
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No manifest file found to audit.")

    result = audit_manifest(manifest_path, use_mock=config.MOCK_MODE)
    return {
        "verified": result.verified,
        "record_id": result.record_id,
        "local_hash": result.local_hash,
        "chain_hash": result.chain_hash,
        "error": result.error,
        "manifest_path": str(result.manifest_path),
    }


from app.audit.verifier import audit_manifest, demonstrate_tamper

# (rest of code)
@app.post("/api/tamper-demo")
async def tamper_demo():
    """Run interactive tamper demonstration."""
    manifest_path = config.MANIFEST_PATH
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No manifest file found.")

    res = demonstrate_tamper(manifest_path)
    return {
        "original_hash": res["original_hash"],
        "tampered_hash": res["tampered_hash"],
        "hashes_differ": res["hashes_differ"],
        "status": "TAMPER_DETECTED" if res["hashes_differ"] else "MATCH",
        "details": f"{res['field_modified']} ({res['original_value']} -> {res['tampered_value']})",
    }


# Serve sample images from data directory
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
if _DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(_DATA_DIR)), name="data")

# Serve frontend static assets
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
