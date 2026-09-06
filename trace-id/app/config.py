"""
TRACE-ID — Configuration
All settings loaded from environment variables via python-dotenv.
Never hard-code secrets or credentials here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root (trace-id/) from this file's location (trace-id/app/)
_PROJECT_ROOT = Path(__file__).parent.parent

# Load .env from project root; silently skip if not present
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ── API Keys ───────────────────────────────────────────────────────────────────

SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
"""SerpApi API key. Get yours at https://serpapi.com/"""

GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
"""Path to Google Cloud service account JSON. Used by google-cloud-vision."""

# ── Blockchain ─────────────────────────────────────────────────────────────────

POLYGON_RPC_URL: str = os.getenv(
    "POLYGON_RPC_URL", "https://rpc-amoy.polygon.technology/"
)
"""Polygon Amoy testnet RPC endpoint."""

PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
"""Wallet private key for Polygon Amoy transactions. NEVER print or log."""

CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")
"""Deployed EvidenceRegistry contract address."""

# ── Similarity Thresholds ──────────────────────────────────────────────────────

FACE_SIMILARITY_THRESHOLD: float = float(
    os.getenv("FACE_SIMILARITY_THRESHOLD", "0.38")
)
"""
Cosine similarity threshold for InsightFace buffalo_l / ArcFace.
Typical range: 0.3 (lenient) – 0.7 (strict).
Must be calibrated for your chosen model and test set.
Default 0.45 is a conservative starting point.
"""

VISUAL_SIMILARITY_THRESHOLD: float = float(
    os.getenv("VISUAL_SIMILARITY_THRESHOLD", "0.60")
)
"""
Cosine similarity threshold for DINOv2 ViT-S/14 visual embeddings.
Used for independent visual verification of candidates.
"""

# ── Scoring Weights ────────────────────────────────────────────────────────────

FACE_WEIGHT: float = float(os.getenv("FACE_WEIGHT", "0.60"))
VISUAL_WEIGHT: float = float(os.getenv("VISUAL_WEIGHT", "0.30"))
DISCOVERY_WEIGHT: float = float(os.getenv("DISCOVERY_WEIGHT", "0.10"))

"""
Final score formula (documented publicly in README):
  final_score = FACE_WEIGHT * face_similarity
              + VISUAL_WEIGHT * visual_similarity
              + DISCOVERY_WEIGHT * discovery_confidence
Face verification PASS is mandatory regardless of final score.
"""

# ── Pipeline Settings ──────────────────────────────────────────────────────────

MAX_CANDIDATES: int = int(os.getenv("MAX_CANDIDATES", "10"))
"""Maximum number of candidates to evaluate per pipeline run."""

MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")
"""
When True: uses mocked search results — no API keys required.
All mock output is clearly labeled [MOCK MODE].
Never use mock results for real evidence claims.
"""

# ── File Paths ─────────────────────────────────────────────────────────────────

DATA_DIR = _PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
CANDIDATES_DIR = DATA_DIR / "candidates"
EVIDENCE_DIR = DATA_DIR / "evidence"
RESULTS_DIR = DATA_DIR / "results"

MANIFEST_PATH = EVIDENCE_DIR / "manifest.json"
MANIFEST_HASH_PATH = EVIDENCE_DIR / "manifest.sha256"
BLOCKCHAIN_RECEIPT_PATH = RESULTS_DIR / "blockchain_receipt.json"

# ── InsightFace ────────────────────────────────────────────────────────────────

INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
"""
InsightFace model pack. buffalo_l includes:
  - detection: SCRFD 10G
  - recognition: ArcFace R100 (512-dim embeddings)
Downloaded automatically on first run (~200 MB).
"""

INSIGHTFACE_CTX_ID: int = -1
"""
GPU context ID for InsightFace.
-1 = CPU inference. Set to 0 for first CUDA GPU.
"""

# ── DINOv2 ────────────────────────────────────────────────────────────────────

DINOV2_MODEL_NAME: str = "dinov2_vits14"
"""
DINOv2 variant. dinov2_vits14 = ViT-S/14 (21M params, 384-dim).
Loaded via torch.hub. Downloaded on first run (~80 MB).
Larger alternatives: dinov2_vitb14, dinov2_vitl14, dinov2_vitg14.
"""

DINOV2_IMAGE_SIZE: int = 224
"""Input resolution for DINOv2 preprocessing."""

# ── Download Settings ──────────────────────────────────────────────────────────

DOWNLOAD_TIMEOUT_SECONDS: int = 5
DOWNLOAD_MAX_RETRIES: int = 1
DOWNLOAD_USER_AGENT: str = (
    "TRACE-ID/1.0 (research pipeline; contact: trace-id@example.com)"
)

# ── Social Media Domains ───────────────────────────────────────────────────────

SOCIAL_MEDIA_DOMAINS: frozenset[str] = frozenset(
    {
        "x.com",
        "twitter.com",
        "instagram.com",
        "facebook.com",
        "fb.com",
        "linkedin.com",
        "threads.net",
        "tiktok.com",
        "youtube.com",
        "reddit.com",
        "pinterest.com",
        "snapchat.com",
        "tumblr.com",
        "flickr.com",
        "vk.com",
        "weibo.com",
    }
)


def validate_config() -> list[str]:
    """
    Return a list of missing/invalid configuration warnings.
    Does not raise — caller decides whether to abort.
    """
    warnings: list[str] = []
    if not MOCK_MODE:
        if not SERPAPI_API_KEY:
            warnings.append("SERPAPI_API_KEY is not set (required for Google Lens search)")
        if not GOOGLE_APPLICATION_CREDENTIALS:
            warnings.append(
                "GOOGLE_APPLICATION_CREDENTIALS is not set (required for Cloud Vision)"
            )
        if not POLYGON_RPC_URL:
            warnings.append("POLYGON_RPC_URL is not set (required for blockchain anchoring)")
        if not PRIVATE_KEY:
            warnings.append("PRIVATE_KEY is not set (required for blockchain anchoring)")
        if not CONTRACT_ADDRESS:
            warnings.append(
                "CONTRACT_ADDRESS is not set "
                "(deploy contract first: python -m app.blockchain.deploy)"
            )
    if abs(FACE_WEIGHT + VISUAL_WEIGHT + DISCOVERY_WEIGHT - 1.0) > 1e-6:
        warnings.append(
            f"Scoring weights do not sum to 1.0: "
            f"FACE={FACE_WEIGHT} VISUAL={VISUAL_WEIGHT} DISCOVERY={DISCOVERY_WEIGHT}"
        )
    return warnings
