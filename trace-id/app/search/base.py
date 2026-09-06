"""
TRACE-ID — Search Base
Defines the common Candidate dataclass and abstract BaseSearchEngine.
All search engine implementations must conform to this interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, urlunparse


@dataclass
class Candidate:
    """
    Unified candidate record — produced by any search engine.

    All fields that cannot be determined are left as empty strings / None.
    No field is hardcoded — all values originate from live API responses.
    """

    source_engine: str
    """Which search engine discovered this candidate: 'google_lens' | 'google_cloud_vision'"""

    image_url: str
    """Direct URL to an image associated with this candidate."""

    page_url: str = ""
    """URL of the web/social page where this image was found."""

    domain: str = ""
    """Extracted domain (e.g. 'x.com', 'instagram.com', 'example.com')."""

    title: str = ""
    """Title or description from search result."""

    candidate_type: str = "web"
    """'social' if domain is a known social platform, else 'web'."""

    thumbnail_url: str = ""
    """Small preview image URL if available."""

    discovery_sources: list[str] = field(default_factory=list)
    """List of engines that found this candidate (populated by aggregator)."""

    discovery_confidence: float = 1.0
    """
    Base confidence from discovery.
    1.0 = found by one engine; bumped toward 1.0 when found by multiple engines.
    Used in final scoring formula.
    """

    # ── Verification results (populated later in pipeline) ──────────────────
    face_count: int = 0
    face_similarity: Optional[float] = None
    face_passed: Optional[bool] = None
    visual_similarity: Optional[float] = None
    visual_passed: Optional[bool] = None
    final_score: Optional[float] = None


    # ── Download state ───────────────────────────────────────────────────────
    image_available: bool = False
    """Set to True after the image has been successfully downloaded."""

    local_image_path: str = ""
    """Local filesystem path to the downloaded candidate image."""

    image_sha256: str = ""
    """SHA-256 of the raw downloaded image bytes."""

    def to_dict(self) -> dict:
        """Serializable dict for JSON/manifest output. Excludes internal state."""
        return {
            "source_engine": self.source_engine,
            "page_url": self.page_url,
            "image_url": self.image_url,
            "domain": self.domain,
            "title": self.title,
            "candidate_type": self.candidate_type,
            "discovery_sources": self.discovery_sources,
            "discovery_confidence": self.discovery_confidence,
            "face_similarity": self.face_similarity,
            "face_passed": self.face_passed,
            "visual_similarity": self.visual_similarity,
            "visual_passed": self.visual_passed,
            "final_score": self.final_score,
        }


class BaseSearchEngine:
    """Abstract base class for search engines."""

    engine_name: str = "base"

    def search(self, image_path: str) -> list[Candidate]:
        """
        Perform reverse image search for the given image path.

        Args:
            image_path: Absolute path to the input image file.

        Returns:
            List of Candidate objects (may be empty).

        Raises:
            RuntimeError: On unrecoverable API errors.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.search() not implemented")


# ── URL Utilities ─────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication purposes.

    - Strip fragments (#...)
    - Lowercase scheme and host
    - Remove trailing slash from path
    - Remove common tracking parameters
    """
    if not url:
        return ""

    # Add scheme if missing (case-insensitive check)
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        return url

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

    # Strip tracking query params
    _TRACKING_PARAMS = frozenset(
        {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "ref", "referrer", "source", "_ga",
        }
    )
    from urllib.parse import parse_qs, urlencode
    qs = parse_qs(parsed.query, keep_blank_values=False)
    clean_qs = {k: v for k, v in qs.items() if k.lower() not in _TRACKING_PARAMS}
    query = urlencode(clean_qs, doseq=True)

    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    return normalized


def extract_domain(url: str) -> str:
    """Extract the registered domain from a URL (e.g. 'x.com' from 'https://x.com/...')."""
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        netloc = urlparse(url).netloc.lower()
        # Strip www. prefix
        netloc = re.sub(r"^www\.", "", netloc)
        # Strip port
        netloc = netloc.split(":")[0]
        return netloc
    except Exception:
        return ""


def classify_candidate_type(domain: str) -> str:
    """Return 'social' if domain is a known social platform, else 'web'."""
    from app.config import SOCIAL_MEDIA_DOMAINS
    # Check exact match and subdomain match
    if domain in SOCIAL_MEDIA_DOMAINS:
        return "social"
    for social in SOCIAL_MEDIA_DOMAINS:
        if domain.endswith("." + social):
            return "social"
    return "web"
