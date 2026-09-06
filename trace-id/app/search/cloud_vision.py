"""
TRACE-ID — Google Cloud Vision Web Detection Engine
Secondary reverse image search source.

Uses the Google Cloud Vision API Web Detection feature to extract:
  - pages with matching images
  - full matching images
  - partial matching images
  - visually similar images
  - web entities

All candidates originate from the live API response — nothing is hardcoded.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from app import config
from app.search.base import (
    BaseSearchEngine,
    Candidate,
    classify_candidate_type,
    extract_domain,
    normalize_url,
)

logger = logging.getLogger(__name__)

# ── Mock data for MOCK_MODE ───────────────────────────────────────────────────
_MOCK_PAGES: list[dict] = [
    {
        "url": "https://example.com/news/mock-article",
        "page_title": "[MOCK] News Article About Person",
        "full_matching_images": [{"url": "https://via.placeholder.com/200"}],
    },
]
_MOCK_FULL_IMAGES: list[dict] = [
    {"url": "https://via.placeholder.com/500"},
]
_MOCK_SIMILAR_IMAGES: list[dict] = [
    {"url": "https://via.placeholder.com/300"},
]


class CloudVisionEngine(BaseSearchEngine):
    """
    Google Cloud Vision Web Detection.

    Credentials are read from GOOGLE_APPLICATION_CREDENTIALS env var
    (path to service account JSON key file), which google-cloud-vision
    picks up automatically via Application Default Credentials.

    Result sections parsed:
      pages_with_matching_images → page_url + best matching image_url
      full_matching_images       → image_url (exact image match)
      partial_matching_images    → image_url (partial match)
      visually_similar_images    → image_url (visually similar)
      web_entities               → used for title/metadata only
    """

    engine_name = "google_cloud_vision"

    def __init__(self) -> None:
        self.last_results_breakdown: dict[str, int] = {
            "matching_pages": 0,
            "matching_images": 0,
        }

    def search(self, image_path: str) -> list[Candidate]:

        """
        Send image to Google Cloud Vision Web Detection and return candidates.

        Args:
            image_path: Absolute path to input image file.

        Returns:
            List of Candidate objects from the API response.
        """
        if config.MOCK_MODE:
            return self._mock_results()

        if not config.GOOGLE_APPLICATION_CREDENTIALS:
            logger.warning(
                "GOOGLE_APPLICATION_CREDENTIALS not set — "
                "Google Cloud Vision will attempt Application Default Credentials."
            )

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found for Cloud Vision: {path}")

        try:
            from google.cloud import vision
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-vision not installed. "
                "Run: pip install google-cloud-vision"
            ) from exc

        logger.info("Google Cloud Vision: sending image '%s' for web detection", path.name)

        try:
            client = vision.ImageAnnotatorClient()
            with open(path, "rb") as f:
                content = f.read()

            image = vision.Image(content=content)
            response = client.web_detection(image=image)

            if response.error.message:
                raise RuntimeError(
                    f"Google Cloud Vision API error: {response.error.message}"
                )

            web = response.web_detection
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Google Cloud Vision Web Detection error: {exc}"
            ) from exc

        return self._parse_web_detection(web)

    def _parse_web_detection(self, web: Any) -> list[Candidate]:
        """Parse a WebDetection proto object into Candidate objects."""
        candidates: list[Candidate] = []

        # ── pages with matching images ────────────────────────────────────────
        for page in web.pages_with_matching_images:
            page_url = normalize_url(page.url)
            domain = extract_domain(page_url)

            # Take the best image URL from this page's matching images
            image_url = ""
            for full_img in page.full_matching_images:
                image_url = normalize_url(full_img.url)
                break
            if not image_url:
                for part_img in page.partial_matching_images:
                    image_url = normalize_url(part_img.url)
                    break

            candidates.append(
                Candidate(
                    source_engine=self.engine_name,
                    image_url=image_url,
                    page_url=page_url,
                    domain=domain,
                    title=page.page_title if hasattr(page, "page_title") else "",
                    candidate_type=classify_candidate_type(domain),
                    discovery_sources=[self.engine_name],
                )
            )

        # ── full matching images (direct image URL matches) ───────────────────
        for img in web.full_matching_images:
            image_url = normalize_url(img.url)
            domain = extract_domain(image_url)
            candidates.append(
                Candidate(
                    source_engine=self.engine_name,
                    image_url=image_url,
                    page_url="",
                    domain=domain,
                    title="Full matching image",
                    candidate_type=classify_candidate_type(domain),
                    discovery_sources=[self.engine_name],
                )
            )

        # ── partial matching images ────────────────────────────────────────────
        for img in web.partial_matching_images:
            image_url = normalize_url(img.url)
            domain = extract_domain(image_url)
            candidates.append(
                Candidate(
                    source_engine=self.engine_name,
                    image_url=image_url,
                    page_url="",
                    domain=domain,
                    title="Partial matching image",
                    candidate_type=classify_candidate_type(domain),
                    discovery_sources=[self.engine_name],
                )
            )

        # ── visually similar images ───────────────────────────────────────────
        for img in web.visually_similar_images:
            image_url = normalize_url(img.url)
            domain = extract_domain(image_url)
            candidates.append(
                Candidate(
                    source_engine=self.engine_name,
                    image_url=image_url,
                    page_url="",
                    domain=domain,
                    title="Visually similar image",
                    candidate_type=classify_candidate_type(domain),
                    discovery_sources=[self.engine_name],
                )
            )

        pages_cnt = len(list(web.pages_with_matching_images)) if hasattr(web, "pages_with_matching_images") else 0
        matching_imgs_cnt = (
            (len(list(web.full_matching_images)) if hasattr(web, "full_matching_images") else 0)
            + (len(list(web.partial_matching_images)) if hasattr(web, "partial_matching_images") else 0)
            + (len(list(web.visually_similar_images)) if hasattr(web, "visually_similar_images") else 0)
        )
        self.last_results_breakdown = {
            "matching_pages": pages_cnt,
            "matching_images": matching_imgs_cnt,
        }

        logger.info(
            "Google Cloud Vision: parsed %d candidates from web detection",
            len(candidates),
        )
        return candidates

    def _mock_results(self) -> list[Candidate]:
        """Return clearly labeled mock candidates for MOCK_MODE."""
        logger.warning(
            "[MOCK MODE] Cloud Vision returning mock results — "
            "NOT real web detection results"
        )
        candidates: list[Candidate] = []
        for page in _MOCK_PAGES:
            page_url = normalize_url(page["url"])
            domain = extract_domain(page_url)
            img_list = page.get("full_matching_images", [])
            image_url = normalize_url(img_list[0]["url"]) if img_list else ""
            candidates.append(
                Candidate(
                    source_engine="google_cloud_vision [MOCK]",
                    image_url=image_url,
                    page_url=page_url,
                    domain=domain,
                    title=page.get("page_title", ""),
                    candidate_type=classify_candidate_type(domain),
                    discovery_sources=["google_cloud_vision [MOCK]"],
                )
            )
        for img in _MOCK_FULL_IMAGES:
            image_url = normalize_url(img["url"])
            domain = extract_domain(image_url)
            candidates.append(
                Candidate(
                    source_engine="google_cloud_vision [MOCK]",
                    image_url=image_url,
                    page_url="",
                    domain=domain,
                    title="[MOCK] Full matching image",
                    candidate_type=classify_candidate_type(domain),
                    discovery_sources=["google_cloud_vision [MOCK]"],
                )
            )
        return candidates
