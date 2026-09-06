"""
TRACE-ID — SerpApi Google Lens Search Engine
Primary reverse image search source.

Uses the official SerpApi Google Lens API:
  https://serpapi.com/google-lens-api

Image is uploaded (multipart/form-data) according to the current API contract.
All candidate URLs originate from the live API response — nothing is hardcoded.
"""

from __future__ import annotations

import logging
import os
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
# These are structurally realistic but entirely fabricated candidates.
# They are returned ONLY when MOCK_MODE=true and are clearly labeled.
_MOCK_VISUAL_MATCHES: list[dict] = [
    {
        "title": "[MOCK] Example Public Profile",
        "link": "https://example.com/profile/mock-user",
        "thumbnail": "https://via.placeholder.com/100",
        "source": "example.com",
    },
    {
        "title": "[MOCK] Public Post",
        "link": "https://example.com/posts/mock-post-123",
        "thumbnail": "https://via.placeholder.com/100",
        "source": "example.com",
    },
]


class SerpApiLensEngine(BaseSearchEngine):
    """
    Google Lens reverse image search via SerpApi.

    Parses the following result sections dynamically:
      - visual_matches   (primary reverse-image results)
      - reverse_image    (direct image metadata)
      - knowledge_graph  (entity information if available)

    No social-media URLs, usernames, or post IDs are hardcoded here.
    All candidates originate from the live API response.
    """

    engine_name = "google_lens"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or config.SERPAPI_API_KEY
        self.last_results_breakdown: dict[str, int] = {
            "exact_matches": 0,
            "visual_matches": 0,
            "pages": 0,
        }


    def search(self, image_path: str) -> list[Candidate]:
        """
        Upload the image to SerpApi Google Lens and return parsed candidates.

        Args:
            image_path: Absolute path to input image file.

        Returns:
            List of Candidate objects from the API response.

        Raises:
            RuntimeError: On API authentication failure or network error.
        """
        if config.MOCK_MODE:
            return self._mock_results()

        if not self._api_key:
            raise RuntimeError(
                "SERPAPI_API_KEY is not set. "
                "Get your key at https://serpapi.com/ and add it to .env"
            )

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found for SerpApi search: {path}")

        try:
            from serpapi import GoogleSearch
        except ImportError as exc:
            raise RuntimeError(
                "google-search-results package not installed. "
                "Run: pip install google-search-results"
            ) from exc

        logger.info("SerpApi Google Lens: uploading image '%s'", path.name)

        # SerpApi Google Lens API — image upload via url_image or image_file
        # We use the file upload method as per current API contract.
        # Reference: https://serpapi.com/google-lens-api
        try:
            params: dict[str, Any] = {
                "engine": "google_lens",
                "api_key": self._api_key,
            }

            # Upload the image file using the current SerpApi contract
            with open(path, "rb") as f:
                image_bytes = f.read()

            # SerpApi Google Lens Image Upload:
            # 1. Upload local image to https://serpapi.com/image to obtain temporary image_id
            # 2. Query engine=google_lens with image_id
            import requests

            upload_url = "https://serpapi.com/image"
            with open(path, "rb") as f:
                upload_resp = requests.post(
                    upload_url,
                    files={"image": (path.name, f, self._mime_type(path))},
                    params={"api_key": self._api_key},
                    timeout=30,
                )

            if upload_resp.status_code != 200:
                raise RuntimeError(
                    f"SerpApi image upload failed: HTTP {upload_resp.status_code} — "
                    f"{upload_resp.text[:300]}"
                )

            upload_data = upload_resp.json()
            image_id = upload_data.get("image_id")

            if not image_id:
                raise RuntimeError(f"SerpApi did not return image_id: {upload_data}")

            params["image_id"] = image_id
            search = GoogleSearch(params)
            results = search.get_dict()

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"SerpApi Google Lens search error: {exc}") from exc

        return self._parse_results(results)

    @staticmethod
    def _mime_type(path: Path) -> str:
        ext = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "image/jpeg")

    def _parse_results(self, results: dict) -> list[Candidate]:
        """
        Parse the raw SerpApi response dict into Candidate objects.

        Parses dynamically — no hardcoded URLs, domains, or usernames.
        """
        candidates: list[Candidate] = []

        # ── visual_matches section ────────────────────────────────────────────
        visual_matches = results.get("visual_matches", [])
        for match in visual_matches:
            page_url = match.get("link", "") or match.get("url", "")
            image_url = (
                match.get("thumbnail", "")
                or match.get("image", "")
                or match.get("original", "")
            )
            if not page_url and not image_url:
                continue

            domain = extract_domain(page_url or image_url)
            candidate = Candidate(
                source_engine=self.engine_name,
                image_url=normalize_url(image_url),
                page_url=normalize_url(page_url),
                domain=domain,
                title=match.get("title", ""),
                candidate_type=classify_candidate_type(domain),
                thumbnail_url=normalize_url(match.get("thumbnail", "")),
                discovery_sources=[self.engine_name],
            )
            candidates.append(candidate)

        # ── reverse_image section (metadata/direct links) ─────────────────────
        reverse_image = results.get("reverse_image", {})
        if isinstance(reverse_image, dict):
            for img_url in reverse_image.get("image_sources", []):
                if not img_url:
                    continue
                domain = extract_domain(img_url)
                candidate = Candidate(
                    source_engine=self.engine_name,
                    image_url=normalize_url(img_url),
                    page_url="",
                    domain=domain,
                    title="Direct image match",
                    candidate_type=classify_candidate_type(domain),
                    discovery_sources=[self.engine_name],
                )
                candidates.append(candidate)

        # ── knowledge_graph section ───────────────────────────────────────────
        kg = results.get("knowledge_graph", {})
        if isinstance(kg, dict) and kg.get("images"):
            for kg_img in kg["images"]:
                img_url = kg_img.get("url", "") or kg_img.get("image", "")
                if not img_url:
                    continue
                domain = extract_domain(img_url)
                candidate = Candidate(
                    source_engine=self.engine_name,
                    image_url=normalize_url(img_url),
                    page_url=normalize_url(kg.get("knowledge_graph_search_link", "")),
                    domain=domain,
                    title=kg.get("title", ""),
                    candidate_type=classify_candidate_type(domain),
                    discovery_sources=[self.engine_name],
                )
                candidates.append(candidate)

        # ── image_results section (some lens responses include this) ──────────
        image_results = results.get("image_results", [])
        for ir in image_results:
            page_url = ir.get("link", "") or ir.get("url", "")
            image_url = ir.get("original", "") or ir.get("thumbnail", "") or ir.get("image", "")
            if not page_url and not image_url:
                continue
            domain = extract_domain(page_url or image_url)
            candidate = Candidate(
                source_engine=self.engine_name,
                image_url=normalize_url(image_url),
                page_url=normalize_url(page_url),
                domain=domain,
                title=ir.get("title", ""),
                candidate_type=classify_candidate_type(domain),
                thumbnail_url=normalize_url(ir.get("thumbnail", "")),
                discovery_sources=[self.engine_name],
            )
            candidates.append(candidate)

        # ── matching_pages section ────────────────────────────────────────────
        matching_pages = results.get("matching_pages", [])
        for mp in matching_pages:
            page_url = mp.get("link", "") or mp.get("url", "")
            image_url = mp.get("thumbnail", "") or mp.get("image", "") or mp.get("original", "")
            if not page_url and not image_url:
                continue
            domain = extract_domain(page_url or image_url)
            candidate = Candidate(
                source_engine=self.engine_name,
                image_url=normalize_url(image_url),
                page_url=normalize_url(page_url),
                domain=domain,
                title=mp.get("title", "Matching page"),
                candidate_type=classify_candidate_type(domain),
                thumbnail_url=normalize_url(mp.get("thumbnail", "")),
                discovery_sources=[self.engine_name],
            )
            candidates.append(candidate)

        # ── exact_matches / exact image results if provided ───────────────────
        exact_matches = results.get("exact_matches", [])
        for em in exact_matches:
            page_url = em.get("link", "") or em.get("url", "")
            image_url = em.get("thumbnail", "") or em.get("image", "") or em.get("original", "")
            if not page_url and not image_url:
                continue
            domain = extract_domain(page_url or image_url)
            candidate = Candidate(
                source_engine=self.engine_name,
                image_url=normalize_url(image_url),
                page_url=normalize_url(page_url),
                domain=domain,
                title=em.get("title", "Exact match"),
                candidate_type=classify_candidate_type(domain),
                thumbnail_url=normalize_url(em.get("thumbnail", "")),
                discovery_sources=[self.engine_name],
            )
            candidates.append(candidate)

        pages_count = sum(1 for c in candidates if c.page_url)
        self.last_results_breakdown = {
            "exact_matches": len(exact_matches),
            "visual_matches": len(visual_matches),
            "matching_pages": len(matching_pages),
            "pages": pages_count,
        }

        logger.info(
            "SerpApi Google Lens: parsed %d candidates", len(candidates)
        )
        return candidates

    def _mock_results(self) -> list[Candidate]:
        """
        Return clearly labeled mock candidates for MOCK_MODE.
        These are NOT real search results. Used only for development/testing.
        """
        logger.warning(
            "[MOCK MODE] SerpApi Google Lens returning mock results — "
            "NOT real reverse-image search results"
        )
        candidates: list[Candidate] = []
        for match in _MOCK_VISUAL_MATCHES:
            page_url = normalize_url(match.get("link", ""))
            image_url = normalize_url(match.get("thumbnail", ""))
            domain = extract_domain(page_url)
            candidates.append(
                Candidate(
                    source_engine="google_lens [MOCK]",
                    image_url=image_url,
                    page_url=page_url,
                    domain=domain,
                    title=match.get("title", ""),
                    candidate_type=classify_candidate_type(domain),
                    thumbnail_url=image_url,
                    discovery_sources=["google_lens [MOCK]"],
                )
            )
        return candidates
