"""
TRACE-ID — SerpApi Google Reverse Image Search Engine
Secondary reverse image search source using SerpApi Google Reverse Image API.
https://serpapi.com/google-reverse-image-api

Discovers candidates from:
  - image_results (matching web pages with titles, snippets, and images)
  - inline_images (direct image results)
  - associated page URLs

All candidates originate from live API responses — nothing is hardcoded.
"""

from __future__ import annotations

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


class SerpApiReverseImageEngine(BaseSearchEngine):
    """
    Google Reverse Image search engine via SerpApi.
    """

    engine_name = "google_reverse_image"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or config.SERPAPI_API_KEY
        self.last_results_breakdown: dict[str, int] = {
            "image_results": 0,
            "inline_images": 0,
            "pages": 0,
        }

    def search(self, image_path: str) -> list[Candidate]:
        """
        Upload image or query SerpApi Google Reverse Image API and return parsed candidates.
        """
        if config.MOCK_MODE:
            return self._mock_results()

        if not self._api_key:
            logger.warning("SERPAPI_API_KEY not set for Google Reverse Image search.")
            return []

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found for Reverse Image search: {path}")

        try:
            from serpapi import GoogleSearch
            import requests

            # 1. Upload local image to https://serpapi.com/image to get image_id
            upload_url = "https://serpapi.com/image"
            ext = path.suffix.lower()
            mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else ("image/png" if ext == ".png" else "image/webp")

            with open(path, "rb") as f:
                upload_resp = requests.post(
                    upload_url,
                    files={"image": (path.name, f, mime)},
                    params={"api_key": self._api_key},
                    timeout=30,
                )

            if upload_resp.status_code != 200:
                logger.warning(
                    "SerpApi image upload failed for reverse image: HTTP %d — %s",
                    upload_resp.status_code,
                    upload_resp.text[:200],
                )
                return []

            upload_data = upload_resp.json()
            image_id = upload_data.get("image_id")
            if not image_id:
                logger.warning("SerpApi did not return image_id for reverse image search")
                return []

            params: dict[str, Any] = {
                "engine": "google_reverse_image",
                "image_id": image_id,
                "api_key": self._api_key,
            }

            search = GoogleSearch(params)
            results = search.get_dict()
            return self._parse_results(results)

        except Exception as exc:
            logger.warning("SerpApi Google Reverse Image search error: %s", exc)
            return []

    def _parse_results(self, results: dict) -> list[Candidate]:
        """Parse raw SerpApi Google Reverse Image API results into Candidate objects."""
        candidates: list[Candidate] = []
        img_res_count = 0
        inline_img_count = 0
        pages_count = 0

        # 1. image_results section
        image_results = results.get("image_results", [])
        for item in image_results:
            img_res_count += 1
            page_url = item.get("link", "") or item.get("url", "")
            image_url = item.get("thumbnail", "") or item.get("original", "") or item.get("image", "")
            if not page_url and not image_url:
                continue

            domain = extract_domain(page_url or image_url)
            if page_url:
                pages_count += 1

            candidates.append(
                Candidate(
                    source_engine=self.engine_name,
                    image_url=normalize_url(image_url),
                    page_url=normalize_url(page_url),
                    domain=domain,
                    title=item.get("title", "") or item.get("snippet", ""),
                    candidate_type=classify_candidate_type(domain),
                    thumbnail_url=normalize_url(item.get("thumbnail", "")),
                    discovery_sources=[self.engine_name],
                )
            )

        # 2. inline_images section
        inline_images = results.get("inline_images", [])
        for item in inline_images:
            inline_img_count += 1
            page_url = item.get("link", "") or item.get("source", "")
            image_url = item.get("original", "") or item.get("thumbnail", "")
            if not page_url and not image_url:
                continue

            domain = extract_domain(page_url or image_url)
            if page_url:
                pages_count += 1

            candidates.append(
                Candidate(
                    source_engine=self.engine_name,
                    image_url=normalize_url(image_url),
                    page_url=normalize_url(page_url),
                    domain=domain,
                    title=item.get("title", "Inline image match"),
                    candidate_type=classify_candidate_type(domain),
                    thumbnail_url=normalize_url(item.get("thumbnail", "")),
                    discovery_sources=[self.engine_name],
                )
            )

        self.last_results_breakdown = {
            "image_results": img_res_count,
            "inline_images": inline_img_count,
            "pages": pages_count,
        }

        logger.info(
            "SerpApi Google Reverse Image: parsed %d candidates", len(candidates)
        )
        return candidates

    def _mock_results(self) -> list[Candidate]:
        """Return mock candidates for MOCK_MODE."""
        self.last_results_breakdown = {
            "image_results": 1,
            "inline_images": 1,
            "pages": 1,
        }
        return [
            Candidate(
                source_engine="google_reverse_image [MOCK]",
                image_url="https://via.placeholder.com/200",
                page_url="https://example.com/mock-reverse",
                domain="example.com",
                title="[MOCK] Reverse Image Match",
                candidate_type="web",
                thumbnail_url="https://via.placeholder.com/200",
                discovery_sources=["google_reverse_image [MOCK]"],
            )
        ]
