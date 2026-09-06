"""
TRACE-ID — Contextual Web Discovery (OCR & Watermark Signal)
Performs contextual web discovery using text/watermarks detected by OCR.

Discovers candidate web and social posts matching creator watermarks or usernames.
This layer augments reverse-image search without replacing it.
"""

from __future__ import annotations

import logging
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


class ContextSearchEngine(BaseSearchEngine):
    """
    Search engine that queries SerpApi Google Search for contextual keywords/watermarks.
    """

    engine_name = "ocr_context"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or config.SERPAPI_API_KEY

    def search_queries(self, queries: list[str]) -> list[Candidate]:
        """
        Execute search across all provided contextual queries and return aggregated candidates.
        """
        if config.MOCK_MODE or not self._api_key or not queries:
            return []

        try:
            from serpapi import GoogleSearch
        except ImportError:
            return []

        candidates: list[Candidate] = []

        for q in queries[:2]:  # Limit to top 2 queries to maintain high precision
            try:
                params: dict[str, Any] = {
                    "engine": "google",
                    "q": q,
                    "api_key": self._api_key,
                    "num": 8,
                }
                search = GoogleSearch(params)
                results = search.get_dict()

                # Parse organic_results
                for item in results.get("organic_results", []):
                    page_url = normalize_url(item.get("link", ""))
                    if not page_url:
                        continue
                    domain = extract_domain(page_url)
                    thumbnail = item.get("thumbnail", "") or item.get("rich_snippet", {}).get("top", {}).get("extensions", [""])[0] if isinstance(item.get("rich_snippet"), dict) else ""

                    candidates.append(
                        Candidate(
                            source_engine=self.engine_name,
                            image_url=normalize_url(thumbnail) if thumbnail.startswith("http") else "",
                            page_url=page_url,
                            domain=domain,
                            title=item.get("title", ""),
                            candidate_type=classify_candidate_type(domain),
                            thumbnail_url=normalize_url(thumbnail) if thumbnail.startswith("http") else "",
                            discovery_sources=[self.engine_name],
                            discovery_confidence=0.85,
                        )
                    )

                # Parse inline_images if present
                for img in results.get("inline_images", []):
                    page_url = normalize_url(img.get("link", ""))
                    img_url = normalize_url(img.get("original", "") or img.get("thumbnail", ""))
                    if not page_url and not img_url:
                        continue
                    domain = extract_domain(page_url or img_url)
                    candidates.append(
                        Candidate(
                            source_engine=self.engine_name,
                            image_url=img_url,
                            page_url=page_url,
                            domain=domain,
                            title=img.get("title", "Context image match"),
                            candidate_type=classify_candidate_type(domain),
                            thumbnail_url=img_url,
                            discovery_sources=[self.engine_name],
                            discovery_confidence=0.85,
                        )
                    )

            except Exception as exc:
                logger.debug("Contextual search query '%s' notice: %s", q, exc)

        logger.info("OCR Context Discovery: found %d candidates", len(candidates))
        return candidates
