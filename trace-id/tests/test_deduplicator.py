"""
Tests for candidate deduplication.

Verifies:
  - Exact URL deduplication
  - Discovery source merging
  - URL normalization before dedup
  - Empty input handling
"""

from __future__ import annotations

import pytest

from app.candidates.deduplicator import deduplicate_candidates
from app.search.base import Candidate


def make_candidate(image_url: str, page_url: str = "", source: str = "test") -> Candidate:
    return Candidate(
        source_engine=source,
        image_url=image_url,
        page_url=page_url,
        domain="example.com",
        discovery_sources=[source],
    )


class TestDeduplicateCandidates:
    def test_empty_input_returns_empty(self):
        assert deduplicate_candidates([]) == []

    def test_single_candidate_unchanged(self):
        c = make_candidate("https://example.com/img.jpg")
        result = deduplicate_candidates([c])
        assert len(result) == 1

    def test_exact_duplicate_removed(self):
        url = "https://example.com/img.jpg"
        c1 = make_candidate(url, source="engine_a")
        c2 = make_candidate(url, source="engine_b")
        result = deduplicate_candidates([c1, c2])
        assert len(result) == 1

    def test_sources_merged_on_duplicate(self):
        url = "https://example.com/img.jpg"
        c1 = make_candidate(url, source="google_lens")
        c2 = make_candidate(url, source="google_cloud_vision")
        result = deduplicate_candidates([c1, c2])
        assert len(result) == 1
        assert "google_lens" in result[0].discovery_sources
        assert "google_cloud_vision" in result[0].discovery_sources

    def test_different_urls_not_deduplicated(self):
        c1 = make_candidate("https://example.com/img1.jpg")
        c2 = make_candidate("https://example.com/img2.jpg")
        result = deduplicate_candidates([c1, c2])
        assert len(result) == 2

    def test_normalized_url_deduplication(self):
        """Different formatting, same URL → deduped."""
        c1 = make_candidate("https://example.com/path/")
        c2 = make_candidate("https://example.com/path")  # no trailing slash
        # Both normalize to same URL
        from app.search.base import normalize_url
        assert normalize_url("https://example.com/path/") == normalize_url("https://example.com/path")
        result = deduplicate_candidates([c1, c2])
        assert len(result) == 1

    def test_empty_url_candidates_skipped(self):
        """Candidates with no URL are skipped."""
        c = make_candidate("", "")
        result = deduplicate_candidates([c])
        assert len(result) == 0

    def test_page_url_used_when_no_image_url(self):
        """page_url used as fallback key when image_url is empty."""
        c1 = Candidate(
            source_engine="a",
            image_url="",
            page_url="https://example.com/post/1",
            discovery_sources=["a"],
        )
        c2 = Candidate(
            source_engine="b",
            image_url="",
            page_url="https://example.com/post/1",
            discovery_sources=["b"],
        )
        result = deduplicate_candidates([c1, c2])
        assert len(result) == 1
        assert len(result[0].discovery_sources) == 2
