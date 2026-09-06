"""
TRACE-ID — URL Deduplicator
Normalizes and deduplicates candidate URLs to avoid redundant processing.
"""

from __future__ import annotations

import logging
from typing import Iterable

from app.search.base import Candidate, normalize_url

logger = logging.getLogger(__name__)


def deduplicate_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """
    Remove duplicate candidates based on normalized image_url.

    When duplicates are found, merges discovery_sources from all duplicates
    into the first-seen representative candidate.

    Args:
        candidates: Iterable of Candidate objects (possibly with duplicate URLs).

    Returns:
        List of unique Candidate objects.
    """
    seen: dict[str, Candidate] = {}  # normalized_key → representative

    for c in candidates:
        # Use image_url as primary key, page_url as secondary
        key = normalize_url(c.image_url) if c.image_url else normalize_url(c.page_url)
        if not key:
            continue

        if key in seen:
            existing = seen[key]
            # Merge sources
            for src in c.discovery_sources:
                if src not in existing.discovery_sources:
                    existing.discovery_sources.append(src)
            # Fill missing data
            if not existing.page_url and c.page_url:
                existing.page_url = c.page_url
            if not existing.title and c.title:
                existing.title = c.title
        else:
            seen[key] = c

    result = list(seen.values())
    logger.debug(
        "Deduplicator: %d input → %d unique candidates",
        sum(1 for _ in seen),  # already consumed — use len(seen)
        len(result),
    )
    return result


def normalize_candidate_urls(candidates: list[Candidate]) -> list[Candidate]:
    """
    In-place URL normalization for all candidates.
    Returns the same list (mutated).
    """
    for c in candidates:
        if c.image_url:
            c.image_url = normalize_url(c.image_url)
        if c.page_url:
            c.page_url = normalize_url(c.page_url)
        if c.thumbnail_url:
            c.thumbnail_url = normalize_url(c.thumbnail_url)
    return candidates
