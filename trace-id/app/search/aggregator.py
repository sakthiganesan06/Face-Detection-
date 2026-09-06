"""
TRACE-ID — Search Result Aggregator
Merges candidates from multiple search engines, deduplicates, and normalizes.

Multi-engine discovery increases confidence but does NOT remove the requirement
for independent image/face verification. Candidates from both engines must still
pass face and visual verification before being accepted.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from app.search.base import Candidate, normalize_url

logger = logging.getLogger(__name__)


def aggregate_candidates(
    candidates_per_engine: dict[str, list[Candidate]],
    max_candidates: int | None = None,
) -> list[Candidate]:
    """
    Merge candidates from multiple search engines.

    Deduplication logic:
      - Normalize image_url and page_url
      - Group by normalized_image_url (primary key) or normalized_page_url (secondary key)
      - Merge discovery_sources from all engines that found the same candidate
      - Increase discovery_confidence when multiple engines agree

    Args:
        candidates_per_engine: Dict mapping engine_name → list[Candidate]
        max_candidates: Maximum number of candidates to return (None = no limit)

    Returns:
        Deduplicated, merged list of Candidate objects.
    """
    from app import config

    if max_candidates is None:
        max_candidates = config.MAX_CANDIDATES

    # ── Normalize all URLs first ──────────────────────────────────────────────
    all_candidates: list[Candidate] = []
    for engine_name, candidates in candidates_per_engine.items():
        for c in candidates:
            c.image_url = normalize_url(c.image_url) if c.image_url else ""
            c.page_url = normalize_url(c.page_url) if c.page_url else ""
            all_candidates.append(c)

    # ── Build dedup map ───────────────────────────────────────────────────────
    # Key: (normalized_image_url, normalized_page_url) — use whichever is available
    seen: dict[str, Candidate] = {}  # dedup_key → representative Candidate

    def dedup_key(c: Candidate) -> str:
        """Canonical key for deduplication."""
        # Prefer image_url as primary key; fall back to page_url
        primary = c.image_url or c.page_url
        return primary

    for candidate in all_candidates:
        key = dedup_key(candidate)
        if not key:
            continue  # Skip candidates with no usable URL

        if key in seen:
            existing = seen[key]
            # Merge discovery sources
            for src in candidate.discovery_sources:
                if src not in existing.discovery_sources:
                    existing.discovery_sources.append(src)
            # Merge engine name if different
            if candidate.source_engine not in existing.source_engine:
                existing.source_engine = (
                    existing.source_engine + "+" + candidate.source_engine
                )
            # Fill in missing fields from this candidate
            if not existing.page_url and candidate.page_url:
                existing.page_url = candidate.page_url
            if not existing.title and candidate.title:
                existing.title = candidate.title
            if not existing.thumbnail_url and candidate.thumbnail_url:
                existing.thumbnail_url = candidate.thumbnail_url
        else:
            seen[key] = candidate

    deduplicated = list(seen.values())

    # ── Assign discovery_confidence ───────────────────────────────────────────
    for c in deduplicated:
        num_sources = len(c.discovery_sources)
        if num_sources == 1:
            c.discovery_confidence = 1.0
        elif num_sources == 2:
            c.discovery_confidence = 1.0  # Multi-engine agreement — maximum confidence
        else:
            c.discovery_confidence = 1.0  # Can't exceed 1.0

    # ── Filter empty candidates ────────────────────────────────────────────────
    valid = [c for c in deduplicated if c.image_url or c.page_url]

    # ── Prioritize: social first, then by number of sources ──────────────────
    valid.sort(
        key=lambda c: (
            -(len(c.discovery_sources)),   # more sources = higher priority
            c.candidate_type != "social",  # social = True (0) before web (1)
        )
    )

    result = valid[:max_candidates]

    logger.info(
        "Aggregator: %d total → %d after dedup → %d returned (max=%d)",
        len(all_candidates),
        len(valid),
        len(result),
        max_candidates,
    )

    return result


def print_aggregation_summary(
    candidates_per_engine: dict[str, list[Candidate]],
    merged: list[Candidate],
) -> None:
    """Print a concise summary of aggregation results to the logger."""
    for engine, cands in candidates_per_engine.items():
        logger.info("  %-30s %d results", engine, len(cands))
    logger.info("  %-30s %d merged (after dedup)", "Total unique candidates:", len(merged))
