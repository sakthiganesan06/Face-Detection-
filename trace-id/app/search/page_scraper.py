"""
TRACE-ID — Public Web Page Image Scraper (Layer 6 Discovery)
When reverse search or contextual search discovers a relevant public web page,
this module extracts high-quality candidate image URLs directly from the HTML.

Allows the pipeline to discover different photos of the person embedded within
the discovered article, social post, or public profile.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from app import config
from app.search.base import Candidate, classify_candidate_type, extract_domain, normalize_url

logger = logging.getLogger(__name__)


def extract_images_from_page(page_url: str, max_images: int = 4) -> list[str]:
    """
    Fetch a public web page and extract prominent image URLs.

    Extracts:
      - OpenGraph meta tags: og:image
      - Twitter card meta tags: twitter:image
      - <img> tags in main body (ignoring tiny tracking icons and svg/gif)

    Args:
        page_url: Full URL of the web page to parse.
        max_images: Maximum number of image URLs to return.

    Returns:
        List of absolute, normalized image URLs.
    """
    if not page_url or not page_url.startswith(("http://", "https://")):
        return []

    # Extract YouTube video thumbnails directly from URL structure if applicable
    if "youtube.com" in domain or "youtu.be" in domain:
        yt_id = None
        if "watch?v=" in page_url:
            yt_id = page_url.split("watch?v=")[1].split("&")[0].split("#")[0]
        elif "youtu.be/" in page_url:
            yt_id = page_url.split("youtu.be/")[1].split("?")[0].split("#")[0]
        elif "/shorts/" in page_url:
            yt_id = page_url.split("/shorts/")[1].split("?")[0].split("#")[0]
        if yt_id:
            return [f"https://img.youtube.com/vi/{yt_id}/maxresdefault.jpg", f"https://img.youtube.com/vi/{yt_id}/hqdefault.jpg"]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    try:
        resp = requests.get(page_url, headers=headers, timeout=5, allow_redirects=True)
        if resp.status_code != 200 or not resp.text:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        extracted: list[str] = []
        seen = set()

        # 1. Check OpenGraph and Twitter meta tags
        for prop in ["og:image", "twitter:image", "og:image:secure_url", "image", "twitter:image:src"]:
            metas = soup.find_all("meta", property=prop) + soup.find_all("meta", attrs={"name": prop})
            for meta in metas:
                if meta and meta.get("content"):
                    img_url = urljoin(page_url, meta["content"].strip())
                    norm = normalize_url(img_url)
                    if norm and norm not in seen:
                        seen.add(norm)
                        extracted.append(norm)

        # 2. Check <img> and <figure> tags
        img_tags = soup.find_all("img")
        for img in img_tags:
            src = (
                img.get("src")
                or img.get("data-src")
                or img.get("data-original")
                or img.get("data-highres")
                or img.get("data-lazy-src")
                or img.get("srcset")
            )
            if not src:
                continue

            # Handle srcset (take first or highest res URL)
            if " " in src and "," in src:
                parts = [p.strip().split(" ")[0] for p in src.split(",") if p.strip()]
                src = parts[-1] if parts else src

            full_url = urljoin(page_url, src.strip())
            norm = normalize_url(full_url)
            if not norm or norm in seen:
                continue

            # Filter out icons, svgs, trackers, badges
            low_url = norm.lower()
            if any(ext in low_url for ext in [".svg", ".ico", "1x1", "tracking", "pixel", "logo", "favicon", "avatar", "badge"]):
                continue

            seen.add(norm)
            extracted.append(norm)
            if len(extracted) >= max_images:
                break

        return extracted[:max_images]

    except Exception as exc:
        logger.debug("Error extracting images from page %s: %s", page_url[:50], exc)
        return []


def expand_candidates_from_pages(
    candidates: list[Candidate], max_pages_to_scrape: int = 8
) -> list[Candidate]:
    """
    Expand candidate list by inspecting discovered pages and extracting their embedded photos.
    """
    expanded: list[Candidate] = []
    scraped_pages = set()

    for cand in candidates:
        if not cand.page_url or cand.page_url in scraped_pages:
            continue
        if len(scraped_pages) >= max_pages_to_scrape:
            break

        scraped_pages.add(cand.page_url)
        extracted_imgs = extract_images_from_page(cand.page_url)
        for img_url in extracted_imgs:
            # Avoid re-adding identical image URL
            if cand.image_url and normalize_url(img_url) == normalize_url(cand.image_url):
                continue

            domain = extract_domain(cand.page_url)
            expanded.append(
                Candidate(
                    source_engine=f"{cand.source_engine}_page_extract",
                    image_url=img_url,
                    page_url=cand.page_url,
                    domain=domain,
                    title=cand.title or "Page Image",
                    candidate_type=classify_candidate_type(domain),
                    thumbnail_url=img_url,
                    discovery_sources=[cand.source_engine, "page_scraper"],
                    discovery_confidence=cand.discovery_confidence * 0.95,
                )
            )

    if expanded:
        logger.info("Page Scraper (Layer 6): extracted %d additional candidate images from %d pages", len(expanded), len(scraped_pages))

    return expanded
