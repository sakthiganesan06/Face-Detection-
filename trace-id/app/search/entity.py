"""
TRACE-ID — Entity Recognition & Social Media Match Classifier
Extracts recognized person/entity names and structures verified social media posts (Instagram, X, Facebook, LinkedIn, etc.)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


def extract_person_name(titles: list[str]) -> Optional[str]:
    """
    Extract the most likely person/entity name from search match titles.

    Uses title frequency analysis and regex patterns for common title formats:
    - 'Sundar Pichai said WHAT?!'
    - 'Google CEO Sundar Pichai to now head...'
    - 'Elon Musk on X...'
    """
    if not titles:
        return None

    # Clean titles: strip site suffixes like ' - Wikipedia', ' | LinkedIn', etc.
    cleaned_titles = []
    for title in titles:
        if not title:
            continue
        cleaned = re.split(r"[-|–—•:]", title)[0].strip()
        cleaned_titles.append(cleaned)

    # Look for common 2-3 word capitalized proper nouns in titles
    name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
    name_counts: Counter[str] = Counter()

    # Common stop words/titles to ignore
    ignore_names = {
        "Google Ceo", "New Ceo", "Chief Executive", "World Government",
        "Asia Society", "Los Angeles", "New York", "United States",
        "Public Profile", "Example Profile", "High Quality",
    }

    for title in titles:
        if not title:
            continue
        matches = name_pattern.findall(title)
        for name in matches:
            if name not in ignore_names and len(name.split()) >= 2:
                name_counts[name] += 1

    if name_counts:
        top_name, top_count = name_counts.most_common(1)[0]
        return top_name

    return None


def categorize_social_post(url: str, domain: str) -> dict[str, str]:
    """
    Extract specific platform and post type information from a URL.
    
    Supports:
      - Instagram (post, reel, story, profile)
      - X / Twitter (tweet, status, profile)
      - LinkedIn (pulse article, post, profile)
      - Facebook (video, post, photo)
      - YouTube (video, short, channel)
      - Threads, TikTok, Reddit, Pinterest
    """
    url_lower = url.lower()
    platform = "Web"
    post_type = "Web Page"

    if "instagram.com" in domain or "instagram.com" in url_lower:
        platform = "Instagram"
        if "/p/" in url_lower:
            post_type = "Post"
        elif "/reel/" in url_lower:
            post_type = "Reel"
        elif "/stories/" in url_lower:
            post_type = "Story"
        else:
            post_type = "Profile / Page"

    elif "x.com" in domain or "twitter.com" in domain or "x.com" in url_lower or "twitter.com" in url_lower:
        platform = "X (Twitter)"
        if "/status/" in url_lower or "/statuses/" in url_lower:
            post_type = "Tweet / Post"
        else:
            post_type = "Profile"

    elif "linkedin.com" in domain or "linkedin.com" in url_lower:
        platform = "LinkedIn"
        if "/pulse/" in url_lower:
            post_type = "Article"
        elif "/posts/" in url_lower:
            post_type = "Post"
        else:
            post_type = "Profile / Update"

    elif "facebook.com" in domain or "fb.com" in domain or "facebook.com" in url_lower:
        platform = "Facebook"
        if "/videos/" in url_lower or "/video/" in url_lower:
            post_type = "Video"
        elif "/posts/" in url_lower or "/story.php" in url_lower:
            post_type = "Post"
        elif "/photos/" in url_lower or "/photo.php" in url_lower:
            post_type = "Photo"
        else:
            post_type = "Page / Post"

    elif "threads.net" in domain or "threads.net" in url_lower:
        platform = "Threads"
        post_type = "Post"

    elif "tiktok.com" in domain or "tiktok.com" in url_lower:
        platform = "TikTok"
        post_type = "Video"

    elif "youtube.com" in domain or "youtu.be" in domain or "youtube.com" in url_lower:
        platform = "YouTube"
        if "/shorts/" in url_lower:
            post_type = "Short"
        else:
            post_type = "Video"

    elif "reddit.com" in domain or "reddit.com" in url_lower:
        platform = "Reddit"
        post_type = "Post / Comment"

    return {
        "platform": platform,
        "post_type": post_type,
    }
