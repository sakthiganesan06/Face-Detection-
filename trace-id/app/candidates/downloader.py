"""
TRACE-ID — Candidate Image Downloader
Downloads candidate images from public URLs for local verification.

Rules:
  - Respect HTTP errors and timeouts
  - Never bypass authentication or access controls
  - Mark unavailable images as IMAGE_UNAVAILABLE
  - Never crash the pipeline due to a single failed download
  - Save images to data/candidates/ with content-hash filenames
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from app import config

logger = logging.getLogger(__name__)

# ── Allowed MIME types for download ──────────────────────────────────────────
_ALLOWED_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
    }
)

_MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB limit per candidate image


class DownloadResult:
    """Result of a candidate image download attempt."""

    __slots__ = ("url", "success", "local_path", "sha256", "reason", "content_type")

    def __init__(
        self,
        url: str,
        success: bool,
        local_path: str = "",
        sha256: str = "",
        reason: str = "",
        content_type: str = "",
    ) -> None:
        self.url = url
        self.success = success
        self.local_path = local_path
        self.sha256 = sha256
        self.reason = reason
        self.content_type = content_type

    @property
    def status(self) -> str:
        return "DOWNLOADED" if self.success else "IMAGE_UNAVAILABLE"


def download_image(
    url: str,
    output_dir: Path | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> DownloadResult:
    """
    Download a publicly accessible image from a URL.

    Args:
        url: Public URL to the image.
        output_dir: Directory to save the image. Defaults to data/candidates/.
        timeout: Request timeout in seconds.
        max_retries: Number of retry attempts on transient failures.

    Returns:
        DownloadResult with success status, local path, and SHA-256.
    """
    if output_dir is None:
        output_dir = config.CANDIDATES_DIR
    if timeout is None:
        timeout = config.DOWNLOAD_TIMEOUT_SECONDS
    if max_retries is None:
        max_retries = config.DOWNLOAD_MAX_RETRIES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not url:
        return DownloadResult(url=url, success=False, reason="Empty URL")

    # Validate URL scheme
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return DownloadResult(
            url=url,
            success=False,
            reason=f"Unsupported URL scheme: {parsed.scheme}",
        )

    headers = {
        "User-Agent": config.DOWNLOAD_USER_AGENT,
        "Accept": "image/*, */*;q=0.8",
    }

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )

            # ── HTTP error handling ───────────────────────────────────────────
            if response.status_code == 401:
                return DownloadResult(
                    url=url,
                    success=False,
                    reason="HTTP 401 Unauthorized — access-controlled content, skipping",
                )
            if response.status_code == 403:
                return DownloadResult(
                    url=url,
                    success=False,
                    reason="HTTP 403 Forbidden — access-controlled content, skipping",
                )
            if response.status_code == 404:
                return DownloadResult(url=url, success=False, reason="HTTP 404 Not Found")
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 2))
                if attempt < max_retries:
                    logger.warning(
                        "Rate-limited by %s (HTTP 429). Waiting %ds before retry.",
                        parsed.netloc,
                        retry_after,
                    )
                    time.sleep(min(retry_after, 10))
                    continue
                return DownloadResult(url=url, success=False, reason="HTTP 429 Rate Limited")
            if response.status_code >= 400:
                return DownloadResult(
                    url=url,
                    success=False,
                    reason=f"HTTP {response.status_code}",
                )

            # ── Content-Type check ────────────────────────────────────────────
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
            if content_type and content_type not in _ALLOWED_MIME_TYPES:
                return DownloadResult(
                    url=url,
                    success=False,
                    reason=f"Unsupported content type: {content_type}",
                    content_type=content_type,
                )

            # ── Content-Length check ──────────────────────────────────────────
            content_length = int(response.headers.get("Content-Length", 0))
            if content_length > _MAX_CONTENT_LENGTH:
                return DownloadResult(
                    url=url,
                    success=False,
                    reason=f"Image too large: {content_length / 1024 / 1024:.1f} MB > 20 MB limit",
                )

            # ── Read response ─────────────────────────────────────────────────
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=65536):
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_CONTENT_LENGTH:
                    return DownloadResult(
                        url=url,
                        success=False,
                        reason="Image too large (exceeded 20 MB during streaming)",
                    )

            data = b"".join(chunks)
            if not data:
                return DownloadResult(url=url, success=False, reason="Empty response body")

            # ── Compute SHA-256 ───────────────────────────────────────────────
            sha256 = hashlib.sha256(data).hexdigest()

            # ── Detect extension ──────────────────────────────────────────────
            ext = _detect_extension(content_type, url)
            filename = f"{sha256[:16]}{ext}"
            local_path = output_dir / filename

            if not local_path.exists():
                local_path.write_bytes(data)
                logger.debug("Downloaded: %s → %s", url[:60], filename)
            else:
                logger.debug("Cache hit: %s already exists", filename)

            return DownloadResult(
                url=url,
                success=True,
                local_path=str(local_path),
                sha256=sha256,
                content_type=content_type,
            )

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {timeout}s"
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.ConnectionError as exc:
            last_error = f"Connection error: {exc}"
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.RequestException as exc:
            last_error = f"Request error: {exc}"
            break

    return DownloadResult(url=url, success=False, reason=last_error or "Unknown error")


def _detect_extension(content_type: str, url: str) -> str:
    """Determine file extension from Content-Type or URL."""
    ct_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    if content_type in ct_map:
        return ct_map[content_type]
    # Fallback: try to get extension from URL
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}:
        return suffix if suffix != ".jpeg" else ".jpg"
    return ".jpg"  # Safe default
