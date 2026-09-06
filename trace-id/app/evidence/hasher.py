"""
TRACE-ID — Evidence Hasher
Computes SHA-256 of the canonical JSON evidence manifest.

The hash is deterministic: the same manifest always produces the same hash.
Used for blockchain anchoring and tamper detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from app.evidence.canonicalize import canonicalize

logger = logging.getLogger(__name__)


def hash_manifest(manifest: dict) -> str:
    """
    Compute the SHA-256 hash of a manifest dict's canonical JSON form.

    Steps:
      1. Canonicalize the manifest (sorted keys, compact JSON, UTF-8)
      2. Compute SHA-256 of the resulting bytes

    Returns:
        Lowercase hex-encoded SHA-256 digest (64 characters).
    """
    canonical_bytes = canonicalize(manifest)
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    logger.debug("Manifest hash: %s", digest)
    return digest


def hash_bytes(data: bytes) -> str:
    """
    Compute SHA-256 of arbitrary bytes.
    Used for hashing downloaded candidate images.
    """
    return hashlib.sha256(data).hexdigest()


def save_manifest(
    manifest: dict,
    manifest_path: Path | None = None,
    hash_path: Path | None = None,
) -> tuple[str, str]:
    """
    Save the evidence manifest and its SHA-256 to disk.

    Args:
        manifest:       Evidence manifest dict.
        manifest_path:  Path to save manifest.json. Defaults to config location.
        hash_path:      Path to save manifest.sha256. Defaults to config location.

    Returns:
        Tuple of (evidence_hash, canonical_json_string).
    """
    from app import config

    if manifest_path is None:
        manifest_path = config.MANIFEST_PATH
    if hash_path is None:
        hash_path = config.MANIFEST_HASH_PATH

    manifest_path = Path(manifest_path)
    hash_path = Path(hash_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute canonical form and hash
    from app.evidence.canonicalize import canonicalize, canonicalize_str
    canonical_str = canonicalize_str(manifest)
    evidence_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    # Save human-readable manifest (pretty-printed for readability)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Save hash file
    hash_path.write_text(evidence_hash + "\n", encoding="utf-8")

    logger.info("Manifest saved: %s", manifest_path)
    logger.info("Hash saved:     %s", hash_path)
    logger.info("Evidence hash:  %s", evidence_hash)

    return evidence_hash, canonical_str


def load_manifest(manifest_path: Path | str) -> dict:
    """
    Load a manifest JSON file from disk.

    Args:
        manifest_path: Path to manifest.json file.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in manifest file {path}: {exc}") from exc
