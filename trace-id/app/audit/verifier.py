"""
TRACE-ID — Audit / Replay Verifier
Verifies whether a locally stored evidence manifest has been tampered with
after blockchain anchoring.

Algorithm:
  1. Load manifest.json from disk
  2. Canonicalize it (sorted keys, compact UTF-8 JSON)
  3. Recompute SHA-256 (local_hash)
  4. Retrieve stored evidence_hash from Polygon Amoy blockchain (chain_hash)
  5. Compare: local_hash == chain_hash → VERIFIED | TAMPERED

This is the core tamper-detection feature of TRACE-ID.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from app.evidence.canonicalize import canonicalize
from app.evidence.hasher import hash_manifest, load_manifest

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    """Result of an audit/replay verification."""

    manifest_path: str
    record_id: str
    local_hash: str
    chain_hash: str
    verified: bool
    error: str = ""

    @property
    def status(self) -> str:
        return "VERIFIED" if self.verified else ("TAMPERED" if not self.error else "ERROR")

    def print_report(self) -> None:
        """Print a formatted audit report to stdout."""
        width = 60
        print("=" * width)
        print("TRACE-ID — AUDIT REPORT")
        print("=" * width)
        print(f"  Manifest:       {self.manifest_path}")
        print(f"  Record ID:      {self.record_id}")
        print()
        print(f"  Local SHA-256:  {self.local_hash}")
        print(f"  On-chain SHA-256: {self.chain_hash}")
        print()
        if self.error:
            print(f"  Status:         ERROR")
            print(f"  Reason:         {self.error}")
        else:
            status_str = "[OK] VERIFIED" if self.verified else "[FAIL] TAMPERED"
            print(f"  Status:         {status_str}")
            if not self.verified:
                print()
                print("  WARNING: The local manifest does not match the on-chain record.")
                print("  This indicates the manifest was modified after anchoring.")
        print("=" * width)


def audit_manifest(
    manifest_path: Path | str,
    use_mock: bool = False,
) -> AuditResult:
    """
    Perform a full audit of a manifest file against the blockchain.

    Args:
        manifest_path: Path to the manifest.json file to audit.
        use_mock:      If True, skip blockchain query and use stored hash from
                       the manifest's blockchain_receipt if available.
                       Used in tests and when blockchain is unavailable.

    Returns:
        AuditResult with VERIFIED/TAMPERED/ERROR status.
    """
    path = Path(manifest_path)

    # ── Step 1: Load manifest ─────────────────────────────────────────────────
    try:
        manifest = load_manifest(path)
    except (FileNotFoundError, ValueError) as exc:
        return AuditResult(
            manifest_path=str(path),
            record_id="",
            local_hash="",
            chain_hash="",
            verified=False,
            error=f"Cannot load manifest: {exc}",
        )

    record_id = manifest.get("record_id", "")
    if not record_id:
        return AuditResult(
            manifest_path=str(path),
            record_id="",
            local_hash="",
            chain_hash="",
            verified=False,
            error="Manifest does not contain 'record_id' field",
        )

    # ── Step 2: Recompute local hash ──────────────────────────────────────────
    local_hash = hash_manifest(manifest)

    # ── Step 3: Retrieve chain hash ───────────────────────────────────────────
    if use_mock:
        # In mock mode, compare against the saved .sha256 file (if available)
        chain_hash = _load_saved_hash(path)
        if not chain_hash:
            # No saved hash — treat as TAMPERED (no anchor exists)
            return AuditResult(
                manifest_path=str(path),
                record_id=record_id,
                local_hash=local_hash,
                chain_hash="",
                verified=False,
                error="[MOCK MODE] No saved SHA-256 found for comparison.",
            )
    else:
        try:
            from app import config
            from app.blockchain.client import BlockchainClient

            client = BlockchainClient()
            on_chain = client.get_evidence(record_id)
            chain_hash = on_chain.get("evidence_hash", "")
        except Exception as exc:
            return AuditResult(
                manifest_path=str(path),
                record_id=record_id,
                local_hash=local_hash,
                chain_hash="",
                verified=False,
                error=f"Blockchain query failed: {exc}",
            )

    # ── Step 4: Compare ───────────────────────────────────────────────────────
    verified = local_hash.lower() == chain_hash.lower()

    return AuditResult(
        manifest_path=str(path),
        record_id=record_id,
        local_hash=local_hash,
        chain_hash=chain_hash,
        verified=verified,
    )


def _load_saved_hash(manifest_path: Path) -> str:
    """
    Try to load a saved .sha256 file from the same directory as the manifest.
    Used in mock/offline audit mode.
    """
    sha256_path = manifest_path.parent / (manifest_path.stem + ".sha256")
    if sha256_path.exists():
        return sha256_path.read_text(encoding="utf-8").strip()
    # Also try the canonical location
    from app import config
    if config.MANIFEST_HASH_PATH.exists():
        return config.MANIFEST_HASH_PATH.read_text(encoding="utf-8").strip()
    return ""


def demonstrate_tamper(manifest_path: Path | str, field: str = "visual_similarity") -> dict:
    """
    Demonstrate tamper detection by modifying a manifest field.

    Creates a MODIFIED copy of the manifest (does NOT modify the original).
    Used for the tamper demo in README/screen recording.

    Args:
        manifest_path: Path to the original manifest.json.
        field:         Field to modify for demonstration.

    Returns:
        Dict with original_hash, tampered_hash, field_modified.
    """
    path = Path(manifest_path)
    manifest = load_manifest(path)
    original_hash = hash_manifest(manifest)

    # Modify the demo field without changing anything important
    original_value = manifest.get(field)
    if isinstance(original_value, float):
        tampered_value = round(original_value + 0.001, 6)
    elif isinstance(original_value, str):
        tampered_value = original_value + "_tampered"
    else:
        tampered_value = str(original_value) + "_tampered"

    tampered_manifest = dict(manifest)
    tampered_manifest[field] = tampered_value

    tampered_hash = hash_manifest(tampered_manifest)

    # Save tampered copy (never overwrites original)
    tampered_path = path.parent / (path.stem + "_TAMPERED_DEMO.json")
    with open(tampered_path, "w", encoding="utf-8") as f:
        json.dump(tampered_manifest, f, indent=2)

    return {
        "original_manifest": str(path),
        "tampered_manifest": str(tampered_path),
        "field_modified": field,
        "original_value": original_value,
        "tampered_value": tampered_value,
        "original_hash": original_hash,
        "tampered_hash": tampered_hash,
        "hashes_differ": original_hash != tampered_hash,
    }
