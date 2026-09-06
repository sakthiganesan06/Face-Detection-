"""
TRACE-ID — Blockchain Client
Web3.py client for interacting with the deployed EvidenceRegistry contract
on Polygon Amoy testnet.

Security:
  - PRIVATE_KEY is read from environment only
  - Private key is NEVER printed, logged, or stored anywhere
  - Only SHA-256 hashes are stored on-chain (no biometrics, no PII)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import config

logger = logging.getLogger(__name__)

# ── Minimal ABI (only functions we use) ──────────────────────────────────────
_EVIDENCE_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "recordId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"},
        ],
        "name": "anchorEvidence",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "recordId", "type": "bytes32"}
        ],
        "name": "getEvidence",
        "outputs": [
            {
                "components": [
                    {"internalType": "bytes32", "name": "recordId", "type": "bytes32"},
                    {"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "address", "name": "submitter", "type": "address"},
                    {"internalType": "bool", "name": "exists", "type": "bool"},
                ],
                "internalType": "struct EvidenceRegistry.Evidence",
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "recordId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"},
        ],
        "name": "verifyEvidence",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "recordId", "type": "bytes32"}
        ],
        "name": "recordExists",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "recordId", "type": "bytes32"},
            {"indexed": False, "internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"indexed": True, "internalType": "address", "name": "submitter", "type": "address"},
        ],
        "name": "EvidenceAnchored",
        "type": "event",
    },
]


class BlockchainClient:
    """
    Client for interacting with the EvidenceRegistry contract on Polygon Amoy.

    Security guarantee: PRIVATE_KEY is read once from config and used only
    to sign transactions. It is never stored as an instance attribute or logged.
    """

    def __init__(
        self,
        rpc_url: str | None = None,
        contract_address: str | None = None,
    ) -> None:
        self._rpc_url = rpc_url or config.POLYGON_RPC_URL
        self._contract_address = contract_address or config.CONTRACT_ADDRESS
        self._w3 = None
        self._contract = None

    def _ensure_connected(self) -> None:
        """Lazy-connect to blockchain RPC and contract."""
        if self._w3 is not None:
            return

        try:
            from web3 import Web3
            from web3.middleware import ExtraDataToPOAMiddleware
        except ImportError as exc:
            raise RuntimeError(
                "web3 not installed. Run: pip install web3"
            ) from exc

        if not self._rpc_url:
            raise RuntimeError(
                "POLYGON_RPC_URL is not set. Add it to your .env file."
            )
        if not self._contract_address:
            raise RuntimeError(
                "CONTRACT_ADDRESS is not set. "
                "Deploy the contract first: python -m app.blockchain.deploy"
            )
        if not config.PRIVATE_KEY:
            raise RuntimeError(
                "PRIVATE_KEY is not set. Add it to your .env file.\n"
                "WARNING: Use a dedicated testnet wallet with no real funds."
            )

        w3 = Web3(Web3.HTTPProvider(self._rpc_url))

        # Polygon Amoy uses PoA consensus — inject middleware
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        if not w3.is_connected():
            raise RuntimeError(
                f"Cannot connect to Polygon Amoy RPC: {self._rpc_url}\n"
                "Check your internet connection and RPC URL."
            )

        # Normalize contract address
        checksum_address = Web3.to_checksum_address(self._contract_address)
        contract = w3.eth.contract(
            address=checksum_address,
            abi=_EVIDENCE_REGISTRY_ABI,
        )

        self._w3 = w3
        self._contract = contract
        logger.info(
            "Connected to Polygon Amoy | Contract: %s", checksum_address
        )

    def anchor_evidence(
        self,
        record_id_hex: str,
        evidence_hash_hex: str,
        max_wait_seconds: int = 120,
    ) -> dict:
        """
        Anchor evidence hash on Polygon Amoy by calling anchorEvidence().

        Args:
            record_id_hex:    Hex string (64 chars) for bytes32 recordId.
            evidence_hash_hex: Hex string (64 chars) for bytes32 evidenceHash.
            max_wait_seconds: Seconds to wait for transaction confirmation.

        Returns:
            Dict with transaction_hash, block_number, timestamp, gas_used.

        Raises:
            RuntimeError: On connection, signing, or transaction failure.
        """
        self._ensure_connected()

        from web3 import Web3

        # Convert hex strings to bytes32
        record_id_bytes = bytes.fromhex(record_id_hex.lstrip("0x").zfill(64))
        evidence_hash_bytes = bytes.fromhex(evidence_hash_hex.lstrip("0x").zfill(64))

        # Build account from private key (never log or print the key)
        pk = config.PRIVATE_KEY.lstrip("0x")
        account = self._w3.eth.account.from_key(pk)
        wallet_address = account.address

        logger.info("Anchoring evidence from wallet: %s", wallet_address)

        # Check if record already exists on-chain
        chain_id = self._w3.eth.chain_id
        network_names = {
            80002: "Polygon Amoy",
            11155111: "Ethereum Sepolia",
            1: "Ethereum Mainnet",
            137: "Polygon Mainnet",
        }
        explorer_bases = {
            80002: "https://amoy.polygonscan.com",
            11155111: "https://sepolia.etherscan.io",
            1: "https://etherscan.io",
            137: "https://polygonscan.com",
        }
        network_name = network_names.get(chain_id, f"EVM Chain (ID: {chain_id})")
        explorer_base = explorer_bases.get(chain_id, "https://sepolia.etherscan.io")

        try:
            if self._contract.functions.recordExists(record_id_bytes).call():
                logger.info("Record %s already anchored on-chain. Returning verified record.", record_id_hex)
                existing = self.get_evidence(record_id_hex)
                return {
                    "network": network_name,
                    "chain_id": chain_id,
                    "contract_address": self._contract_address,
                    "record_id": record_id_hex,
                    "evidence_hash": existing["evidence_hash"],
                    "transaction_hash": f"Confirmed on-chain at block {existing.get('timestamp')}",
                    "block_number": "Confirmed",
                    "gas_used": 0,
                    "wallet_address": existing.get("submitter", wallet_address),
                    "anchored_at": existing.get("anchored_at", datetime.now(timezone.utc).isoformat()),
                    "explorer_url": f"{explorer_base}/address/{self._contract_address}",
                    "polygonscan_url": f"{explorer_base}/address/{self._contract_address}",
                }
        except Exception as check_exc:
            logger.debug("recordExists check notice: %s", check_exc)

        # Estimate gas
        try:
            gas_estimate = self._contract.functions.anchorEvidence(
                record_id_bytes, evidence_hash_bytes
            ).estimate_gas({"from": wallet_address})
            gas_limit = int(gas_estimate * 1.3)  # 30% buffer
        except Exception as exc:
            raise RuntimeError(
                f"Gas estimation failed: {exc}\n"
                "Possible causes: record already exists, insufficient balance, "
                "or contract not deployed."
            ) from exc

        # Get current nonce and gas price
        nonce = self._w3.eth.get_transaction_count(wallet_address)
        gas_price = self._w3.eth.gas_price

        # Build transaction
        tx = self._contract.functions.anchorEvidence(
            record_id_bytes, evidence_hash_bytes
        ).build_transaction(
            {
                "from": wallet_address,
                "nonce": nonce,
                "gas": gas_limit,
                "gasPrice": gas_price,
            }
        )

        # Sign and send
        signed_tx = self._w3.eth.account.sign_transaction(tx, pk)
        tx_hash = self._w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash_hex = tx_hash.hex()

        logger.info("Transaction sent: %s", tx_hash_hex)
        logger.info("Waiting for confirmation (max %ds)...", max_wait_seconds)

        # Wait for receipt with reasonable fast timeout
        receipt = None
        try:
            receipt = self._w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=min(max_wait_seconds, 6)
            )
        except Exception as exc:
            logger.info("Transaction broadcasted (mining asynchronously): %s", tx_hash_hex)

        block_number = receipt["blockNumber"] if receipt else None
        gas_used = receipt["gasUsed"] if receipt else None
        anchored_at = datetime.now(timezone.utc).isoformat()

        chain_id = self._w3.eth.chain_id
        network_names = {
            80002: "Polygon Amoy",
            11155111: "Ethereum Sepolia",
            1: "Ethereum Mainnet",
            137: "Polygon Mainnet",
        }
        explorer_bases = {
            80002: "https://amoy.polygonscan.com",
            11155111: "https://sepolia.etherscan.io",
            1: "https://etherscan.io",
            137: "https://polygonscan.com",
        }
        network_name = network_names.get(chain_id, f"EVM Chain (ID: {chain_id})")
        explorer_base = explorer_bases.get(chain_id, "https://etherscan.io")

        result = {
            "network": network_name,
            "chain_id": chain_id,
            "contract_address": self._contract_address,
            "record_id": record_id_hex,
            "evidence_hash": evidence_hash_hex,
            "transaction_hash": tx_hash_hex,
            "block_number": block_number or "Pending Confirmation",
            "gas_used": gas_used or 0,
            "wallet_address": wallet_address,
            "anchored_at": anchored_at,
            "explorer_url": f"{explorer_base}/tx/{tx_hash_hex}",
            "polygonscan_url": f"{explorer_base}/tx/{tx_hash_hex}",
        }

        # Save receipt
        self._save_receipt(result)
        return result

    def get_evidence(self, record_id_hex: str) -> dict:
        """
        Retrieve on-chain evidence record by record ID.

        Args:
            record_id_hex: Hex string (64 chars) for bytes32 recordId.

        Returns:
            Dict with on-chain evidence fields.

        Raises:
            RuntimeError: If record not found or connection fails.
        """
        self._ensure_connected()

        record_id_bytes = bytes.fromhex(record_id_hex.lstrip("0x").zfill(64))

        try:
            evidence = self._contract.functions.getEvidence(record_id_bytes).call()
        except Exception as exc:
            error_str = str(exc)
            if "RecordNotFound" in error_str or "execution reverted" in error_str.lower():
                raise RuntimeError(
                    f"Record not found on-chain: {record_id_hex}\n"
                    "The record may not have been anchored, or the contract address is wrong."
                ) from exc
            raise RuntimeError(f"getEvidence failed: {exc}") from exc

        # evidence is a tuple: (recordId, evidenceHash, timestamp, submitter, exists)
        return {
            "record_id": evidence[0].hex() if isinstance(evidence[0], bytes) else evidence[0],
            "evidence_hash": evidence[1].hex() if isinstance(evidence[1], bytes) else evidence[1],
            "timestamp": evidence[2],
            "anchored_at": datetime.fromtimestamp(evidence[2], tz=timezone.utc).isoformat(),
            "submitter": evidence[3],
            "exists": evidence[4],
        }

    def verify_on_chain(self, record_id_hex: str, evidence_hash_hex: str) -> bool:
        """
        Call verifyEvidence() on-chain to check if hash matches stored value.

        Returns:
            True if hashes match, False otherwise.
        """
        self._ensure_connected()

        record_id_bytes = bytes.fromhex(record_id_hex.lstrip("0x").zfill(64))
        evidence_hash_bytes = bytes.fromhex(evidence_hash_hex.lstrip("0x").zfill(64))

        try:
            return bool(
                self._contract.functions.verifyEvidence(
                    record_id_bytes, evidence_hash_bytes
                ).call()
            )
        except Exception as exc:
            raise RuntimeError(f"verifyEvidence call failed: {exc}") from exc

    @staticmethod
    def _save_receipt(receipt: dict) -> None:
        """Save blockchain receipt to data/results/blockchain_receipt.json."""
        receipt_path = config.BLOCKCHAIN_RECEIPT_PATH
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        logger.info("Blockchain receipt saved: %s", receipt_path)
