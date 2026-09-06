// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * TRACE-ID — EvidenceRegistry
 * Minimal on-chain registry for cryptographic evidence anchoring.
 *
 * Purpose:
 *   Stores SHA-256 hashes of evidence manifests on Polygon Amoy testnet.
 *   Provides tamper-evident timestamping: if the off-chain manifest is modified
 *   after anchoring, its recomputed hash will differ from the stored on-chain hash.
 *
 * Security properties:
 *   - Only stores bytes32 hashes — NO raw biometric data
 *   - NO personal information stored on-chain
 *   - Each recordId can only be anchored once (immutable after anchoring)
 *   - Emits EvidenceAnchored event for off-chain indexing
 *
 * Limitations:
 *   - Deployed on Polygon Amoy testnet (not mainnet)
 *   - recordId uniqueness enforced on-chain; collisions must be handled off-chain
 */
contract EvidenceRegistry {

    // ── Structs ───────────────────────────────────────────────────────────────

    struct Evidence {
        bytes32 recordId;       // Deterministic ID from evidence manifest
        bytes32 evidenceHash;   // SHA-256 of canonical evidence manifest JSON
        uint256 timestamp;      // block.timestamp at anchoring (Unix seconds)
        address submitter;      // Wallet address that anchored the evidence
        bool exists;            // Sentinel for existence checks
    }

    // ── State ─────────────────────────────────────────────────────────────────

    mapping(bytes32 => Evidence) private _records;

    // ── Events ────────────────────────────────────────────────────────────────

    /**
     * Emitted when new evidence is successfully anchored.
     *
     * @param recordId      The deterministic record identifier.
     * @param evidenceHash  SHA-256 hash of the canonical evidence manifest.
     * @param timestamp     Block timestamp when the evidence was anchored.
     * @param submitter     Address that submitted the transaction.
     */
    event EvidenceAnchored(
        bytes32 indexed recordId,
        bytes32 evidenceHash,
        uint256 timestamp,
        address indexed submitter
    );

    // ── Errors ────────────────────────────────────────────────────────────────

    error RecordAlreadyExists(bytes32 recordId);
    error RecordNotFound(bytes32 recordId);
    error InvalidHash();

    // ── Functions ─────────────────────────────────────────────────────────────

    /**
     * Anchor evidence on-chain.
     *
     * @param recordId      Deterministic 32-byte record identifier.
     * @param evidenceHash  SHA-256 hash of the canonical evidence manifest (bytes32).
     *
     * Requirements:
     *   - recordId must not already exist in the registry
     *   - evidenceHash must not be zero
     *
     * Emits {EvidenceAnchored}.
     */
    function anchorEvidence(
        bytes32 recordId,
        bytes32 evidenceHash
    ) external {
        if (_records[recordId].exists) {
            revert RecordAlreadyExists(recordId);
        }
        if (evidenceHash == bytes32(0)) {
            revert InvalidHash();
        }

        _records[recordId] = Evidence({
            recordId: recordId,
            evidenceHash: evidenceHash,
            timestamp: block.timestamp,
            submitter: msg.sender,
            exists: true
        });

        emit EvidenceAnchored(
            recordId,
            evidenceHash,
            block.timestamp,
            msg.sender
        );
    }

    /**
     * Retrieve a stored evidence record by recordId.
     *
     * @param recordId  The record identifier to query.
     * @return          The Evidence struct.
     *
     * Reverts with RecordNotFound if the recordId has not been anchored.
     */
    function getEvidence(bytes32 recordId)
        external
        view
        returns (Evidence memory)
    {
        if (!_records[recordId].exists) {
            revert RecordNotFound(recordId);
        }
        return _records[recordId];
    }

    /**
     * Check whether a recordId has been anchored.
     *
     * @param recordId  The record identifier to check.
     * @return          True if anchored, false otherwise.
     */
    function recordExists(bytes32 recordId) external view returns (bool) {
        return _records[recordId].exists;
    }

    /**
     * Verify that a given evidenceHash matches what is stored on-chain.
     *
     * @param recordId      The record to verify against.
     * @param evidenceHash  The hash to compare.
     * @return              True if hashes match, false otherwise.
     *
     * Reverts with RecordNotFound if the recordId has not been anchored.
     */
    function verifyEvidence(bytes32 recordId, bytes32 evidenceHash)
        external
        view
        returns (bool)
    {
        if (!_records[recordId].exists) {
            revert RecordNotFound(recordId);
        }
        return _records[recordId].evidenceHash == evidenceHash;
    }
}
