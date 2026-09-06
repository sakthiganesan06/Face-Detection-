TRACE-ID
Traceable Identity & Digital Evidence
"Discover. Verify. Anchor."
Python 3.10+ Polygon Amoy Ethereum Sepolia License: MIT

TRACE-ID is a biometric identity tracing & digital evidence anchoring pipeline designed for Hacker House Goa 2026 Task #3:

Face Detection & Prominence Selection: Detects all human faces in an input image using InsightFace SCRFD-10G, calculates bounding box area, proximity to center, and detection quality, selecting the primary foreground subject while logging all faces.
ArcFace Embedding: Generates a 512-dimensional L2-normalized biometric embedding vector using ArcFace R100 (buffalo_l).
OCR & Watermark Context Discovery: Extracts visible creator watermarks, handles, and text overlays (using EasyOCR and Google Cloud Vision) to augment public search discovery.
Genuine Multi-Source Web Discovery:
Google Lens via SerpApi: Full-image + isolated face-crop discovery.
Google Reverse Image via SerpApi: Image results, inline images, matching pages.
Google Cloud Vision Web Detection: Matching pages, full/partial images, visual matches.
Public Web Page Scraper (Layer 6): Extracts embedded high-resolution candidate images from discovered web articles and posts.
Different-Photo Same-Person Face Verification:
Downloads accessible candidate images.
Detects all candidate faces and evaluates each against the input reference embedding using ArcFace.
Does NOT require the exact input photo to exist online: Successfully verifies different photos, poses, lighting, and environments of the same person.
Secondary DINOv2 Visual Verification: Computes 384-d vision transformer embeddings (ViT-S/14) as a secondary whole-image similarity signal without overriding biometric face matches.
Deterministic Evidence Manifest: Assembles a canonical JSON manifest (version 1.0, sorted keys, deterministic formatting) and computes an immutable SHA-256 fingerprint.
Blockchain Anchoring: Anchors the cryptographic evidence hash to an EVM Smart Contract (EvidenceRegistry.sol) on Polygon Amoy & Ethereum Sepolia Testnets.
On-Chain Audit & Tamper Demonstration: Recomputes the SHA-256 hash from local JSON and verifies it against the immutable blockchain registry, instantly identifying any tampering.
Important

Core Principle: Reverse-image search performs web discovery. ArcFace performs independent facial verification of discovered candidate images. The system does not require the exact input image to exist online.

Identity Disclaimer: Face similarity indicates biometric visual correspondence between the input image and discovered public content. This does NOT prove account ownership, authorship, or legal real-world identity. Blockchain anchoring cryptographically proves the integrity of the evidence record at a specific timestamp; it does not prove identity or ownership. Zero raw biometric vectors are stored on-chain.

Table of Contents
Architecture
Technologies
Candidate Extraction Layers
Setup & Installation
Environment Configuration
Usage & CLI Commands
11-Stage Pipeline Overview
Evidence Manifest & Hashing
Blockchain Integration
Audit & Tamper Demonstration
Testing & Validation
Ethical, Privacy & Safety Considerations
Hacker House Goa Task #3 Compliance
Architecture
INPUT PHOTO
     │
     ▼
┌──────────────────────────────────────────────┐
│  [1] Face Detection & Prominence Selection   │
│  InsightFace buffalo_l (SCRFD-10G)           │
│  Calculates area, centrality, quality        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  [2] ArcFace Biometric Embedding             │
│  ArcFace R100 (512-dim L2-normalized vector) │
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┼───────────────┬────────────────┐
       ▼               ▼               ▼                ▼
┌──────────────┐┌──────────────┐┌──────────────┐ ┌──────────────┐
│ Google Lens  ││ Google Rev   ││ Google Cloud │ │ OCR / Text   │
│ via SerpApi  ││ Image        ││ Vision Web   │ │ Context      │
│ (Full+Crop)  ││ (SerpApi)    ││ Detection    │ │ (EasyOCR)    │
└──────┬───────┘└──────┬───────┘└──────┬───────┘ └───┬──────────┘
       │               │               │             │
       └───────────────┼───────────────┴─────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│  [3] Public Web Page Scraper (Layer 6)       │
│  Extracts embedded images from news/articles │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  [4] URL Normalization & Deduplication       │
│  Group by canonical URL and image content    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  [5] Candidate Download & Validation         │
│  Concurrent 8-worker thread pool             │
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│ Candidate Face Verification │ │ DINOv2 Visual Verification  │
│ Detects ALL candidate faces │ │ (ViT-S/14 384-dim)          │
│ ArcFace MAX face similarity │ │ Secondary supporting signal │
└──────────────┬──────────────┘ └──────────────┬──────────────┘
               └───────────────┬───────────────┘
                               ▼
┌──────────────────────────────────────────────┐
│  [6] Invariant Biometric Decision Gate       │
│  IF no face → REJECT_NO_FACE                 │
│  ELSE IF ArcFace >= 0.38 → MATCH             │
│  ELSE → NO_MATCH                             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  [7] Deterministic Canonical Manifest        │
│  Sorted keys, UTF-8 JSON                     │
│  SHA-256 Content Hash                        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  [8] Polygon Amoy / Ethereum Sepolia         │
│  EvidenceRegistry.sol: anchorEvidence()      │
│  TX Hash + Block Number Confirmed            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  [9] Blockchain Audit / Tamper Verification  │
│  Recompute SHA-256 vs. On-Chain Registry     │
│  → VERIFIED / TAMPERED                       │
└──────────────────────────────────────────────┘
Technologies
Component	Technology	Purpose
Face Detection & Recognition	InsightFace buffalo_l (SCRFD-10G + ArcFace R100)	512-dim face embeddings, multi-face prominence scoring, cosine similarity
OCR & Text Extraction	EasyOCR + Google Cloud Vision Text Detection	Extracts visible creator watermarks, page names, and usernames
Primary Reverse Search	Google Lens via SerpApi	Live full-image and face-crop reverse image discovery
Secondary Reverse Search	Google Reverse Image via SerpApi	Direct matching image results and inline images
Web Detection Search	Google Cloud Vision Web Detection	Matching pages, visually similar images, and web entities
Page Image Extractor	BeautifulSoup4 + Requests	Extracts embedded images from discovered public news/web pages
Secondary Visual Verification	DINOv2 ViT-S/14 (Facebook Research)	384-dim whole-image visual resemblance check
Evidence Hashing	SHA-256 (hashlib)	Cryptographic fingerprinting of canonical JSON manifest
Smart Contract	Solidity (EvidenceRegistry.sol)	On-chain registry mapping recordId 
→
 evidenceHash
Blockchain Client	Web3.py	Polygon Amoy & Ethereum Sepolia transaction execution
CLI & Diagnostics	Click + Rich	Structured terminal pipeline reporting
Candidate Extraction Layers
Discovery executes in strict hierarchical order:

Layer 1: Exact reverse-image results (Google Lens & Google Reverse Image).
Layer 2: Matching web pages (Google Lens & SerpApi).
Layer 3: Visual matches (Google Lens visual approximations).
Layer 4: Cloud Vision web detection (pages with matching images, full/partial images).
Layer 5: OCR / Contextual discovery (searches public pages for detected watermark strings).
Layer 6: Public web page scraper (extracts embedded photos from discovered web articles).
Setup & Installation
1. Prerequisites
Python 3.10+
Git
Dedicated testnet wallet with Polygon Amoy or Ethereum Sepolia testnet tokens (from public faucets)
SerpApi API key (from serpapi.com)
Google Cloud Service Account with Vision API enabled
2. Installation
# Clone the repository
git clone https://github.com/your-username/trace-id.git
cd trace-id

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
Environment Configuration
Copy .env.example to .env and fill in your keys:

# API Keys
SERPAPI_API_KEY=your_serpapi_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Blockchain (Polygon Amoy or Ethereum Sepolia)
POLYGON_RPC_URL=https://rpc-amoy.polygon.technology/
PRIVATE_KEY=your_testnet_wallet_private_key
CONTRACT_ADDRESS=0x4978328Ff7665676e6BBC055FcC7c53E8Fd3F1BA

# Biometric & Scoring Thresholds
FACE_SIMILARITY_THRESHOLD=0.38
VISUAL_SIMILARITY_THRESHOLD=0.60
FACE_WEIGHT=0.60
VISUAL_WEIGHT=0.30
DISCOVERY_WEIGHT=0.10

# Pipeline Settings
MAX_CANDIDATES=12
MOCK_MODE=false
Usage & CLI Commands
1. Full Pipeline Run (Real Live APIs)
python -m app.main --image data/input/sundar_pichai.png
If the input image contains multiple people, select a specific face:

python -m app.main --image data/input/group_photo.jpg --face-index 0
2. Blockchain Audit Mode
Verify an existing evidence manifest against the on-chain registry:

python -m app.main --audit data/evidence/manifest.json
Or run offline verification against the local .sha256 hash:

python -m app.main --audit data/evidence/manifest.json --mock-blockchain
3. Tamper Detection Demonstration
Simulate an attacker altering the manifest and verify the VERIFIED -> TAMPERED detection:

python -m app.main --tamper-demo data/evidence/manifest.json
4. Deploy Smart Contract
To deploy your own EvidenceRegistry contract to Polygon Amoy:

python -m app.blockchain.deploy
11-Stage Pipeline Overview
========================================
TRACE-ID PIPELINE
========================================

[1] INPUT
Image: data/input/sundar_pichai.png
Resolution: 640x873

[2] FACE DETECTION
Faces detected: 1
Face 0:
  bbox = (135, 168, 502, 673)
  area = 185335
  confidence = 0.86

Primary face:
  Face 0
  Area: 185335
  Confidence: 0.86

[3] FACE EMBEDDING
Model: InsightFace / ArcFace (buffalo_l)
Embedding dimension: 512

[4] OCR
Detected text: None

[5] REVERSE IMAGE SEARCH
Google Lens:
Exact matches: 0
Visual matches: 118
Matching pages: 118

Google Reverse Image:
Image results: 0
Inline images: 0

Google Cloud Vision:
Matching pages: 0
Full matches: 0
Partial matches: 0
Visual matches: 0

[6] CANDIDATE COLLECTION
Total raw candidates: 118
After deduplication: 10

[7] FACE VERIFICATION

Candidate 1
URL: https://www.linkedin.com/pulse/...
Faces detected: 1
Best face similarity: 0.68
DINOv2 similarity: 0.69
Result: MATCH

Candidate 2
URL: https://www.facebook.com/Stanford/posts/...
Faces detected: 1
Best face similarity: 0.98
DINOv2 similarity: 0.95
Result: MATCH

...

[8] VERIFIED MATCH
Platform: Facebook
Post URL: https://www.facebook.com/Stanford/posts/...
Face similarity: 0.98
Visual similarity: 0.95

All Verified Matches (10 photos):
  [1] facebook.com: https://www.facebook.com/... (face_sim: 0.98)
  [2] youtube.com: https://www.youtube.com/... (face_sim: 0.77)
  [3] flickr.com: https://www.flickr.com/... (face_sim: 0.72)
  [4] instagram.com: https://www.instagram.com/reel/DZ9XinruHTc (face_sim: 0.71)
  [5] instagram.com: https://www.instagram.com/reel/Dau5czOIvvm (face_sim: 0.64)

[9] EVIDENCE
Content SHA-256: fb2e5a7787a41b66398330ae85ca6c2ee89a202ac7afb2b77d7bcac28249bb05

[10] BLOCKCHAIN
Network: Ethereum Sepolia
Contract: 0x4978328Ff7665676e6BBC055FcC7c53E8Fd3F1BA
Transaction hash: 86029889cd5ffba14397cde2eb8ebb737ec38f7e165fc152199daf30edc9606c
Block number: 11649074

[11] FINAL RESULT

MATCH FOUND
Evidence anchored on Ethereum Sepolia
========================================
Evidence Manifest & Hashing
When a verified match is found, TRACE-ID builds a canonical JSON document:

{
  "version": "1.0",
  "record_id": "86029889cd5ffba14397cde2eb8ebb737ec38f7e165fc152199daf30edc9606c",
  "platform": "facebook.com",
  "post_url": "https://www.facebook.com/Stanford/posts/...",
  "image_url": "https://scontent.xx.fbcdn.net/...",
  "face_model": "InsightFace buffalo_l (ArcFace R100)",
  "face_similarity": 0.981245,
  "face_verification": "PASS",
  "visual_model": "DINOv2 ViT-S/14",
  "visual_similarity": 0.946633,
  "visual_verification": "PASS",
  "final_score": 0.972635,
  "score_weights": {
    "face_weight": 0.6,
    "visual_weight": 0.3,
    "discovery_weight": 0.1
  },
  "discovery_sources": [
    "google_lens"
  ],
  "verified_social_posts": [
    {
      "platform": "Facebook",
      "post_type": "Post",
      "url": "https://www.facebook.com/Stanford/posts/...",
      "face_similarity": 0.9812,
      "visual_similarity": 0.9466,
      "score": 0.9726
    },
    {
      "platform": "Instagram",
      "post_type": "Reel",
      "url": "https://www.instagram.com/reel/DZ9XinruHTc",
      "face_similarity": 0.7124,
      "visual_similarity": 0.6721,
      "score": 0.7289
    }
  ],
  "verification_result": "MATCH",
  "verification_statement": "Candidate image passed face similarity verification. This verifies visual correspondence, NOT legal identity or account ownership.",
  "discovered_at": "2026-09-06T18:48:14+00:00",
  "pipeline": "TRACE-ID v1.0"
}
The canonical hash is computed: 
'_' allowed only in math mode
$$\text{evidence_hash} = \text{SHA-256}(\text{canonicalize}(\text{manifest}))$$

Blockchain Integration
The smart contract EvidenceRegistry.sol provides an immutable on-chain record:

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract EvidenceRegistry {
    struct EvidenceRecord {
        bytes32 evidenceHash;
        uint256 timestamp;
        address submitter;
        bool exists;
    }

    mapping(bytes32 => EvidenceRecord) public records;
    event EvidenceAnchored(bytes32 indexed recordId, bytes32 evidenceHash, address indexed submitter, uint256 timestamp);

    function anchorEvidence(bytes32 recordId, bytes32 evidenceHash) external {
        require(!records[recordId].exists, "Record ID already exists");
        records[recordId] = EvidenceRecord(evidenceHash, block.timestamp, msg.sender, true);
        emit EvidenceAnchored(recordId, evidenceHash, msg.sender, block.timestamp);
    }

    function recordExists(bytes32 recordId) external view returns (bool) {
        return records[recordId].exists;
    }

    function verifyEvidence(bytes32 recordId, bytes32 evidenceHash) external view returns (bool) {
        if (!records[recordId].exists) return false;
        return records[recordId].evidenceHash == evidenceHash;
    }
}
Audit & Tamper Demonstration
1. Unmodified Verification
python -m app.main --audit data/evidence/manifest.json --mock-blockchain
Output:

========================================
BLOCKCHAIN AUDIT
========================================
Manifest: data/evidence/manifest.json
Mode: offline (comparing against local .sha256 file)

Local evidence hash:
fb2e5a7787a41b66398330ae85ca6c2ee89a202ac7afb2b77d7bcac28249bb05

On-chain evidence hash:
fb2e5a7787a41b66398330ae85ca6c2ee89a202ac7afb2b77d7bcac28249bb05

Status:
VERIFIED — EVIDENCE UNCHANGED
========================================
2. Malicious Modification Tamper Detection
python -m app.main --tamper-demo data/evidence/manifest.json
Output:

========================================
TRACE-ID — TAMPER DETECTION DEMO
========================================
Original manifest: data/evidence/manifest.json

1. Original Evidence Verification:
  Field: visual_similarity = 0.946633
  Local SHA-256:    fb2e5a7787a41b66398330ae85ca6c2ee89a202ac7afb2b77d7bcac28249bb05
  On-chain SHA-256: fb2e5a7787a41b66398330ae85ca6c2ee89a202ac7afb2b77d7bcac28249bb05
  Status: VERIFIED — EVIDENCE UNCHANGED

2. Simulating Malicious Manifest Modification:
  Altered: visual_similarity = 0.947633
  Tampered SHA-256: f22e08baa1c50020b7691d9573f9927c0c73f603e8c1634f92c156932c6e01e8

3. Re-running Blockchain Audit on Tampered Evidence:
  Local evidence hash:
  f22e08baa1c50020b7691d9573f9927c0c73f603e8c1634f92c156932c6e01e8
  On-chain evidence hash:
  fb2e5a7787a41b66398330ae85ca6c2ee89a202ac7afb2b77d7bcac28249bb05
  Status: TAMPERED — EVIDENCE HASH DOES NOT MATCH

Result: VERIFIED -> TAMPERED successfully demonstrated.
========================================
Testing & Validation
Run the automated test suite:

python -m pytest tests/ -v
============================= test session starts =============================
tests/test_audit.py (11 tests) ........................................ PASSED
tests/test_canonicalize.py (15 tests) ................................. PASSED
tests/test_deduplicator.py (8 tests) .................................. PASSED
tests/test_different_photo_matching.py (7 tests) ...................... PASSED
  - test_1_exact_same_image_match                                      PASSED
  - test_2_different_photo_same_person_match                           PASSED
  - test_3_different_person_rejected                                   PASSED
  - test_4_input_multiple_people_primary_face_selected                 PASSED
  - test_4_no_face_detected_state                                      PASSED
  - test_5_multi_person_candidate_selection                            PASSED
  - test_6_tamper_detection_after_anchoring                            PASSED
tests/test_hasher.py (12 tests) ....................................... PASSED
tests/test_scoring.py (10 tests) ...................................... PASSED
tests/test_url_normalization.py (30 tests) ............................ PASSED

============================= 93 passed in 0.61s ==============================
Ethical, Privacy & Safety Considerations
Consenting Test Data: Use only your own photos, consenting test subjects, or public/authorized content.
No Mass Surveillance: The system is designed for verifiable proof of public presence, not unrestricted facial recognition of private citizens.
No Biometrics On-Chain: Raw 512-d embeddings are discarded from RAM after verification and are never stored on disk or written to the blockchain.
Zero Fabrication: If discovery APIs return no matches, the system reports NO_VERIFIED_MATCH and explains the bottleneck rather than hallucinating links.
Hacker House Goa Task #3 Compliance
Requirement	Implementation in TRACE-ID	Status
Detect and encode face	InsightFace SCRFD-10G + ArcFace R100 512-d embeddings	✅ Complete
Genuine reverse-image search	Live Google Lens + Google Reverse Image + Google Cloud Vision + EasyOCR	✅ Complete
Find real matching public posts	Dynamically extracted social media URLs (Instagram, X, LinkedIn, YouTube, Facebook)	✅ Complete
Different-photo verification	ArcFace invariant biometric decision gate (independent of background/pose)	✅ Complete
Tamper-evident blockchain record	Canonical JSON SHA-256 hash anchored to Polygon Amoy / Ethereum Sepolia smart contract	✅ Complete
Audit & Tamper Demonstration	Live blockchain replay + local verification (VERIFIED -> TAMPERED)	✅ Complete
Zero Mocking / Fabrication	All candidate links originate from live API discovery	✅ Complete
TRACE-ID — Hacker House Goa 2026 Submission
