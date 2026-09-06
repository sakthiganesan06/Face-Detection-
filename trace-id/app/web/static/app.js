/**
 * TRACE-ID — Frontend Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const dropPrompt = document.getElementById("dropPrompt");
  const previewWrapper = document.getElementById("previewWrapper");
  const imagePreview = document.getElementById("imagePreview");
  const removeImgBtn = document.getElementById("removeImgBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const maxCandidatesSelect = document.getElementById("maxCandidates");
  const sampleBtns = document.querySelectorAll(".sample-btn");

  const statusCard = document.getElementById("statusCard");
  const stepperTitle = document.getElementById("stepperTitle");
  const resultsSection = document.getElementById("resultsSection");
  const networkName = document.getElementById("networkName");

  const personName = document.getElementById("personName");
  const resultThumb = document.getElementById("resultThumb");
  const bestScore = document.getElementById("bestScore");
  const pipelineTiming = document.getElementById("pipelineTiming");
  const socialGrid = document.getElementById("socialGrid");
  const socialCount = document.getElementById("socialCount");

  const proofRecordId = document.getElementById("proofRecordId");
  const proofHash = document.getElementById("proofHash");
  const proofContract = document.getElementById("proofContract");
  const proofTx = document.getElementById("proofTx");
  const etherscanTxLink = document.getElementById("etherscanTxLink");

  const runAuditBtn = document.getElementById("runAuditBtn");
  const runTamperDemoBtn = document.getElementById("runTamperDemoBtn");
  const auditResultBox = document.getElementById("auditResultBox");
  const candidatesTable = document.getElementById("candidatesTable").querySelector("tbody");

  let currentFile = null;

  // 1. Fetch system status on load
  fetchStatus();

  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        networkName.textContent = `${data.network} • ${data.contract_address.slice(0, 6)}...${data.contract_address.slice(-4)}`;
      }
    } catch (err) {
      console.warn("Could not fetch status:", err);
      networkName.textContent = "Offline / Local Mode";
    }
  }

  // 2. Drag & Drop events
  ["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  browseBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  removeImgBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

  // 3. Quick Sample Picker
  sampleBtns.forEach(btn => {
    btn.addEventListener("click", async () => {
      try {
        const res = await fetch("/data/input/sundar_pichai.png");
        const blob = await res.blob();
        const sampleFile = new File([blob], "sundar_pichai.png", { type: "image/png" });
        handleFile(sampleFile);
      } catch (err) {
        // Fallback local fetch
        const sampleFile = new File(["dummy"], "sundar_pichai.png", { type: "image/png" });
        handleFile(sampleFile);
      }
    });
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (JPG, PNG, WebP).");
      return;
    }
    currentFile = file;
    dropPrompt.style.display = "none";
    previewWrapper.style.display = "block";
    analyzeBtn.disabled = false;

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      resultThumb.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    currentFile = null;
    fileInput.value = "";
    imagePreview.src = "";
    dropPrompt.style.display = "block";
    previewWrapper.style.display = "none";
    analyzeBtn.disabled = true;
    resultsSection.style.display = "none";
    statusCard.style.display = "none";
  }

  // 4. Analyze Button Action
  analyzeBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    // UI state: analyzing
    analyzeBtn.disabled = true;
    resultsSection.style.display = "none";
    statusCard.style.display = "block";
    auditResultBox.style.display = "none";

    // Start animated stepper progress
    resetSteps();
    startStepperAnimation();

    const formData = new FormData();
    formData.append("file", currentFile);
    formData.append("max_candidates", maxCandidatesSelect.value);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        stepperTitle.textContent = `🛡️ ${data.message || data.detail || "No Biometric Match Found"}`;
        
        if (data.all_candidates && data.all_candidates.length > 0) {
          // Render the candidates table so user can inspect what Google Lens scanned
          candidatesTable.innerHTML = "";
          data.all_candidates.forEach(cand => {
            const row = document.createElement("tr");
            row.innerHTML = `
              <td>
                <div style="display: flex; align-items: center; gap: 8px;">
                  ${cand.image_url ? `<img src="${cand.image_url}" style="width: 32px; height: 32px; border-radius: 6px; object-fit: cover; flex-shrink: 0;" onerror="this.style.display='none'">` : ''}
                  <strong>${escapeHtml(cand.domain || "Web")}</strong>
                </div>
              </td>
              <td><a href="${cand.page_url}" target="_blank" style="color: var(--primary-cyan); text-decoration: none;">${escapeHtml(cand.title || "Match")}</a></td>
              <td>${(cand.face_similarity * 100).toFixed(1)}%</td>
              <td>${(cand.visual_similarity * 100).toFixed(1)}%</td>
              <td><strong>${(cand.final_score * 100).toFixed(1)}%</strong></td>
              <td><span style="color: var(--accent-pink); font-weight: bold;">REJECTED</span></td>
            `;
            candidatesTable.appendChild(row);
          });
          resultsSection.style.display = "block";
          document.querySelector(".person-banner").style.display = "none";
          document.querySelector(".blockchain-proof-card").style.display = "none";
          document.querySelector(".social-grid").innerHTML = `
            <div style="grid-column: 1 / -1; padding: 20px; background: rgba(255, 68, 68, 0.08); border: 1px solid rgba(255, 68, 68, 0.25); border-radius: var(--radius-md); color: #ff8888;">
              <strong>🛡️ Zero False-Positive Biometric Protection:</strong><br>
              Google Lens found ${data.all_candidates.length} visual approximations on the web, but ArcFace compared their facial geometry with the uploaded face and rejected all of them (highest match was ${data.all_candidates[0] ? (data.all_candidates[0].face_similarity * 100).toFixed(1) : 0}%).
            </div>
          `;
          resultsSection.scrollIntoView({ behavior: "smooth" });
        } else {
          alert(data.message || data.detail || "No candidates found online for this image.");
        }
        analyzeBtn.disabled = false;
        return;
      }

      // Complete all steps
      finishAllSteps();
      stepperTitle.textContent = "✅ Verification & Anchoring Complete!";

      // Render Results
      setTimeout(() => {
        renderResults(data);
        resultsSection.style.display = "block";
        analyzeBtn.disabled = false;
        resultsSection.scrollIntoView({ behavior: "smooth" });
      }, 600);

    } catch (err) {
      console.error("API error:", err);
      stepperTitle.textContent = "❌ Connection or Server Error";
      alert("Error connecting to backend server. Make sure the server is running.");
      analyzeBtn.disabled = false;
    }
  });

  // 5. Render Results
  function renderResults(data) {
    document.querySelector(".person-banner").style.display = "flex";
    document.querySelector(".blockchain-proof-card").style.display = "block";
    personName.textContent = data.recognized_person || "Identified Subject";
    bestScore.textContent = `${(data.best_match.final_score * 100).toFixed(1)}%`;
    pipelineTiming.textContent = `Pipeline completed in ${data.elapsed_seconds}s • Verified match on ${data.best_match.domain}`;

    if (imagePreview.src) {
      resultThumb.src = imagePreview.src;
    }

    // Render Social Grid
    socialGrid.innerHTML = "";
    const socials = data.verified_social_posts || [];
    socialCount.textContent = `${socials.length} Verified Posts Found`;

    if (socials.length === 0) {
      socialGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 24px; text-align: center; color: var(--text-muted); background: var(--bg-surface); border-radius: var(--radius-md);">
          No direct social media posts found in top results. Web matches were verified.
        </div>
      `;
    } else {
      socials.forEach(post => {
        const card = document.createElement("div");
        card.className = "social-card";

        let pillClass = "pill-x";
        const platLower = post.platform.toLowerCase();
        if (platLower.includes("instagram")) pillClass = "pill-instagram";
        else if (platLower.includes("facebook")) pillClass = "pill-facebook";
        else if (platLower.includes("linkedin")) pillClass = "pill-linkedin";
        else if (platLower.includes("youtube")) pillClass = "pill-youtube";

        card.innerHTML = `
          <div style="display: flex; gap: 14px; align-items: flex-start;">
            ${post.thumbnail ? `<img src="${post.thumbnail}" style="width: 58px; height: 58px; border-radius: 8px; object-fit: cover; border: 1px solid rgba(0,240,255,0.2); flex-shrink: 0;" onerror="this.style.display='none'">` : ''}
            <div style="flex: 1; min-width: 0;">
              <div class="social-card-top">
                <span class="platform-pill ${pillClass}">${post.platform} • ${post.post_type}</span>
                <span class="face-sim-badge">Face: ${(post.face_similarity * 100).toFixed(1)}%</span>
              </div>
              <h4 class="social-title" style="margin-top: 6px;">${escapeHtml(post.title || "Social Media Post")}</h4>
            </div>
          </div>
          <a href="${post.url}" target="_blank" rel="noopener noreferrer" class="social-link-btn" style="margin-top: 12px;">
            <span>View Original Post</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3"/></svg>
          </a>
        `;
        socialGrid.appendChild(card);
      });
    }

    // Blockchain Proof
    proofRecordId.textContent = data.evidence.record_id || "-";
    proofHash.textContent = data.evidence.evidence_hash || "-";
    
    if (data.blockchain_receipt) {
      proofContract.textContent = data.blockchain_receipt.contract_address || "-";
      proofTx.textContent = data.blockchain_receipt.transaction_hash || "Anchored on Sepolia";
      
      const txHash = data.blockchain_receipt.transaction_hash;
      if (txHash && txHash.startsWith("0x")) {
        etherscanTxLink.href = `https://sepolia.etherscan.io/tx/${txHash}`;
        etherscanTxLink.style.display = "inline-flex";
      } else {
        etherscanTxLink.href = `https://sepolia.etherscan.io/address/${data.blockchain_receipt.contract_address || ''}`;
        etherscanTxLink.style.display = "inline-flex";
      }
    }

    // All candidates table
    candidatesTable.innerHTML = "";
    (data.all_candidates || []).forEach(cand => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>
          <div style="display: flex; align-items: center; gap: 8px;">
            ${cand.image_url ? `<img src="${cand.image_url}" style="width: 32px; height: 32px; border-radius: 6px; object-fit: cover; flex-shrink: 0;" onerror="this.style.display='none'">` : ''}
            <strong>${escapeHtml(cand.domain || "Web")}</strong>
          </div>
        </td>
        <td><a href="${cand.page_url}" target="_blank" style="color: var(--primary-cyan); text-decoration: none;">${escapeHtml(cand.title || "Match")}</a></td>
        <td>${(cand.face_similarity * 100).toFixed(1)}%</td>
        <td>${(cand.visual_similarity * 100).toFixed(1)}%</td>
        <td><strong>${(cand.final_score * 100).toFixed(1)}%</strong></td>
        <td><span style="color: ${cand.accepted ? 'var(--accent-emerald)' : 'var(--accent-pink)'}; font-weight: bold;">${cand.accepted ? 'PASS' : 'FAIL'}</span></td>
      `;
      candidatesTable.appendChild(row);
    });
  }

  // 6. Stepper Controls
  let stepInterval;
  function resetSteps() {
    for (let i = 1; i <= 5; i++) {
      const step = document.getElementById(`step${i}`);
      step.className = "step-item";
    }
    clearInterval(stepInterval);
  }

  function startStepperAnimation() {
    let currentStep = 1;
    document.getElementById("step1").classList.add("active");
    stepperTitle.textContent = "Extracting Face Biometrics...";

    stepInterval = setInterval(() => {
      if (currentStep < 4) {
        document.getElementById(`step${currentStep}`).classList.remove("active");
        document.getElementById(`step${currentStep}`).classList.add("done");
        currentStep++;
        document.getElementById(`step${currentStep}`).classList.add("active");
        
        const titles = [
          "",
          "Searching Live Web & Social Networks...",
          "Running Dual AI Verification...",
          "Generating Cryptographic SHA-256 Manifest...",
          "Anchoring Evidence to Ethereum Blockchain..."
        ];
        stepperTitle.textContent = titles[currentStep] || "Processing...";
      }
    }, 2800);
  }

  function finishAllSteps() {
    clearInterval(stepInterval);
    for (let i = 1; i <= 5; i++) {
      const step = document.getElementById(`step${i}`);
      step.className = "step-item done";
    }
  }

  // 7. On-Chain Integrity Audit
  runAuditBtn.addEventListener("click", async () => {
    auditResultBox.style.display = "block";
    auditResultBox.className = "audit-result-box";
    auditResultBox.textContent = "Querying Ethereum blockchain contract...";

    try {
      const res = await fetch("/api/audit", { method: "POST" });
      const data = await res.json();

      if (data.verified) {
        auditResultBox.className = "audit-result-box audit-pass";
        auditResultBox.innerHTML = `
          <strong>✅ ON-CHAIN AUDIT VERIFIED: MATCH</strong><br>
          • Record ID: <code>${data.record_id}</code><br>
          • Local Evidence Hash: <code>${data.local_hash}</code><br>
          • On-Chain Stored Hash: <code>${data.chain_hash}</code><br>
          <em>The evidence manifest has been verified unmodified against the immutable smart contract record.</em>
        `;
      } else {
        auditResultBox.className = "audit-result-box audit-fail";
        auditResultBox.innerHTML = `
          <strong>❌ AUDIT FAILED: TAMPER DETECTED</strong><br>
          • Reason: ${data.error || "Hash mismatch with on-chain record"}
        `;
      }
    } catch (err) {
      auditResultBox.className = "audit-result-box audit-fail";
      auditResultBox.textContent = "Failed to communicate with blockchain audit endpoint.";
    }
  });

  // 8. Tamper Demo Simulation
  runTamperDemoBtn.addEventListener("click", async () => {
    auditResultBox.style.display = "block";
    auditResultBox.className = "audit-result-box";
    auditResultBox.textContent = "Simulating data tampering & hash recomputation...";

    try {
      const res = await fetch("/api/tamper-demo", { method: "POST" });
      const data = await res.json();

      auditResultBox.className = "audit-result-box audit-fail";
      auditResultBox.innerHTML = `
        <strong>🛡️ TAMPER DETECTION DEMO PASSED</strong><br>
        • Field Modified: <code>${data.details}</code><br>
        • Original Hash: <code>${data.original_hash}</code><br>
        • Tampered Hash: <code>${data.tampered_hash}</code><br>
        • Verdict: <strong>${data.status}</strong> (Cryptographic integrity check rejected the altered evidence!)
      `;
    } catch (err) {
      auditResultBox.className = "audit-result-box audit-fail";
      auditResultBox.textContent = "Tamper demo execution error.";
    }
  });

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
});
