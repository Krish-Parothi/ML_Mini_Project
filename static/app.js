/**
 * CyberShield Security - Production UI Controller
 * Standard Enterprise Architecture (Scanner, Batch Telemetry, Benchmarking, Report)
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initScanner();
  initBatchAudit();
  loadModelBenchmarks();
  loadReportMarkdown();
});

/* ==========================================================================
   1. Tab Navigation System
   ========================================================================== */
function initTabs() {
  const navBtns = document.querySelectorAll(".nav-link, .nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      
      navBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

/* ==========================================================================
   2. Single URL Threat Scanner Controller
   ========================================================================== */
function initScanner() {
  const scanForm = document.getElementById("scanForm");
  const urlInput = document.getElementById("urlInput");
  const submitBtn = document.getElementById("scanSubmitBtn");
  const btnText = submitBtn.querySelector(".btn-text");
  const btnLoader = submitBtn.querySelector(".btn-loader");
  const activeModelSelect = document.getElementById("activeModelSelect");
  const resultArea = document.getElementById("scanResultArea");
  
  // Quick Sample Chips
  document.querySelectorAll(".test-chip, .sample-tag").forEach(tag => {
    tag.addEventListener("click", () => {
      const url = tag.getAttribute("data-url");
      urlInput.value = url;
      triggerScan(url);
    });
  });

  // Form Submission
  scanForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const url = urlInput.value.trim();
    if (!url) return;
    triggerScan(url);
  });

  // Toggle 51 Features Drawer
  const toggleFeaturesBtn = document.getElementById("toggleAllFeaturesBtn");
  const fullDrawer = document.getElementById("fullFeatureDrawer");
  toggleFeaturesBtn.addEventListener("click", () => {
    const isHidden = fullDrawer.classList.contains("hidden");
    if (isHidden) {
      fullDrawer.classList.remove("hidden");
      toggleFeaturesBtn.textContent = "Hide 51 Vector";
    } else {
      fullDrawer.classList.add("hidden");
      toggleFeaturesBtn.textContent = "View Full Vector";
    }
  });

  async function triggerScan(url) {
    btnText.textContent = "Analyzing...";
    btnLoader.classList.remove("hidden");
    submitBtn.disabled = true;

    try {
      const selectedModel = activeModelSelect.value;
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url, model_name: selectedModel })
      });

      if (!res.ok) {
        throw new Error(`HTTP Error ${res.status}`);
      }

      const data = await res.json();
      renderScanResult(data);
    } catch (err) {
      console.error("Scan error:", err);
      alert("Failed to inspect URL. Ensure server is running on http://127.0.0.1:8000.");
    } finally {
      btnText.textContent = "Analyze URL";
      btnLoader.classList.add("hidden");
      submitBtn.disabled = false;
    }
  }

  function renderScanResult(data) {
    resultArea.classList.remove("hidden");

    // Telemetry & Verdict Header
    document.getElementById("latencyBadge").textContent = 
      `Inference: ${data.latency_ms.model_inference} ms | Total: ${data.latency_ms.total} ms`;
    document.getElementById("usedModelTag").textContent = data.model_used;

    const verdictStatus = document.getElementById("verdictStatus");
    const riskLevelBadge = document.getElementById("riskLevelBadge");
    const verdictSummary = document.getElementById("verdictSummary");
    const riskScoreValue = document.getElementById("riskScoreValue");
    const circularProgress = document.getElementById("circularProgress");

    const phishProb = data.phishing_probability;
    riskScoreValue.textContent = `${phishProb}%`;

    // Enterprise color states
    let accentColor = "#10b981"; // Emerald
    if (phishProb >= 75) {
      accentColor = "#f43f5e"; // Rose Red
    } else if (phishProb >= 40) {
      accentColor = "#f59e0b"; // Amber
    } else if (phishProb > 10) {
      accentColor = "#3b82f6"; // Blue
    }

    riskScoreValue.style.color = accentColor;
    circularProgress.style.background = `conic-gradient(${accentColor} ${phishProb * 3.6}deg, var(--color-canvas) ${phishProb * 3.6}deg)`;

    if (data.is_phishing) {
      verdictStatus.textContent = "Phishing Threat Detected";
      verdictStatus.className = "verdict-status-pill status-threat";
      riskLevelBadge.textContent = data.risk_level;
      riskLevelBadge.className = "risk-pill threat";
      verdictSummary.textContent = "High probability of deceptive credential harvesting, spoofing, or topological evasion.";
    } else {
      verdictStatus.textContent = "Legitimate URL";
      verdictStatus.className = "verdict-status-pill status-safe";
      riskLevelBadge.textContent = data.risk_level;
      riskLevelBadge.className = "risk-pill";
      verdictSummary.textContent = "Lexical and topological indicators align with standard authentic web properties.";
    }

    // Explainability Indicators
    const flagsList = document.getElementById("flagsList");
    const flagsCountBadge = document.getElementById("flagsCountBadge");
    flagsList.innerHTML = "";
    flagsCountBadge.textContent = `${data.flags.length} Indicators`;

    if (data.flags.length === 0) {
      flagsList.innerHTML = `
        <div class="empty-flags">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
          </svg>
          <span>No malicious lexical anomalies or heuristic flags detected.</span>
        </div>
      `;
    } else {
      data.flags.forEach(flag => {
        const item = document.createElement("div");
        item.className = "flag-row";
        item.innerHTML = `
          <span>${flag.message}</span>
          <span class="flag-badge">${flag.level.toUpperCase()}</span>
        `;
        flagsList.appendChild(item);
      });
    }

    // Key Features Grid
    const topGrid = document.getElementById("topFeaturesGrid");
    topGrid.innerHTML = "";
    for (const [key, val] of Object.entries(data.top_features)) {
      const cell = document.createElement("div");
      cell.className = "metric-cell";
      cell.innerHTML = `
        <div class="metric-label">${key}</div>
        <div class="metric-value">${typeof val === 'boolean' ? (val ? 'True' : 'False') : val}</div>
      `;
      topGrid.appendChild(cell);
    }

    // Complete 51-Dimensional Feature Drawer
    renderAllFeatures(data.url);
  }

  async function renderAllFeatures(url) {
    const allGrid = document.getElementById("allFeaturesGrid");
    allGrid.innerHTML = '<div style="color:var(--color-text-tertiary);padding:0.5rem;font-size:0.8rem;">Populating full 51 vector...</div>';
    
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url, model_name: "Best (Auto)" })
      });
      const d = await res.json();
      allGrid.innerHTML = "";
      
      const featureKeys = [
        "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex", "CharContinuationRate",
        "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain", "HasObfuscation",
        "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL",
        "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS", "LineOfCode",
        "LargestLineLength", "HasTitle", "DomainTitleMatchScore", "URLTitleMatchScore", "HasFavicon",
        "Robots", "IsResponsive", "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
        "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton", "HasHiddenFields",
        "HasPasswordField", "Bank", "Pay", "Crypto", "HasCopyrightInfo", "NoOfImage", "NoOfCSS",
        "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef", "TLD_Code"
      ];

      featureKeys.forEach(k => {
        const item = document.createElement("div");
        item.className = "drawer-cell";
        const val = d.top_features[k] !== undefined ? d.top_features[k] : 0;
        item.innerHTML = `
          <span>${k}</span>
          <span>${typeof val === 'boolean' ? (val ? 1 : 0) : val}</span>
        `;
        allGrid.appendChild(item);
      });
    } catch(e) {
      allGrid.innerHTML = '<div style="color:var(--color-danger);font-size:0.8rem;">Failed to expand vector.</div>';
    }
  }
}

/* ==========================================================================
   3. Batch URL Audit Controller
   ========================================================================== */
function initBatchAudit() {
  const batchInput = document.getElementById("batchInput");
  const runBtn = document.getElementById("runBatchBtn");
  const clearBtn = document.getElementById("clearBatchBtn");
  const sampleBtn = document.getElementById("loadBatchSampleBtn");
  const resultsWrapper = document.getElementById("batchResultsWrapper");
  const exportCsvBtn = document.getElementById("exportCsvBtn");

  const sampleUrls = [
    "https://www.google.com",
    "https://en.wikipedia.org/wiki/Phishing",
    "http://192.168.1.105/paypal-login/update.php?account=verify",
    "https://www.github.com/login",
    "http://www.f0519141.xsph.ru/bank-auth/signin.html",
    "https://service-mitld.firebaseapp.com/login?token=%20%40hack",
    "https://aws.amazon.com/console/",
    "http://verify-bankofamerica.top/security/update.asp",
    "https://stackoverflow.com/questions",
    "http://apple-support-icloud-lock.xyz/auth/findmy"
  ];

  sampleBtn.addEventListener("click", () => {
    batchInput.value = sampleUrls.join("\n");
  });

  clearBtn.addEventListener("click", () => {
    batchInput.value = "";
    resultsWrapper.classList.add("hidden");
  });

  let currentBatchData = null;

  runBtn.addEventListener("click", async () => {
    const rawText = batchInput.value.trim();
    if (!rawText) {
      alert("Please enter at least one URL.");
      return;
    }

    const urls = rawText.split("\n").map(u => u.trim()).filter(u => u.length > 0);
    if (urls.length === 0) return;

    runBtn.textContent = "Processing...";
    runBtn.disabled = true;

    try {
      const activeModelSelect = document.getElementById("activeModelSelect");
      const res = await fetch("/api/batch-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: urls, model_name: activeModelSelect.value })
      });

      if (!res.ok) throw new Error("Batch execution failed");
      const data = await res.json();
      currentBatchData = data;
      renderBatchResults(data);
    } catch (err) {
      console.error("Batch error:", err);
      alert("Batch audit failed. Please try again.");
    } finally {
      runBtn.textContent = "Execute Batch Scan";
      runBtn.disabled = false;
    }
  });

  function renderBatchResults(data) {
    resultsWrapper.classList.remove("hidden");
    document.getElementById("batchTotalCount").textContent = data.total_urls;
    document.getElementById("batchPhishCount").textContent = data.phishing_count;
    document.getElementById("batchSafeCount").textContent = data.safe_count;
    document.getElementById("batchThroughput").textContent = data.throughput_urls_per_sec;

    const tbody = document.getElementById("batchTableBody");
    tbody.innerHTML = "";

    data.results.forEach((item, idx) => {
      const tr = document.createElement("tr");
      const isPhish = item.status === "Phishing";
      tr.innerHTML = `
        <td style="color:var(--color-text-tertiary);font-family:var(--font-mono);">${idx + 1}</td>
        <td style="max-width:320px;overflow:hidden;text-overflow:ellipsis;font-family:var(--font-mono);">${escapeHtml(item.url)}</td>
        <td>
          <span style="font-weight:600;color:${isPhish ? 'var(--color-danger)' : 'var(--color-success)'}">
            ${item.status}
          </span>
        </td>
        <td style="font-family:var(--font-mono);font-weight:600;">${item.phishing_probability}%</td>
        <td><span class="count-tag">${item.flag_count} flags</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Export to CSV
  exportCsvBtn.addEventListener("click", () => {
    if (!currentBatchData || !currentBatchData.results) return;
    let csv = "Index,URL,Status,Phishing_Probability,Flag_Count\n";
    currentBatchData.results.forEach((r, idx) => {
      csv += `${idx + 1},"${r.url}",${r.status},${r.phishing_probability}%,${r.flag_count}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `cybershield_batch_audit_${Date.now()}.csv`;
    link.click();
  });
}

/* ==========================================================================
   4. Model Benchmarks Matrix Loader
   ========================================================================== */
async function loadModelBenchmarks() {
  const tbody = document.getElementById("benchmarkTableBody");
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const comparison = data.comparison;
    tbody.innerHTML = "";

    for (const [name, metrics] of Object.entries(comparison)) {
      const isBest = name === "LightGBM";
      const tr = document.createElement("tr");
      if (isBest) tr.className = "row-best";

      tr.innerHTML = `
        <td>
          <strong>${name}</strong>
          ${isBest ? ' <span class="tag-neutral" style="font-size:0.65rem;color:var(--color-primary);border-color:var(--color-primary-border);">Production Best</span>' : ''}
        </td>
        <td>${metrics.model_type}</td>
        <td style="font-family:var(--font-mono);font-weight:600;">${(metrics.accuracy * 100).toFixed(2)}%</td>
        <td style="font-family:var(--font-mono);">${(metrics.precision * 100).toFixed(2)}%</td>
        <td style="font-family:var(--font-mono);">${(metrics.recall * 100).toFixed(2)}%</td>
        <td style="font-family:var(--font-mono);">${(metrics.specificity * 100).toFixed(2)}%</td>
        <td style="font-family:var(--font-mono);">${(metrics.f1_score * 100).toFixed(2)}%</td>
        <td style="font-family:var(--font-mono);">${metrics.roc_auc.toFixed(4)}</td>
        <td style="font-family:var(--font-mono);">${metrics.train_time_sec}s</td>
        <td style="font-family:var(--font-mono);">${metrics.latency_ms_per_sample.toFixed(4)} ms</td>
        <td style="font-family:var(--font-mono);">${metrics.model_size_mb} MB</td>
      `;
      tbody.appendChild(tr);
    }
  } catch(e) {
    console.error("Failed to fetch model metrics:", e);
    tbody.innerHTML = '<tr><td colspan="11">Failed to load telemetry.</td></tr>';
  }
}

/* ==========================================================================
   5. Technical Report Loader
   ========================================================================== */
async function loadReportMarkdown() {
  const container = document.getElementById("reportContainer");
  const copyBtn = document.getElementById("copyReportBtn");

  try {
    const res = await fetch("/REPORT.md");
    if (!res.ok) throw new Error("Could not find REPORT.md");
    const md = await res.text();
    
    container.innerHTML = parseSimpleMarkdown(md);

    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(md).then(() => {
        copyBtn.textContent = "Copied to Clipboard!";
        setTimeout(() => copyBtn.textContent = "Copy Full Report Text", 2000);
      });
    });
  } catch (err) {
    container.innerHTML = `
      <div style="padding:1.5rem;color:var(--color-text-secondary);">
        <h3>Academic Report Available</h3>
        <p>The comprehensive 10-page academic report is located at <code>REPORT.md</code> in the repository root.</p>
      </div>
    `;
  }
}

function parseSimpleMarkdown(md) {
  let html = md
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
    .replace(/\*(.*)\*/gim, '<em>$1</em>')
    .replace(/`([^`]+)`/gim, '<code>$1</code>')
    .replace(/\n\n/gim, '<p></p>')
    .replace(/\n/gim, '<br>');
  return html;
}

function escapeHtml(text) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return text.replace(/[&<>"']/g, m => map[m]);
}
