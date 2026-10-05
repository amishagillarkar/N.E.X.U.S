/**
 * NEXUS AI - Reactive Frontend Controller
 * ========================================
 * Powers the interactive dashboard, presets, elimination matrix,
 * circular gauges, copilot drawer, and API key rotation manager.
 */

// Application State
const state = {
  currentProfile: null,
  latestDiagnosis: null,
  knownSkills: ["Python", "DSA", "SQL"],
  targetPaths: ["PLACEMENTS", "GATE_CS"],
  copilotHistory: [],
  llmConfig: null,
};

// Preset Scenarios
const PRESETS = {
  overwhelmed_3rd: {
    name: "Rahul Sharma",
    branch: "CSE",
    year_of_study: "3rd Year",
    cgpa: 7.8,
    runway: 6,
    hours: 25,
    paths: ["PLACEMENTS", "GATE_CS"],
    skills: ["Python", "DSA", "SQL", "OOPs"],
    constraints: "College 9am-4pm, no unpaid internships",
  },
  gate_da: {
    name: "Ananya Iyer",
    branch: "Data Science & AI",
    year_of_study: "4th Year / Final Year",
    cgpa: 8.5,
    runway: 5,
    hours: 32,
    paths: ["GATE_DA", "PLACEMENTS"],
    skills: ["Python", "Linear Algebra", "Machine Learning", "SQL", "DBMS"],
    constraints: "Targeting top IIT M.Tech DA program",
  },
  cat_pivot: {
    name: "Vikram Malhotra",
    branch: "Mechanical Engineering",
    year_of_study: "4th Year / Final Year",
    cgpa: 7.2,
    runway: 8,
    hours: 22,
    paths: ["CAT", "PLACEMENTS"],
    skills: ["Quantitative Aptitude", "Verbal Ability", "Basic Python"],
    constraints: "Non-CS background, pivoting towards management consulting",
  },
  study_abroad: {
    name: "Meera Patel",
    branch: "IT",
    year_of_study: "3rd Year",
    cgpa: 9.1,
    runway: 12,
    hours: 20,
    paths: ["STUDY_ABROAD", "PLACEMENTS"],
    skills: ["Python", "Java", "DSA", "Projects & Portfolio", "Git"],
    constraints: "Needs research paper + GRE 325+ score",
  },
  fresh_2nd: {
    name: "Rohan Verma",
    branch: "CSE",
    year_of_study: "2nd Year",
    cgpa: 8.0,
    runway: 18,
    hours: 18,
    paths: ["PLACEMENTS", "GATE_CS"],
    skills: ["C++", "DSA Basics"],
    constraints: "Early starter, building long-term foundation",
  },
};

// DOM Elements
document.addEventListener("DOMContentLoaded", () => {
  initUI();
  fetchLLMStatus();
});

function initUI() {
  // Range sliders
  const cgpaInput = document.getElementById("cgpaInput");
  const cgpaVal = document.getElementById("cgpaVal");
  cgpaInput.addEventListener("input", (e) => {
    cgpaVal.textContent = parseFloat(e.target.value).toFixed(1);
  });

  const runwayMonths = document.getElementById("runwayMonths");
  const runwayVal = document.getElementById("runwayVal");
  runwayMonths.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    const weeks = Math.round(val * 4.345);
    runwayVal.textContent = `${val} Month${val > 1 ? "s" : ""} (~${weeks} Weeks)`;
  });

  const weeklyHours = document.getElementById("weeklyHours");
  const weeklyHoursVal = document.getElementById("weeklyHoursVal");
  const burnoutAlert = document.getElementById("burnoutAlert");
  weeklyHours.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    weeklyHoursVal.textContent = `${val} hrs / week`;
    if (val >= 40) {
      burnoutAlert.classList.add("active");
    } else {
      burnoutAlert.classList.remove("active");
    }
  });

  // Skills tag handling
  renderSkillTags();
  document.getElementById("btnAddSkill").addEventListener("click", addCustomSkill);
  document.getElementById("customSkillInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addCustomSkill();
    }
  });

  // Quick skill pills
  document.querySelectorAll(".quick-skill-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      const skill = pill.dataset.skill;
      if (!state.knownSkills.includes(skill)) {
        state.knownSkills.push(skill);
        renderSkillTags();
      }
    });
  });

  // Path selection
  document.querySelectorAll(".path-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const pathId = chip.dataset.path;
      const idx = state.targetPaths.indexOf(pathId);
      if (idx > -1) {
        if (state.targetPaths.length > 1) {
          state.targetPaths.splice(idx, 1);
        }
      } else {
        if (state.targetPaths.length < 4) {
          state.targetPaths.push(pathId);
        }
      }
      renderPathSelection();
    });
  });
  renderPathSelection();

  // Presets
  document.querySelectorAll(".preset-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.preset;
      if (PRESETS[key]) {
        applyPreset(PRESETS[key]);
      }
    });
  });

  // Form submit
  document.getElementById("diagnosticForm").addEventListener("submit", (e) => {
    e.preventDefault();
    runDiagnosis();
  });

  document.getElementById("btnQuickDemo").addEventListener("click", () => {
    applyPreset(PRESETS.overwhelmed_3rd);
    runDiagnosis();
  });

  document.getElementById("btnResetForm").addEventListener("click", () => {
    applyPreset(PRESETS.overwhelmed_3rd);
  });

  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });

  // Copy & Print Report
  document.getElementById("btnCopyReport").addEventListener("click", () => {
    if (state.latestDiagnosis && state.latestDiagnosis.report) {
      navigator.clipboard.writeText(state.latestDiagnosis.report);
      alert("Diagnostic Report copied to clipboard!");
    }
  });

  document.getElementById("btnPrintReport").addEventListener("click", () => {
    window.print();
  });

  // Copilot Drawer
  const copilotDrawer = document.getElementById("copilotDrawer");
  document.getElementById("btnOpenCopilot").addEventListener("click", () => {
    copilotDrawer.classList.toggle("open");
  });
  document.getElementById("btnCloseCopilot").addEventListener("click", () => {
    copilotDrawer.classList.remove("open");
  });
  document.getElementById("copilotForm").addEventListener("submit", (e) => {
    e.preventDefault();
    sendCopilotMessage();
  });

  // Settings Modal
  const settingsModal = document.getElementById("settingsModal");
  document.getElementById("btnOpenSettings").addEventListener("click", () => {
    openSettingsModal();
  });
  document.getElementById("providerStatusPill").addEventListener("click", () => {
    openSettingsModal();
  });
  document.getElementById("btnCloseSettings").addEventListener("click", () => {
    settingsModal.classList.remove("open");
  });
  document.getElementById("btnCancelSettings").addEventListener("click", () => {
    settingsModal.classList.remove("open");
  });
  document.getElementById("btnSaveSettings").addEventListener("click", saveSettings);
  document.getElementById("btnTestPreflight").addEventListener("click", testPreflight);

  document.getElementById("providerSelect").addEventListener("change", (e) => {
    updateProviderModalHelper(e.target.value);
  });
}

// Render Skill Tags
function renderSkillTags() {
  const container = document.getElementById("skillsTagContainer");
  container.innerHTML = "";
  state.knownSkills.forEach((skill, idx) => {
    const tag = document.createElement("span");
    tag.className = "skill-tag";
    tag.innerHTML = `
      ${skill}
      <span class="remove" onclick="removeSkill(${idx})">✕</span>
    `;
    container.appendChild(tag);
  });
  document.getElementById("skillCountBadge").textContent = `${state.knownSkills.length} Skills`;
}

window.removeSkill = function (index) {
  state.knownSkills.splice(index, 1);
  renderSkillTags();
};

function addCustomSkill() {
  const input = document.getElementById("customSkillInput");
  const val = input.value.trim();
  if (val && !state.knownSkills.includes(val)) {
    state.knownSkills.push(val);
    renderSkillTags();
    input.value = "";
  }
}

// Render Path Selection
function renderPathSelection() {
  document.querySelectorAll(".path-chip").forEach((chip) => {
    const pathId = chip.dataset.path;
    const orderIdx = state.targetPaths.indexOf(pathId);
    const orderSpan = chip.querySelector(".path-chip-order");
    if (orderIdx > -1) {
      chip.classList.add("selected");
      orderSpan.textContent = `#${orderIdx + 1}`;
    } else {
      chip.classList.remove("selected");
      orderSpan.textContent = "";
    }
  });
}

// Apply Preset
function applyPreset(preset) {
  document.getElementById("studentName").value = preset.name;
  document.getElementById("studentBranch").value = preset.branch;
  document.getElementById("yearOfStudy").value = preset.year_of_study;
  document.getElementById("cgpaInput").value = preset.cgpa;
  document.getElementById("cgpaVal").textContent = preset.cgpa.toFixed(1);

  document.getElementById("runwayMonths").value = preset.runway;
  document.getElementById("runwayVal").textContent = `${preset.runway} Months (~${Math.round(preset.runway * 4.345)} Weeks)`;

  document.getElementById("weeklyHours").value = preset.hours;
  document.getElementById("weeklyHoursVal").textContent = `${preset.hours} hrs / week`;
  if (preset.hours >= 40) {
    document.getElementById("burnoutAlert").classList.add("active");
  } else {
    document.getElementById("burnoutAlert").classList.remove("active");
  }

  document.getElementById("constraintsInput").value = preset.constraints;

  state.knownSkills = [...preset.skills];
  renderSkillTags();

  state.targetPaths = [...preset.paths];
  renderPathSelection();
}

// Fetch LLM Provider Status
async function fetchLLMStatus() {
  try {
    const res = await fetch("/api/v1/config/llm");
    if (res.ok) {
      const data = await res.json();
      state.llmConfig = data;
      updateProviderBadge(data);
    }
  } catch (err) {
    console.warn("Could not fetch LLM config:", err);
  }
}

function updateProviderBadge(config) {
  const dot = document.getElementById("llmStatusDot");
  const txt = document.getElementById("providerStatusText");

  if (config.provider === "offline" || !config.is_configured) {
    dot.className = "status-indicator offline";
    txt.textContent = "⚡ Deterministic Engine (Offline)";
  } else {
    dot.className = "status-indicator";
    const keysCount = config.total_keys;
    const model = config.model;
    txt.textContent = `🟢 ${config.provider_name.split(" ")[0]} (${keysCount} Key${keysCount > 1 ? "s" : ""}) • ${model}`;
  }
}

// Run Full Diagnosis
async function runDiagnosis() {
  const emptyState = document.getElementById("emptyStateView");
  const loadingState = document.getElementById("loadingStateView");
  const resultsView = document.getElementById("resultsView");
  const btnSubmit = document.getElementById("btnSubmitDiagnose");

  emptyState.style.display = "none";
  resultsView.style.display = "none";
  loadingState.style.display = "block";
  btnSubmit.disabled = true;

  // Scroll right panel into view on mobile/after preset
  loadingState.scrollIntoView({ behavior: "smooth", block: "start" });

  const paths = state.targetPaths.map(p => p.replace("_", " ")).join(" + ");
  const loadingTitle = document.getElementById("loadingTitle");
  const loadingSubtext = document.getElementById("loadingSubtext");
  if (loadingTitle) loadingTitle.textContent = `Auditing ${paths} Multi-Path Matrix...`;
  if (loadingSubtext) loadingSubtext.textContent = `Analysing ${state.knownSkills.length} declared skills against ${paths} requirements`;

  const payload = {
    name: document.getElementById("studentName").value.trim() || "Anonymous",
    branch: document.getElementById("studentBranch").value.trim() || "Engineering",
    year_of_study: document.getElementById("yearOfStudy").value,
    cgpa: parseFloat(document.getElementById("cgpaInput").value),
    target_paths: state.targetPaths,
    known_skills: state.knownSkills,
    months_available: parseInt(document.getElementById("runwayMonths").value),
    weekly_hours: parseInt(document.getElementById("weeklyHours").value),
    constraints: document.getElementById("constraintsInput").value.trim() || null,
  };

  state.currentProfile = payload;

  try {
    const response = await fetch("/api/v1/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${response.status}`);  
    }

    const data = await response.json();
    state.latestDiagnosis = data;

    // Render results
    renderDiagnosisResults(data);

    loadingState.style.display = "none";
    resultsView.style.display = "flex";
    // Scroll results into view
    setTimeout(() => resultsView.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  } catch (err) {
    loadingState.style.display = "none";
    emptyState.style.display = "block";
    console.error("Diagnosis error:", err);
    alert("Error running diagnosis: " + err.message);
  } finally {
    btnSubmit.disabled = false;
  }
}

// Render Diagnosis Results
function renderDiagnosisResults(data) {
  const coverage = data.coverage || {};
  const primaryPath = data.primary_path || (Object.keys(coverage)[0] || state.targetPaths[0]);
  const primaryCoverage = coverage[primaryPath] || {};
  const primaryFeasibility = primaryCoverage.feasibility
    || (data.feasibility && data.feasibility[primaryPath])
    || "MODERATE";

  // 1. Verdict Banner
  const banner = document.getElementById("verdictBanner");
  banner.className = `verdict-banner ${primaryFeasibility.toLowerCase()}`;

  const iconMap = { HIGH: "✨", MODERATE: "⚡", CRITICAL_TIMELINE: "🚨" };
  document.getElementById("verdictIcon").textContent = iconMap[primaryFeasibility] || "🎯";

  const titleMap = {
    HIGH: "HIGH FEASIBILITY — EXCELLENT FIT",
    MODERATE: "MODERATE FEASIBILITY — ACTION REQUIRED",
    CRITICAL_TIMELINE: "CRITICAL TIMELINE — NARROW SLACK",
  };
  document.getElementById("verdictTitle").textContent = titleMap[primaryFeasibility] || primaryFeasibility;
  document.getElementById("primaryPathLabel").textContent = primaryPath.replace("_", " ");
  document.getElementById("executionModeBadge").textContent = data.mode === "AGENT" ? "AI Agent" : "Deterministic Engine";

  // Utilisation and Buffer
  const matrix = data.elimination_matrix || {};
  const utilisation = matrix.utilisation_percent ? `${matrix.utilisation_percent}%` : "75%";
  document.getElementById("statSaturation").textContent = utilisation;

  const buffer = primaryCoverage.buffer_months !== undefined ? `${primaryCoverage.buffer_months >= 0 ? "+" : ""}${primaryCoverage.buffer_months} mo` : "0.0 mo";
  document.getElementById("statBuffer").textContent = buffer;

  // 2. Per-Path Comparison Cards
  const cardsContainer = document.getElementById("pathCardsContainer");
  cardsContainer.innerHTML = "";

  Object.entries(coverage).forEach(([pathKey, cov]) => {
    const isPrimary = pathKey === primaryPath;
    const feasClass = (cov.feasibility || "MODERATE").toLowerCase();
    const pct = Math.round(cov.weighted_coverage_percent || 0);

    // SVG Circular gauge dashoffset: circumference ~ 188.5
    const circumference = 188.5;
    const offset = circumference - (pct / 100) * circumference;

    const card = document.createElement("div");
    card.className = `path-card ${isPrimary ? "is-primary" : ""}`;
    card.innerHTML = `
      <div class="path-card-header">
        <div class="path-title">${pathKey.replace("_", " ")} ${isPrimary ? "★" : ""}</div>
        <span class="badge-feasibility ${feasClass}">${cov.feasibility}</span>
      </div>

      <div class="gauge-container">
        <svg class="gauge-svg" viewBox="0 0 70 70">
          <circle class="gauge-bg" cx="35" cy="35" r="30"></circle>
          <circle class="gauge-progress" cx="35" cy="35" r="30" style="stroke-dashoffset: ${offset};"></circle>
        </svg>
        <div>
          <div class="gauge-text">${pct}%</div>
          <div class="gauge-label">Weighted Coverage</div>
        </div>
      </div>

      <div class="path-metrics">
        <div class="metric-box">
          <div class="metric-lbl">Est. Prep Hours</div>
          <div class="metric-val">${Math.round(cov.estimated_hours || 0)} hrs</div>
        </div>
        <div class="metric-box">
          <div class="metric-lbl">Months Req.</div>
          <div class="metric-val">${cov.months_required || "-"} mo</div>
        </div>
      </div>
    `;
    cardsContainer.appendChild(card);
  });

  // 3. Elimination Matrix Table
  const tbody = document.getElementById("matrixTableBody");
  tbody.innerHTML = "";

  const matrixRows = Array.isArray(matrix.elimination_matrix) ? matrix.elimination_matrix : [];
  if (matrixRows.length > 0) {
    const sharedCount = (matrix.skill_overlap && matrix.skill_overlap.shared_topic_count) || 0;
    matrixRows.forEach((row) => {
      const tr = document.createElement("tr");
      // The API returns `feasibility_alone` on each row and `verdict`
      // (KEEP_AS_PRIMARY / DROP_OR_DEFER), not `feasibility` / `action`.
      const feasibility = row.feasibility_alone || "MODERATE";
      const verdict = row.verdict || row.action || "";
      const verdictClass = /KEEP/.test(verdict) ? "keep" : (/DROP/.test(verdict) ? "drop" : "secondary");
      const weekly = row.required_weekly_hours != null ? row.required_weekly_hours : row.solo_weekly_hours;
      const overlap = row.skill_overlap_percent != null ? row.skill_overlap_percent : sharedCount;
      const penalty = row.conflict_penalty_percent != null ? row.conflict_penalty_percent : matrix.conflict_penalty_percent;
      tr.innerHTML = `
        <td><strong>${String(row.path || "").replace("_", " ")}</strong></td>
        <td><span class="badge-feasibility ${feasibility.toLowerCase()}">${feasibility}</span></td>
        <td>${weekly != null ? weekly : "-"}h</td>
        <td>${overlap != null ? overlap : "-"}</td>
        <td>${penalty != null ? penalty : "-"}%</td>
        <td><span class="badge-verdict ${verdictClass}">${verdict}</span></td>
      `;
      if (row.reason) tr.title = row.reason;
      tbody.appendChild(tr);
    });
  } else {
    // Fallback row
    state.targetPaths.forEach((path) => {
      const cov = coverage[path] || {};
      const fbFeasibility = cov.feasibility || "MODERATE";
      const months = (state.currentProfile && state.currentProfile.months_available) || 6;
      const weeklyHours = Math.round((cov.estimated_hours || 0) / (months * 4.345));
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${path.replace("_", " ")}</strong></td>
        <td><span class="badge-feasibility ${fbFeasibility.toLowerCase()}">${fbFeasibility}</span></td>
        <td>${weeklyHours > 0 ? weeklyHours : 15}h</td>
        <td>${Math.round(cov.coverage_percent || 30)}%</td>
        <td>0%</td>
        <td><span class="badge-verdict keep">KEEP PRIMARY</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Recommendation Text
  const recText = matrix.recommendation || (
    primaryFeasibility === "HIGH" 
      ? `Full steam ahead on ${primaryPath}. Your runway is well aligned with the estimated preparation effort.`
      : `Prioritize ${primaryPath} first to hit critical placement/exam milestones. Secondary paths should only receive weekend audit time to avoid timeline compression.`
  );
  document.getElementById("recommendationText").textContent = recText;

  // 4. Tab: Roadmap
  const timeline = document.getElementById("roadmapTimeline");
  timeline.innerHTML = "";

  if (data.roadmap && data.roadmap.length > 0) {
    data.roadmap.forEach((item, idx) => {
      const milestone = document.createElement("div");
      milestone.className = "milestone-item";
      
      let tierHtml = "";
      // `sequence` is an array of tier objects from the KB tool, each with
      // tier / tier_name / objective / modules / estimated_hours.
      if (Array.isArray(item.sequence)) {
        item.sequence.forEach((tier) => {
          const mods = Array.isArray(tier.modules) ? tier.modules : [];
          if (!mods.length) return;
          tierHtml += `<div class="tier-item"><strong>${tier.tier_name || tier.tier}:</strong> ${mods.slice(0, 3).join(", ")}</div>`;
        });
      } else if (item.sequence && typeof item.sequence === "object") {
        Object.entries(item.sequence).forEach(([tier, topics]) => {
          if (Array.isArray(topics) && topics.length > 0) {
            tierHtml += `<div class="tier-item"><strong>${tier}:</strong> ${topics.slice(0, 3).join(", ")}</div>`;
          }
        });
      }

      milestone.innerHTML = `
        <div class="milestone-dot"></div>
        <div class="milestone-header">
          <div class="milestone-title">Phase ${idx + 1}: ${item.label || item.topic}</div>
          <span class="milestone-hours">${item.total_hours} Hours</span>
        </div>
        <div style="font-size: 0.78rem; color: var(--text-dim);">Prerequisites: ${item.prerequisites && item.prerequisites.length ? item.prerequisites.join(", ") : "None (Foundational)"}</div>
        <div class="tier-list">${tierHtml}</div>
      `;
      timeline.appendChild(milestone);
    });
  } else {
    timeline.innerHTML = "<p style='color: var(--text-muted); font-size: 0.85rem;'>Roadmap generation complete in report view.</p>";
  }

  // 5. Tab: Skill Gaps & Prerequisites
  const masteredList = document.getElementById("masteredSkillsList");
  masteredList.innerHTML = "";
  state.knownSkills.forEach((s) => {
    const pill = document.createElement("div");
    pill.className = "gap-pill";
    pill.innerHTML = `<span>${s}</span><span style="color: var(--emerald); font-weight: bold;">Recognized</span>`;
    masteredList.appendChild(pill);
  });

  const missingList = document.getElementById("missingTopicsList");
  missingList.innerHTML = "";
  const missing = primaryCoverage.missing_topics || [];
  const blocking = primaryCoverage.blocking_topics || [];

  missing.slice(0, 8).forEach((t) => {
    const isBlocking = blocking.includes(t);
    const pill = document.createElement("div");
    pill.className = `gap-pill ${isBlocking ? "blocking" : ""}`;
    pill.innerHTML = `
      <span>${t}</span>
      <span style="font-size: 0.7rem; font-weight: 700;">${isBlocking ? "BLOCKING PREREQ" : "GAP"}</span>
    `;
    missingList.appendChild(pill);
  });

  // 6. Tab: Full Report
  const reportBox = document.getElementById("reportMarkdownContainer");
  reportBox.innerHTML = renderSimpleMarkdown(data.report);
}

// Simple Markdown Renderer
function renderSimpleMarkdown(md) {
  if (!md) return "<p>No report generated.</p>";
  let html = md
    .replace(/^### (.*$)/gim, "<h3>$1</h3>")
    .replace(/^## (.*$)/gim, "<h2>$1</h2>")
    .replace(/^# (.*$)/gim, "<h1>$1</h1>")
    .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/gim, "<em>$1</em>")
    .replace(/`([^`]+)`/gim, "<code>$1</code>")
    .replace(/^\s*-\s+(.*$)/gim, "<li>$1</li>")
    .replace(/\n\n/gim, "<br><br>");
  return html;
}

// Copilot Messaging
async function sendCopilotMessage() {
  const input = document.getElementById("copilotInput");
  const msg = input.value.trim();
  if (!msg) return;

  appendChatMessage("user", msg);
  input.value = "";

  const container = document.getElementById("copilotMessages");
  const loadingBubble = document.createElement("div");
  loadingBubble.className = "chat-bubble bot";
  loadingBubble.textContent = "Analyzing roadmap and thinking...";
  container.appendChild(loadingBubble);
  container.scrollTop = container.scrollHeight;

  try {
    const res = await fetch("/api/v1/copilot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        context: state.latestDiagnosis ? {
          primary_path: state.latestDiagnosis.primary_path,
          feasibility: state.latestDiagnosis.feasibility,
          coverage: state.latestDiagnosis.coverage,
          elimination_matrix: state.latestDiagnosis.elimination_matrix,
        } : null,
        chat_history: state.copilotHistory,
      }),
    });

    loadingBubble.remove();

    if (res.ok) {
      const data = await res.json();
      appendChatMessage("bot", data.reply);
      state.copilotHistory.push({ role: "user", content: msg });
      state.copilotHistory.push({ role: "assistant", content: data.reply });
    } else {
      appendChatMessage("bot", "I am currently running in offline assistance mode. Focus on completing your highest weighted topic gaps first!");
    }
  } catch (err) {
    loadingBubble.remove();
    appendChatMessage("bot", "Network error. Deterministic advice: Stick to the 70/30 time allocation between primary and secondary paths.");
  }
}

function appendChatMessage(role, text) {
  const container = document.getElementById("copilotMessages");
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.innerHTML = renderSimpleMarkdown(text);
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

// Settings Modal Management
function openSettingsModal() {
  const modal = document.getElementById("settingsModal");
  modal.classList.add("open");

  if (state.llmConfig) {
    document.getElementById("providerSelect").value = state.llmConfig.provider || "gemini";
    updateProviderModalHelper(state.llmConfig.provider || "gemini");
    renderRotationStatusTable(state.llmConfig.keys || []);
  }
}

function updateProviderModalHelper(provider) {
  const desc = document.getElementById("providerDescription");
  const link = document.getElementById("getKeyLink");
  const keySection = document.getElementById("keyInputSection");

  const helpers = {
    gemini: {
      desc: "Google AI Studio (Gemini 2.0 Flash) — 15 RPM, 1M TPM, 1,500 req/day completely free.",
      url: "https://aistudio.google.com/app/apikey",
      label: "👉 Get free Gemini API key (Google AI Studio)",
    },
    groq: {
      desc: "Groq Cloud (Llama 3.3 70B) — 500+ tokens/sec, 100% free tier with fast tool-calling.",
      url: "https://console.groq.com/keys",
      label: "👉 Get free Groq API key (Groq Cloud)",
    },
    openrouter: {
      desc: "OpenRouter (Free open-weights tier) — Multiple free frontier models.",
      url: "https://openrouter.ai/keys",
      label: "👉 Get free OpenRouter API key",
    },
    openai: {
      desc: "OpenAI Platform — GPT-4o mini (requires paid balance/credits).",
      url: "https://platform.openai.com/api-keys",
      label: "👉 OpenAI API Keys Dashboard",
    },
    offline: {
      desc: "Deterministic Mathematical Engine — 100% offline, 0 API keys required, zero latency.",
      url: "",
      label: "",
    },
  };

  const current = helpers[provider] || helpers.gemini;
  desc.textContent = current.desc;

  if (provider === "offline") {
    keySection.style.display = "none";
    link.style.display = "none";
  } else {
    keySection.style.display = "block";
    link.style.display = "inline-block";
    link.href = current.url;
    link.textContent = current.label;
  }
}

function renderRotationStatusTable(keys) {
  const tbody = document.getElementById("rotationStatusBody");
  tbody.innerHTML = "";

  if (!keys || keys.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-dim);">No external keys configured (Offline Engine active)</td></tr>`;
    return;
  }

  keys.forEach((k) => {
    const tr = document.createElement("tr");
    const statusColor = k.status === "active" ? "var(--emerald)" : "var(--amber)";
    tr.innerHTML = `
      <td>${k.index}</td>
      <td><code>${k.masked}</code> ${k.is_active ? "<strong>(Current)</strong>" : ""}</td>
      <td><span style="color: ${statusColor}; font-weight: 600;">${k.status}</span></td>
      <td>${k.uses}</td>
      <td>${k.cooldown_seconds > 0 ? `${k.cooldown_seconds}s` : "-"}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Save Settings
async function saveSettings() {
  const provider = document.getElementById("providerSelect").value;
  const rawKeys = document.getElementById("apiKeysInput").value.trim();

  try {
    const res = await fetch("/api/v1/config/llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: provider,
        keys: rawKeys,
      }),
    });

    if (res.ok) {
      const data = await res.json();
      state.llmConfig = data;
      updateProviderBadge(data);
      renderRotationStatusTable(data.keys || []);
      document.getElementById("settingsModal").classList.remove("open");
      alert(`Settings saved! Active provider: ${data.provider_name}`);
    } else {
      alert("Failed to save settings.");
    }
  } catch (err) {
    alert("Error updating settings: " + err.message);
  }
}

// Test Preflight
async function testPreflight() {
  const resultDiv = document.getElementById("preflightResult");
  resultDiv.style.display = "block";
  resultDiv.style.background = "rgba(99, 102, 241, 0.15)";
  resultDiv.style.border = "1px solid rgba(99, 102, 241, 0.4)";
  resultDiv.innerHTML = "Testing LLM round-trip and tool calling capability...";

  try {
    const res = await fetch("/api/v1/llm-check", { method: "POST" });
    const data = await res.json();

    if (data.ok) {
      resultDiv.style.background = "var(--emerald-bg)";
      resultDiv.style.border = "1px solid var(--emerald-border)";
      resultDiv.innerHTML = `<strong>✓ Preflight Succeeded!</strong> Roundtrip: ${data.elapsed_ms}ms • Provider: ${data.provider.name} • Model: ${data.provider.model} • Tool calling: OK!`;
    } else {
      resultDiv.style.background = "var(--rose-bg)";
      resultDiv.style.border = "1px solid var(--rose-border)";
      resultDiv.innerHTML = `<strong>✕ Preflight Notice:</strong> ${data.remedy || "Tool-calling failed. Offline fallback engine will be used automatically."}`;
    }
  } catch (err) {
    resultDiv.style.background = "var(--rose-bg)";
    resultDiv.style.border = "1px solid var(--rose-border)";
    resultDiv.innerHTML = `<strong>✕ Error:</strong> ${err.message}`;
  }
}
