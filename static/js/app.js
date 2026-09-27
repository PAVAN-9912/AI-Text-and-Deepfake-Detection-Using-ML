/* ============================================================
   AI Text and Deepfake Detection Using ML
   Vanilla JavaScript Client Application - Result UX Refinements
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    initTabNavigation();
    initTextDetector();
    initVideoDetector();
    initAccordion();
    initHistory();
    fetchSystemHealth();
    fetchSystemMetrics();
});

// ============================================================
// 1. TAB NAVIGATION
// ============================================================

function initTabNavigation() {
    const navLinks = document.querySelectorAll(".nav-link");

    navLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const targetTab = link.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });

    // Hash navigation support
    const hash = window.location.hash.replace("#", "");
    if (hash && document.getElementById(hash)) {
        switchTab(hash);
    }
}

function switchTab(tabId) {
    const navLinks = document.querySelectorAll(".nav-link");
    const tabPanes = document.querySelectorAll(".tab-pane");

    navLinks.forEach(link => {
        if (link.getAttribute("data-tab") === tabId) {
            link.classList.add("active");
        } else {
            link.classList.remove("active");
        }
    });

    tabPanes.forEach(pane => {
        if (pane.id === tabId) {
            pane.classList.add("active");
        } else {
            pane.classList.remove("active");
        }
    });

    window.location.hash = tabId;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Global window reference for inline HTML clicks
window.switchTab = switchTab;


// ============================================================
// 2. SYSTEM HEALTH & METRICS
// ============================================================

async function fetchSystemHealth() {
    try {
        const response = await fetch("/api/health");
        const data = await response.json();

        const dot = document.getElementById("status-dot");
        const text = document.getElementById("status-text");
        const feats = document.getElementById("status-features");

        if (data.status === "healthy" && data.text_model && data.video_model) {
            dot.className = "status-dot green";
            text.textContent = "READY";
        } else {
            dot.className = "status-dot";
            text.textContent = "PARTIAL";
        }

        if (data.feature_count && feats) {
            feats.textContent = Number(data.feature_count).toLocaleString() + " FEATURES";
        }
    } catch (err) {
        console.warn("Could not retrieve system health:", err);
    }
}

async function fetchSystemMetrics() {
    try {
        const response = await fetch("/api/metrics");
        const data = await response.json();

        if (data.text_metrics) {
            const acc = document.getElementById("metric-text-acc");
            const f1 = document.getElementById("metric-text-f1");
            const auc = document.getElementById("metric-text-auc");
            const prec = document.getElementById("metric-text-prec");

            if (acc) acc.textContent = data.text_metrics.accuracy + "%";
            if (f1) f1.textContent = data.text_metrics.f1 + "%";
            if (auc) auc.textContent = data.text_metrics.roc_auc + "%";
            if (prec) prec.textContent = data.text_metrics.precision + "%";
        }

        if (data.video_metrics && typeof data.video_metrics === "object") {
            const acc = document.getElementById("metric-video-acc");
            const f1 = document.getElementById("metric-video-f1");
            const auc = document.getElementById("metric-video-auc");
            const prec = document.getElementById("metric-video-prec");

            if (acc) acc.textContent = data.video_metrics.accuracy + "%";
            if (f1) f1.textContent = data.video_metrics.f1 + "%";
            if (auc) auc.textContent = data.video_metrics.roc_auc + "%";
            if (prec) prec.textContent = data.video_metrics.precision + "%";
        }
    } catch (err) {
        console.warn("Could not retrieve metrics:", err);
    }
}


// ============================================================
// 3. TEXT DETECTION LOGIC
// ============================================================

const HUMAN_SAMPLE = "I forgot my laptop charger at home again. Could you please send me the notes from the morning lecture when you get a chance? Thanks!";
const AI_SAMPLE = "In summary, the rapid evolution of artificial intelligence represents a profound paradigm shift across modern technological frameworks, necessitating robust algorithmic architectures and multi-layered verification protocols.";

function initTextDetector() {
    const textInput = document.getElementById("text-input");
    const wordBadge = document.getElementById("word-count-badge");
    const charBadge = document.getElementById("char-count-badge");

    const btnAnalyze = document.getElementById("btn-analyze-text");
    const btnClear = document.getElementById("btn-clear-text");
    const btnPaste = document.getElementById("btn-paste-text");
    const btnPresetHuman = document.getElementById("preset-human");
    const btnPresetAi = document.getElementById("preset-ai");

    // Real-time counter & reset
    function updateCounters() {
        const val = textInput.value.trim();
        const words = val ? val.split(/\s+/).length : 0;
        const chars = textInput.value.length;

        wordBadge.textContent = `${words} word${words !== 1 ? 's' : ''}`;
        charBadge.textContent = `${chars} character${chars !== 1 ? 's' : ''}`;
    }

    textInput.addEventListener("input", () => {
        updateCounters();
        resetTextResultUI();
    });

    // Presets & Controls
    btnPresetHuman.addEventListener("click", () => {
        textInput.value = HUMAN_SAMPLE;
        updateCounters();
        resetTextResultUI();
    });

    btnPresetAi.addEventListener("click", () => {
        textInput.value = AI_SAMPLE;
        updateCounters();
        resetTextResultUI();
    });

    btnClear.addEventListener("click", () => {
        textInput.value = "";
        updateCounters();
        resetTextResultUI();
    });

    btnPaste.addEventListener("click", async () => {
        try {
            const clipText = await navigator.clipboard.readText();
            if (clipText) {
                textInput.value = clipText;
                updateCounters();
                resetTextResultUI();
            }
        } catch (err) {
            showTextError("Clipboard access permission denied. Please paste manually.");
        }
    });

    // Analyze Click
    btnAnalyze.addEventListener("click", () => analyzeText());
}

async function analyzeText() {
    const textInput = document.getElementById("text-input");
    const text = textInput.value.trim();

    if (!text) {
        showTextError("Please enter text before analyzing.");
        return;
    }

    const placeholder = document.getElementById("text-placeholder");
    const loading = document.getElementById("text-loading");
    const resultContent = document.getElementById("text-result-content");
    const errorState = document.getElementById("text-error-state");
    const stageTxt = document.getElementById("text-loading-stage");

    if (placeholder) placeholder.classList.add("hidden");
    if (resultContent) resultContent.classList.add("hidden");
    if (errorState) errorState.classList.add("hidden");
    if (loading) loading.classList.remove("hidden");

    // Concise, accurate processing stages
    const stages = [
        "Analyzing text...",
        "Extracting TF-IDF n-grams...",
        "Generating MiniLM embeddings...",
        "Calculating linguistic features...",
        "Evaluating classification verdict..."
    ];

    let stageIdx = 0;
    if (stageTxt) stageTxt.textContent = stages[0];
    const stageTimer = setInterval(() => {
        stageIdx = (stageIdx + 1) % stages.length;
        if (stageTxt) stageTxt.textContent = stages[stageIdx];
    }, 400);

    try {
        const response = await fetch("/api/analyze/text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        clearInterval(stageTimer);
        const data = await response.json();

        if (loading) loading.classList.add("hidden");

        if (!data.success) {
            showTextError(data.error || "Unable to analyze this text. Please try again.");
            return;
        }

        renderTextResult(data);
        saveHistoryEntry({
            media_type: "TEXT",
            result: data.result || "UNKNOWN",
            confidence: (data.confidence != null) ? `${data.confidence}%` : (data.result === "INSUFFICIENT_EVIDENCE" ? "Insufficient context" : (data.result === "UNCERTAIN" ? "Borderline" : "N/A")),
            details: `${data.word_count || 0} words`
        });

    } catch (err) {
        clearInterval(stageTimer);
        if (loading) loading.classList.add("hidden");
        showTextError("Server request failed: " + err.message);
    }
}

function showTextError(message) {
    const placeholder = document.getElementById("text-placeholder");
    const loading = document.getElementById("text-loading");
    const resultContent = document.getElementById("text-result-content");
    const errorState = document.getElementById("text-error-state");
    const errorMsg = document.getElementById("text-error-msg");

    if (placeholder) placeholder.classList.add("hidden");
    if (loading) loading.classList.add("hidden");
    if (resultContent) resultContent.classList.add("hidden");

    if (errorState) {
        if (errorMsg) errorMsg.textContent = message;
        errorState.classList.remove("hidden");
    }
}

function renderTextResult(data) {
    const resultContent = document.getElementById("text-result-content");
    const errorState = document.getElementById("text-error-state");
    const shortAlert = document.getElementById("short-text-alert");
    const alertIcon = document.getElementById("short-text-icon");
    const alertTitle = document.getElementById("short-text-title");
    const alertReason = document.getElementById("short-text-reason");

    const badge = document.getElementById("text-result-badge");
    const confidenceVal = document.getElementById("text-confidence-val");

    const probContainer = document.getElementById("text-prob-container");
    const probAi = document.getElementById("prob-ai-text");
    const probHuman = document.getElementById("prob-human-text");
    const barAi = document.getElementById("bar-ai-text");
    const barHuman = document.getElementById("bar-human-text");

    const resWords = document.getElementById("res-word-count");
    const resGroup = document.getElementById("res-length-group");
    const resThresh = document.getElementById("res-threshold");
    const resMethod = document.getElementById("res-method");

    const detLogWord = document.getElementById("det-log-word");
    const detSentences = document.getElementById("det-sentences");
    const detAvgSent = document.getElementById("det-avg-sentence");
    const detDiversity = document.getElementById("det-diversity");
    const detBucket = document.getElementById("det-bucket");

    if (!resultContent || !badge) return;

    if (errorState) errorState.classList.add("hidden");
    resultContent.classList.remove("hidden");

    // Extract exact API values
    const result = data.result || "UNKNOWN";
    const wordCount = (data.word_count != null) ? data.word_count : "--";
    const lengthGroup = data.length_group || "unknown";
    const threshold = (data.threshold != null) ? data.threshold : 0.41;
    const aiProb = (data.ai_probability != null) ? Number(data.ai_probability) : null;
    const humanProb = (data.human_probability != null) ? Number(data.human_probability) : null;
    const confidence = (data.confidence != null) ? Number(data.confidence) : null;

    // Set summary info
    if (resWords) resWords.textContent = wordCount;
    if (resGroup) resGroup.textContent = lengthGroup;
    if (resThresh) resThresh.textContent = threshold;

    // Set linguistic details if available
    if (data.linguistic_features) {
        if (detLogWord) detLogWord.textContent = data.linguistic_features.log_word_count ?? "--";
        if (detSentences) detSentences.textContent = data.linguistic_features.sentence_count ?? "--";
        if (detAvgSent) detAvgSent.textContent = data.linguistic_features.avg_sentence_length ?? "--";
        if (detDiversity) detDiversity.textContent = data.linguistic_features.vocabulary_diversity ?? "--";
        if (detBucket) detBucket.textContent = data.linguistic_features.length_bucket ?? "--";
    } else {
        if (detLogWord) detLogWord.textContent = "--";
        if (detSentences) detSentences.textContent = "--";
        if (detAvgSent) detAvgSent.textContent = "--";
        if (detDiversity) detDiversity.textContent = "--";
        if (detBucket) detBucket.textContent = "--";
    }

    // STATE 1: INSUFFICIENT EVIDENCE (1-2 words)
    if (result === "INSUFFICIENT_EVIDENCE") {
        if (shortAlert) {
            shortAlert.className = "alert alert-info";
            shortAlert.classList.remove("hidden");
            if (alertIcon) alertIcon.textContent = "ℹ️";
            if (alertTitle) alertTitle.textContent = "Insufficient Evidence";
            if (alertReason) alertReason.textContent = data.reason || "The input text is too brief for reliable AI detection.";
        }

        badge.textContent = "INSUFFICIENT EVIDENCE";
        badge.className = "badge insufficient";

        if (confidenceVal) confidenceVal.textContent = "N/A";
        if (probContainer) probContainer.classList.add("hidden");
        if (resMethod) resMethod.textContent = "Context Verification";
    }
    // STATE 2: UNCERTAIN / BORDERLINE
    else if (result === "UNCERTAIN") {
        if (shortAlert) {
            shortAlert.className = "alert alert-warning";
            shortAlert.classList.remove("hidden");
            if (alertIcon) alertIcon.textContent = "⚠️";
            if (alertTitle) alertTitle.textContent = "Uncertain / Borderline Prediction";
            if (alertReason) alertReason.textContent = data.reason || "The available text provides limited evidence, resulting in a borderline prediction.";
        }

        badge.textContent = "UNCERTAIN / BORDERLINE";
        badge.className = "badge uncertain";

        if (confidenceVal) confidenceVal.textContent = "Inconclusive";

        if (aiProb != null && humanProb != null) {
            if (probContainer) probContainer.classList.remove("hidden");
            if (probAi) probAi.textContent = `${aiProb.toFixed(2)}%`;
            if (probHuman) probHuman.textContent = `${humanProb.toFixed(2)}%`;
            if (barAi) barAi.style.width = `${Math.min(Math.max(aiProb, 0), 100)}%`;
            if (barHuman) barHuman.style.width = `${Math.min(Math.max(humanProb, 0), 100)}%`;
        } else {
            if (probContainer) probContainer.classList.add("hidden");
        }

        if (resMethod) resMethod.textContent = "ML Inference (Borderline Range)";
    }
    // STATE 3: AI-GENERATED
    else if (result === "AI-GENERATED") {
        if (data.is_short_text && data.reason) {
            if (shortAlert) {
                shortAlert.className = "alert alert-info";
                shortAlert.classList.remove("hidden");
                if (alertIcon) alertIcon.textContent = "ℹ️";
                if (alertTitle) alertTitle.textContent = "Short Text Notice";
                if (alertReason) alertReason.textContent = data.reason;
            }
        } else {
            if (shortAlert) shortAlert.classList.add("hidden");
        }

        badge.textContent = "AI-GENERATED";
        badge.className = "badge ai";

        if (confidenceVal) {
            confidenceVal.textContent = (confidence != null) ? `${confidence.toFixed(2)}%` : (aiProb != null ? `${aiProb.toFixed(2)}%` : "--");
        }

        if (probContainer) probContainer.classList.remove("hidden");
        if (probAi) probAi.textContent = `${aiProb != null ? aiProb.toFixed(2) : '--'}%`;
        if (probHuman) probHuman.textContent = `${humanProb != null ? humanProb.toFixed(2) : '--'}%`;
        if (barAi) barAi.style.width = `${aiProb != null ? Math.min(Math.max(aiProb, 0), 100) : 0}%`;
        if (barHuman) barHuman.style.width = `${humanProb != null ? Math.min(Math.max(humanProb, 0), 100) : 0}%`;

        if (resMethod) resMethod.textContent = "TextDetector_V3.4 (194K)";
    }
    // STATE 4: HUMAN-WRITTEN
    else {
        if (data.is_short_text && data.reason) {
            if (shortAlert) {
                shortAlert.className = "alert alert-info";
                shortAlert.classList.remove("hidden");
                if (alertIcon) alertIcon.textContent = "ℹ️";
                if (alertTitle) alertTitle.textContent = "Short Text Notice";
                if (alertReason) alertReason.textContent = data.reason;
            }
        } else {
            if (shortAlert) shortAlert.classList.add("hidden");
        }

        badge.textContent = "HUMAN-WRITTEN";
        badge.className = "badge human";

        if (confidenceVal) {
            confidenceVal.textContent = (confidence != null) ? `${confidence.toFixed(2)}%` : (humanProb != null ? `${humanProb.toFixed(2)}%` : "--");
        }

        if (probContainer) probContainer.classList.remove("hidden");
        if (probAi) probAi.textContent = `${aiProb != null ? aiProb.toFixed(2) : '--'}%`;
        if (probHuman) probHuman.textContent = `${humanProb != null ? humanProb.toFixed(2) : '--'}%`;
        if (barAi) barAi.style.width = `${aiProb != null ? Math.min(Math.max(aiProb, 0), 100) : 0}%`;
        if (barHuman) barHuman.style.width = `${humanProb != null ? Math.min(Math.max(humanProb, 0), 100) : 0}%`;

        if (resMethod) resMethod.textContent = "TextDetector_V3.4 (194K)";
    }
}

function resetTextResultUI() {
    const placeholder = document.getElementById("text-placeholder");
    const resultContent = document.getElementById("text-result-content");
    const loading = document.getElementById("text-loading");
    const errorState = document.getElementById("text-error-state");
    const barAi = document.getElementById("bar-ai-text");
    const barHuman = document.getElementById("bar-human-text");

    if (placeholder) placeholder.classList.remove("hidden");
    if (resultContent) resultContent.classList.add("hidden");
    if (loading) loading.classList.add("hidden");
    if (errorState) errorState.classList.add("hidden");

    if (barAi) barAi.style.width = "0%";
    if (barHuman) barHuman.style.width = "0%";
}
window.resetTextResultUI = resetTextResultUI;


// ============================================================
// 4. VIDEO DETECTION LOGIC
// ============================================================

let selectedVideoFile = null;

function initVideoDetector() {
    const dropzone = document.getElementById("video-dropzone");
    const fileInput = document.getElementById("video-file-input");
    const btnAnalyzeVideo = document.getElementById("btn-analyze-video");
    const btnRemoveVideo = document.getElementById("btn-remove-video");

    dropzone.addEventListener("click", (e) => {
        if (e.target !== btnRemoveVideo && !btnRemoveVideo.contains(e.target)) {
            fileInput.click();
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleVideoSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleVideoSelect(e.target.files[0]);
        }
    });

    btnRemoveVideo.addEventListener("click", (e) => {
        e.stopPropagation();
        removeVideoFile();
    });

    btnAnalyzeVideo.addEventListener("click", () => analyzeVideo());
}

function handleVideoSelect(file) {
    const validExts = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
    const fileExt = "." + file.name.split('.').pop().toLowerCase();

    if (!validExts.includes(fileExt)) {
        showVideoError("Invalid file format. Please upload an MP4, MOV, AVI, MKV, or WEBM video.");
        return;
    }

    if (file.size > 100 * 1024 * 1024) {
        showVideoError("File size exceeds 100MB limit.");
        return;
    }

    selectedVideoFile = file;

    const promptBox = document.getElementById("dropzone-prompt");
    const previewBox = document.getElementById("video-preview-box");
    const videoPlayer = document.getElementById("video-player");
    const fileNameTxt = document.getElementById("file-name-txt");
    const fileSizeTxt = document.getElementById("file-size-txt");
    const btnAnalyze = document.getElementById("btn-analyze-video");

    if (fileNameTxt) fileNameTxt.textContent = file.name;
    if (fileSizeTxt) fileSizeTxt.textContent = (file.size / (1024 * 1024)).toFixed(1) + " MB";

    if (videoPlayer) videoPlayer.src = URL.createObjectURL(file);

    if (promptBox) promptBox.classList.add("hidden");
    if (previewBox) previewBox.classList.remove("hidden");
    if (btnAnalyze) btnAnalyze.disabled = false;

    // Reset previous analysis states immediately
    resetVideoResultUI();
}

function removeVideoFile() {
    selectedVideoFile = null;
    const fileInput = document.getElementById("video-file-input");
    if (fileInput) fileInput.value = "";

    const promptBox = document.getElementById("dropzone-prompt");
    const previewBox = document.getElementById("video-preview-box");
    const videoPlayer = document.getElementById("video-player");
    const btnAnalyze = document.getElementById("btn-analyze-video");

    if (videoPlayer) videoPlayer.src = "";
    if (previewBox) previewBox.classList.add("hidden");
    if (promptBox) promptBox.classList.remove("hidden");
    if (btnAnalyze) btnAnalyze.disabled = true;

    resetVideoResultUI();
}

async function analyzeVideo() {
    if (!selectedVideoFile) return;

    const placeholder = document.getElementById("video-placeholder");
    const loading = document.getElementById("video-loading");
    const resultContent = document.getElementById("video-result-content");
    const errorState = document.getElementById("video-error-state");
    const stageTxt = document.getElementById("video-loading-stage");

    if (placeholder) placeholder.classList.add("hidden");
    if (resultContent) resultContent.classList.add("hidden");
    if (errorState) errorState.classList.add("hidden");
    if (loading) loading.classList.remove("hidden");

    const stages = [
        "Analyzing video...",
        "Extracting 8 uniform temporal frames...",
        "Detecting facial landmarks with YuNet...",
        "Extracting ResNet-18 spatial features...",
        "Evaluating Temporal Conv1D + BiGRU Attention..."
    ];

    let stageIdx = 0;
    if (stageTxt) stageTxt.textContent = stages[0];
    const stageTimer = setInterval(() => {
        stageIdx = (stageIdx + 1) % stages.length;
        if (stageTxt) stageTxt.textContent = stages[stageIdx];
    }, 600);

    try {
        const formData = new FormData();
        formData.append("file", selectedVideoFile);

        const response = await fetch("/api/analyze/video", {
            method: "POST",
            body: formData
        });

        clearInterval(stageTimer);
        const data = await response.json();

        if (loading) loading.classList.add("hidden");

        if (!data.success) {
            showVideoError(data.error || "Unable to analyze this video. Please check the file and try again.");
            return;
        }

        renderVideoResult(data);
        saveHistoryEntry({
            media_type: "VIDEO",
            result: data.result,
            confidence: `${data.confidence}%`,
            details: selectedVideoFile.name
        });

    } catch (err) {
        clearInterval(stageTimer);
        if (loading) loading.classList.add("hidden");
        showVideoError("Video analysis failed: " + err.message);
    }
}

function showVideoError(message) {
    const placeholder = document.getElementById("video-placeholder");
    const loading = document.getElementById("video-loading");
    const resultContent = document.getElementById("video-result-content");
    const errorState = document.getElementById("video-error-state");
    const errorMsg = document.getElementById("video-error-msg");

    if (placeholder) placeholder.classList.add("hidden");
    if (loading) loading.classList.add("hidden");
    if (resultContent) resultContent.classList.add("hidden");

    if (errorState) {
        if (errorMsg) errorMsg.textContent = message;
        errorState.classList.remove("hidden");
    }
}

function renderVideoResult(data) {
    const resultContent = document.getElementById("video-result-content");
    const errorState = document.getElementById("video-error-state");
    const badge = document.getElementById("video-result-badge");
    const confidenceVal = document.getElementById("video-confidence-val");

    const probReal = document.getElementById("prob-real-video");
    const probFake = document.getElementById("prob-fake-video");
    const barReal = document.getElementById("bar-real-video");
    const barFake = document.getElementById("bar-fake-video");

    const resFrames = document.getElementById("res-video-frames");
    const resThreshold = document.getElementById("res-video-threshold");

    const decisionAlert = document.getElementById("video-decision-alert");
    const decisionIcon = document.getElementById("video-decision-icon");
    const decisionTitle = document.getElementById("video-decision-title");
    const decisionReason = document.getElementById("video-decision-reason");

    if (!resultContent || !badge) return;

    if (errorState) errorState.classList.add("hidden");
    resultContent.classList.remove("hidden");

    // Standardize result check directly from the API result field
    const rawResult = String(data.result || "").toUpperCase().trim();
    const isReal = (rawResult === "REAL" || rawResult === "REAL (AUTHENTIC)" || rawResult === "REAL / HUMAN");

    const realProb = Number(data.real_probability);
    const fakeProb = Number(data.fake_probability);
    const confidence = Number(data.confidence);

    // Dynamic threshold extraction
    const rawThresh = data.threshold != null ? Number(data.threshold) : 0.32;
    const threshPct = rawThresh <= 1.0 ? (rawThresh * 100).toFixed(2) : rawThresh.toFixed(2);

    if (isReal) {
        badge.textContent = "REAL (AUTHENTIC)";
        badge.className = "badge real";
        if (decisionAlert) {
            decisionAlert.className = "alert alert-info";
            decisionAlert.classList.remove("hidden");
            if (decisionIcon) decisionIcon.textContent = "ℹ️";
            if (decisionTitle) decisionTitle.textContent = "Calibrated Decision (Authentic)";
            if (decisionReason) {
                decisionReason.textContent = `V4 Decision: Real probability (${realProb.toFixed(2)}%) meets or exceeds the calibrated real threshold (${threshPct}%).`;
            }
        }
    } else {
        badge.textContent = "FAKE / DEEPFAKE";
        badge.className = "badge fake";
        if (decisionAlert) {
            decisionAlert.className = "alert alert-warning";
            decisionAlert.classList.remove("hidden");
            if (decisionIcon) decisionIcon.textContent = "⚠️";
            if (decisionTitle) decisionTitle.textContent = "Calibrated Decision (Deepfake)";
            if (decisionReason) {
                decisionReason.textContent = `V4 Decision: Real probability (${realProb.toFixed(2)}%) is below the calibrated real threshold (${threshPct}%).`;
            }
        }
    }

    if (confidenceVal) {
        confidenceVal.textContent = (!isNaN(confidence) ? confidence.toFixed(2) : "--") + "%";
    }

    if (probReal) probReal.textContent = (!isNaN(realProb) ? realProb.toFixed(2) : "--") + "%";
    if (probFake) probFake.textContent = (!isNaN(fakeProb) ? fakeProb.toFixed(2) : "--") + "%";
    if (barReal) barReal.style.width = Math.min(Math.max(!isNaN(realProb) ? realProb : 0, 0), 100) + "%";
    if (barFake) barFake.style.width = Math.min(Math.max(!isNaN(fakeProb) ? fakeProb : 0, 0), 100) + "%";

    if (resFrames) resFrames.textContent = `${data.frames_analyzed || 8} Frames`;
    if (resThreshold) resThreshold.textContent = `${threshPct}%`;
}

function resetVideoResultUI() {
    const placeholder = document.getElementById("video-placeholder");
    const resultContent = document.getElementById("video-result-content");
    const loading = document.getElementById("video-loading");
    const errorState = document.getElementById("video-error-state");
    const barReal = document.getElementById("bar-real-video");
    const barFake = document.getElementById("bar-fake-video");

    if (placeholder) placeholder.classList.remove("hidden");
    if (resultContent) resultContent.classList.add("hidden");
    if (loading) loading.classList.add("hidden");
    if (errorState) errorState.classList.add("hidden");

    if (barReal) barReal.style.width = "0%";
    if (barFake) barFake.style.width = "0%";
}
window.resetVideoResultUI = resetVideoResultUI;


// ============================================================
// 5. ACCORDION
// ============================================================

function initAccordion() {
    const toggleBtn = document.getElementById("accordion-toggle");
    const body = document.getElementById("accordion-body");

    if (toggleBtn && body) {
        toggleBtn.addEventListener("click", () => {
            body.classList.toggle("hidden");
            const icon = toggleBtn.querySelector(".accordion-icon");
            if (icon) icon.textContent = body.classList.contains("hidden") ? "▼" : "▲";
        });
    }
}


// ============================================================
// 6. LOCAL STORAGE HISTORY
// ============================================================

const HISTORY_KEY = "ai_detection_history";

function initHistory() {
    const btnClearHistory = document.getElementById("btn-clear-history");
    if (btnClearHistory) {
        btnClearHistory.addEventListener("click", () => {
            if (confirm("Are you sure you want to clear your analysis session history?")) {
                localStorage.removeItem(HISTORY_KEY);
                renderHistoryTable();
            }
        });
    }

    renderHistoryTable();
}

function saveHistoryEntry(entry) {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const fullEntry = {
        time: timestamp,
        media_type: entry.media_type,
        result: entry.result,
        confidence: entry.confidence,
        details: entry.details
    };

    let history = [];
    try {
        const stored = localStorage.getItem(HISTORY_KEY);
        if (stored) history = JSON.parse(stored);
    } catch (e) { history = []; }

    history.unshift(fullEntry);
    if (history.length > 20) history = history.slice(0, 20);

    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistoryTable();
}

function renderHistoryTable() {
    const tbody = document.getElementById("history-table-body");
    if (!tbody) return;

    let history = [];
    try {
        const stored = localStorage.getItem(HISTORY_KEY);
        if (stored) history = JSON.parse(stored);
    } catch (e) { history = []; }

    if (history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-history">No analysis history recorded yet in this session.</td></tr>`;
        return;
    }

    tbody.innerHTML = history.map(item => {
        let badgeClass = "badge";
        const res = String(item.result || "").toUpperCase();
        if (res === "HUMAN-WRITTEN" || res === "REAL" || res === "REAL (AUTHENTIC)") {
            badgeClass += " human";
        } else if (res === "AI-GENERATED" || res === "FAKE" || res === "FAKE / DEEPFAKE") {
            badgeClass += " ai";
        } else {
            badgeClass += " insufficient";
        }

        return `
            <tr>
                <td>${item.time}</td>
                <td><strong>${item.media_type}</strong></td>
                <td>${item.details || '--'}</td>
                <td><span class="${badgeClass}">${item.result}</span></td>
                <td><strong>${item.confidence}</strong></td>
            </tr>
        `;
    }).join("");
}
