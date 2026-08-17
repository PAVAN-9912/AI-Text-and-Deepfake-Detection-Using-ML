/* ============================================================
   AI-Text-and-Deepfake-Detection-Using-ML
   Vanilla JavaScript Client Application
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
            text.textContent = "All Engines Online";
        } else {
            dot.className = "status-dot";
            text.textContent = "Partial System Ready";
        }

        if (data.feature_count) {
            feats.textContent = Number(data.feature_count).toLocaleString();
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
            document.getElementById("metric-text-acc").textContent = data.text_metrics.accuracy + "%";
            document.getElementById("metric-text-f1").textContent = data.text_metrics.f1 + "%";
            document.getElementById("metric-text-auc").textContent = data.text_metrics.roc_auc + "%";
            document.getElementById("metric-text-prec").textContent = data.text_metrics.precision + "%";
        }

        if (data.video_metrics && typeof data.video_metrics === "object") {
            document.getElementById("metric-video-acc").textContent = data.video_metrics.accuracy + "%";
            document.getElementById("metric-video-f1").textContent = data.video_metrics.f1 + "%";
            document.getElementById("metric-video-auc").textContent = data.video_metrics.roc_auc + "%";
            document.getElementById("metric-video-prec").textContent = data.video_metrics.precision + "%";
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

    // Real-time counter
    function updateCounters() {
        const val = textInput.value.trim();
        const words = val ? val.split(/\s+/).length : 0;
        const chars = textInput.value.length;

        wordBadge.textContent = `${words} word${words !== 1 ? 's' : ''}`;
        charBadge.textContent = `${chars} character${chars !== 1 ? 's' : ''}`;
    }

    textInput.addEventListener("input", updateCounters);

    // Presets & Controls
    btnPresetHuman.addEventListener("click", () => {
        textInput.value = HUMAN_SAMPLE;
        updateCounters();
    });

    btnPresetAi.addEventListener("click", () => {
        textInput.value = AI_SAMPLE;
        updateCounters();
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
            }
        } catch (err) {
            alert("Clipboard access permission denied.");
        }
    });

    // Analyze Click
    btnAnalyze.addEventListener("click", () => analyzeText());
}

async function analyzeText() {
    const textInput = document.getElementById("text-input");
    const text = textInput.value.trim();

    if (!text) {
        alert("Please enter text before analyzing.");
        return;
    }

    const placeholder = document.getElementById("text-placeholder");
    const loading = document.getElementById("text-loading");
    const resultContent = document.getElementById("text-result-content");
    const stageTxt = document.getElementById("text-loading-stage");

    placeholder.classList.add("hidden");
    resultContent.classList.add("hidden");
    loading.classList.remove("hidden");

    // UI Processing Animation Stages
    const stages = [
        "Preparing Text...",
        "Extracting TF-IDF Features...",
        "Creating MiniLM Embeddings...",
        "Building Linguistic Features...",
        "Running Logistic Regression Classifier..."
    ];

    let stageIdx = 0;
    const stageTimer = setInterval(() => {
        stageIdx = (stageIdx + 1) % stages.length;
        stageTxt.textContent = stages[stageIdx];
    }, 400);

    try {
        const response = await fetch("/api/analyze/text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        clearInterval(stageTimer);
        const data = await response.json();

        loading.classList.add("hidden");

        if (!data.success) {
            alert(data.error || "Text analysis failed.");
            placeholder.classList.remove("hidden");
            return;
        }

        renderTextResult(data);
        saveHistoryEntry({
            media_type: "TEXT",
            result: data.result,
            confidence: data.confidence ? `${data.confidence}%` : "Rule-based",
            details: `${data.word_count} words`
        });

    } catch (err) {
        clearInterval(stageTimer);
        loading.classList.add("hidden");
        placeholder.classList.remove("hidden");
        alert("Server request failed: " + err.message);
    }
}

function renderTextResult(data) {
    const resultContent = document.getElementById("text-result-content");
    const shortAlert = document.getElementById("short-text-alert");
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

    resultContent.classList.remove("hidden");

    if (data.is_short_text) {
        // Short text rule presentation
        shortAlert.classList.remove("hidden");
        document.getElementById("short-text-reason").textContent = data.reason;

        badge.textContent = "HUMAN-WRITTEN";
        badge.className = "badge rule";
        confidenceVal.textContent = "Rule-based";

        probContainer.classList.add("hidden");

        resWords.textContent = data.word_count;
        resGroup.textContent = "very_short";
        resThresh.textContent = "N/A";
        resMethod.textContent = "Rule-based (<=5 words)";

        detLogWord.textContent = "--";
        detSentences.textContent = "--";
        detAvgSent.textContent = "--";
        detDiversity.textContent = "--";
        detBucket.textContent = "--";
    } else {
        // Full ML model presentation
        shortAlert.classList.add("hidden");
        probContainer.classList.remove("hidden");

        if (data.result === "AI-GENERATED") {
            badge.textContent = "AI-GENERATED";
            badge.className = "badge ai";
        } else {
            badge.textContent = "HUMAN-WRITTEN";
            badge.className = "badge human";
        }

        confidenceVal.textContent = data.confidence + "%";

        probAi.textContent = data.ai_probability + "%";
        probHuman.textContent = data.human_probability + "%";
        barAi.style.width = data.ai_probability + "%";
        barHuman.style.width = data.human_probability + "%";

        resWords.textContent = data.word_count;
        resGroup.textContent = data.length_group;
        resThresh.textContent = data.threshold;
        resMethod.textContent = "V2 Improved ML Model";

        if (data.linguistic_features) {
            detLogWord.textContent = data.linguistic_features.log_word_count;
            detSentences.textContent = data.linguistic_features.sentence_count;
            detAvgSent.textContent = data.linguistic_features.avg_sentence_length;
            detDiversity.textContent = data.linguistic_features.vocabulary_diversity;
            detBucket.textContent = data.linguistic_features.length_bucket;
        }
    }
}

function resetTextResultUI() {
    document.getElementById("text-placeholder").classList.remove("hidden");
    document.getElementById("text-result-content").classList.add("hidden");
    document.getElementById("text-loading").classList.add("hidden");
}


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
        alert("Invalid file format. Please upload an MP4, MOV, AVI, MKV, or WEBM video.");
        return;
    }

    if (file.size > 100 * 1024 * 1024) {
        alert("File size exceeds 100MB limit.");
        return;
    }

    selectedVideoFile = file;

    const promptBox = document.getElementById("dropzone-prompt");
    const previewBox = document.getElementById("video-preview-box");
    const videoPlayer = document.getElementById("video-player");
    const fileNameTxt = document.getElementById("file-name-txt");
    const fileSizeTxt = document.getElementById("file-size-txt");
    const btnAnalyze = document.getElementById("btn-analyze-video");

    fileNameTxt.textContent = file.name;
    fileSizeTxt.textContent = (file.size / (1024 * 1024)).toFixed(1) + " MB";

    videoPlayer.src = URL.createObjectURL(file);

    promptBox.classList.add("hidden");
    previewBox.classList.remove("hidden");
    btnAnalyze.disabled = false;
}

function removeVideoFile() {
    selectedVideoFile = null;
    const fileInput = document.getElementById("video-file-input");
    fileInput.value = "";

    const promptBox = document.getElementById("dropzone-prompt");
    const previewBox = document.getElementById("video-preview-box");
    const videoPlayer = document.getElementById("video-player");
    const btnAnalyze = document.getElementById("btn-analyze-video");

    videoPlayer.src = "";
    previewBox.classList.add("hidden");
    promptBox.classList.remove("hidden");
    btnAnalyze.disabled = true;

    document.getElementById("video-placeholder").classList.remove("hidden");
    document.getElementById("video-result-content").classList.add("hidden");
    document.getElementById("video-loading").classList.add("hidden");
}

async function analyzeVideo() {
    if (!selectedVideoFile) return;

    const placeholder = document.getElementById("video-placeholder");
    const loading = document.getElementById("video-loading");
    const resultContent = document.getElementById("video-result-content");
    const stageTxt = document.getElementById("video-loading-stage");

    placeholder.classList.add("hidden");
    resultContent.classList.add("hidden");
    loading.classList.remove("hidden");

    const stages = [
        "Loading Video File...",
        "Extracting 8 Uniform Frames...",
        "Extracting ResNet-18 Spatial Features...",
        "Running BiLSTM Temporal Attention Model...",
        "Evaluating Deepfake Probability..."
    ];

    let stageIdx = 0;
    const stageTimer = setInterval(() => {
        stageIdx = (stageIdx + 1) % stages.length;
        stageTxt.textContent = stages[stageIdx];
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

        loading.classList.add("hidden");

        if (!data.success) {
            alert(data.error || "Video analysis failed.");
            placeholder.classList.remove("hidden");
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
        loading.classList.add("hidden");
        placeholder.classList.remove("hidden");
        alert("Video server request failed: " + err.message);
    }
}

function renderVideoResult(data) {
    const resultContent = document.getElementById("video-result-content");
    const badge = document.getElementById("video-result-badge");
    const confidenceVal = document.getElementById("video-confidence-val");

    const probReal = document.getElementById("prob-real-video");
    const probFake = document.getElementById("prob-fake-video");
    const barReal = document.getElementById("bar-real-video");
    const barFake = document.getElementById("bar-fake-video");

    const resFrames = document.getElementById("res-video-frames");

    resultContent.classList.remove("hidden");

    if (data.result === "REAL") {
        badge.textContent = "REAL (AUTHENTIC)";
        badge.className = "badge real";
    } else {
        badge.textContent = "FAKE / DEEPFAKE";
        badge.className = "badge fake";
    }

    confidenceVal.textContent = data.confidence + "%";

    probReal.textContent = data.real_probability + "%";
    probFake.textContent = data.fake_probability + "%";
    barReal.style.width = data.real_probability + "%";
    barFake.style.width = data.fake_probability + "%";

    resFrames.textContent = `${data.frames_analyzed} Frames`;
}


// ============================================================
// 5. ACCORDION
// ============================================================

function initAccordion() {
    const toggleBtn = document.getElementById("accordion-toggle");
    const body = document.getElementById("accordion-body");

    toggleBtn.addEventListener("click", () => {
        body.classList.toggle("hidden");
        const icon = toggleBtn.querySelector(".accordion-icon");
        icon.textContent = body.classList.contains("hidden") ? "▼" : "▲";
    });
}


// ============================================================
// 6. LOCAL STORAGE HISTORY
// ============================================================

const HISTORY_KEY = "ai_detection_history";

function initHistory() {
    const btnClearHistory = document.getElementById("btn-clear-history");
    btnClearHistory.addEventListener("click", () => {
        if (confirm("Are you sure you want to clear your analysis session history?")) {
            localStorage.removeItem(HISTORY_KEY);
            renderHistoryTable();
        }
    });

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
    let history = [];
    try {
        const stored = localStorage.getItem(HISTORY_KEY);
        if (stored) history = JSON.parse(stored);
    } catch (e) { history = []; }

    if (history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-history">No analysis history recorded yet.</td></tr>`;
        return;
    }

    tbody.innerHTML = history.map(item => {
        let badgeClass = "badge";
        if (item.result === "HUMAN-WRITTEN" || item.result === "REAL") badgeClass += " human";
        else if (item.result === "AI-GENERATED" || item.result === "FAKE") badgeClass += " ai";

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
