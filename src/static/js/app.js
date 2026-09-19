/**
 * MythosAI - Local Fiction Co-Author Client Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // ---------------- State ----------------
    const state = {
        temperature: 0.70,
        top_p: 0.90,
        repeat_penalty: 1.10,
        top_k: 3,
        max_tokens: 512,
        selectedModel: "llama3.1:8b",
        streaming: true,
        loreList: [],
        isGenerating: false
    };

    // ---------------- DOM Elements ----------------
    // Status
    const chromaStatusDot = document.querySelector("#chromaStatusBadge .status-dot");
    const chromaStatusText = document.getElementById("chromaStatusText");
    const ollamaStatusDot = document.querySelector("#ollamaStatusBadge .status-dot");
    const ollamaStatusText = document.getElementById("ollamaStatusText");
    const modelSelector = document.getElementById("modelSelector");

    // Lorebook
    const loreContentInput = document.getElementById("loreContentInput");
    const loreCategoryInput = document.getElementById("loreCategoryInput");
    const addLoreBtn = document.getElementById("addLoreBtn");
    const loreSearchInput = document.getElementById("loreSearchInput");
    const seedSampleLoreBtn = document.getElementById("seedSampleLoreBtn");
    const clearLoreBtn = document.getElementById("clearLoreBtn");
    const loreListContainer = document.getElementById("loreListContainer");
    const loreCountBadge = document.getElementById("loreCountBadge");

    // Manuscript Editor
    const manuscriptText = document.getElementById("manuscriptText");
    const promptInput = document.getElementById("promptInput");
    const generateBtn = document.getElementById("generateBtn");
    const generateBtnText = document.getElementById("generateBtnText");
    const clearStoryBtn = document.getElementById("clearStoryBtn");
    const streamToggle = document.getElementById("streamToggle");
    const wordCount = document.getElementById("wordCount");
    const charCount = document.getElementById("charCount");

    // Parameter Controls
    const tempSlider = document.getElementById("tempSlider");
    const tempVal = document.getElementById("tempVal");
    const topPSlider = document.getElementById("topPSlider");
    const topPVal = document.getElementById("topPVal");
    const repPenaltySlider = document.getElementById("repPenaltySlider");
    const repPenaltyVal = document.getElementById("repPenaltyVal");
    const topKSlider = document.getElementById("topKSlider");
    const topKVal = document.getElementById("topKVal");
    const maxTokensSlider = document.getElementById("maxTokensSlider");
    const maxTokensVal = document.getElementById("maxTokensVal");
    const presetChips = document.querySelectorAll(".preset-chips .chip");

    // Tabs & Inspector
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const retrievedDocsList = document.getElementById("retrievedDocsList");
    const personaPreview = document.getElementById("personaPreview");
    const toastContainer = document.getElementById("toastContainer");

    // ---------------- Toast Notifications ----------------
    function showToast(message, duration = 3000) {
        const toast = document.createElement("div");
        toast.className = "toast";
        toast.textContent = message;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // ---------------- Health & Models ----------------
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                
                // Update Chroma status
                if (data.chromadb_connected) {
                    chromaStatusDot.classList.add("connected");
                    chromaStatusText.textContent = "Online";
                } else {
                    chromaStatusDot.classList.remove("connected");
                    chromaStatusText.textContent = "Offline";
                }

                // Update Ollama status
                if (data.ollama_connected) {
                    ollamaStatusDot.classList.add("connected");
                    ollamaStatusText.textContent = "Ready";
                } else {
                    ollamaStatusDot.classList.remove("connected");
                    ollamaStatusText.textContent = "Standalone/Mock";
                }
            }
        } catch (e) {
            chromaStatusDot.classList.remove("connected");
            chromaStatusText.textContent = "Error";
            ollamaStatusDot.classList.remove("connected");
            ollamaStatusText.textContent = "Error";
        }
    }

    async function loadModels() {
        try {
            const res = await fetch("/api/models");
            if (res.ok) {
                const data = await res.json();
                if (data.models && data.models.length > 0) {
                    modelSelector.innerHTML = "";
                    data.models.forEach(m => {
                        const opt = document.createElement("option");
                        opt.value = m.name;
                        opt.textContent = m.name;
                        modelSelector.appendChild(opt);
                    });
                    state.selectedModel = data.models[0].name;
                }
            }
        } catch (e) {
            console.warn("Could not fetch models list:", e);
        }
    }

    async function loadPersona() {
        try {
            const res = await fetch("/api/persona");
            if (res.ok) {
                const data = await res.json();
                personaPreview.textContent = data.persona || "No persona prompt configured.";
            }
        } catch (e) {
            personaPreview.textContent = "Error loading persona prompt.";
        }
    }

    // ---------------- Lorebook API Operations ----------------
    async function fetchLore() {
        try {
            const res = await fetch("/api/lore");
            if (res.ok) {
                const data = await res.json();
                state.loreList = data.items || [];
                renderLoreList();
            }
        } catch (e) {
            console.error("Failed to fetch lore:", e);
        }
    }

    function renderLoreList(filterText = "") {
        const query = filterText.toLowerCase().trim();
        const filtered = state.loreList.filter(item => 
            item.content.toLowerCase().includes(query) ||
            (item.metadata && JSON.stringify(item.metadata).toLowerCase().includes(query))
        );

        loreCountBadge.textContent = `${state.loreList.length} entries`;

        if (filtered.length === 0) {
            loreListContainer.innerHTML = `
                <div class="empty-state">
                    <p>${state.loreList.length === 0 ? "No canon entries yet. Add worldbuilding notes or click Seed Sample Canon." : "No matching lore found."}</p>
                </div>
            `;
            return;
        }

        loreListContainer.innerHTML = "";
        filtered.forEach(item => {
            const card = document.createElement("div");
            card.className = "lore-item-card";

            const tag = item.metadata?.category || item.metadata?.tag || "Canon";
            card.innerHTML = `
                <div class="lore-item-header">
                    <span class="lore-tag-chip">${tag}</span>
                    <button class="delete-lore-btn" data-id="${item.id}" title="Delete entry">&times;</button>
                </div>
                <div class="lore-item-text">${escapeHtml(item.content)}</div>
            `;
            loreListContainer.appendChild(card);
        });

        // Attach delete listeners
        loreListContainer.querySelectorAll(".delete-lore-btn").forEach(btn => {
            btn.addEventListener("click", async (e) => {
                const id = e.target.getAttribute("data-id");
                await deleteLore(id);
            });
        });
    }

    async function addLoreEntry(content, category = "General") {
        if (!content.trim()) {
            showToast("Please enter lore content.");
            return;
        }

        try {
            const res = await fetch("/api/lore", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    content: content.trim(),
                    metadata: { category: category.trim() || "General" }
                })
            });

            if (res.status === 201) {
                showToast("Lore added to vector database!");
                loreContentInput.value = "";
                loreCategoryInput.value = "";
                await fetchLore();
            } else {
                const err = await res.json();
                showToast(`Failed: ${err.detail || "Error adding lore"}`);
            }
        } catch (e) {
            showToast("Network error while adding lore.");
        }
    }

    async function deleteLore(id) {
        try {
            const res = await fetch(`/api/lore/${id}`, { method: "DELETE" });
            if (res.ok) {
                showToast("Lore entry deleted.");
                await fetchLore();
            }
        } catch (e) {
            showToast("Failed to delete lore.");
        }
    }

    async function clearAllLore() {
        if (!confirm("Are you sure you want to clear all lore canon from ChromaDB?")) return;
        try {
            const res = await fetch("/api/lore", { method: "DELETE" });
            if (res.ok) {
                showToast("Lorebook cleared.");
                await fetchLore();
            }
        } catch (e) {
            showToast("Failed to clear lorebook.");
        }
    }

    async function seedSampleLore() {
        const sampleLore = [
            {
                content: "The ancient sword is named 'Aethelgard' and it glows with a faint blue light whenever dark sorcery or danger is near.",
                category: "Artifact"
            },
            {
                content: "The Obsidian Citadel of Val-Theron was built three centuries ago atop the ruins of the Sunken Pyre by the Dragon-Priests.",
                category: "Location"
            },
            {
                content: "Commander Vance suffers from a cursed phantom scar across his left eye that burns when void beasts draw close.",
                category: "Character"
            },
            {
                content: "The Whispering Fog of the Ashen Hollows drains magical energy from standard relics unless shielded by silver runes.",
                category: "Magic/Rules"
            }
        ];

        for (const item of sampleLore) {
            await fetch("/api/lore", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    content: item.content,
                    metadata: { category: item.category }
                })
            });
        }
        showToast("Sample fantasy canon seeded!");
        await fetchLore();
    }

    // ---------------- Story Generation ----------------
    async function generateStory() {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            showToast("Please provide a prompt to continue the story.");
            promptInput.focus();
            return;
        }

        if (state.isGenerating) return;
        state.isGenerating = true;
        generateBtn.disabled = true;
        generateBtnText.textContent = "Writing...";

        const payload = {
            prompt: prompt,
            parameters: {
                temperature: state.temperature,
                top_p: state.top_p,
                repeat_penalty: state.repeat_penalty,
                max_tokens: state.max_tokens
            },
            top_k: state.top_k,
            model: state.selectedModel
        };

        try {
            if (state.streaming) {
                // Stream generation
                const res = await fetch("/api/generate/stream", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) throw new Error(`HTTP ${res.status}`);

                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                
                // Add separator if manuscript already has text
                if (manuscriptText.innerText.trim().length > 0) {
                    manuscriptText.innerText += "\n\n";
                }

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value, { stream: true });
                    manuscriptText.innerText += chunk;
                    updateMetrics();
                    manuscriptText.scrollTop = manuscriptText.scrollHeight;
                }

                promptInput.value = "";
                showToast("Generation complete.");
            } else {
                // Standard non-streaming generation
                const res = await fetch("/api/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                if (res.ok) {
                    const data = await res.json();
                    if (manuscriptText.innerText.trim().length > 0) {
                        manuscriptText.innerText += "\n\n";
                    }
                    manuscriptText.innerText += data.story_segment;
                    updateMetrics();
                    promptInput.value = "";

                    // Render retrieved docs in context tab
                    renderRetrievedDocs(data.retrieved_lore || []);
                    showToast("Generation complete.");
                } else {
                    const err = await res.json();
                    showToast(`Error: ${err.detail || "Generation failed"}`);
                }
            }
        } catch (e) {
            console.error(e);
            showToast("Failed to generate story segment.");
        } finally {
            state.isGenerating = false;
            generateBtn.disabled = false;
            generateBtnText.textContent = "Co-Author Next Scene";
        }
    }

    function renderRetrievedDocs(docs) {
        if (!docs || docs.length === 0) {
            retrievedDocsList.innerHTML = `
                <div class="empty-state">
                    <p>No lore was retrieved for the latest turn.</p>
                </div>
            `;
            return;
        }

        retrievedDocsList.innerHTML = "";
        docs.forEach((doc, idx) => {
            const card = document.createElement("div");
            card.className = "retrieved-doc-card";
            const score = doc.relevance_score ? (doc.relevance_score * 100).toFixed(1) : "N/A";
            card.innerHTML = `
                <div class="lore-item-header">
                    <span class="lore-tag-chip">Doc #${idx + 1}</span>
                    <span class="doc-relevance-badge">Relevance: ${score}%</span>
                </div>
                <div class="lore-item-text">${escapeHtml(doc.content)}</div>
            `;
            retrievedDocsList.appendChild(card);
        });
    }

    // ---------------- Metrics & UI Utilities ----------------
    function updateMetrics() {
        const text = manuscriptText.innerText.trim();
        const chars = text.length;
        const words = text ? text.split(/\s+/).length : 0;
        wordCount.textContent = `${words} words`;
        charCount.textContent = `${chars} characters`;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // ---------------- Event Listeners ----------------
    // Lore creation
    addLoreBtn.addEventListener("click", () => {
        addLoreEntry(loreContentInput.value, loreCategoryInput.value);
    });

    loreSearchInput.addEventListener("input", (e) => {
        renderLoreList(e.target.value);
    });

    seedSampleLoreBtn.addEventListener("click", seedSampleLore);
    clearLoreBtn.addEventListener("click", clearAllLore);

    // Generation
    generateBtn.addEventListener("click", generateStory);
    promptInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            generateStory();
        }
    });

    clearStoryBtn.addEventListener("click", () => {
        if (confirm("Clear manuscript content?")) {
            manuscriptText.innerText = "";
            updateMetrics();
        }
    });

    manuscriptText.addEventListener("input", updateMetrics);

    // Quick Prompts
    document.querySelectorAll(".quick-prompt").forEach(el => {
        el.addEventListener("click", () => {
            promptInput.value = el.getAttribute("data-prompt");
            promptInput.focus();
        });
    });

    // Model & Stream toggles
    modelSelector.addEventListener("change", (e) => {
        state.selectedModel = e.target.value;
    });

    streamToggle.addEventListener("change", (e) => {
        state.streaming = e.target.checked;
    });

    // Sliders
    tempSlider.addEventListener("input", (e) => {
        state.temperature = parseFloat(e.target.value);
        tempVal.textContent = state.temperature.toFixed(2);
    });

    topPSlider.addEventListener("input", (e) => {
        state.top_p = parseFloat(e.target.value);
        topPVal.textContent = state.top_p.toFixed(2);
    });

    repPenaltySlider.addEventListener("input", (e) => {
        state.repeat_penalty = parseFloat(e.target.value);
        repPenaltyVal.textContent = state.repeat_penalty.toFixed(2);
    });

    topKSlider.addEventListener("input", (e) => {
        state.top_k = parseInt(e.target.value, 10);
        topKVal.textContent = state.top_k;
    });

    maxTokensSlider.addEventListener("input", (e) => {
        state.max_tokens = parseInt(e.target.value, 10);
        maxTokensVal.textContent = state.max_tokens;
    });

    // Presets
    presetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            presetChips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");

            const t = parseFloat(chip.getAttribute("data-temp"));
            const p = parseFloat(chip.getAttribute("data-topp"));
            const r = parseFloat(chip.getAttribute("data-rep"));

            state.temperature = t;
            state.top_p = p;
            state.repeat_penalty = r;

            tempSlider.value = t;
            tempVal.textContent = t.toFixed(2);
            topPSlider.value = p;
            topPVal.textContent = p.toFixed(2);
            repPenaltySlider.value = r;
            repPenaltyVal.textContent = r.toFixed(2);

            showToast(`Applied preset: ${chip.textContent}`);
        });
    });

    // Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const target = document.getElementById(btn.getAttribute("data-tab"));
            if (target) target.classList.add("active");
        });
    });

    // ---------------- Initial Boot ----------------
    checkHealth();
    loadModels();
    loadPersona();
    fetchLore();
    setInterval(checkHealth, 10000);
});
